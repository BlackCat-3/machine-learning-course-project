"""Fixed-update four-fold study and gated full-data residual refinement."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from contracts import coordinate_metrics, read_submission, write_submission
from refinement_study import read_json, sha256
from robust_heatmap import RobustHeatmap, update_ema
from run import ROOT, save_json
from residual_refinement import ResidualRefiner, patch_batch, to_original, loss_original, PATCH
from prepare_residual_v4 import (OUT, ROBUST, MANIFEST, BEST, BEST_HASH, setup,
                                 infer_context, native_cache)

PRETRAINED = ROOT.parent / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
SEEDS = (43, 44)
STEPS = (40, 80, 120, 160, 240)


def load_cache(path, records):
    z = np.load(path)
    if list(z['ids']) != [r['id'] for r in records]:
        raise ValueError('Native crop cache ID order mismatch')
    crops = np.asarray([r['crop'] for r in records], dtype=np.float32)
    return {
        'patches': torch.from_numpy(z['patches'].transpose(0, 3, 1, 2).copy()).cuda(),
        'context': torch.from_numpy(z['context'].copy()).cuda(),
        'centers': torch.tensor(z['centers'], dtype=torch.float32, device='cuda'),
        'origins': torch.tensor(z['origins'], dtype=torch.float32, device='cuda'),
        'retina_origin': torch.tensor(crops[:, :2], device='cuda'),
        'retina_size': torch.tensor(crops[:, 2:] - crops[:, :2], device='cuda'),
    }


@torch.inference_mode()
def predict_residual(model, data, indices):
    model.eval()
    predictions = []
    for start in range(0, len(indices), 8):
        batch = indices[start:start + 8]
        views = []
        for flip_value in (1., -1.):
            x, context, cxy, centers, flip = patch_batch(data, batch, flip_value=flip_value)
            with torch.autocast('cuda'):
                _, local = model(x, context, cxy)
            views.append(to_original(local, centers, flip))
        predictions.append(torch.stack(views).mean(0).cpu().numpy())
    return np.concatenate(predictions)


def source_hashes():
    names = ['residual_refinement.py', 'prepare_residual_v4.py', 'residual_study_v4.py',
             'robust_heatmap.py', 'model.py', 'run.py', 'contracts.py', 'data.py']
    return {name: sha256(ROOT / name) for name in names}


def train_one(data, records, fit, validation, run, seed, max_steps, cache_hash):
    config = {
        'seed': seed, 'steps': max_steps, 'schedule_steps': 240, 'batch': 8, 'native_patch_pixels': PATCH,
        'train_ids': [records[i]['id'] for i in fit],
        'validation_ids': [records[i]['id'] for i in validation],
        'native_cache_sha256': cache_hash, 'pretrained_sha256': sha256(PRETRAINED),
        'source_hashes': source_hashes(), 'encoder_lr': 1e-5, 'decoder_lr': 5e-4,
        'weight_decay': .001, 'ema_decay': .95,
        'loss': 'original pixel MSE / 1024 + 0.2 gaussian CE',
        'coarse_branch': 'frozen precomputed out-of-fold features and coordinates',
        'checkpoint_steps': list(STEPS) if validation else [max_steps],
        'tta': 'horizontal flip of native patch with separately encoded flipped global image',
    }
    if (run / 'status.json').exists():
        status = read_json(run / 'status.json')
        if status['state'] != 'completed' or read_json(run / 'config.json') != config:
            raise RuntimeError(f'Incomplete/incompatible run: {run}')
        return
    run.mkdir(parents=True, exist_ok=True)
    save_json(run / 'config.json', config)
    save_json(run / 'status.json', {'state': 'starting', 'steps': 0})
    torch.manual_seed(seed); np.random.seed(seed)
    model = ResidualRefiner(PRETRAINED).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    enc = list(model.encoder.parameters()); enc_ids = {id(p) for p in enc}
    opt = torch.optim.AdamW([{'params': enc, 'lr': 1e-5},
                           {'params': [p for p in model.parameters() if id(p) not in enc_ids], 'lr': 5e-4}],
                          weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=240, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    targets = torch.full((len(records), 2), float('nan'), device='cuda')
    targets[fit] = torch.tensor([records[i]['xy'] for i in fit], device='cuda', dtype=torch.float32)
    truth = np.asarray([records[i]['xy'] for i in validation]) if validation else None
    # Zero initialization must reproduce the frozen coarse predictions.
    initial_ids = validation or fit[:8]
    initial = predict_residual(ema, data, initial_ids)
    initial_error = float(np.abs(initial - data['centers'][initial_ids].cpu().numpy()).max())
    if initial_error > .01: raise RuntimeError(f'Nonzero initial residual: {initial_error}')
    queue, history, predictions, pred_steps = [], [], [], []
    seen_ids = set(); skipped_steps = 0
    started = time.perf_counter()
    step = 0
    while step < max_steps:
        tick = time.perf_counter()
        if len(queue) < 8:
            queue.extend(np.random.permutation(fit).tolist())
        idx, queue = queue[:8], queue[8:]
        seen_ids.update(records[i]['id'] for i in idx)
        model.train()
        x, context, cxy, centers, flip = patch_batch(data, idx, training=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits, local = model(x, context, cxy)
            loss = loss_original(logits, local, centers, flip, targets[idx])
        if not torch.isfinite(loss): raise FloatingPointError(f'{run.name} step {step}')
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        previous = scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale() < previous:
            skipped_steps += 1
            continue
        update_ema(ema, model, .95)
        step += 1
        scheduler.step()
        row = {'step': step, 'loss': float(loss.detach()), 'rmse': '',
               'seconds': time.perf_counter() - tick}
        save_at_step = step in STEPS if validation else step == max_steps
        if save_at_step:
            if validation:
                pred = predict_residual(ema, data, validation)
                row['rmse'] = coordinate_metrics(truth, pred)['rmse']
                predictions.append(pred); pred_steps.append(step)
                np.savez_compressed(run / 'validation_history.npz', steps=np.array(pred_steps),
                                    ids=np.array(config['validation_ids']), predictions=np.stack(predictions))
            torch.save({'model': ema.state_dict(), 'step': step, 'config': config,
                        'weight_type': 'ema'}, run / f'step{step:03d}.pt')
        history.append(row)
        if step % 20 == 0 or step == 1:
            with (run / 'history.csv').open('w', newline='', encoding='utf8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
            save_json(run / 'status.json', {'state': 'running', 'steps': step, 'skipped_optimizer_steps': skipped_steps})
            print(f'{run.name} step {step}/{max_steps} loss={row["loss"]:.4f} rmse={row["rmse"]} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    if seen_ids != set(config['train_ids']): raise RuntimeError('Some fit records were never trained')
    save_json(run / 'status.json', {'state': 'completed', 'steps': max_steps,
              'skipped_optimizer_steps': skipped_steps, 'seconds': time.perf_counter() - started,
              'initial_max_residual_pixels': initial_error, 'all_training_ids_seen': sorted(seen_ids)})
    del model, ema, opt; torch.cuda.empty_cache()


def select(records):
    truth = np.asarray([r['xy'] for r in records])
    folds = np.asarray([r['fold'] for r in records])
    baseline = np.load(ROBUST / 'oof_predictions.npz')['robust_only']
    base_score = coordinate_metrics(truth, baseline)['rmse']
    base_folds = [coordinate_metrics(truth[folds == f], baseline[folds == f])['rmse'] for f in range(4)]
    by_seed = []
    for seed in SEEDS:
        pred = np.full((len(STEPS), len(records), 2), np.nan)
        for fold in range(4):
            run = OUT / f'fold{fold}_seed{seed}'
            if read_json(run / 'status.json')['state'] != 'completed': raise RuntimeError(run)
            z = np.load(run / 'validation_history.npz')
            assert list(z['steps']) == list(STEPS)
            idx = np.where(folds == fold)[0]
            assert list(z['ids']) == [records[i]['id'] for i in idx]
            pred[:, idx] = z['predictions']
        assert np.isfinite(pred).all()
        by_seed.append(pred)
    combined = np.mean(by_seed, axis=0)
    candidates = []
    for k, step in enumerate(STEPS):
        for weight in (1., .5):
            pred = baseline + weight * (combined[k] - baseline)
            metrics = coordinate_metrics(truth, pred)
            fs = [coordinate_metrics(truth[folds == f], pred[folds == f])['rmse'] for f in range(4)]
            count = sum(x < y for x, y in zip(fs, base_folds))
            reduction = 1 - metrics['rmse'] / base_score
            candidates.append({'steps': step, 'residual_weight': weight, 'metrics': metrics,
                               'fold_rmse': fs, 'improved_folds': count, 'relative_reduction': reduction,
                               'qualified': reduction >= .05 and count >= 3})
    passing = [c for c in candidates if c['qualified']]
    selected = min(passing, key=lambda c: c['metrics']['rmse']) if passing else None
    best = min(candidates, key=lambda c: c['metrics']['rmse'])
    np.savez_compressed(OUT / 'oof_predictions.npz', truth=truth, baseline=baseline, folds=folds,
                        ids=np.array([r['id'] for r in records]), steps=np.array(STEPS),
                        seed43=by_seed[0], seed44=by_seed[1], refined=combined)
    result = {'selected': selected, 'best_candidate': best, 'candidates': candidates,
              'baseline_rmse': base_score, 'baseline_fold_rmse': base_folds,
              'gate': read_json(OUT / 'plan.json')['gate'],
              'note': 'Shared update count selected on existing model-selection folds; no online score.'}
    save_json(OUT / 'selection.json', result)
    save_json(OUT / 'status.json', {'state': 'validated_qualified' if selected else 'completed_not_qualified',
              'validation_runs': 8, 'steps_per_run': 240, 'new_submission_created': False,
              'best_submission_unchanged': sha256(BEST) == BEST_HASH})
    print(json.dumps({'baseline_rmse': base_score, 'best_candidate': best, 'selected': selected}, indent=2), flush=True)
    return result


def full_cache(records):
    target = OUT / 'full_data.npz'
    if target.exists(): return target
    centers = np.full((len(records), 2), np.nan)
    context = np.full((len(records), 2, 512, 2, 2), np.nan, dtype=np.float16)
    for fold in range(4):
        z = np.load(OUT / f'coarse_context/fold{fold}.npz')
        centers[z['indices']] = z['centers']; context[z['indices']] = z['context']
    images = torch.from_numpy(np.load(ROBUST / 'images448.npz')['images'].transpose(0, 3, 1, 2).copy()).cuda()
    test = [i for i, r in enumerate(records) if r['split'] == 'test']
    contexts, predictions, sources = [], [], []
    for seed in SEEDS:
        path = ROBUST / f'full_seed{seed}/best.pt'
        cp = torch.load(path, map_location='cpu', weights_only=True)
        assert set(cp['config']['train_ids']) == {r['id'] for r in records if r['split'] == 'train'}
        assert cp['config']['validation_ids'] == []
        model = RobustHeatmap(False).cuda().eval().requires_grad_(False)
        model.load_state_dict(cp['model'])
        xy, ctx = infer_context(model, images, test, records)
        contexts.append(ctx.astype(np.float32)); predictions.append(xy)
        sources.append({'checkpoint': str(path), 'sha256': sha256(path)})
        del model
    old = read_submission(BEST)
    established = np.asarray([old[records[i]['id']] for i in test])
    difference = float(np.abs(np.mean(predictions, axis=0) - established).max())
    if difference > .25: raise RuntimeError(f'Full coarse reproduction error {difference}')
    centers[test] = established; context[test] = np.mean(contexts, axis=0)
    assert np.isfinite(centers).all() and np.isfinite(context).all()
    patches, origins = native_cache(records, centers)
    np.savez_compressed(target, ids=np.array([r['id'] for r in records]), centers=centers,
                        context=context, patches=patches, origins=origins)
    save_json(target.with_suffix('.json'), {'test_coarse_sources': sources,
              'fit_coarse_centers': 'four outer OOF models, each excluding the record group',
              'max_test_coarse_reproduction_error': difference, 'sha256': sha256(target)})
    del images; torch.cuda.empty_cache()
    return target


def finalize(records):
    selection = read_json(OUT / 'selection.json')
    chosen = selection['selected']
    if chosen is None:
        print('Validation gate not passed; no final refit or new submission.', flush=True)
        return
    cache = full_cache(records)
    data = load_cache(cache, records)
    fit = [i for i, r in enumerate(records) if r['split'] == 'train']
    test = [i for i, r in enumerate(records) if r['split'] == 'test']
    predictions, sources = [], []
    for seed in SEEDS:
        run = OUT / f'full_seed{seed}'
        train_one(data, records, fit, [], run, seed, chosen['steps'], sha256(cache))
        checkpoint = run / f'step{chosen["steps"]:03d}.pt'
        cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = ResidualRefiner().cuda().eval()
        model.load_state_dict(cp['model'])
        pred = predict_residual(model, data, test)
        np.save(run / 'test_predictions.npy', pred)
        predictions.append(pred)
        sources.append({'path': str(checkpoint), 'sha256': sha256(checkpoint), 'seed': seed,
                        'steps': chosen['steps'], 'train_count': len(fit)})
        del model
    baseline = data['centers'][test].cpu().numpy().astype(np.float64)
    final = baseline + chosen['residual_weight'] * (np.mean(predictions, axis=0) - baseline)
    path = OUT / 'submission_residual_v4.csv'
    values = {records[i]['id']: final[k] for k, i in enumerate(test)}
    sizes = {records[i]['id']: (records[i]['width'], records[i]['height']) for i in test}
    write_submission(ROOT / 'data/detection/sample_submission.csv', values, path, sizes)
    save_json(OUT / 'predictions.json', {str(k): v.tolist() for k, v in values.items()})
    save_json(OUT / 'result.json', {'selection': chosen, 'coarse': read_json(cache.with_suffix('.json')),
              'refinement_models': sources, 'submission': str(path), 'submission_sha256': sha256(path),
              'rows': 40, 'train_count': 80, 'online_score': None,
              'old_best_unchanged': sha256(BEST) == BEST_HASH})
    save_json(OUT / 'status.json', {'state': 'completed_candidate_ready', 'new_submission_created': True,
              'best_submission_unchanged': sha256(BEST) == BEST_HASH, 'online_score': None})
    print(f'New candidate: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['validate', 'final', 'all'], default='all')
    args = parser.parse_args()
    setup()
    if read_json(OUT / 'preparation_status.json')['state'] != 'completed':
        raise RuntimeError('Nested coarse preparation is incomplete')
    snapshot = OUT / 'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for name, digest in source_hashes().items():
        if (snapshot / name).exists() and sha256(snapshot / name) != digest:
            raise RuntimeError('Source changed after experiment began')
        if not (snapshot / name).exists(): shutil.copyfile(ROOT / name, snapshot / name)
    save_json(snapshot / 'hashes.json', source_hashes())
    records = read_json(MANIFEST)['records']
    train = [r for r in records if r['split'] == 'train']
    if args.stage in ('validate', 'all'):
        for fold in range(4):
            path = OUT / f'outer{fold}_data.npz'
            cache_hash = sha256(path)
            if read_json(path.with_suffix('.json'))['cache_sha256'] != cache_hash:
                raise RuntimeError('Native cache changed')
            data = load_cache(path, train)
            fit = [i for i, r in enumerate(train) if r['fold'] != fold]
            val = [i for i, r in enumerate(train) if r['fold'] == fold]
            fit_truth = np.asarray([train[i]['xy'] for i in fit])
            max_offset = np.abs(data['centers'][fit].cpu().numpy() - fit_truth).max()
            if max_offset + 16 >= PATCH / 2:
                raise ValueError('Geometry preflight failed: target plus jitter outside patch')
            for seed in SEEDS:
                train_one(data, train, fit, val, OUT / f'fold{fold}_seed{seed}', seed, 240, cache_hash)
            del data; torch.cuda.empty_cache()
        select(train)
    if args.stage in ('final', 'all'):
        finalize(records)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    main()
