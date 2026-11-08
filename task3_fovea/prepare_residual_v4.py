"""Leakage-controlled nested coarse predictions and frozen context features.

For each pair of outer folds, train on the other two folds. Each held-out fold
selects a checkpoint independently, exclusively for use as the INNER held-out
fold. For outer fold A, the A/B pair uses selection on B, never on A.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from contracts import coordinate_metrics
from data import original_coordinates
from model import heatmap_coordinates, heatmap_loss
from refinement_study import read_json, sha256
from robust_heatmap import RobustHeatmap, illumination_augment, update_ema, WEIGHTS
from run import ROOT, augment, normalize, save_json
from residual_refinement import CACHE, PATCH

OUT = ROOT / 'outputs/residual_v4_nested_20261024'
ROBUST = ROOT / 'outputs/improvement_20261024/robust_heatmap'
MANIFEST = ROOT / 'outputs/baseline/prepared/manifest.json'
BEST = ROOT / 'outputs/improvement_20261024/submission_improved.csv'
BEST_HASH = '20d24c207d73e8ef0af3de589e9e3becd0416b14c8b30c02068815451f136a8b'
COARSE_STEPS = 320
COARSE_CHECKS = (40, 80, 120, 160, 240, 320)


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(BEST) != BEST_HASH:
        raise RuntimeError('Current best submission changed')
    plan = {
        'method': f'frozen coarse global context + native {PATCH}px residual heatmap',
        'outer_folds': 4, 'seeds': [43, 44], 'coarse_inner_steps': COARSE_STEPS,
        'coarse_inner_schedule_steps': 320,
        'coarse_inner_split': 'leave two original folds out; choose on inner fold only, never outer fold',
        'coarse_inner_candidate_steps': list(COARSE_CHECKS),
        'refiner_steps': 240, 'candidate_steps': [40, 80, 120, 160, 240],
        'batch_size': 8, 'local_patch_pixels': PATCH, 'unresized_cache_pixels': CACHE,
        'refiner_encoder_lr': 1e-5, 'refiner_head_lr': 5e-4, 'weight_decay': .001,
        'ema_decay': .95, 'coordinate_loss': 'original pixel MSE / 1024',
        'heatmap_ce_weight': .2, 'heatmap_sigma_cells': 2,
        'center_prior_sigma_original_pixels': 64, 'extra_center_jitter_sd_pixels': 8,
        'selection': 'one shared step count, averaged two seeds; residual weights 1 and 0.5',
        'gate': {'relative_rmse_reduction': .05, 'minimum_improved_folds': 3},
        'baseline_sha256': BEST_HASH, 'manifest_sha256': sha256(MANIFEST),
        'validation_note': 'Same model-selection folds as previous work, not an independent test.',
        'full_refit': 'same selected update count, frozen full-data coarse ensemble; outer OOF fit centers',
        'external_data': False, 'manual_test_labels': False,
    }
    if (OUT / 'plan.json').exists():
        previous = read_json(OUT / 'plan.json')
        if previous != plan:
            changed = {k for k in set(previous) | set(plan) if previous.get(k) != plan.get(k)}
            if changed <= {'method', 'local_patch_pixels'} and not list(OUT.glob('fold*_seed*/status.json')):
                save_json(OUT / 'plan_before_geometry_check.json', previous)
                save_json(OUT / 'preflight_amendment.json', {
                    'changed_fields': sorted(changed), 'before_refiner_training': True,
                    'reason': 'Nested-training coarse error reached 280.75px; enlarge native window to 640px so every fit target plus jitter remains visible. No images removed.'})
                save_json(OUT / 'plan.json', plan)
            else:
                raise RuntimeError('Plan changed: use a new output directory')
    else:
        save_json(OUT / 'plan.json', plan)
        shutil.copyfile(BEST, OUT / 'preserved_best_14.20909.csv')
    return plan


@torch.inference_mode()
def infer_context(model, images, indices, records):
    model.eval()
    predictions, features = [], []
    for start in range(0, len(indices), 8):
        ids = indices[start:start + 8]
        x = images[ids].float() / 255
        xy_views, ctx_views = [], []
        for flip in (False, True):
            inp = normalize(x.flip(-1) if flip else x)
            with torch.autocast('cuda'):
                e = model.encoder
                c = e.maxpool(e.relu(e.bn1(e.conv1(inp))))
                c1 = e.layer1(c); c2 = e.layer2(c1); c3 = e.layer3(c2); c4 = e.layer4(c3)
                p = model.refine[3](model.lateral[3](c4))
                levels = (c1, c2, c3, c4)
                for level in (2, 1, 0):
                    lateral = model.lateral[level](levels[level])
                    p = model.refine[level](lateral + F.interpolate(
                        p, size=lateral.shape[-2:], mode='bilinear', align_corners=False))
                xy = heatmap_coordinates(model.head(p))
                ctx = F.adaptive_avg_pool2d(c4.float(), (2, 2))
            if flip:
                xy[:, 0] = 1 - xy[:, 0]
            xy_views.append(xy.float()); ctx_views.append(ctx.float())
        predictions.append(torch.stack(xy_views).mean(0).cpu().numpy())
        features.append(torch.stack(ctx_views, dim=1).cpu().numpy())
    xy = original_coordinates(np.concatenate(predictions), [records[i] for i in indices])
    return xy, np.concatenate(features).astype(np.float16)


def train_pair(images, records, excluded, seed):
    name = f'pair{excluded[0]}{excluded[1]}_seed{seed}'
    run = OUT / 'nested_coarse' / name
    fit = [i for i, r in enumerate(records) if r['split'] == 'train' and r['fold'] not in excluded]
    held = [i for i, r in enumerate(records) if r['split'] == 'train' and r['fold'] in excluded]
    config = {'seed': seed, 'train_ids': [records[i]['id'] for i in fit],
              'heldout_ids': [records[i]['id'] for i in held], 'excluded_folds': list(excluded),
              'optimizer_steps': COARSE_STEPS, 'schedule_steps': 320, 'batch': 8,
              'pretrained_sha256': sha256(WEIGHTS), 'manifest_sha256': sha256(MANIFEST),
              'selection': 'separate best checkpoint on each inner fold; never select on outer fold',
              'candidate_steps': list(COARSE_CHECKS), 'ema_decay': .95}
    if (run / 'status.json').exists():
        if read_json(run / 'config.json') != config or read_json(run / 'status.json')['state'] != 'completed':
            raise RuntimeError(f'Incomplete or incompatible coarse run: {run}')
        return run
    run.mkdir(parents=True, exist_ok=True)
    save_json(run / 'config.json', config)
    save_json(run / 'status.json', {'state': 'running', 'steps': 0})
    torch.manual_seed(seed); np.random.seed(seed)
    model = RobustHeatmap().cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    enc = list(model.encoder.parameters()); enc_ids = {id(p) for p in enc}
    opt = torch.optim.AdamW([{'params': enc, 'lr': 5e-5},
                            {'params': [p for p in model.parameters() if id(p) not in enc_ids], 'lr': 5e-4}],
                           weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=320, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    targets = torch.full((len(records), 2), float('nan'), device='cuda')
    targets[fit] = torch.tensor([records[i]['xy_crop'] for i in fit], device='cuda')
    queue, losses = [], []
    start = time.perf_counter()
    best = {f: float('inf') for f in excluded}
    selected_steps = {}
    history = []
    step = 0
    skipped = 0
    while step < COARSE_STEPS:
        if len(queue) < 8:
            queue.extend(np.random.permutation(fit).tolist())
        idx, queue = queue[:8], queue[8:]
        model.train()
        x, xy = augment(images[idx].float() / 255, targets[idx])
        x = illumination_augment(x)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits = model(normalize(x))
            loss = heatmap_loss(logits, xy, 1.5) + 95 * F.mse_loss(heatmap_coordinates(logits), xy)
        if not torch.isfinite(loss): raise FloatingPointError(name)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        previous = scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale() < previous:
            skipped += 1
            continue
        update_ema(ema, model)
        scheduler.step(); losses.append(float(loss.detach())); step += 1
        if step in COARSE_CHECKS:
            xy, ctx = infer_context(ema, images, held, records)
            row = {'step': step}
            for inner in excluded:
                rows = [k for k, i in enumerate(held) if records[i]['fold'] == inner]
                ids = [held[k] for k in rows]
                truth = np.asarray([records[i]['xy'] for i in ids])
                score = coordinate_metrics(truth, xy[rows])['rmse']
                row[f'fold{inner}_rmse'] = score
                if score < best[inner]:
                    best[inner] = score; selected_steps[inner] = step
                    cp_config = dict(config, inner_selection_ids=[records[i]['id'] for i in ids],
                                     inner_fold=inner, outer_fold=next(f for f in excluded if f != inner))
                    torch.save({'model': ema.state_dict(), 'config': cp_config, 'steps': step},
                               run / f'inner{inner}.pt')
                    np.savez_compressed(run / f'inner{inner}.npz', indices=np.array(ids),
                                        centers=xy[rows], context=ctx[rows])
            history.append(row)
            save_json(run / 'selection_history.json', history)
    save_json(run / 'status.json', {'state': 'completed', 'steps': COARSE_STEPS,
              'seconds': time.perf_counter() - start, 'losses': losses,
              'selected_steps_by_inner_fold': selected_steps, 'best_rmse_by_inner_fold': best,
              'skipped_attempts': skipped,
              'checkpoint_sha256': {str(f): sha256(run / f'inner{f}.pt') for f in excluded}})
    print(f'{name} {COARSE_STEPS} updates complete, inner selection={selected_steps}, rmse={best}, seconds={time.perf_counter()-start:.1f}', flush=True)
    del model, ema, opt; torch.cuda.empty_cache()
    return run


def outer_context(images, records, fold):
    target = OUT / 'coarse_context' / f'fold{fold}.npz'
    target.parent.mkdir(exist_ok=True)
    indices = [i for i, r in enumerate(records) if r['split'] == 'train' and r['fold'] == fold]
    paths = [ROBUST / f'fold{fold}_seed{s}/best.pt' for s in (43, 44)]
    sources = [{'path': str(p), 'sha256': sha256(p)} for p in paths]
    if target.exists():
        if read_json(target.with_suffix('.json'))['sources'] != sources:
            raise RuntimeError('Coarse feature checkpoint changed')
        return target
    contexts, centers = [], []
    for path in paths:
        cp = torch.load(path, map_location='cpu', weights_only=True)
        assert set(cp['config']['train_ids']).isdisjoint(records[i]['id'] for i in indices)
        assert set(cp['config']['validation_ids']) == {records[i]['id'] for i in indices}
        model = RobustHeatmap(False).cuda().eval().requires_grad_(False)
        model.load_state_dict(cp['model'])
        xy, ctx = infer_context(model, images, indices, records)
        contexts.append(ctx.astype(np.float32)); centers.append(xy)
        del model
    # Preserve the exact established OOF coarse centers for a paired comparison.
    established = np.load(ROBUST / 'oof_predictions.npz')['robust_only'][indices]
    delta = float(np.max(np.abs(np.mean(centers, axis=0) - established)))
    if delta > .25:
        raise RuntimeError(f'Coarse inference reproduction difference: {delta}')
    np.savez_compressed(target, indices=np.array(indices), centers=established,
                        context=np.mean(contexts, axis=0).astype(np.float16))
    save_json(target.with_suffix('.json'), {'sources': sources, 'max_reproduction_error_pixels': delta})
    return target


def native_cache(records, centers):
    patches, origins = [], []
    for i, r in enumerate(records):
        origin = np.floor(centers[i] - CACHE / 2).astype(int)
        with Image.open(r['file']) as image:
            patch = image.convert('RGB').crop((*origin, *(origin + CACHE)))
            patches.append(np.asarray(patch, dtype=np.uint8))
        origins.append(origin)
    return np.stack(patches), np.asarray(origins, dtype=np.float32)


def prepare_all():
    setup()
    records = read_json(MANIFEST)['records']
    train = [r for r in records if r['split'] == 'train']
    assert len(train) == 80 and all(records[i]['id'] == train[i]['id'] for i in range(80))
    for r in records:
        if sha256(r['file']) != r['sha256']: raise RuntimeError(f'Changed image {r["id"]}')
    images = torch.from_numpy(np.load(ROBUST / 'images448.npz')['images'].transpose(0, 3, 1, 2).copy()).cuda()
    for pair in itertools.combinations(range(4), 2):
        for seed in (43, 44):
            train_pair(images, records, pair, seed)
    for f in range(4): outer_context(images, records, f)
    del images; torch.cuda.empty_cache()
    truth = np.asarray([r['xy'] for r in train])
    for outer in range(4):
        target = OUT / f'outer{outer}_data.npz'
        if target.exists(): continue
        centers = np.full((80, 2), np.nan)
        context = np.full((80, 2, 512, 2, 2), np.nan, dtype=np.float16)
        source_by_id = {}
        for inner in range(4):
            idx = [i for i, r in enumerate(train) if r['fold'] == inner]
            if inner == outer:
                path = OUT / f'coarse_context/fold{outer}.npz'
                z = np.load(path); centers[idx] = z['centers']; context[idx] = z['context']
                for i in idx: source_by_id[str(train[i]['id'])] = {'kind': 'outer-heldout', 'cache': str(path)}
            else:
                a, b = sorted((inner, outer)); c, ctx = [], []
                sources = []
                for seed in (43, 44):
                    run = OUT / f'nested_coarse/pair{a}{b}_seed{seed}'
                    cfg = read_json(run / 'config.json')
                    excluded_ids = {r['id'] for r in train if r['fold'] in (outer, inner)}
                    if not set(cfg['train_ids']).isdisjoint(excluded_ids):
                        raise RuntimeError('Nested coarse leakage')
                    cp_path = run / f'inner{inner}.pt'
                    cp_config = torch.load(cp_path, map_location='cpu', weights_only=True)['config']
                    outer_ids = {r['id'] for r in train if r['fold'] == outer}
                    assert cp_config['outer_fold'] == outer
                    assert set(cp_config['inner_selection_ids']).isdisjoint(outer_ids)
                    z = np.load(run / f'inner{inner}.npz'); rows = [list(z['indices']).index(i) for i in idx]
                    c.append(z['centers'][rows]); ctx.append(z['context'][rows].astype(np.float32))
                    sources.append({'run': str(run), 'train_ids': cfg['train_ids'],
                                    'inner_selection_ids': cp_config['inner_selection_ids'],
                                    'checkpoint': str(cp_path), 'checkpoint_sha256': sha256(cp_path)})
                centers[idx] = np.mean(c, axis=0); context[idx] = np.mean(ctx, axis=0)
                for i in idx: source_by_id[str(train[i]['id'])] = {'kind': 'inner-heldout', 'sources': sources}
        assert np.isfinite(centers).all() and np.isfinite(context).all()
        patches, origins = native_cache(train, centers)
        np.savez_compressed(target, ids=np.array([r['id'] for r in train]), centers=centers,
                            context=context, patches=patches, origins=origins)
        fit = [i for i, r in enumerate(train) if r['fold'] != outer]
        meta = {'outer_fold': outer, 'fit_ids': [train[i]['id'] for i in fit],
                'validation_ids': [r['id'] for r in train if r['fold'] == outer],
                'training_coarse_metrics': coordinate_metrics(truth[fit], centers[fit]),
                'max_training_axis_error_pixels': float(np.abs(truth[fit] - centers[fit]).max()),
                'sources_by_image': source_by_id, 'cache_sha256': sha256(target)}
        save_json(target.with_suffix('.json'), meta)
        print(f'outer{outer} native cache ready, train coarse RMSE={meta["training_coarse_metrics"]["rmse"]:.3f}, max axis error={meta["max_training_axis_error_pixels"]:.1f}', flush=True)
    save_json(OUT / 'preparation_status.json', {'state': 'completed', 'nested_models': 12,
              'coarse_steps_each': COARSE_STEPS, 'native_cache_size': CACHE,
              'best_submission_unchanged': sha256(BEST) == BEST_HASH})


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    prepare_all()
