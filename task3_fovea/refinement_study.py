"""OOF-gated local refinement; no test annotations or human-adjusted predictions.

Run --stage validate first. --stage final requires a passing selection.json;
--stage all runs both stages only when the fixed validation gate passes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from contracts import coordinate_metrics, write_submission
from data import original_coordinates
from refinement_model import LocalRefiner, refinement_loss
from run import ROOT, normalize, save_json

MANIFEST = ROOT / 'outputs/baseline/prepared/manifest.json'
OLD_STUDY = ROOT / 'outputs/study'
EPOCHS, BATCH, PATCH, CACHE_SIZE = 80, 16, 256, 1024
EMA_DECAY, BASELINE_RMSE = 0.97, 28.1537
FRACTIONS = (0.34, 0.42)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def crop_coordinates(xy, records):
    crops = np.asarray([r['crop'] for r in records], dtype=np.float64)
    return (np.asarray(xy) - crops[:, :2]) / (crops[:, 2:] - crops[:, :2])


def load_images(records, output, manifest_hash, device):
    cache = output / 'retina_1024.npz'
    if not cache.exists():
        arrays = []
        for record in records:
            path = Path(record['file'])
            if sha256(path) != record['sha256']:
                raise ValueError(f'Image changed since manifest: {path}')
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert('RGB').crop(record['crop']).resize(
                    (CACHE_SIZE, CACHE_SIZE), Image.Resampling.BILINEAR)))
        np.savez_compressed(cache, images=np.stack(arrays), manifest_sha256=manifest_hash)
    with np.load(cache, allow_pickle=False) as stored:
        if str(stored['manifest_sha256'].item()) != manifest_hash:
            raise ValueError('Cached retina images do not match the manifest')
        images = stored['images']
    if images.dtype != np.uint8 or images.shape != (len(records), CACHE_SIZE, CACHE_SIZE, 3):
        raise ValueError('Invalid retina image cache')
    return torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)


def make_theta(centers, fractions, angles=None, flips=None):
    """Map patch grid [-1,1] through A to retina grid [-1,1]."""
    n, device = len(centers), centers.device
    angles = torch.zeros(n, device=device) if angles is None else angles
    flips = torch.ones(n, device=device) if flips is None else flips
    a = torch.zeros(n, 2, 2, device=device)
    a[:, 0, 0], a[:, 0, 1] = fractions * angles.cos() * flips, -fractions * angles.sin()
    a[:, 1, 0], a[:, 1, 1] = fractions * angles.sin() * flips, fractions * angles.cos()
    return torch.cat((a, (2 * centers - 1)[:, :, None]), dim=2)


def extract_patches(images, indices, theta):
    x = images[indices].float().div_(255)
    grid = F.affine_grid(theta, (len(indices), 3, PATCH, PATCH), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def patch_to_retina(xy, theta):
    return (torch.bmm(theta[:, :, :2], (2 * xy.float() - 1).unsqueeze(-1)).squeeze(-1)
            + theta[:, :, 2] + 1) / 2


def training_patches(images, indices, targets):
    """Only fit targets determine training crops; coarse OOF is never consulted."""
    xy, n = targets[indices], len(indices)
    rand = lambda *shape: torch.rand(*shape, device=images.device)
    centers = xy + (rand(n, 2) * 2 - 1) * 0.10
    theta = make_theta(centers, 0.32 + 0.14 * rand(n),
                       (rand(n) * 2 - 1) * math.radians(20),
                       torch.where(rand(n) < 0.5, -1., 1.))
    local_xy = (torch.linalg.solve(theta[:, :, :2],
                (2 * xy - 1 - theta[:, :, 2]).unsqueeze(-1)).squeeze(-1) + 1) / 2
    if not ((local_xy >= 0) & (local_xy <= 1)).all():
        raise ValueError('Training augmentation moved a target outside its patch')
    x = extract_patches(images, indices, theta)
    brightness, contrast = 0.8 + 0.4 * rand(n, 1, 1, 1), 0.85 + 0.3 * rand(n, 1, 1, 1)
    color, gamma = 0.9 + 0.2 * rand(n, 3, 1, 1), 0.85 + 0.3 * rand(n, 1, 1, 1)
    mean = x.mean((2, 3), keepdim=True)
    x = (((x - mean) * contrast + mean) * brightness * color).clamp(0, 1).pow(gamma)
    return normalize(x), local_xy


@torch.inference_mode()
def predict_local(model, images, indices, centers, records):
    """Centers are supplied coarse predictions; records are used only for crop geometry."""
    model.eval()
    result = {'heatmap': [], 'dual': []}
    for start in range(0, len(indices), BATCH):
        batch = indices[start:start + BATCH]
        center = centers[batch]
        if not torch.isfinite(center).all():
            raise ValueError('Missing coarse predictions for inference')
        variants = {'heatmap': [], 'dual': []}
        for fraction in FRACTIONS:
            for flip in (1., -1.):
                theta = make_theta(center, center.new_full((len(batch),), fraction),
                                   flips=center.new_full((len(batch),), flip))
                x = normalize(extract_patches(images, batch, theta))
                with torch.autocast(images.device.type, enabled=images.is_cuda):
                    output = model(x)
                variants['heatmap'].append(patch_to_retina(output['xy'], theta))
                variants['dual'].append(patch_to_retina((output['xy'].float() + output['reg_xy'].float()) / 2, theta))
        for head in result:
            normalized = torch.stack(variants[head]).mean(0).cpu().numpy()
            result[head].append(original_coordinates(normalized, [records[i] for i in batch]))
    return {head: np.concatenate(values) for head, values in result.items()}


@torch.no_grad()
def update_ema(ema, model):
    current = model.state_dict()
    for name, value in ema.state_dict().items():
        # Frozen encoder running statistics are copied exactly, not smoothed.
        if value.is_floating_point() and not name.endswith(('running_mean', 'running_var')):
            value.mul_(EMA_DECAY).add_(current[name], alpha=1 - EMA_DECAY)
        else:
            value.copy_(current[name])


def train_run(images, records, fit, validation, centers, out, seed, epochs, pretrained, sources):
    config = {'seed': seed, 'epochs': epochs, 'schedule_epochs': EPOCHS, 'batch_size': BATCH,
              'patch_size': PATCH, 'train_ids': [records[i]['id'] for i in fit],
              'validation_ids': [records[i]['id'] for i in validation], 'sources': sources,
              'encoder_lr': 1e-4, 'decoder_lr': 5e-4, 'weight_decay': 1e-4,
              'ema_decay': EMA_DECAY, 'checkpoint_selection': 'EMA dual-head original-pixel RMSE',
              'train_patches_per_record': 2, 'train_width_range': [0.32, 0.46],
              'center_jitter': 0.10, 'rotation_degrees': 20, 'tta_widths': list(FRACTIONS), 'tta_hflip': True}
    if (out / 'status.json').exists():
        status = read_json(out / 'status.json')
        if status['state'] != 'completed' or status['completed_epochs'] != epochs or read_json(out / 'config.json') != config:
            raise RuntimeError(f'Incomplete or incompatible run; use a fresh output directory: {out}')
        if not (out / 'best.pt').exists() or (validation and not (out / 'best_predictions.npz').exists()):
            raise FileNotFoundError(f'Missing completed run artifacts: {out}')
        return status
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Unrecognized existing run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'config.json', config)
    save_json(out / 'status.json', {'state': 'starting', 'completed_epochs': 0})
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = LocalRefiner(pretrained).to(images.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': 1e-4},
        {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': 5e-4}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=images.is_cuda)
    # Fill only fit targets so validation coordinates cannot enter augmentation.
    targets = torch.full((len(records), 2), float('nan'), device=images.device)
    targets[fit] = torch.tensor([records[i]['xy_crop'] for i in fit], device=images.device)
    truth = np.asarray([records[i]['xy'] for i in validation]) if validation else None
    best_score, best_epoch, history = float('inf'), 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tick, total_loss = time.perf_counter(), 0.
        model.train()
        order = np.random.permutation(np.tile(fit, 2))
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH].tolist()
            x, target = training_patches(images, batch, targets)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(images.device.type, enabled=images.is_cuda):
                loss = refinement_loss(model(x), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {out}, epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                update_ema(ema, model)
            total_loss += float(loss.detach()) * len(batch)
        scheduler.step()
        row = {'epoch': epoch, 'loss': total_loss / len(order)}
        if validation:
            predictions = predict_local(ema, images, validation, centers, records)
            row.update({f'{head}_rmse': coordinate_metrics(truth, values)['rmse'] for head, values in predictions.items()})
            if row['dual_rmse'] < best_score:
                best_score, best_epoch = row['dual_rmse'], epoch
                torch.save({'model': ema.state_dict(), 'epoch': epoch, 'config': config}, out / 'best.pt')
                np.savez(out / 'best_predictions.npz', ids=np.array(config['validation_ids']), **predictions)
        row['seconds'] = time.perf_counter() - tick
        history.append(row)
        with (out / 'history.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        save_json(out / 'status.json', {'state': 'running', 'completed_epochs': epoch,
            'best_epoch': best_epoch, 'best_rmse': best_score if validation else None})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'{out.name} {epoch}/{epochs} loss={row["loss"]:.4f} dual_rmse={row.get("dual_rmse")} best={best_score:.3f}', flush=True)
    if not validation:
        best_epoch = epochs
        torch.save({'model': ema.state_dict(), 'epoch': epochs, 'config': config}, out / 'best.pt')
    status = {'state': 'completed', 'completed_epochs': epochs, 'best_epoch': best_epoch,
              'best_rmse': best_score if validation else None, 'seconds': time.perf_counter() - started}
    save_json(out / 'status.json', status)
    return status


def validate(images, records, train, coarse, centers, output, pretrained, sources):
    truth = np.asarray([records[i]['xy'] for i in train])
    positions = {index: position for position, index in enumerate(train)}
    folds = np.asarray([records[i]['fold'] for i in train])
    oof = {head: np.full_like(truth, np.nan) for head in ('heatmap', 'dual')}
    best_epochs = []
    for fold in range(4):
        fit = [i for i in train if records[i]['fold'] != fold]
        validation = [i for i in train if records[i]['fold'] == fold]
        out = output / f'fold{fold}_seed42'
        status = train_run(images, records, fit, validation, centers, out, 42, EPOCHS, pretrained, sources)
        with np.load(out / 'best_predictions.npz') as predictions:
            if predictions['ids'].tolist() != [records[i]['id'] for i in validation]:
                raise ValueError('Validation prediction IDs differ from their fold')
            for head in oof:
                oof[head][[positions[i] for i in validation]] = predictions[head]
        best_epochs.append(status['best_epoch'])
    candidates = {'local_heatmap': oof['heatmap'], 'local_dual': oof['dual'],
                  'baseline_local_heatmap_mean': (coarse + oof['heatmap']) / 2,
                  'baseline_local_dual_mean': (coarse + oof['dual']) / 2}
    baseline_folds = [coordinate_metrics(truth[folds == f], coarse[folds == f]) for f in range(4)]
    metrics = {name: coordinate_metrics(truth, xy) for name, xy in candidates.items()}
    fold_metrics = {name: [coordinate_metrics(truth[folds == f], xy[folds == f]) for f in range(4)] for name, xy in candidates.items()}
    gates = {name: {'pooled_pass': metrics[name]['rmse'] <= BASELINE_RMSE * 0.9,
                   'folds_improved': sum(fold_metrics[name][f]['rmse'] < baseline_folds[f]['rmse'] for f in range(4))} for name in candidates}
    eligible = [name for name in candidates if gates[name]['pooled_pass'] and gates[name]['folds_improved'] >= 3]
    selected = min(eligible, key=lambda name: metrics[name]['rmse']) if eligible else None
    selection = {'gate_passed': bool(eligible), 'selected': selected, 'metrics': metrics, 'fold_metrics': fold_metrics,
        'baseline': coordinate_metrics(truth, coarse), 'baseline_folds': baseline_folds, 'gates': gates,
        'gate_rule': {'baseline_rmse': BASELINE_RMSE, 'required_relative_reduction': 0.10, 'required_improved_folds': 3},
        'head': ('heatmap' if 'heatmap' in selected else 'dual') if selected else None,
        'baseline_weight': (0.5 if selected.startswith('baseline_') else 0.) if selected else None,
        'best_epochs': best_epochs, 'refit_epochs': int(np.median(best_epochs)), 'sources': sources,
        'all_training_ids': [records[i]['id'] for i in train], 'model_selection_validation': True}
    np.savez(output / 'oof_predictions.npz', ids=np.array(selection['all_training_ids']), folds=folds, truth=truth, coarse=coarse, **candidates)
    save_json(output / 'selection.json', selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    return selection


def final(images, records, train, test, centers, output, pretrained, sources, selection):
    if not selection['gate_passed']:
        print('Validation gate failed; no refined CSV generated.', flush=True)
        return
    if selection['sources'] != sources or selection['all_training_ids'] != [records[i]['id'] for i in train]:
        raise ValueError('Final settings or source artifacts differ from validation')
    destination = output / 'submission_refined.csv'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite an existing submission: {destination}')
    old_prediction_path = OLD_STUDY / 'predictions.json'
    saved = read_json(old_prediction_path)
    if set(saved) != {str(records[i]['id']) for i in test}:
        raise ValueError('Coarse test prediction IDs do not match the manifest')
    coarse = np.asarray([saved[str(records[i]['id'])] for i in test], dtype=np.float64)
    if coarse.shape != (len(test), 2) or not np.isfinite(coarse).all():
        raise ValueError('Invalid coarse test predictions')
    centers[test] = torch.tensor(crop_coordinates(coarse, [records[i] for i in test]), device=images.device, dtype=torch.float32)
    predictions, models = [], []
    for seed in (43, 44):
        out = output / f'full_seed{seed}'
        epochs = selection['refit_epochs']
        train_run(images, records, train, [], centers, out, seed, epochs, pretrained, sources)
        checkpoint = torch.load(out / 'best.pt', map_location=images.device, weights_only=True)
        if checkpoint['epoch'] != epochs or checkpoint['config']['train_ids'] != [records[i]['id'] for i in train]:
            raise ValueError('Invalid full-data checkpoint')
        model = LocalRefiner(None).to(images.device)
        model.load_state_dict(checkpoint['model'])
        predicted = predict_local(model, images, test, centers, records)
        np.savez(out / 'test_predictions.npz', ids=np.array([records[i]['id'] for i in test]), **predicted)
        predictions.append(predicted[selection['head']])
        models.append({'seed': seed, 'epochs': epochs, 'checkpoint': str(out / 'best.pt'), 'sha256': sha256(out / 'best.pt')})
        del model, checkpoint
    weight = selection['baseline_weight']
    xy = weight * coarse + (1 - weight) * np.mean(predictions, axis=0)
    prediction_map = {records[i]['id']: xy[j].tolist() for j, i in enumerate(test)}
    sizes = {records[i]['id']: (records[i]['width'], records[i]['height']) for i in test}
    write_submission(ROOT / 'data/detection/sample_submission.csv', prediction_map, destination, sizes)
    save_json(output / 'predictions_refined.json', prediction_map)
    save_json(output / 'result.json', {'selection': selection, 'models': models, 'sources': sources,
        'coarse_predictions_source': str(old_prediction_path), 'coarse_predictions_sha256': sha256(old_prediction_path),
        'cache_sha256': sha256(output / 'retina_1024.npz'), 'submission_sha256': sha256(destination),
        'refit_train_count': len(train), 'test_count': len(test), 'submission_rows': 2 * len(test),
        'online_score': None, 'note': 'OOF selected validation models; the two-seed full-data ensemble has no held-out score.'})
    print(f'New submission: {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('validate', 'final', 'all'), default='validate')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/improvement_20261024/refinement')
    parser.add_argument('--pretrained', type=Path, default=ROOT.parent / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_json(MANIFEST)['records']
    train, test = ([i for i, r in enumerate(records) if r['split'] == split] for split in ('train', 'test'))
    if len(train) != 80 or not {39, 77}.issubset({records[i]['id'] for i in train}) or {records[i]['fold'] for i in train} != set(range(4)):
        raise ValueError('Expected all 80 original training records and their four unchanged folds')
    truth = np.asarray([records[i]['xy'] for i in train])
    with np.load(OLD_STUDY / 'oof_predictions.npz', allow_pickle=False) as old:
        coarse = old['equal_mean'].copy()
        if not np.array_equal(old['truth'], truth) or coarse.shape != truth.shape or not np.isfinite(coarse).all():
            raise ValueError('Old OOF order/coordinates differ from the current manifest')
    if abs(coordinate_metrics(truth, coarse)['rmse'] - BASELINE_RMSE) > 1e-3:
        raise ValueError('The required coarse OOF baseline changed')
    for folder in ('baseline', 'regression'):
        for fold in range(4):
            config = read_json(ROOT / 'outputs' / folder / f'validation_fold{fold}_seed42/config.json')
            if config['validation_ids'] != [records[i]['id'] for i in train if records[i]['fold'] == fold]:
                raise ValueError('Coarse model fold does not match refinement validation fold')
    source_paths = [MANIFEST, OLD_STUDY / 'oof_predictions.npz', args.pretrained, Path(__file__), ROOT / 'refinement_model.py', ROOT / 'model.py']
    sources = {str(path.resolve()): sha256(path) for path in source_paths}
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = load_images(records, args.output, sources[str(MANIFEST.resolve())], device)
    centers = torch.full((len(records), 2), float('nan'), device=device)
    centers[train] = torch.tensor(crop_coordinates(coarse, [records[i] for i in train]), device=device, dtype=torch.float32)
    if args.stage in ('validate', 'all'):
        selection = validate(images, records, train, coarse, centers, args.output, args.pretrained, sources)
    else:
        selection = read_json(args.output / 'selection.json')
    if args.stage in ('final', 'all'):
        final(images, records, train, test, centers, args.output, args.pretrained, sources, selection)


if __name__ == '__main__':
    main()
