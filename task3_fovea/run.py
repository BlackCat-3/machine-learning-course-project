"""Reproducible validation, full-data refit, and Kaggle CSV generation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from contracts import coordinate_metrics, write_submission
from data import original_coordinates, prepare
from model import FoveaNet, heatmap_coordinates, heatmap_loss

ROOT = Path(__file__).resolve().parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def normalize(x):
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    return (x - mean) / std


def make_model(architecture, pretrained):
    if architecture == 'heatmap':
        return FoveaNet(pretrained)
    from model_regression import FoveaRegressor
    return FoveaRegressor(pretrained)


def model_coordinates(model, x):
    output = model(x)
    return heatmap_coordinates(output) if output.ndim == 4 else output.float()


def augment(x, xy):
    n = len(x)
    angle = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(12)
    scale = 0.9 + torch.rand(n, device=x.device) * 0.2
    flip = torch.where(torch.rand(n, device=x.device) < 0.5, -1., 1.)
    a = torch.zeros(n, 2, 2, device=x.device)
    a[:, 0, 0] = scale * angle.cos() * flip
    a[:, 0, 1] = -scale * angle.sin()
    a[:, 1, 0] = scale * angle.sin() * flip
    a[:, 1, 1] = scale * angle.cos()
    translation = (torch.rand(n, 2, device=x.device) * 2 - 1) * 0.10
    theta = torch.cat((a, translation[:, :, None]), dim=2)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    xy = (torch.linalg.solve(a, (xy * 2 - 1 - translation)[:, :, None]).squeeze(-1) + 1) / 2
    brightness = 0.8 + torch.rand(n, 1, 1, 1, device=x.device) * 0.4
    contrast = 0.85 + torch.rand(n, 1, 1, 1, device=x.device) * 0.3
    color = 0.9 + torch.rand(n, 3, 1, 1, device=x.device) * 0.2
    center = x.mean((2, 3), keepdim=True)
    x = ((x - center) * contrast + center) * brightness * color
    return x.clamp(0, 1), xy


@torch.inference_mode()
def predict(model, images, indices, batch_size=16, tta=False):
    model.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        x = images[indices[start:start + batch_size]].float() / 255
        with torch.autocast(device_type=images.device.type, enabled=images.is_cuda):
            xy = model_coordinates(model, normalize(x))
            if tta:
                other = model_coordinates(model, normalize(x.flip(-1)))
                other[:, 0] = 1 - other[:, 0]
                xy = (xy + other) / 2
        predictions.append(xy.cpu().numpy())
    return np.concatenate(predictions)


def train_one(images, records, train_idx, val_idx, out, args, seed, epochs):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'best.pt').exists():
        raise FileExistsError(f'Existing checkpoint: use a new output directory: {out}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = make_model(args.arch, args.pretrained).to(images.device)
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': 0.0002},
                                   {'params': decoder, 'lr': 0.001}], weight_decay=0.0001)
    # Keep identical schedule in validation and refit, even if refit stops at best epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=images.is_cuda)
    all_targets = torch.tensor([r.get('xy_crop', [0, 0]) for r in records], device=images.device, dtype=torch.float32)
    config = {'seed': seed, 'epochs_planned': epochs, 'schedule_epochs': args.epochs,
              'train_ids': [records[i]['id'] for i in train_idx],
              'validation_ids': [records[i]['id'] for i in val_idx],
              'size': args.size, 'batch_size': args.batch_size, 'architecture': args.arch,
              'pretrained': str(Path(args.pretrained).resolve()), 'tta': 'horizontal-flip mean',
              'optimizer': 'AdamW', 'encoder_lr': 0.0002, 'decoder_lr': 0.001,
              'metric': 'sqrt(mean squared error of all X and Y scalars), original pixels'}
    save_json(out / 'config.json', config)
    best_score, best_epoch = float('inf'), 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = np.random.permutation(train_idx)
        total_loss = 0.
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            x, xy = augment(images[idx].float() / 255, all_targets[idx])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=images.device.type, enabled=images.is_cuda):
                output = model(normalize(x))
                loss = heatmap_loss(output, xy) if output.ndim == 4 else 100 * F.mse_loss(output.float(), xy)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(idx)
        scheduler.step()
        row = {'epoch': epoch, 'loss': total_loss / len(train_idx)}
        if val_idx:
            val_records = [records[i] for i in val_idx]
            raw = predict(model, images, val_idx, args.batch_size, tta=True)
            pred = original_coordinates(raw, val_records)
            metrics = coordinate_metrics(np.array([r['xy'] for r in val_records]), pred)
            row.update(metrics)
            if metrics['rmse'] < best_score:
                best_score, best_epoch = metrics['rmse'], epoch
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'config': config}, out / 'best.pt')
                np.save(out / 'validation_predictions.npy', pred)
        row['seconds'] = time.perf_counter() - t0
        history.append(row)
        with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        save_json(out / 'status.json', {'state': 'running', 'completed_epochs': epoch,
                                      'best_epoch': best_epoch, 'best_rmse': best_score if val_idx else None})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'{out.name} epoch {epoch}/{epochs} loss={row["loss"]:.4f} '
                  f'rmse={row.get("rmse", float("nan")):.2f} best={best_score:.2f} sec={row["seconds"]:.1f}', flush=True)
    if not val_idx:
        best_epoch = epochs
        torch.save({'model': model.state_dict(), 'epoch': epochs, 'config': config}, out / 'best.pt')
    result = {'state': 'completed', 'completed_epochs': epochs, 'best_epoch': best_epoch,
              'best_rmse': best_score if val_idx else None, 'seconds': time.perf_counter() - started}
    save_json(out / 'status.json', result)
    del model, optimizer
    if images.is_cuda:
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'validate', 'final', 'all'], default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/detection')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/baseline')
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', choices=['heatmap', 'regression'], default='heatmap')
    parser.add_argument('--pretrained', default=str(ROOT.parent / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = args.output / 'prepared'
    if args.stage in ('prepare', 'all') or not (prepared / 'manifest.json').exists():
        manifest = prepare(args.data_dir, prepared, args.size)
    else:
        manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['size'] != args.size:
        raise ValueError('Cached image size differs; use a new output directory')
    if args.stage == 'prepare':
        print('Prepared', manifest['fold_counts'], flush=True)
        return
    records = manifest['records']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = torch.from_numpy(np.load(prepared / 'images.npz')['images'].transpose(0, 3, 1, 2).copy()).to(device)
    train_idx = [i for i, r in enumerate(records) if r['split'] == 'train' and r['fold'] != args.fold]
    val_idx = [i for i, r in enumerate(records) if r['split'] == 'train' and r['fold'] == args.fold]
    validation_dir = args.output / f'validation_fold{args.fold}_seed{args.seed}'
    if args.stage in ('validate', 'all'):
        true = np.array([records[i]['xy'] for i in val_idx])
        train_mean = np.mean([records[i]['xy_crop'] for i in train_idx], axis=0)
        baseline = original_coordinates(np.tile(train_mean, (len(val_idx), 1)), [records[i] for i in val_idx])
        save_json(args.output / 'constant_baseline.json', coordinate_metrics(true, baseline))
        train_one(images, records, train_idx, val_idx, validation_dir, args, args.seed, args.epochs)
    if args.stage in ('final', 'all'):
        selection = json.loads((validation_dir / 'status.json').read_text(encoding='utf-8'))
        if selection['state'] != 'completed':
            raise ValueError('Validation run incomplete')
        selected_config = json.loads((validation_dir / 'config.json').read_text(encoding='utf-8'))
        expected = {'seed': args.seed, 'schedule_epochs': args.epochs, 'size': args.size,
                    'batch_size': args.batch_size, 'pretrained': str(Path(args.pretrained).resolve())}
        if any(selected_config.get(k) != v for k, v in expected.items()):
            raise ValueError('Refit arguments do not match the validation configuration')
        if selected_config.get('architecture', 'heatmap') != args.arch:
            raise ValueError('Refit architecture differs from validation')
        all_train = [i for i, r in enumerate(records) if r['split'] == 'train']
        test_idx = [i for i, r in enumerate(records) if r['split'] == 'test']
        final_dir = args.output / f'full_seed{args.seed}'
        train_one(images, records, all_train, [], final_dir, args, args.seed, selection['best_epoch'])
        model = make_model(args.arch, None).to(device)
        checkpoint = torch.load(final_dir / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        test_records = [records[i] for i in test_idx]
        xy = original_coordinates(predict(model, images, test_idx, args.batch_size, tta=True), test_records)
        predictions = {r['id']: xy[j].tolist() for j, r in enumerate(test_records)}
        image_sizes = {r['id']: (r['width'], r['height']) for r in test_records}
        output = args.output / 'submission.csv'
        write_submission(args.data_dir / 'sample_submission.csv', predictions, output, image_sizes)
        save_json(args.output / 'predictions.json', predictions)
        save_json(args.output / 'result.json', {'validation': selection, 'refit_epochs': selection['best_epoch'],
                  'refit_train_count': len(all_train), 'test_count': len(test_idx), 'submission_rows': 2 * len(test_idx),
                  'submission_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                  'online_score': None, 'limitations': manifest['limitations']})
        print('Submission:', output, flush=True)


if __name__ == '__main__':
    main()
