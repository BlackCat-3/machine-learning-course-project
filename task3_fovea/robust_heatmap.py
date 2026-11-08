"""Higher-resolution ResNet34 heatmaps with EMA and illumination augmentation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34

from contracts import coordinate_metrics, write_submission
from data import original_coordinates
from model import FoveaNet, heatmap_coordinates, heatmap_loss
from run import ROOT, augment, normalize, predict, save_json

OUT = ROOT / 'outputs/improvement_20261024/robust_heatmap'
WEIGHTS = ROOT.parent / '.cache/torch/hub/checkpoints/resnet34-b627a593.pth'


class RobustHeatmap(FoveaNet):
    def __init__(self, pretrained=True):
        super().__init__(None)
        self.encoder = resnet34(weights=None)
        if pretrained:
            self.encoder.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self._freeze_encoder_batchnorm()


def illumination_augment(x):
    n, _, h, w = x.shape
    gamma = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.65, 1.5)
    x = x.clamp_min(1e-5).pow(gamma)
    # Smooth directional lighting variation independent of landmark position.
    xx = torch.linspace(-1, 1, w, device=x.device)[None, None, None, :]
    yy = torch.linspace(-1, 1, h, device=x.device)[None, None, :, None]
    dx = torch.empty(n, 1, 1, 1, device=x.device).uniform_(-0.25, 0.25)
    dy = torch.empty(n, 1, 1, 1, device=x.device).uniform_(-0.25, 0.25)
    x = x * (1 + dx * xx + dy * yy)
    if torch.rand(()).item() < 0.25:
        x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, 1)
    return x.clamp(0, 1)


@torch.no_grad()
def update_ema(ema, model, decay=0.95):
    for p, src in zip(ema.parameters(), model.parameters()):
        p.mul_(decay).add_(src, alpha=1 - decay)
    for p, src in zip(ema.buffers(), model.buffers()):
        p.copy_(src)


def train(images, records, fit, val, out, seed, epochs=40):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'best.pt').exists():
        raise FileExistsError(out)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RobustHeatmap().to(images.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    opt = torch.optim.AdamW([{'params': enc, 'lr': 0.00005},
                            {'params': [p for p in model.parameters() if id(p) not in enc_ids], 'lr': 0.0005}], weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=images.is_cuda)
    target = torch.tensor([r.get('xy_crop', [0, 0]) for r in records], device=images.device)
    config = {'architecture': 'ResNet34 FPN', 'size': 448, 'seed': seed, 'epochs_planned': epochs,
              'schedule_epochs': 40, 'ema_decay': 0.95, 'batch_size': 8,
              'train_ids': [records[i]['id'] for i in fit], 'validation_ids': [records[i]['id'] for i in val],
              'coordinate_mse_weight': 100, 'pretrained': str(WEIGHTS),
              'augmentation': 'affine flip brightness contrast color gamma directional lighting blur',
              'metric': 'original-pixel scalar RMSE', 'validation_tta': 'horizontal flip'}
    save_json(out / 'config.json', config)
    best, best_epoch = float('inf'), 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tick = time.perf_counter()
        model.train()
        total = 0.
        perm = np.random.permutation(fit)
        for j in range(0, len(perm), 8):
            idx = perm[j:j + 8]
            x, xy = augment(images[idx].float() / 255, target[idx])
            x = illumination_augment(x)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=images.device.type, enabled=images.is_cuda):
                logits = model(normalize(x))
                loss = heatmap_loss(logits, xy, sigma=1.5) + 95 * F.mse_loss(heatmap_coordinates(logits), xy)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model)
            total += float(loss.detach()) * len(idx)
        scheduler.step()
        row = {'epoch': epoch, 'loss': total / len(fit)}
        if val:
            vr = [records[i] for i in val]
            pred = original_coordinates(predict(ema, images, val, batch_size=8, tta=True), vr)
            row.update(coordinate_metrics(np.array([r['xy'] for r in vr]), pred))
            if row['rmse'] < best:
                best, best_epoch = row['rmse'], epoch
                torch.save({'model': ema.state_dict(), 'epoch': epoch, 'config': config, 'weight_type': 'ema'}, out / 'best.pt')
                np.save(out / 'validation_predictions.npy', pred)
        row['seconds'] = time.perf_counter() - tick
        history.append(row)
        with (out / 'history.csv').open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        save_json(out / 'status.json', {'state': 'running', 'completed_epochs': epoch,
                  'best_epoch': best_epoch, 'best_rmse': best if val else None})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'{out.name} epoch {epoch}/{epochs} loss={row["loss"]:.4f} '
                  + (f'RMSE={row["rmse"]:.2f} best={best:.2f} ' if val else '')
                  + f'sec={row["seconds"]:.1f}', flush=True)
    if not val:
        best_epoch = epochs
        torch.save({'model': ema.state_dict(), 'epoch': epochs, 'config': config, 'weight_type': 'ema'}, out / 'best.pt')
    save_json(out / 'status.json', {'state': 'completed', 'completed_epochs': epochs,
              'best_epoch': best_epoch, 'best_rmse': best if val else None, 'seconds': time.perf_counter() - t0})
    del model, ema, opt
    torch.cuda.empty_cache()


def load_data():
    manifest = json.loads((ROOT / 'outputs/baseline/prepared/manifest.json').read_text(encoding='utf8'))
    records = manifest['records']
    cache = OUT / 'images448.npz'
    if not cache.exists():
        arrays = []
        for r in records:
            with Image.open(r['file']) as image:
                arrays.append(np.asarray(image.convert('RGB').crop(r['crop']).resize((448, 448), Image.Resampling.BILINEAR)))
        np.savez_compressed(cache, images=np.stack(arrays))
    images = torch.from_numpy(np.load(cache)['images'].transpose(0, 3, 1, 2).copy()).cuda()
    return images, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['validate', 'final', 'all'], default='validate')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    plan = {'seeds': [43, 44], 'folds': 4, 'epochs': 40, 'size': 448,
            'candidates': ['robust_only', 'robust_baseline_mean'],
            'gate': {'relative_rmse_reduction': 0.10, 'minimum_improved_folds': 3},
            'validation_note': 'Same model-selection folds as baseline; not an independent test.',
            'training': 'ResNet34; lower backbone LR; EMA; gamma/lighting/blur; 100x coordinate MSE',
            'refit': 'same two seeds; all80; median chosen epoch across folds for each seed'}
    if not (OUT / 'plan.json').exists(): save_json(OUT / 'plan.json', plan)
    images, records = load_data()
    fit_all = [i for i, r in enumerate(records) if r['split'] == 'train']
    test = [i for i, r in enumerate(records) if r['split'] == 'test']
    truth = np.array([records[i]['xy'] for i in fit_all])
    baseline = np.load(ROOT / 'outputs/study/oof_predictions.npz')['equal_mean']
    if args.stage in ('validate', 'all'):
        by_seed, chosen_epochs = {}, {}
        for seed in (43, 44):
            by_seed[seed] = np.zeros_like(truth)
            chosen_epochs[seed] = []
            for fold in range(4):
                fit = [i for i in fit_all if records[i]['fold'] != fold]
                val = [i for i in fit_all if records[i]['fold'] == fold]
                out = OUT / f'fold{fold}_seed{seed}'
                if not (out / 'status.json').exists(): train(images, records, fit, val, out, seed)
                status = json.loads((out / 'status.json').read_text(encoding='utf8'))
                if status['state'] != 'completed': raise RuntimeError(f'Incomplete: {out}')
                by_seed[seed][val] = np.load(out / 'validation_predictions.npy')
                chosen_epochs[seed].append(status['best_epoch'])
        robust = (by_seed[43] + by_seed[44]) / 2
        candidates = {'robust_only': robust, 'robust_baseline_mean': (robust + baseline) / 2}
        baseline_rmse = coordinate_metrics(truth, baseline)['rmse']
        baseline_folds = [coordinate_metrics(truth[[i for i in fit_all if records[i]['fold'] == f]],
                          baseline[[i for i in fit_all if records[i]['fold'] == f]])['rmse'] for f in range(4)]
        details, qualified = {}, []
        for name, prediction in candidates.items():
            metrics = coordinate_metrics(truth, prediction)
            fm = [coordinate_metrics(truth[[i for i in fit_all if records[i]['fold'] == f]],
                  prediction[[i for i in fit_all if records[i]['fold'] == f]])['rmse'] for f in range(4)]
            improved = sum(v < b for v, b in zip(fm, baseline_folds))
            passed = metrics['rmse'] <= baseline_rmse * 0.9 and improved >= 3
            details[name] = {'metrics': metrics, 'fold_rmse': fm, 'improved_folds': improved, 'passed_gate': passed}
            if passed: qualified.append(name)
        chosen = min(qualified, key=lambda name: details[name]['metrics']['rmse']) if qualified else None
        result = {'selected': chosen, 'candidates': details, 'best_epochs': chosen_epochs,
                  'refit_epochs': {str(s): int(np.median(v)) for s, v in chosen_epochs.items()},
                  'baseline_rmse': baseline_rmse, 'baseline_fold_rmse': baseline_folds, 'plan': plan}
        np.savez(OUT / 'oof_predictions.npz', truth=truth, baseline=baseline, seed43=by_seed[43], seed44=by_seed[44], **candidates)
        save_json(OUT / 'selection.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage in ('final', 'all'):
        selection = json.loads((OUT / 'selection.json').read_text(encoding='utf8'))
        if selection['selected'] is None:
            print('Gate not passed; preserving baseline submission.', flush=True)
            return
        predictions, sources = [], []
        for seed in (43, 44):
            out = OUT / f'full_seed{seed}'
            epochs = selection['refit_epochs'][str(seed)]
            if not (out / 'status.json').exists(): train(images, records, fit_all, [], out, seed, epochs)
            status = json.loads((out / 'status.json').read_text(encoding='utf8'))
            if status['state'] != 'completed' or status['completed_epochs'] != epochs: raise RuntimeError('Incomplete refit')
            model = RobustHeatmap(False).cuda()
            checkpoint = torch.load(out / 'best.pt', map_location='cuda', weights_only=True)
            model.load_state_dict(checkpoint['model'])
            pred = original_coordinates(predict(model, images, test, 8, tta=True), [records[i] for i in test])
            np.save(out / 'test_predictions.npy', pred)
            predictions.append(pred)
            sources.append({'seed': seed, 'epochs': epochs, 'checkpoint': str(out / 'best.pt'),
                            'sha256': hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest()})
            del model
        final = np.mean(predictions, axis=0)
        if selection['selected'] == 'robust_baseline_mean':
            original = json.loads((ROOT / 'outputs/study/predictions.json').read_text(encoding='utf8'))
            final = (final + np.array([original[str(records[i]['id'])] for i in test])) / 2
        values = {records[i]['id']: final[k] for k, i in enumerate(test)}
        sizes = {records[i]['id']: (records[i]['width'], records[i]['height']) for i in test}
        csv_path = OUT / 'submission_robust.csv'
        write_submission(ROOT / 'data/detection/sample_submission.csv', values, csv_path, sizes)
        save_json(OUT / 'result.json', {'selection': selection, 'models': sources, 'train_count': len(fit_all),
                  'rows': 2 * len(test), 'submission_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  'online_score': None, 'note': 'Validation score is not final ensemble independent test or online score.'})
        print(csv_path, flush=True)


if __name__ == '__main__':
    main()
