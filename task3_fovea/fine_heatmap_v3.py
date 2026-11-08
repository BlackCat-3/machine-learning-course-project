"""Isolated ResNet50/512px fovea experiment with a shared OOF-selected epoch.

Validation saves every epoch's predictions. Only epochs 6/10/16/24/32/40 are
eligible for shared epoch selection; an individual fold's best checkpoint is
audit-only. Both full-data seeds use the same selected epoch and the original
40-epoch LR schedule. No downloads or external training data are used.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50

from contracts import coordinate_metrics, load_labels, load_template, read_submission, write_submission
from data import original_coordinates
from model import FoveaNet, heatmap_coordinates, heatmap_loss
from robust_heatmap import illumination_augment
from run import ROOT, augment, normalize, predict


SIZE, BATCH_SIZE, SCHEDULE_EPOCHS = 512, 8, 40
SEEDS, CANDIDATE_EPOCHS = (43, 44), (6, 10, 16, 24, 32, 40)
EMA_DECAY, SIGMA, COORDINATE_WEIGHT = 0.95, 1.0, 100.0
ENCODER_LR, HEAD_LR, WEIGHT_DECAY, ETA_MIN = 2.5e-5, 5e-4, 2e-4, 1e-6
EXPECTED_BASELINE_RMSE = 16.334263567891036
MANIFEST = ROOT / 'outputs/baseline/prepared/manifest.json'
BASELINE_DIR = ROOT / 'outputs/improvement_20261024/robust_heatmap'
WEIGHTS = ROOT.parent / '.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'
DEFAULT_OUTPUT = ROOT / 'outputs/improvement_20261024/fine_heatmap_v3'
TEMPLATE = ROOT / 'data/detection/sample_submission.csv'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def save_checkpoint(path, model, epoch, config):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'config': config, 'weight_type': 'ema'}, temporary)
    os.replace(temporary, path)


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def require_same_json(path, expected):
    if Path(path).exists():
        if read_json(path) != expected:
            raise ValueError(f'Incompatible saved configuration; use a new output directory: {path}')
    else:
        write_json(path, expected)


class FineHeatmap50(FoveaNet):
    """ResNet50 encoder with correctly sized bottleneck-to-FPN projections."""

    def __init__(self, pretrained=True):
        super().__init__(None)
        self.encoder = resnet50(weights=None)
        if pretrained:
            self.encoder.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.lateral = nn.ModuleList(nn.Conv2d(channels, 64, 1) for channels in (256, 512, 1024, 2048))
        self._freeze_encoder_batchnorm()


def fine_loss(logits, targets):
    # heatmap_loss already includes five times coordinate MSE.
    return heatmap_loss(logits, targets, sigma=SIGMA) + (COORDINATE_WEIGHT - 5) * F.mse_loss(
        heatmap_coordinates(logits), targets.float())


@torch.no_grad()
def update_ema(ema, model):
    for parameter, current in zip(ema.parameters(), model.parameters()):
        parameter.mul_(EMA_DECAY).add_(current, alpha=1 - EMA_DECAY)
    for buffer, current in zip(ema.buffers(), model.buffers()):
        buffer.copy_(current)


def experiment_plan():
    return {
        'architecture': 'ResNet50 FPN', 'encoder_channels': [256, 512, 1024, 2048], 'decoder_channels': 64,
        'size': SIZE, 'batch_size': BATCH_SIZE, 'schedule_epochs': SCHEDULE_EPOCHS,
        'seeds': list(SEEDS), 'folds': 4, 'candidate_epochs': list(CANDIDATE_EPOCHS),
        'shared_epoch_rule': 'minimum pooled scalar RMSE of the two-seed mean over all four folds; ties choose earlier epoch',
        'candidate_rule': 'after shared epoch selection, compare new_only and equal-mean new_robust_mean only',
        'checkpoint_note': 'per-fold best checkpoints are audit-only and never used in shared OOF selection or refit',
        'encoder_lr': ENCODER_LR, 'head_lr': HEAD_LR, 'weight_decay': WEIGHT_DECAY, 'eta_min': ETA_MIN,
        'ema_decay': EMA_DECAY, 'gaussian_sigma_cells': SIGMA, 'normalized_coordinate_mse_weight': COORDINATE_WEIGHT,
        'augmentation': 'same affine/flip/brightness/contrast/color plus gamma/directional-lighting/blur as robust baseline',
        'validation_tta': 'horizontal flip; normalized coordinates restored before averaging',
        'gate': {'relative_rmse_reduction': 0.05, 'minimum_improved_folds': 3},
        'refit_rule': 'all 80 records; both seeds use the same shared epoch; cosine schedule stays at 40',
        'validation_scope': 'model-selection validation on the original grouped folds; not an independent test',
    }


def load_inputs():
    manifest = read_json(MANIFEST)
    records = manifest['records']
    train = [index for index, record in enumerate(records) if record['split'] == 'train']
    test = [index for index, record in enumerate(records) if record['split'] == 'test']
    if len(train) != 80 or len(test) != 20:
        raise ValueError('Expected the original 80 training and 20 test images')
    if len({record['id'] for record in records}) != len(records):
        raise ValueError('Duplicate image IDs in the manifest')
    if {records[index]['fold'] for index in train} != set(range(4)):
        raise ValueError('Expected four unchanged folds')
    groups = {}
    for index in train:
        groups.setdefault(records[index]['pixel_sha256'], set()).add(records[index]['fold'])
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError('Exact duplicate training images cross validation folds')
    labels = load_labels(ROOT / 'data/detection/fovea_localization_train_GT.csv')
    if set(labels) != {records[index]['id'] for index in train}:
        raise ValueError('Training label IDs differ from the manifest')
    for index in train:
        record = records[index]
        if not np.array_equal(labels[record['id']], record['xy']):
            raise ValueError(f'Training label changed: {record["id"]}')
    template = load_template(TEMPLATE)
    if {int(identifier.split('_')[0]) for identifier in template} != {records[index]['id'] for index in test}:
        raise ValueError('Test IDs differ from the submission template')
    truth = np.asarray([records[index]['xy'] for index in train], dtype=np.float64)
    with np.load(BASELINE_DIR / 'oof_predictions.npz', allow_pickle=False) as stored:
        if not np.array_equal(stored['truth'], truth):
            raise ValueError('Baseline OOF order or training coordinates differ')
        baseline = stored['robust_only'].copy()
    if baseline.shape != truth.shape or not np.isfinite(baseline).all():
        raise ValueError('Invalid baseline OOF predictions')
    baseline_metrics = coordinate_metrics(truth, baseline)
    if abs(baseline_metrics['rmse'] - EXPECTED_BASELINE_RMSE) > 1e-6:
        raise ValueError('The expected robust OOF reference has changed')
    result = read_json(BASELINE_DIR / 'result.json')
    if result['selection']['selected'] != 'robust_only':
        raise ValueError('Expected the currently submitted robust-only baseline')
    baseline_csv = BASELINE_DIR / 'submission_robust.csv'
    if file_sha256(baseline_csv) != result['submission_sha256']:
        raise ValueError('Robust submission differs from its recorded final artifact')
    sizes = {records[index]['id']: (records[index]['width'], records[index]['height']) for index in test}
    test_baseline = read_submission(baseline_csv, sizes, TEMPLATE)
    if set(test_baseline) != set(sizes):
        raise ValueError('Baseline test prediction IDs differ')
    source_paths = [Path(__file__), ROOT / 'model.py', ROOT / 'run.py', ROOT / 'robust_heatmap.py',
                    ROOT / 'data.py', ROOT / 'contracts.py', MANIFEST, WEIGHTS, TEMPLATE,
                    ROOT / 'data/detection/fovea_localization_train_GT.csv',
                    BASELINE_DIR / 'oof_predictions.npz', BASELINE_DIR / 'result.json', baseline_csv]
    sources = {str(path.resolve()): file_sha256(path) for path in source_paths}
    return records, train, test, truth, baseline, test_baseline, sources


def load_images(records, output, manifest_hash, device):
    # Verify the original images even when a cache already exists.
    for record in records:
        if file_sha256(record['file']) != record['sha256']:
            raise ValueError(f'Original image changed: {record["file"]}')
    cache = output / 'images512.npz'
    ids = np.asarray([record['id'] for record in records])
    if not cache.exists():
        arrays = []
        for record in records:
            with Image.open(record['file']) as image:
                arrays.append(np.asarray(image.convert('RGB').crop(record['crop']).resize(
                    (SIZE, SIZE), Image.Resampling.BILINEAR)))
        save_npz(cache, images=np.stack(arrays), ids=ids, manifest_sha256=manifest_hash)
    with np.load(cache, allow_pickle=False) as stored:
        if str(stored['manifest_sha256'].item()) != manifest_hash or not np.array_equal(stored['ids'], ids):
            raise ValueError('Cached image provenance or order differs')
        images = stored['images']
    if images.shape != (len(records), SIZE, SIZE, 3) or images.dtype != np.uint8:
        raise ValueError('Unexpected image cache shape or dtype')
    return torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device), file_sha256(cache)


def run_config(records, fit, validation, seed, epochs, provenance):
    return {
        **experiment_plan(), 'seed': seed, 'epochs_planned': epochs,
        'train_ids': [records[index]['id'] for index in fit],
        'validation_ids': [records[index]['id'] for index in validation],
        'training_stage': 'validation' if validation else 'full_data_refit',
        'sources': provenance,
    }


def read_epoch_prediction(run_dir, epoch, expected_ids):
    path = run_dir / 'epoch_predictions' / f'epoch_{epoch:03d}.npz'
    with np.load(path, allow_pickle=False) as stored:
        if stored['ids'].tolist() != expected_ids:
            raise ValueError(f'Validation prediction order changed: {path}')
        prediction = stored['predictions'].copy()
    if prediction.shape != (len(expected_ids), 2) or not np.isfinite(prediction).all():
        raise ValueError(f'Invalid validation prediction array: {path}')
    return prediction


def train_run(images, records, fit, validation, output, seed, epochs, provenance):
    if set(fit) & set(validation) or not fit:
        raise ValueError('Training/validation overlap or empty training data')
    if not 1 <= epochs <= SCHEDULE_EPOCHS:
        raise ValueError('Training epochs must lie within the fixed schedule')
    config = run_config(records, fit, validation, seed, epochs, provenance)
    if (output / 'status.json').exists():
        status = read_json(output / 'status.json')
        if read_json(output / 'config.json') != config or status.get('state') != 'completed' or status.get('completed_epochs') != epochs:
            raise RuntimeError(f'Incomplete/incompatible run; use a fresh directory: {output}')
        required = [output / 'audit_best.pt'] if validation else [output / 'final.pt']
        if validation:
            required += [output / f'epoch_{epoch:03d}.pt' for epoch in CANDIDATE_EPOCHS]
            for epoch in range(1, epochs + 1):
                read_epoch_prediction(output, epoch, config['validation_ids'])
        if any(not path.is_file() for path in required):
            raise FileNotFoundError(f'Completed run has missing checkpoints: {output}')
        return status
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Unrecognized existing run directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    if validation:
        (output / 'epoch_predictions').mkdir()
    write_json(output / 'config.json', config)
    write_json(output / 'status.json', {'state': 'starting', 'completed_epochs': 0})
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = FineHeatmap50().to(images.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    optimizer = torch.optim.AdamW([
        {'params': encoder, 'lr': ENCODER_LR},
        {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULE_EPOCHS, eta_min=ETA_MIN)
    scaler = torch.amp.GradScaler('cuda', enabled=images.is_cuda)
    target = torch.full((len(records), 2), float('nan'), device=images.device)
    target[fit] = torch.tensor([records[index]['xy_crop'] for index in fit], device=images.device, dtype=torch.float32)
    validation_records = [records[index] for index in validation]
    truth = np.asarray([record['xy'] for record in validation_records]) if validation else None
    best_score, best_epoch, history = float('inf'), 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tick, loss_sum = time.perf_counter(), 0.0
        encoder_lr, head_lr = [group['lr'] for group in optimizer.param_groups]
        model.train()
        order = np.random.permutation(fit)
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE].tolist()
            x, xy = augment(images[indices].float() / 255, target[indices])
            x = illumination_augment(x)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(images.device.type, enabled=images.is_cuda):
                loss = fine_loss(model(normalize(x)), xy)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at {output.name}, epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                update_ema(ema, model)
            loss_sum += float(loss.detach()) * len(indices)
        scheduler.step()
        row = {'epoch': epoch, 'loss': loss_sum / len(fit), 'encoder_lr': encoder_lr, 'head_lr': head_lr}
        if validation:
            normalized = predict(ema, images, validation, BATCH_SIZE, tta=True)
            prediction = original_coordinates(normalized, validation_records)
            row.update(coordinate_metrics(truth, prediction))
            save_npz(output / 'epoch_predictions' / f'epoch_{epoch:03d}.npz',
                     ids=np.asarray(config['validation_ids']), predictions=prediction)
            if row['rmse'] < best_score:
                best_score, best_epoch = row['rmse'], epoch
                save_checkpoint(output / 'audit_best.pt', ema, epoch, config)
            if epoch in CANDIDATE_EPOCHS:
                save_checkpoint(output / f'epoch_{epoch:03d}.pt', ema, epoch, config)
        row['seconds'] = time.perf_counter() - tick
        history.append(row)
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        write_json(output / 'status.json', {
            'state': 'running', 'completed_epochs': epoch,
            'audit_best_epoch': best_epoch if validation else None,
            'audit_best_rmse': best_score if validation else None,
        })
        if epoch == 1 or epoch % 5 == 0 or epoch in CANDIDATE_EPOCHS or epoch == epochs:
            print(f'{output.name} epoch {epoch}/{epochs} loss={row["loss"]:.4f} '
                  + (f'RMSE={row["rmse"]:.3f} audit_best={best_score:.3f} ' if validation else '')
                  + f'sec={row["seconds"]:.1f}', flush=True)
    if not validation:
        save_checkpoint(output / 'final.pt', ema, epochs, config)
    status = {
        'state': 'completed', 'completed_epochs': epochs,
        'audit_best_epoch': best_epoch if validation else None,
        'audit_best_rmse': best_score if validation else None,
        'candidate_epochs_saved': list(CANDIDATE_EPOCHS) if validation else [],
        'seconds': time.perf_counter() - started,
    }
    write_json(output / 'status.json', status)
    del model, ema, optimizer, scheduler, scaler
    if images.is_cuda:
        torch.cuda.empty_cache()
    return status


def fold_metrics(truth, prediction, folds):
    return [coordinate_metrics(truth[folds == fold], prediction[folds == fold]) for fold in range(4)]


def validate(images, records, train, truth, baseline, output, provenance):
    positions = {index: position for position, index in enumerate(train)}
    folds = np.asarray([records[index]['fold'] for index in train])
    audit = {}
    for seed in SEEDS:
        for fold in range(4):
            fit = [index for index in train if records[index]['fold'] != fold]
            validation = [index for index in train if records[index]['fold'] == fold]
            run_dir = output / f'fold{fold}_seed{seed}'
            audit[run_dir.name] = train_run(images, records, fit, validation, run_dir, seed, SCHEDULE_EPOCHS, provenance)
    prediction_arrays, epoch_metrics, epoch_fold_metrics = {}, {}, {}
    for epoch in CANDIDATE_EPOCHS:
        seed_predictions = []
        for seed in SEEDS:
            prediction = np.full_like(truth, np.nan)
            for fold in range(4):
                validation = [index for index in train if records[index]['fold'] == fold]
                expected_ids = [records[index]['id'] for index in validation]
                values = read_epoch_prediction(output / f'fold{fold}_seed{seed}', epoch, expected_ids)
                prediction[[positions[index] for index in validation]] = values
            if not np.isfinite(prediction).all():
                raise ValueError('Shared OOF prediction coverage is incomplete')
            prediction_arrays[f'epoch_{epoch:03d}_seed{seed}'] = prediction
            seed_predictions.append(prediction)
        mean_prediction = np.mean(seed_predictions, axis=0)
        prediction_arrays[f'epoch_{epoch:03d}_mean'] = mean_prediction
        epoch_metrics[str(epoch)] = coordinate_metrics(truth, mean_prediction)
        epoch_fold_metrics[str(epoch)] = fold_metrics(truth, mean_prediction, folds)
    # One epoch for every fold and both seeds, chosen before the two-way blend comparison.
    shared_epoch = min(CANDIDATE_EPOCHS, key=lambda epoch: (epoch_metrics[str(epoch)]['rmse'], epoch))
    new = prediction_arrays[f'epoch_{shared_epoch:03d}_mean']
    candidates = {'new_only': new, 'new_robust_mean': (new + baseline) / 2}
    baseline_metrics = coordinate_metrics(truth, baseline)
    baseline_folds = fold_metrics(truth, baseline, folds)
    details, eligible = {}, []
    for name, prediction in candidates.items():
        metrics = coordinate_metrics(truth, prediction)
        per_fold = fold_metrics(truth, prediction, folds)
        improved = sum(value['rmse'] < reference['rmse'] for value, reference in zip(per_fold, baseline_folds))
        passed = metrics['rmse'] <= 0.95 * baseline_metrics['rmse'] and improved >= 3
        details[name] = {'metrics': metrics, 'fold_metrics': per_fold, 'improved_folds': improved, 'passed_gate': passed}
        if passed:
            eligible.append(name)
    selected = min(eligible, key=lambda name: details[name]['metrics']['rmse']) if eligible else None
    shared_checkpoints = []
    for seed in SEEDS:
        for fold in range(4):
            path = output / f'fold{fold}_seed{seed}' / f'epoch_{shared_epoch:03d}.pt'
            shared_checkpoints.append({'seed': seed, 'fold': fold, 'epoch': shared_epoch,
                                       'path': str(path.resolve()), 'sha256': file_sha256(path)})
    selection = {
        'selected': selected, 'gate_passed': bool(eligible), 'shared_epoch': shared_epoch,
        'refit_epochs': {str(seed): shared_epoch for seed in SEEDS},
        'baseline': baseline_metrics, 'baseline_folds': baseline_folds, 'candidates': details,
        'epoch_metrics': epoch_metrics, 'epoch_fold_metrics': epoch_fold_metrics,
        'shared_epoch_checkpoints': shared_checkpoints, 'audit_only_fold_best': audit,
        'train_ids': [records[index]['id'] for index in train], 'plan': experiment_plan(), 'provenance': provenance,
    }
    save_npz(output / 'oof_predictions.npz', ids=np.asarray(selection['train_ids']), folds=folds,
             truth=truth, baseline=baseline, **prediction_arrays, **candidates)
    selection['oof_predictions_sha256'] = file_sha256(output / 'oof_predictions.npz')
    write_json(output / 'selection.json', selection)
    print(json.dumps({'shared_epoch': shared_epoch, 'selected': selected, 'candidates': details}, ensure_ascii=False, indent=2), flush=True)
    return selection


def final(images, records, train, test, baseline_test, output, provenance, selection):
    if selection['provenance'] != provenance or selection['plan'] != experiment_plan():
        raise ValueError('Sources, cache, runtime, or plan differ from shared-epoch validation')
    if selection['train_ids'] != [records[index]['id'] for index in train]:
        raise ValueError('Full-data training IDs differ from the validation manifest')
    if file_sha256(output / 'oof_predictions.npz') != selection['oof_predictions_sha256']:
        raise ValueError('OOF selection artifact changed')
    if not selection['gate_passed']:
        print('Shared-epoch candidate did not pass the gate; no new CSV generated.', flush=True)
        return
    selected = selection['selected']
    if selected not in ('new_only', 'new_robust_mean') or not selection['candidates'][selected]['passed_gate']:
        raise ValueError('Invalid selected candidate')
    epoch = selection['shared_epoch']
    if epoch not in CANDIDATE_EPOCHS or any(value != epoch for value in selection['refit_epochs'].values()):
        raise ValueError('Both refits must use the same eligible shared epoch')
    destination = output / 'submission_fine_v3.csv'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite an existing submission: {destination}')
    predictions, sources = [], []
    test_records = [records[index] for index in test]
    for seed in SEEDS:
        run_dir = output / f'full_seed{seed}'
        train_run(images, records, train, [], run_dir, seed, epoch, provenance)
        checkpoint_path = run_dir / 'final.pt'
        checkpoint = torch.load(checkpoint_path, map_location=images.device, weights_only=True)
        expected_config = run_config(records, train, [], seed, epoch, provenance)
        if checkpoint['epoch'] != epoch or checkpoint['config'] != expected_config or checkpoint.get('weight_type') != 'ema':
            raise ValueError('Full-data checkpoint does not match the shared-epoch refit')
        model = FineHeatmap50(pretrained=False).to(images.device)
        model.load_state_dict(checkpoint['model'], strict=True)
        prediction = original_coordinates(predict(model, images, test, BATCH_SIZE, tta=True), test_records)
        save_npz(run_dir / 'test_predictions.npz', ids=np.asarray([record['id'] for record in test_records]), predictions=prediction)
        predictions.append(prediction)
        sources.append({'seed': seed, 'epoch': epoch, 'checkpoint': str(checkpoint_path.resolve()),
                        'sha256': file_sha256(checkpoint_path)})
        del model, checkpoint
        if images.is_cuda:
            torch.cuda.empty_cache()
    final_prediction = np.mean(predictions, axis=0)
    if selected == 'new_robust_mean':
        final_prediction = (final_prediction + np.asarray([baseline_test[record['id']] for record in test_records])) / 2
    values = {record['id']: final_prediction[index] for index, record in enumerate(test_records)}
    sizes = {record['id']: (record['width'], record['height']) for record in test_records}
    write_submission(TEMPLATE, values, destination, sizes)
    write_json(output / 'predictions.json', {key: value.tolist() for key, value in values.items()})
    write_json(output / 'result.json', {
        'selection': selection, 'models': sources, 'refit_epochs': epoch, 'train_count': len(train),
        'test_count': len(test), 'submission_rows': 2 * len(test), 'submission_sha256': file_sha256(destination),
        'online_score': None,
        'note': 'Shared-epoch OOF scores are model-selection validation; the full-data two-seed ensemble has no independent held-out score.',
    })
    print(f'New submission: {destination}', flush=True)


def self_check():
    """CPU-only pretrained loading, FPN channels, backward, and EMA checks."""
    torch.set_num_threads(2)
    torch.manual_seed(43)
    model = FineHeatmap50(pretrained=True).cpu().train()
    assert [layer.in_channels for layer in model.lateral] == [256, 512, 1024, 2048]
    batch_norms = [module for module in model.encoder.modules() if isinstance(module, nn.BatchNorm2d)]
    before = [(module.running_mean.clone(), module.running_var.clone()) for module in batch_norms]
    output = model(torch.randn(2, 3, 64, 64, device='cpu'))
    assert output.shape == (2, 1, 16, 16), output.shape
    target = torch.tensor([[0.3, 0.6], [0.65, 0.4]])
    loss = fine_loss(output, target)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (model.encoder.conv1.weight, model.lateral[3].weight, model.head[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    for module, (mean, variance) in zip(batch_norms, before):
        assert not module.training and torch.equal(module.running_mean, mean) and torch.equal(module.running_var, variance)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    old = ema.head[-1].bias.clone()
    with torch.no_grad():
        model.head[-1].bias.add_(1)
    update_ema(ema, model)
    assert torch.allclose(ema.head[-1].bias, old + (1 - EMA_DECAY), atol=1e-6)
    print(json.dumps({'cpu_self_check': 'passed', 'output_shape': list(output.shape),
                      'loss': float(loss.detach()), 'encoder_channels': [256, 512, 1024, 2048],
                      'frozen_batchnorm_layers': len(batch_norms), 'gpu_used': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'validate', 'final', 'all', 'self-check'), default='validate')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.stage == 'self-check':
        self_check()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    records, train, test, truth, baseline, baseline_test, source_hashes = load_inputs()
    require_same_json(args.output / 'plan.json', experiment_plan())
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    images, cache_hash = load_images(records, args.output, source_hashes[str(MANIFEST.resolve())], device)
    provenance = {
        'source_sha256': source_hashes, 'cache_sha256': cache_hash,
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torchvision': str(torchvision.__version__), 'numpy': str(np.__version__),
                    'cuda_build': torch.version.cuda, 'device_type': device.type, 'cudnn_benchmark': True},
    }
    require_same_json(args.output / 'provenance.json', provenance)
    if args.stage == 'prepare':
        print(json.dumps({'state': 'prepared', 'train_count': len(train), 'test_count': len(test),
                          'baseline_rmse': coordinate_metrics(truth, baseline)['rmse']}), flush=True)
        return
    if args.stage in ('validate', 'all'):
        selection = validate(images, records, train, truth, baseline, args.output, provenance)
    else:
        selection = read_json(args.output / 'selection.json')
    if args.stage in ('final', 'all'):
        final(images, records, train, test, baseline_test, args.output, provenance, selection)


if __name__ == '__main__':
    main()
