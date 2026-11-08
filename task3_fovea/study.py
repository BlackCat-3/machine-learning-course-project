"""Compare three fixed candidates by grouped four-fold out-of-fold RMSE."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from contracts import coordinate_metrics, write_submission
from data import original_coordinates, prepare
from run import ROOT, make_model, predict, save_json, train_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['validate', 'final', 'all'], default='all')
    cli = parser.parse_args()
    study_dir = ROOT / 'outputs/study'
    study_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = ROOT / 'outputs/baseline'
    prepared = baseline_dir / 'prepared'
    data_dir = ROOT / 'data/detection'
    args = SimpleNamespace(epochs=80, size=320, batch_size=16,
                           pretrained=str(ROOT.parent / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'))
    if not (prepared / 'manifest.json').exists():
        prepare(data_dir, prepared, args.size)
    manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    records = manifest['records']
    train_idx = [i for i, r in enumerate(records) if r['split'] == 'train']
    test_idx = [i for i, r in enumerate(records) if r['split'] == 'test']
    truth = np.array([records[i]['xy'] for i in train_idx])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = torch.from_numpy(np.load(prepared / 'images.npz')['images'].transpose(0, 3, 1, 2).copy()).to(device)
    plan = {'candidate_names': ['heatmap', 'regression', 'equal_mean'], 'folds': 4, 'seed': 42,
            'selection': 'lowest pooled original-pixel scalar OOF RMSE',
            'refit': 'two new seeds, all 80 training records, median best epoch across four folds',
            'augmentation': 'same geometric and photometric augmentation for both architectures',
            'public_leaderboard_used': False,
            'caution': 'Checkpoint and candidate selection use these folds; this is model-selection validation, not an untouched test.'}
    if not (study_dir / 'plan.json').exists():
        save_json(study_dir / 'plan.json', plan)
    if cli.stage in ('validate', 'all'):
        oof = {}
        epochs = {}
        fold_metrics = {}
        baseline_oof = np.zeros_like(truth)
        for arch in ('heatmap', 'regression'):
            args.arch = arch
            oof[arch] = np.zeros_like(truth)
            epochs[arch] = []
            fold_metrics[arch] = []
            for fold in range(4):
                fit_idx = [i for i in train_idx if records[i]['fold'] != fold]
                val_idx = [i for i in train_idx if records[i]['fold'] == fold]
                out = (baseline_dir if arch == 'heatmap' else ROOT / 'outputs/regression') / f'validation_fold{fold}_seed42'
                if not (out / 'status.json').exists():
                    train_one(images, records, fit_idx, val_idx, out, args, 42, args.epochs)
                status = json.loads((out / 'status.json').read_text(encoding='utf-8'))
                if status['state'] != 'completed':
                    raise RuntimeError(f'Incomplete run: {out}')
                config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
                if config['train_ids'] != [records[i]['id'] for i in fit_idx] or config['validation_ids'] != [records[i]['id'] for i in val_idx]:
                    raise ValueError(f'Split mismatch: {out}')
                oof[arch][val_idx] = np.load(out / 'validation_predictions.npy')
                epochs[arch].append(status['best_epoch'])
                fold_metrics[arch].append(coordinate_metrics(truth[val_idx], oof[arch][val_idx]))
                center = np.mean([records[i]['xy_crop'] for i in fit_idx], axis=0)
                baseline_oof[val_idx] = original_coordinates(np.tile(center, (len(val_idx), 1)), [records[i] for i in val_idx])
        oof['equal_mean'] = (oof['heatmap'] + oof['regression']) / 2
        metrics = {name: coordinate_metrics(truth, values) for name, values in oof.items()}
        selected = min(metrics, key=lambda name: metrics[name]['rmse'])
        selection = {'selected': selected, 'metrics': metrics, 'fold_metrics': fold_metrics,
                     'best_epochs': epochs, 'refit_epochs': {name: int(np.median(values)) for name, values in epochs.items()},
                     'constant_baseline': coordinate_metrics(truth, baseline_oof), 'plan': plan}
        np.savez(study_dir / 'oof_predictions.npz', truth=truth, baseline=baseline_oof, **oof)
        save_json(study_dir / 'selection.json', selection)
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    if cli.stage in ('final', 'all'):
        selection = json.loads((study_dir / 'selection.json').read_text(encoding='utf-8'))
        architectures = ['heatmap', 'regression'] if selection['selected'] == 'equal_mean' else [selection['selected']]
        all_predictions, sources = [], []
        for arch in architectures:
            args.arch = arch
            for seed in (43, 44):
                out = study_dir / f'full_{arch}_seed{seed}'
                epochs = selection['refit_epochs'][arch]
                if not (out / 'status.json').exists():
                    train_one(images, records, train_idx, [], out, args, seed, epochs)
                status = json.loads((out / 'status.json').read_text(encoding='utf-8'))
                if status['state'] != 'completed' or status['completed_epochs'] != epochs:
                    raise RuntimeError(f'Full-data refit incomplete: {out}')
                model = make_model(arch, None).to(device)
                checkpoint = torch.load(out / 'best.pt', map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model'])
                if checkpoint['config']['train_ids'] != [records[i]['id'] for i in train_idx]:
                    raise ValueError('Refit did not use every training image')
                prediction = original_coordinates(predict(model, images, test_idx, args.batch_size, tta=True), [records[i] for i in test_idx])
                np.save(out / 'test_predictions.npy', prediction)
                all_predictions.append(prediction)
                sources.append({'architecture': arch, 'seed': seed, 'epochs': epochs, 'checkpoint': str(out / 'best.pt'),
                                'checkpoint_sha256': hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest()})
                del model
        final_prediction = np.mean(all_predictions, axis=0)
        predicted = {records[i]['id']: final_prediction[j].tolist() for j, i in enumerate(test_idx)}
        sizes = {records[i]['id']: (records[i]['width'], records[i]['height']) for i in test_idx}
        output = study_dir / 'submission.csv'
        write_submission(data_dir / 'sample_submission.csv', predicted, output, sizes)
        save_json(study_dir / 'predictions.json', predicted)
        save_json(study_dir / 'result.json', {'selection': selection, 'models': sources, 'refit_train_count': len(train_idx),
                  'submission_rows': 2 * len(test_idx), 'submission_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                  'online_score': None, 'limitations': manifest['limitations'],
                  'refit_note': 'Final two-seed refit ensemble is not itself evaluated on a held-out set; OOF scores belong to validation models.'})
        print('Final submission:', output, flush=True)


if __name__ == '__main__':
    main()
