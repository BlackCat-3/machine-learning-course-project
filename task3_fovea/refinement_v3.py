"""Reuse held-out local refiners with robust OOF crop centers, then gated refit.

check is CPU-only. evaluate runs inference only, with no retraining. final needs
both a passing evaluation gate and --confirm-candidate matching selection.json.
All outputs belong to iteration3; earlier artifacts are read-only inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contracts import coordinate_metrics, read_submission, write_submission
from refinement_model import LocalRefiner
from refinement_study import (CACHE_SIZE, MANIFEST, crop_coordinates, make_theta,
                              patch_to_retina, predict_local, read_json, sha256, train_run)
from run import ROOT, save_json

PREVIOUS = ROOT / 'outputs/improvement_20261024'
LOCAL = PREVIOUS / 'refinement'
ROBUST = PREVIOUS / 'robust_heatmap'
DEFAULT_OUTPUT = ROOT / 'outputs/iteration3_20261024/refinement'
WEIGHTS = ROOT.parent / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
TEMPLATE = ROOT / 'data/detection/sample_submission.csv'
RELATIVE_REDUCTION, MINIMUM_FOLDS = 0.05, 3


def candidate_specs():
    return {f'{head}_local{int(weight * 100)}': {'head': head, 'local_weight': weight}
            for head in ('heatmap', 'dual') for weight in (1., 0.25, 0.5, 0.75)}


def checked_run(folder, train_ids, validation_ids, seed, epochs, schedule):
    config, status = read_json(folder / 'config.json'), read_json(folder / 'status.json')
    if (config['train_ids'] != train_ids or config['validation_ids'] != validation_ids
            or config['seed'] != seed or config['schedule_epochs'] != schedule
            or config.get('epochs', config.get('epochs_planned')) != epochs):
        raise ValueError(f'Checkpoint split or training recipe differs: {folder}')
    if (status['state'] != 'completed' or status['completed_epochs'] != epochs
            or not 1 <= status['best_epoch'] <= epochs or not (folder / 'best.pt').is_file()):
        raise RuntimeError(f'Incomplete source run: {folder}')
    if set(train_ids).intersection(validation_ids):
        raise ValueError(f'Train/validation overlap: {folder}')
    return config, status


def load_context():
    records = read_json(MANIFEST)['records']
    train, test = ([i for i, record in enumerate(records) if record['split'] == split]
                   for split in ('train', 'test'))
    ids = [records[i]['id'] for i in train]
    folds = np.array([records[i]['fold'] for i in train])
    if len(ids) != 80 or len(set(ids)) != 80 or not {39, 77}.issubset(ids) or set(folds) != set(range(4)):
        raise ValueError('All 80 original records, including 39 and 77, and four folds are required')
    truth = np.array([records[i]['xy'] for i in train], dtype=np.float64)
    local_selection, robust_selection = read_json(LOCAL / 'selection.json'), read_json(ROBUST / 'selection.json')
    if local_selection['all_training_ids'] != ids or robust_selection['selected'] != 'robust_only':
        raise ValueError('This fixed study expects the original local folds and robust_only baseline')
    source_paths = [MANIFEST, LOCAL / 'selection.json', LOCAL / 'retina_1024.npz',
                    ROBUST / 'selection.json', ROBUST / 'oof_predictions.npz', WEIGHTS, TEMPLATE,
                    Path(__file__), ROOT / 'refinement_study.py', ROOT / 'refinement_model.py',
                    ROOT / 'model.py', ROOT / 'run.py', ROOT / 'data.py', ROOT / 'contracts.py']
    for path in (MANIFEST, WEIGHTS, ROOT / 'refinement_study.py', ROOT / 'refinement_model.py', ROOT / 'model.py'):
        if sha256(path) != local_selection['sources'][str(path.resolve())]:
            raise ValueError(f'Original local recipe/source changed: {path}')
    by_seed = {43: np.full_like(truth, np.nan), 44: np.full_like(truth, np.nan)}
    local_epochs, local_runs = [], []
    for fold in range(4):
        fit_ids = [records[i]['id'] for i in train if records[i]['fold'] != fold]
        validation_ids = [records[i]['id'] for i in train if records[i]['fold'] == fold]
        local_folder = LOCAL / f'fold{fold}_seed42'
        config, status = checked_run(local_folder, fit_ids, validation_ids, 42, 80, 80)
        if config['sources'] != local_selection['sources']:
            raise ValueError('Local source fingerprints differ between folds')
        local_epochs.append(status['best_epoch'])
        local_runs.append({'folder': local_folder, 'config': config, 'status': status})
        source_paths.extend(local_folder / name for name in ('best.pt', 'config.json', 'status.json'))
        for seed in (43, 44):
            folder = ROBUST / f'fold{fold}_seed{seed}'
            _, robust_status = checked_run(folder, fit_ids, validation_ids, seed, 40, 40)
            if robust_status['best_epoch'] != robust_selection['best_epochs'][str(seed)][fold]:
                raise ValueError('Robust checkpoint epoch differs from its selection')
            prediction = np.load(folder / 'validation_predictions.npy', allow_pickle=False)
            if prediction.shape != (len(validation_ids), 2) or not np.isfinite(prediction).all():
                raise ValueError(f'Invalid held-out robust predictions: {folder}')
            by_seed[seed][folds == fold] = prediction
            source_paths.extend(folder / name for name in ('best.pt', 'config.json', 'status.json', 'validation_predictions.npy'))
    coarse = (by_seed[43] + by_seed[44]) / 2
    with np.load(ROBUST / 'oof_predictions.npz', allow_pickle=False) as saved:
        if (not np.array_equal(saved['truth'], truth) or not np.array_equal(saved['robust_only'], coarse)
                or any(not np.array_equal(saved[f'seed{seed}'], by_seed[seed]) for seed in by_seed)):
            raise ValueError('Robust OOF predictions cannot be reconstructed from the same held-out folds')
    baseline = coordinate_metrics(truth, coarse)
    if abs(baseline['rmse'] - robust_selection['candidates']['robust_only']['metrics']['rmse']) > 1e-10:
        raise ValueError('Robust OOF score differs from saved selection')
    if local_epochs != local_selection['best_epochs'] or int(np.median(local_epochs)) != local_selection['refit_epochs']:
        raise ValueError('Local refit epoch does not match original checkpoint selection')
    sources = {str(path.resolve()): sha256(path) for path in source_paths}
    return {'records': records, 'train': train, 'test': test, 'ids': ids, 'folds': folds,
            'truth': truth, 'coarse': coarse, 'baseline': baseline, 'local_runs': local_runs,
            'best_epochs': local_epochs, 'refit_epochs': int(np.median(local_epochs)), 'sources': sources}


def load_cached_images(context, device):
    # Read the old cache without allowing cache creation or any write to its folder.
    with np.load(LOCAL / 'retina_1024.npz', allow_pickle=False) as cache:
        if cache['manifest_sha256'].item() != context['sources'][str(MANIFEST.resolve())]:
            raise ValueError('Retina cache manifest hash differs')
        array = cache['images']
    if array.dtype != np.uint8 or array.shape != (len(context['records']), CACHE_SIZE, CACHE_SIZE, 3):
        raise ValueError('Invalid retina cache dimensions or dtype')
    return torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).to(device)


def score_candidates(truth, folds, coarse, local):
    baseline = coordinate_metrics(truth, coarse)
    baseline_folds = [coordinate_metrics(truth[folds == f], coarse[folds == f])['rmse'] for f in range(4)]
    candidates, details, qualified = {}, {}, []
    for name, spec in candidate_specs().items():
        weight = spec['local_weight']
        prediction = weight * local[spec['head']] + (1 - weight) * coarse
        metrics = coordinate_metrics(truth, prediction)
        fold_rmse = [coordinate_metrics(truth[folds == f], prediction[folds == f])['rmse'] for f in range(4)]
        improved = sum(value < baseline_folds[f] for f, value in enumerate(fold_rmse))
        passed = metrics['rmse'] <= baseline['rmse'] * (1 - RELATIVE_REDUCTION) and improved >= MINIMUM_FOLDS
        candidates[name], details[name] = prediction, {**spec, 'metrics': metrics, 'fold_rmse': fold_rmse,
            'improved_folds': improved, 'passed_gate': passed}
        if passed:
            qualified.append(name)
    selected = min(qualified, key=lambda name: details[name]['metrics']['rmse']) if qualified else None
    return candidates, {'selected': selected, 'gate_passed': bool(qualified), 'candidates': details,
                       'baseline': baseline, 'baseline_fold_rmse': baseline_folds}


def evaluate(context, output, device):
    selection_path = output / 'selection.json'
    if selection_path.exists():
        selection = validate_saved_selection(context, output)
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
        return
    images = load_cached_images(context, device)
    records, train, folds = context['records'], context['train'], context['folds']
    centers = torch.full((len(records), 2), float('nan'), device=device)
    centers[train] = torch.tensor(crop_coordinates(context['coarse'], [records[i] for i in train]), device=device, dtype=torch.float32)
    local = {head: np.full_like(context['truth'], np.nan) for head in ('heatmap', 'dual')}
    for fold, run in enumerate(context['local_runs']):
        validation = [i for i in train if records[i]['fold'] == fold]
        checkpoint = torch.load(run['folder'] / 'best.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != run['config'] or checkpoint['epoch'] != run['status']['best_epoch']:
            raise ValueError('Local checkpoint metadata differs from its audited fold')
        model = LocalRefiner(None).to(device)
        model.load_state_dict(checkpoint['model'])
        predicted = predict_local(model, images, validation, centers, records)
        for head in local:
            local[head][folds == fold] = predicted[head]
        np.savez(output / f'fold{fold}_predictions.npz', ids=np.array([records[i]['id'] for i in validation]),
                 coarse=context['coarse'][folds == fold], **predicted)
        print(f'fold {fold}: reused EMA epoch {checkpoint["epoch"]}, robust centers, inference completed', flush=True)
        del model, checkpoint
    candidates, selection = score_candidates(context['truth'], folds, context['coarse'], local)
    np.savez(output / 'oof_predictions.npz', ids=np.array(context['ids']), folds=folds, truth=context['truth'],
             coarse=context['coarse'], local_heatmap=local['heatmap'], local_dual=local['dual'], **candidates)
    selection.update({'sources': context['sources'], 'train_ids': context['ids'],
        'best_epochs': context['best_epochs'], 'refit_epochs': context['refit_epochs'],
        'gate': {'minimum_relative_reduction': RELATIVE_REDUCTION, 'minimum_improved_folds': MINIMUM_FOLDS},
        'oof_sha256': sha256(output / 'oof_predictions.npz'), 'evaluation_retrained_models': False,
        'note': 'Same model-selection folds and preselected local checkpoints; this is not untouched-test performance.'})
    save_json(selection_path, selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


def validate_saved_selection(context, output):
    selection = read_json(output / 'selection.json')
    if (selection['sources'] != context['sources'] or selection['train_ids'] != context['ids']
            or selection['best_epochs'] != context['best_epochs'] or selection['refit_epochs'] != context['refit_epochs']
            or selection['oof_sha256'] != sha256(output / 'oof_predictions.npz')):
        raise ValueError('Evaluation artifacts or sources changed; use a new study output')
    with np.load(output / 'oof_predictions.npz', allow_pickle=False) as saved:
        if (saved['ids'].tolist() != context['ids'] or not np.array_equal(saved['folds'], context['folds'])
                or not np.array_equal(saved['truth'], context['truth']) or not np.array_equal(saved['coarse'], context['coarse'])):
            raise ValueError('Evaluation OOF records changed')
        candidates, recomputed = score_candidates(context['truth'], context['folds'], context['coarse'],
            {'heatmap': saved['local_heatmap'], 'dual': saved['local_dual']})
        if any(not np.array_equal(saved[name], values) for name, values in candidates.items()):
            raise ValueError('Candidate predictions do not match fixed head/weight combinations')
    if any(selection[key] != value for key, value in recomputed.items()):
        raise ValueError('Selection does not match recomputed fixed candidate scores/gate')
    return selection


def robust_final_centers(context):
    """Reconstruct actual robust-only final predictions and verify their CSV hash."""
    result = read_json(ROBUST / 'result.json')
    selection = read_json(ROBUST / 'selection.json')
    if result['selection'] != selection or selection['selected'] != 'robust_only' or result['train_count'] != 80:
        raise ValueError('Unexpected robust final selection')
    test_records = [context['records'][i] for i in context['test']]
    sizes = {record['id']: (record['width'], record['height']) for record in test_records}
    csv_path = ROBUST / 'submission_robust.csv'
    if result['submission_sha256'] != sha256(csv_path):
        raise ValueError('Actual robust submission hash differs from its result')
    parsed = read_submission(csv_path, sizes, TEMPLATE)
    if set(parsed) != set(sizes):
        raise ValueError('Actual robust prediction IDs differ from test images')
    coarse = np.array([parsed[record['id']] for record in test_records])
    predictions, paths = [], [ROBUST / 'result.json', csv_path]
    models = {item['seed']: item for item in result['models']}
    if set(models) != {43, 44}:
        raise ValueError('Expected both existing robust final seeds')
    for seed in (43, 44):
        folder = ROBUST / f'full_seed{seed}'
        epochs = selection['refit_epochs'][str(seed)]
        _, status = checked_run(folder, context['ids'], [], seed, epochs, 40)
        if status['best_epoch'] != epochs or models[seed]['epochs'] != epochs or models[seed]['sha256'] != sha256(folder / 'best.pt'):
            raise ValueError('Robust final checkpoint is not the saved full-data model')
        pred = np.load(folder / 'test_predictions.npy', allow_pickle=False)
        if pred.shape != coarse.shape or not np.isfinite(pred).all():
            raise ValueError('Invalid robust final per-seed predictions')
        predictions.append(pred)
        paths.extend(folder / name for name in ('best.pt', 'config.json', 'status.json', 'test_predictions.npy'))
    if not np.allclose(np.mean(predictions, axis=0), coarse, rtol=0, atol=1e-10):
        raise ValueError('Robust final submission differs from its actual two-seed predictions')
    return coarse, {str(path.resolve()): sha256(path) for path in paths}


def final(context, output, device, confirmed):
    selection = validate_saved_selection(context, output)
    if not selection['gate_passed'] or not selection['selected']:
        raise RuntimeError('Fixed gate failed; final retraining and CSV generation are disabled')
    if confirmed != selection['selected']:
        raise ValueError(f'Final requires --confirm-candidate {selection["selected"]}')
    destination = output / 'submission_refinement_v3.csv'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite a completed submission: {destination}')
    coarse, coarse_sources = robust_final_centers(context)
    sources = {**context['sources'], **coarse_sources, str((output / 'selection.json').resolve()): sha256(output / 'selection.json')}
    images = load_cached_images(context, device)
    records, train, test = context['records'], context['train'], context['test']
    centers = torch.full((len(records), 2), float('nan'), device=device)
    centers[test] = torch.tensor(crop_coordinates(coarse, [records[i] for i in test]), device=device, dtype=torch.float32)
    specification = candidate_specs()[confirmed]
    predictions, models = [], []
    for seed in (43, 44):
        folder = output / f'full_seed{seed}'
        epochs = context['refit_epochs']
        train_run(images, records, train, [], centers, folder, seed, epochs, WEIGHTS, sources)
        config, status = checked_run(folder, context['ids'], [], seed, epochs, 80)
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        if (checkpoint['config'] != config or checkpoint['epoch'] != epochs or status['best_epoch'] != epochs
                or config['sources'] != sources):
            raise ValueError('Final local checkpoint failed full-data provenance checks')
        model = LocalRefiner(None).to(device)
        model.load_state_dict(checkpoint['model'])
        local = predict_local(model, images, test, centers, records)
        np.savez(folder / 'test_predictions.npz', ids=np.array([records[i]['id'] for i in test]), coarse=coarse, **local)
        predictions.append(local[specification['head']])
        models.append({'seed': seed, 'epochs': epochs, 'weight_type': 'ema', 'checkpoint': str(folder / 'best.pt'),
                       'checkpoint_sha256': sha256(folder / 'best.pt'), 'prediction_sha256': sha256(folder / 'test_predictions.npz')})
        del model, checkpoint
    local_weight = specification['local_weight']
    coordinates = local_weight * np.mean(predictions, axis=0) + (1 - local_weight) * coarse
    mapping = {records[i]['id']: coordinates[j].tolist() for j, i in enumerate(test)}
    sizes = {records[i]['id']: (records[i]['width'], records[i]['height']) for i in test}
    write_submission(TEMPLATE, mapping, destination, sizes)
    save_json(output / 'predictions.json', mapping)
    save_json(output / 'result.json', {'selected': confirmed, **specification, 'selection': selection,
        'sources': sources, 'models': models, 'train_count': len(train), 'test_count': len(test),
        'submission_rows': 2 * len(test), 'submission_sha256': sha256(destination), 'online_score': None,
        'note': 'Two-seed full-data local ensemble is not itself held-out validated; crop centers use the actual robust final predictions.'})
    print(f'New submission: {destination}', flush=True)


def cpu_check():
    center = torch.tensor([[0.4, 0.6], [0.57, 0.43]])
    theta = make_theta(center, torch.tensor([0.34, 0.42]), torch.tensor([0.2, -0.3]), torch.tensor([1., -1.]))
    patch = torch.tensor([[0.2, 0.7], [0.8, 0.1]])
    retina = patch_to_retina(patch, theta)
    recovered = (torch.linalg.solve(theta[:, :, :2], (2 * retina - 1 - theta[:, :, 2]).unsqueeze(-1)).squeeze(-1) + 1) / 2
    if not torch.allclose(recovered, patch, atol=2e-7):
        raise AssertionError('Affine inverse check failed')
    truth, folds = np.zeros((8, 2)), np.repeat(np.arange(4), 2)
    coarse = np.ones_like(truth) * 10
    _, passed = score_candidates(truth, folds, coarse, {'heatmap': truth, 'dual': coarse})
    _, failed = score_candidates(truth, folds, coarse, {'heatmap': coarse, 'dual': coarse})
    if passed['selected'] != 'heatmap_local100' or failed['gate_passed']:
        raise AssertionError('Candidate selection/gate check failed')
    print('CPU affine inversion and fixed-candidate gate checks passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'evaluate', 'final'), default='check')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--confirm-candidate', choices=tuple(candidate_specs()))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    # Enforce a separate iteration3 tree, including any caller-specified output.
    iteration_root = (ROOT / 'outputs/iteration3_20261024').resolve()
    output = args.output.resolve()
    if not output.is_relative_to(iteration_root) or output == iteration_root:
        raise ValueError('Output must be a dedicated subdirectory of outputs/iteration3_20261024')
    torch.set_num_threads(4)
    cpu_check()
    context = load_context()
    if args.stage == 'check':
        _, final_sources = robust_final_centers(context)
        print(json.dumps({'ready': True, 'training_count': len(context['train']),
            'baseline_rmse': context['baseline']['rmse'], 'local_best_epochs': context['best_epochs'],
            'refit_epochs': context['refit_epochs'], 'verified_sources': len(context['sources']) + len(final_sources),
            'gpu_used': False}, indent=2), flush=True)
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
    output.mkdir(parents=True, exist_ok=True)
    plan = {'sources': context['sources'], 'candidates': candidate_specs(),
            'relative_reduction': RELATIVE_REDUCTION, 'minimum_improved_folds': MINIMUM_FOLDS,
            'validation_training': False, 'refit_seeds': [43, 44], 'refit_epochs': context['refit_epochs']}
    if (output / 'plan.json').exists() and read_json(output / 'plan.json') != plan:
        raise ValueError('Existing output uses different sources or a different fixed study plan')
    if not (output / 'plan.json').exists():
        save_json(output / 'plan.json', plan)
    if args.stage == 'evaluate':
        evaluate(context, output, device)
    else:
        final(context, output, device, args.confirm_candidate)


if __name__ == '__main__':
    main()
