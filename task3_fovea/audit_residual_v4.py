"""CPU audit of nested training provenance, fixed updates, and selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from contracts import coordinate_metrics, read_submission
from refinement_study import read_json, sha256
from run import ROOT, save_json
from prepare_residual_v4 import OUT, ROBUST, MANIFEST, BEST, BEST_HASH


def main():
    checks = []

    def require(condition, name):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            save_json(OUT / 'audit.json', {'passed': False, 'checks': checks})
            raise AssertionError(name)

    records = read_json(MANIFEST)['records']
    train = [r for r in records if r['split'] == 'train']
    ids = [r['id'] for r in train]
    folds = np.asarray([r['fold'] for r in train])
    truth = np.asarray([r['xy'] for r in train])
    require(sha256(BEST) == BEST_HASH and sha256(OUT / 'preserved_best_14.20909.csv') == BEST_HASH,
            'current online-best file and backup unchanged')
    require(read_json(OUT / 'plan.json')['manifest_sha256'] == sha256(MANIFEST), 'manifest unchanged')
    require(all(sha256(r['file']) == r['sha256'] for r in records), 'all input images unchanged')
    duplicate_groups = {}
    for r in train: duplicate_groups.setdefault(r['pixel_sha256'], set()).add(r['fold'])
    require(all(len(v) == 1 for v in duplicate_groups.values()), 'duplicate groups never cross folds')
    require(read_json(OUT / 'preparation_status.json')['state'] == 'completed', 'nested preparation completed')
    for outer in range(4):
        meta = read_json(OUT / f'outer{outer}_data.json')
        outer_ids = {r['id'] for r in train if r['fold'] == outer}
        require(set(meta['validation_ids']) == outer_ids and set(meta['fit_ids']) == set(ids) - outer_ids,
                f'outer{outer} fit and validation membership')
        require(sha256(OUT / f'outer{outer}_data.npz') == meta['cache_sha256'], f'outer{outer} native cache hash')
        checked = set()
        for image_id, entry in meta['sources_by_image'].items():
            if entry['kind'] == 'outer-heldout':
                require(int(image_id) in outer_ids, f'outer{outer} heldout source ID {image_id}')
                continue
            for source in entry['sources']:
                checkpoint = source['checkpoint']
                if checkpoint in checked: continue
                checked.add(checkpoint)
                cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
                cfg = cp['config']
                fitted, selected = set(cfg['train_ids']), set(cfg['inner_selection_ids'])
                require(fitted.isdisjoint(outer_ids) and selected.isdisjoint(outer_ids) and fitted.isdisjoint(selected)
                        and len(fitted) == 40 and len(selected) == 20 and fitted | selected == set(ids) - outer_ids,
                        f'outer{outer} nested fitting and selection exclude outer labels: {Path(checkpoint).parent.name}')
                hist = read_json(Path(checkpoint).parent / 'selection_history.json')
                key = f'fold{cfg["inner_fold"]}_rmse'
                chosen = min(hist, key=lambda row: row[key])['step']
                require(cp['steps'] == chosen and sha256(checkpoint) == source['checkpoint_sha256'],
                        f'outer{outer} inner checkpoint selection and hash: {Path(checkpoint).parent.name}')
    hashes = read_json(OUT / 'source_snapshot/hashes.json')
    require(all(sha256(OUT / 'source_snapshot' / name) == digest and sha256(ROOT / name) == digest
                for name, digest in hashes.items()), 'training source snapshot matches executed code')
    z = np.load(OUT / 'oof_predictions.npz')
    require(list(z['ids']) == ids and np.array_equal(z['truth'], truth) and np.array_equal(z['folds'], folds),
            'OOF covers all 80 records with correct targets and folds')
    old = np.load(ROBUST / 'oof_predictions.npz')['robust_only']
    require(np.array_equal(z['baseline'], old), 'paired comparison uses exact existing coarse OOF')
    require(np.allclose(z['refined'], (z['seed43'] + z['seed44']) / 2, atol=0, rtol=0), 'two-seed average correct')
    for fold in range(4):
        fit_ids = set(ids) - {r['id'] for r in train if r['fold'] == fold}
        val_ids = [r['id'] for r in train if r['fold'] == fold]
        indices = [i for i, r in enumerate(train) if r['fold'] == fold]
        for seed in (43, 44):
            run = OUT / f'fold{fold}_seed{seed}'
            cfg, status = read_json(run / 'config.json'), read_json(run / 'status.json')
            require(status['state'] == 'completed' and status['steps'] == 240
                    and set(status['all_training_ids_seen']) == fit_ids
                    and set(cfg['train_ids']) == fit_ids and cfg['validation_ids'] == val_ids,
                    f'fold{fold} seed{seed} complete 240 optimizer updates and correct records')
            vh = np.load(run / 'validation_history.npz')
            require(np.array_equal(vh['steps'], z['steps']) and list(vh['ids']) == val_ids
                    and np.array_equal(vh['predictions'], z[f'seed{seed}'][:, indices]),
                    f'fold{fold} seed{seed} shared candidate steps and prediction provenance')
            for step in z['steps']:
                cp = torch.load(run / f'step{step:03d}.pt', map_location='cpu', weights_only=True)
                require(cp['step'] == int(step) and cp['config'] == cfg and cp['weight_type'] == 'ema',
                        f'fold{fold} seed{seed} checkpoint step{step}')
    selection = read_json(OUT / 'selection.json')
    base_rmse = coordinate_metrics(truth, old)['rmse']
    base_fold = [coordinate_metrics(truth[folds == f], old[folds == f])['rmse'] for f in range(4)]
    passing = []
    for candidate in selection['candidates']:
        k = list(z['steps']).index(candidate['steps'])
        p = old + candidate['residual_weight'] * (z['refined'][k] - old)
        score = coordinate_metrics(truth, p)['rmse']
        fs = [coordinate_metrics(truth[folds == f], p[folds == f])['rmse'] for f in range(4)]
        improved = sum(x < y for x, y in zip(fs, base_fold))
        passed = score <= .95 * base_rmse and improved >= 3
        require(abs(score - candidate['metrics']['rmse']) < 1e-9
                and np.allclose(fs, candidate['fold_rmse'], rtol=0, atol=1e-9)
                and candidate['qualified'] == passed and candidate['improved_folds'] == improved,
                f'candidate step{candidate["steps"]} weight{candidate["residual_weight"]} score and gate')
        if passed: passing.append(candidate)
    chosen = min(passing, key=lambda c: c['metrics']['rmse']) if passing else None
    require(selection['selected'] == chosen, 'selected candidate exactly follows fixed gate')
    submission = OUT / 'submission_residual_v4.csv'
    if chosen is None:
        require(not submission.exists(), 'no unqualified new submission')
        state = 'completed_not_qualified'
    else:
        result = read_json(OUT / 'result.json')
        require(submission.exists() and sha256(submission) == result['submission_sha256'], 'final submission hash')
        test = [r for r in records if r['split'] == 'test']
        sizes = {r['id']: (r['width'], r['height']) for r in test}
        actual = read_submission(submission, sizes, ROOT / 'data/detection/sample_submission.csv')
        preds = []
        for seed in (43, 44):
            run = OUT / f'full_seed{seed}'
            cfg, status = read_json(run / 'config.json'), read_json(run / 'status.json')
            require(status['state'] == 'completed' and status['steps'] == chosen['steps']
                    and set(cfg['train_ids']) == set(ids) and cfg['validation_ids'] == [],
                    f'full seed{seed} all 80 records and matched optimizer steps')
            preds.append(np.load(run / 'test_predictions.npy'))
        coarse = np.load(OUT / 'full_data.npz')['centers'][80:].astype(np.float32).astype(np.float64)
        expected = coarse + chosen['residual_weight'] * (np.mean(preds, axis=0) - coarse)
        require(np.array_equal(expected, np.asarray([actual[r['id']] for r in test])), 'final CSV exact prediction mapping')
        state = 'completed_candidate_ready'
    report = {'passed': True, 'state': state, 'checks_passed': len(checks), 'checks': checks,
              'baseline_rmse': base_rmse, 'best_candidate': selection['best_candidate'],
              'selected': chosen, 'online_score': None, 'current_best_preserved': True}
    save_json(OUT / 'audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
