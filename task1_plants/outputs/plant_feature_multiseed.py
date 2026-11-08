"""Five seeded inner-CV/bagging runs compared on identical outer held-out images."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from plant_feature_classifier import macro_f1, save_head, predict_head
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/dinov2_base336_augmented'
OUT = ROOT / 'outputs/dinov2_multiseed'
FEATURES = ROOT / 'work/features_dinov2_vitb14_336_fg_v2'
SEEDS = [7, 21, 42, 123, 2026]


def main():
    global OUT, SEEDS
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--submission', type=Path, default=ROOT / 'outputs/dinov2_multiseed_best_submission.csv')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    OUT, SEEDS = args.out, args.seeds
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((FEATURES / 'manifest.json').read_text(encoding='utf-8'))
    baseline = json.loads((BASE / 'validation.json').read_text(encoding='utf-8'))
    raw = np.asarray(np.load(FEATURES / 'features.npy'), dtype=np.float64)
    n = manifest['train_count']
    y = np.array([r[1] for r in manifest['records'][:n]])
    fold_ids = np.load(BASE / 'fold_ids.npy')
    configs = baseline['configs']
    assert n == 500 and len(raw) == n + len(manifest['test_ids'])

    def fit(tr, cfg):
        a, b = cfg['start'], cfg['stop']
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['C'], max_iter=600, tol=1e-4))
        pipe.fit(raw[tr, :, a:b].reshape(-1, b-a), np.repeat(y[tr], raw.shape[1]))
        return pipe

    def predict(pipe, va, cfg):
        a, b = cfg['start'], cfg['stop']
        return pipe.predict_proba(raw[va, :, a:b].reshape(-1, b-a)).reshape(len(va), raw.shape[1], 5).mean(1)

    def tune(tr, seed):
        splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(tr, y[tr]))
        scores = []
        for cfg in configs:
            pred = np.empty(len(tr), dtype=int)
            for ti, vi in splits:
                assert not np.intersect1d(tr[ti], tr[vi]).size
                pred[vi] = predict(fit(tr[ti], cfg), tr[vi], cfg).argmax(1)
            scores.append(macro_f1(y[tr], pred)[0])
        return configs[int(np.argmax(scores))], scores

    candidates = {'baseline': np.load(BASE / 'oof_probabilities.npy')}
    all_details = {}
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            seed_dir = OUT / f'seed_{seed}'
            seed_dir.mkdir(exist_ok=True)
            single, bagged = np.zeros((n, 5)), np.zeros((n, 5))
            details = []
            for f in range(5):
                tr, va = np.where(fold_ids != f)[0], np.where(fold_ids == f)[0]
                cache = seed_dir / f'fold_{f}.npz'
                detail_path = seed_dir / f'fold_{f}.json'
                if cache.exists() and detail_path.exists():
                    with np.load(cache) as saved:
                        single[va], bagged[va] = saved['single'], saved['bagged']
                    details.append(json.loads(detail_path.read_text()))
                    continue
                cfg, inner_scores = tune(tr, seed + 10000*f)
                single[va] = predict(fit(tr, cfg), va, cfg)
                pool = []
                train_subsets = []
                for ti, _ in StratifiedKFold(5, shuffle=True, random_state=seed + 20000*f).split(tr, y[tr]):
                    subset = tr[ti]
                    assert not np.intersect1d(subset, va).size
                    train_subsets.append(subset.tolist())
                    pool.append(predict(fit(subset, cfg), va, cfg))
                bagged[va] = np.mean(pool, axis=0)
                detail = {'seed': seed, 'fold': f, 'config': cfg, 'inner_scores': inner_scores,
                          'train_subsets': train_subsets, 'validation_indices': va.tolist(),
                          'single_f1': macro_f1(y[va], single[va].argmax(1))[0],
                          'bagged_f1': macro_f1(y[va], bagged[va].argmax(1))[0]}
                np.savez(cache, single=single[va], bagged=bagged[va])
                detail_path.write_text(json.dumps(detail, indent=2), encoding='utf-8')
                details.append(detail)
                print(f"seed={seed} fold={f+1} single={detail['single_f1']:.4f} bagged={detail['bagged_f1']:.4f}", flush=True)
            candidates[f'single_{seed}'] = single
            candidates[f'bagged_{seed}'] = bagged
            all_details[str(seed)] = details
            print(f"SEED {seed}: single={macro_f1(y, single.argmax(1))[0]:.6f} bagged={macro_f1(y, bagged.argmax(1))[0]:.6f}", flush=True)
        candidates['single_ensemble'] = np.mean([candidates[f'single_{s}'] for s in SEEDS], axis=0)
        candidates['bagged_ensemble'] = np.mean([candidates[f'bagged_{s}'] for s in SEEDS], axis=0)
        metrics = {}
        for name, p in candidates.items():
            assert p.shape == (500, 5) and np.isfinite(p).all() and np.allclose(p.sum(1), 1)
            f1, class_f1, confusion = macro_f1(y, p.argmax(1))
            metrics[name] = {'macro_f1': f1, 'errors': int((y != p.argmax(1)).sum()),
                             'fold_scores': [macro_f1(y[fold_ids == f], p[fold_ids == f].argmax(1))[0] for f in range(5)],
                             'per_class_f1': class_f1, 'confusion_matrix_rows_true': confusion}
            np.save(OUT / f'{name}_oof.npy', p)
        # Prefer the established baseline on exact ties. Select before test prediction.
        winner = max(metrics, key=lambda key: metrics[key]['macro_f1'])
        report = {'seeds': SEEDS, 'metrics': metrics, 'selected': winner,
                  'baseline_kaggle_score_user_reported': 0.9550,
                  'evaluation': 'Fixed outer folds; per-seed inner 3-fold configuration selection; bagging is 5 stratified 80% subsets.',
                  'limitations': f'OOF used to compare {2*len(SEEDS)+2} new candidates; selected score may be optimistic. Kaggle score is unknown.',
                  'labels': 'Only the supplied 500 training labels. All views of an image stay together.',
                  'folds': all_details}
        (OUT / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('SELECTED', winner, json.dumps(metrics[winner]), flush=True)
        if args.evaluate_only:
            return
        target = args.submission
        if winner == 'baseline':
            shutil.copyfile(BASE / 'full500_submission.csv', target)
            probs = np.load(BASE / 'full500_test_probabilities.npy')
        else:
            selected_seeds = SEEDS if winner.endswith('ensemble') else [int(winner.split('_')[1])]
            test_indices = np.arange(n, len(raw))
            members = []
            report['deployment'] = []
            for seed in selected_seeds:
                # Same tuning algorithm, now using all provided training images.
                tr = np.arange(n)
                cfg, scores = tune(tr, seed)
                subsets = [tr]
                if winner.startswith('bagged'):
                    subsets = [tr[ti] for ti, _ in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y)]
                seed_members = []
                for i, subset in enumerate(subsets):
                    pipe = fit(subset, cfg)
                    head_path = OUT / f'deploy_seed_{seed}_head_{i}.npz'
                    save_head(pipe, cfg, head_path)
                    p = predict(pipe, test_indices, cfg)
                    assert np.allclose(p, predict_head(head_path, raw[n:]), atol=1e-10)
                    seed_members.append(p)
                members.append(np.mean(seed_members, axis=0))
                report['deployment'].append({'seed': seed, 'config': cfg, 'inner_scores': scores,
                                              'train_subsets': [t.tolist() for t in subsets]})
            probs = np.mean(members, axis=0)
            with target.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['ID', 'Category'])
                w.writerows((i, manifest['classes'][int(p)]) for i, p in zip(manifest['test_ids'], probs.argmax(1)))
        np.save(OUT / 'selected_test_probabilities.npy', probs)
        with target.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert reader.fieldnames == ['ID', 'Category']
        assert [r['ID'] for r in rows] == manifest['test_ids']
        assert len(rows) == len({r['ID'] for r in rows}) == 378
        assert all(r['Category'] in manifest['classes'] for r in rows)
        report['submission'] = str(target)
        report['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        old = np.load(BASE / 'full500_test_probabilities.npy')
        report['changed_test_predictions'] = int((probs.argmax(1) != old.argmax(1)).sum())
        (OUT / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('COMPLETE', target, 'changed predictions:', report['changed_test_predictions'], flush=True)


if __name__ == '__main__':
    main()
