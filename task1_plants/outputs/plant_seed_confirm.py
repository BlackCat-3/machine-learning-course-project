"""Export the best genuinely different candidate, then compare on two new fold assignments.

New assignments reuse the same 500 images; this is a sensitivity check, not a new test set.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from plant_feature_classifier import macro_f1, save_head, predict_head
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/dinov2_base336_augmented'
FEATURES = ROOT / 'work/features_dinov2_vitb14_336_fg_v2'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--candidate', default=None)
    args = parser.parse_args()
    report = json.loads((args.search / 'comparison.json').read_text(encoding='utf-8'))
    original = json.loads((BASE / 'validation.json').read_text(encoding='utf-8'))
    manifest = json.loads((FEATURES / 'manifest.json').read_text(encoding='utf-8'))
    raw = np.asarray(np.load(FEATURES / 'features.npy'), dtype=np.float64)
    y = np.array([r[1] for r in manifest['records'][:500]])
    configs = original['configs']
    n = len(y)
    old_probs = np.load(BASE / 'full500_test_probabilities.npy')
    out = args.search / 'confirmation'
    out.mkdir(exist_ok=True)

    def fit(tr, cfg):
        a, b = cfg['start'], cfg['stop']
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['C'], max_iter=600, tol=1e-4))
        pipe.fit(raw[tr, :, a:b].reshape(-1, b-a), np.repeat(y[tr], raw.shape[1]))
        return pipe

    def predict(head, va, cfg):
        a, b = cfg['start'], cfg['stop']
        return head.predict_proba(raw[va, :, a:b].reshape(-1, b-a)).reshape(len(va), raw.shape[1], 5).mean(1)

    def tune(tr, seed):
        splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(tr, y[tr]))
        scores = []
        for cfg in configs:
            pred = np.empty(len(tr), dtype=int)
            for ti, vi in splits:
                pred[vi] = predict(fit(tr[ti], cfg), tr[vi], cfg).argmax(1)
            scores.append(macro_f1(y[tr], pred)[0])
        return configs[int(np.argmax(scores))]

    def candidate_predict(tr, va, name, fold=0, export=False):
        seeds = report['seeds'] if name.endswith('ensemble') else [int(name.split('_')[1])]
        seed_probs, metadata = [], []
        for seed in seeds:
            if name.startswith('resample'):
                cfg = original['full_head_config'] if export else tune(tr, 142 + fold)
            else:
                cfg = tune(tr, seed + 10000*fold)
            subsets = [tr]
            if name.startswith(('bagged', 'resample')):
                subsets = [tr[ti] for ti, _ in StratifiedKFold(5, shuffle=True, random_state=seed + 20000*fold).split(tr, y[tr])]
            member_probs = []
            for i, subset in enumerate(subsets):
                assert not np.intersect1d(subset, va).size
                head = fit(subset, cfg)
                p = predict(head, va, cfg)
                if export:
                    path = out / f'{name}_seed_{seed}_head_{i}.npz'
                    save_head(head, cfg, path)
                    assert np.allclose(p, predict_head(path, raw[va]), atol=1e-10)
                member_probs.append(p)
            seed_probs.append(np.mean(member_probs, axis=0))
            metadata.append({'seed': seed, 'config': cfg, 'train_subsets': [s.tolist() for s in subsets]})
        return np.mean(seed_probs, axis=0), metadata

    names = sorted((k for k in report['metrics'] if k != 'baseline'),
                   key=lambda k: report['metrics'][k]['macro_f1'], reverse=True)
    if args.candidate:
        names = [args.candidate]
    result = {'comparison_directory': str(args.search), 'duplicate_candidates': [],
              'baseline_kaggle_score_user_reported': 0.9550,
              'note': 'Confirmatory folds reuse the same labeled images. They are not an independent test set.'}
    with threadpool_limits(limits=1):
        for name in names:
            probs, metadata = candidate_predict(np.arange(n), np.arange(n, len(raw)), name, export=True)
            changed = int((probs.argmax(1) != old_probs.argmax(1)).sum())
            print(f'DEPLOY {name} search_f1={report["metrics"][name]["macro_f1"]:.6f} changed={changed}', flush=True)
            if changed:
                break
            result['duplicate_candidates'].append(name)
        else:
            raise RuntimeError('All candidates produced identical test labels; no different submission was generated.')
        result.update(selected=name, search_metric=report['metrics'][name],
                      changed_test_predictions=changed, deployment=metadata, checks=[])
        np.save(out / 'test_probabilities.npy', probs)
        target = ROOT / 'outputs/dinov2_new_seed_submission.csv'
        with target.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Category'])
            writer.writerows((i, manifest['classes'][int(p)]) for i, p in zip(manifest['test_ids'], probs.argmax(1)))
        for split_seed in [314159, 271828]:
            fold_ids = np.zeros(n, dtype=int)
            for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=split_seed).split(np.arange(n), y)):
                fold_ids[va] = f
            baseline_oof, candidate_oof = np.zeros((n, 5)), np.zeros((n, 5))
            for f in range(5):
                tr, va = np.where(fold_ids != f)[0], np.where(fold_ids == f)[0]
                cfg = tune(tr, 142 + f)
                baseline_oof[va] = predict(fit(tr, cfg), va, cfg)
                candidate_oof[va], _ = candidate_predict(tr, va, name, fold=f)
                print(f'CHECK split={split_seed} fold={f+1}', flush=True)
            np.savez(out / f'check_{split_seed}.npz', baseline=baseline_oof, candidate=candidate_oof, fold_ids=fold_ids)
            check = {'split_seed': split_seed, 'baseline_f1': macro_f1(y, baseline_oof.argmax(1))[0],
                     'candidate_f1': macro_f1(y, candidate_oof.argmax(1))[0],
                     'baseline_errors': int((baseline_oof.argmax(1) != y).sum()),
                     'candidate_errors': int((candidate_oof.argmax(1) != y).sum())}
            result['checks'].append(check)
            print('CHECK RESULT', json.dumps(check), flush=True)
            (out / 'confirmation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        with target.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert reader.fieldnames == ['ID', 'Category']
        assert len(rows) == len({r['ID'] for r in rows}) == 378
        assert [r['ID'] for r in rows] == manifest['test_ids']
        assert all(r['Category'] in manifest['classes'] for r in rows)
        result['submission'] = str(target)
        result['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        (out / 'confirmation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('COMPLETE', json.dumps({k: result[k] for k in ['selected', 'changed_test_predictions', 'checks', 'submission']}), flush=True)


if __name__ == '__main__':
    main()
