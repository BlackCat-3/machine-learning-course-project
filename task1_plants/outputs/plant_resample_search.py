"""Search actual resampling seeds with the established inner-CV selection held fixed."""
import json
from pathlib import Path
from plant_feature_classifier import macro_f1
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/dinov2_base336_augmented'
OUT = ROOT / 'outputs/dinov2_resample_search'
FEATURES = ROOT / 'work/features_dinov2_vitb14_336_fg_v2'


def main():
    OUT.mkdir(exist_ok=True)
    baseline = json.loads((BASE / 'validation.json').read_text(encoding='utf-8'))
    manifest = json.loads((FEATURES / 'manifest.json').read_text(encoding='utf-8'))
    x = np.asarray(np.load(FEATURES / 'features.npy')[:500], dtype=np.float64)
    y = np.array([r[1] for r in manifest['records'][:500]])
    folds = np.load(BASE / 'fold_ids.npy')
    metrics = {'baseline': {'macro_f1': baseline['nested_oof_macro_f1']}}
    tried = []
    with threadpool_limits(limits=1):
        for seed in range(1000, 1100):
            name = f'resample_{seed}'
            path = OUT / f'{name}_oof.npy'
            if path.exists():
                oof = np.load(path)
            else:
                oof = np.zeros((500, 5))
                for f in range(5):
                    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
                    cfg = baseline['folds'][f]['config']
                    a, b = cfg['start'], cfg['stop']
                    pool = []
                    for ti, _ in StratifiedKFold(5, shuffle=True, random_state=seed+20000*f).split(tr, y[tr]):
                        subset = tr[ti]
                        assert not np.intersect1d(subset, va).size
                        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['C'], max_iter=600, tol=1e-4))
                        pipe.fit(x[subset, :, a:b].reshape(-1, b-a), np.repeat(y[subset], x.shape[1]))
                        pool.append(pipe.predict_proba(x[va, :, a:b].reshape(-1, b-a)).reshape(len(va), x.shape[1], 5).mean(1))
                    oof[va] = np.mean(pool, axis=0)
                np.save(path, oof)
            score, per_class, matrix = macro_f1(y, oof.argmax(1))
            metrics[name] = {'macro_f1': score, 'errors': int((y != oof.argmax(1)).sum()),
                             'per_class_f1': per_class, 'confusion_matrix_rows_true': matrix,
                             'fold_scores': [macro_f1(y[folds==f], oof[folds==f].argmax(1))[0] for f in range(5)]}
            tried.append(seed)
            report = {'seeds': tried, 'metrics': metrics, 'selected': max(metrics, key=lambda k: metrics[k]['macro_f1']),
                      'method': 'Original inner-CV configuration selection; 5 stratified 80% subsets per outer training fold; seed changes subsets only.',
                      'warning': 'Seed screening selects on OOF. Selected candidates require a separate split sensitivity check; no guarantee of Kaggle improvement.'}
            (OUT / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'{name}: f1={score:.6f} errors={metrics[name]["errors"]}', flush=True)
            # Stop at the first higher score, before looking at another validation assignment.
            if score > baseline['nested_oof_macro_f1']:
                print('FOUND', name, flush=True)
                break


if __name__ == '__main__':
    main()
