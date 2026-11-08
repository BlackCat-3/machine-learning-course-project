"""Compare pooling and head bagging using only each outer fold's training labels."""
import argparse
import json
from pathlib import Path

from plant_feature_classifier import macro_f1
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.out / 'validation.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.features / 'manifest.json').read_text(encoding='utf-8'))
    n = manifest['train_count']
    x = np.asarray(np.load(args.features / 'features.npy', mmap_mode='r')[:n], dtype=np.float64)
    y = np.array([r[1] for r in manifest['records'][:n]])
    folds = np.load(args.out / 'fold_ids.npy')
    assert np.array_equal(np.bincount(folds), np.full(5, 100))
    bagged = np.zeros((n, 5))
    no_fg = np.zeros((n, 5))
    candidates = report['configs']
    non_fg = [i for i, c in enumerate(candidates)
              if c['representation'] in ('last_cls_patch', 'last4_cls_patch')]

    def predict(tr, va, cfg):
        assert not set(tr) & set(va)
        start, stop = cfg['start'], cfg['stop']
        if cfg['training_views'] == 'augmented':
            xx = x[tr, :, start:stop].reshape(-1, stop - start)
            yy = np.repeat(y[tr], x.shape[1])
            vx = x[va, :, start:stop].reshape(-1, stop - start)
        else:
            xx, yy, vx = x[tr, :, start:stop].mean(1), y[tr], x[va, :, start:stop].mean(1)
        head = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['C'], max_iter=600, tol=1e-4))
        head.fit(xx, yy)
        p = head.predict_proba(vx)
        return p.reshape(len(va), -1, 5).mean(1)

    with threadpool_limits(limits=1):
        for f in range(5):
            tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
            fold_report = report['folds'][f]
            cfg = fold_report['config']
            pool = []
            for inner, _ in StratifiedKFold(4, shuffle=True, random_state=442 + f).split(tr, y[tr]):
                pool.append(predict(tr[inner], va, cfg))
            bagged[va] = np.mean(pool, axis=0)
            # Select the no-foreground model using the previously computed inner scores.
            candidate = max(non_fg, key=lambda i: fold_report['inner_scores'][i])
            no_fg[va] = predict(tr, va, candidates[candidate])
    scores = {'single_head_oof': report['nested_oof_macro_f1'],
              'nested_bagging_oof': macro_f1(y, bagged.argmax(1))[0],
              'without_foreground_oof': macro_f1(y, no_fg.argmax(1))[0],
              'heldout_images': n,
              'note': 'All model selection, scaling and fitting excluded the outer heldout images.'}
    (args.out / 'audit.json').write_text(json.dumps(scores, indent=2), encoding='utf-8')
    np.save(args.out / 'no_foreground_oof.npy', no_fg)
    np.save(args.out / 'bagging_oof.npy', bagged)
    print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    main()
