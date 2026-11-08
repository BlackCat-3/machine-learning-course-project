"""Nested five-fold evaluation of a learned linear head on frozen deep features.

Run --stage cv after the 500 training images have features; --stage predict after
all images have features. Every scaler and classifier is fit only on its training
partition. The frozen feature extractor itself never uses task labels.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
_deps = _root / 'work' / 'probe_deps'
if _deps.is_dir():
    sys.path.insert(0, str(_deps))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def macro_f1(y, pred):
    matrix = np.zeros((5, 5), dtype=int)
    np.add.at(matrix, (y, pred), 1)
    tp = np.diag(matrix)
    denominator = matrix.sum(0) + matrix.sum(1)
    scores = np.divide(2 * tp, denominator, out=np.zeros(5, float), where=denominator != 0)
    return float(scores.mean()), scores.tolist(), matrix.tolist()


def save_head(pipe, config, path):
    scaler, model = pipe.steps[0][1], pipe.steps[1][1]
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_, bias=model.intercept_,
             start=config['start'], stop=config['stop'], C=config['C'], training_views=config['training_views'])


def predict_head(path, features):
    with np.load(path) as head:
        mode = str(head['training_views']) if 'training_views' in head else 'mean'
        x = features.mean(1) if mode == 'mean' else features
        x = x[..., int(head['start']):int(head['stop'])]
        logits = ((x - head['mean']) / head['scale']) @ head['coef'].T + head['bias']
    logits -= logits.max(-1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(-1, keepdims=True)
    return probs.mean(1) if probs.ndim == 3 else probs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['cv', 'predict'], default='cv')
    parser.add_argument('--training-views', choices=['mean', 'augmented'], default='mean')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.features / 'manifest.json').read_text(encoding='utf-8'))
    completed = json.loads((args.features / 'progress.json').read_text())['completed']
    n = manifest['train_count']
    raw = np.load(args.features / 'features.npy', mmap_mode='r')
    assert completed >= n
    x_views = np.asarray(raw[:n], dtype=np.float64)
    x = x_views.mean(1)
    y = np.array([r[1] for r in manifest['records'][:n]])
    if args.stage == 'predict':
        assert completed == len(raw), 'Test features are incomplete'
        test = np.asarray(raw[n:], dtype=np.float64)
        variants = {'full500': predict_head(args.out / 'full_head.npz', test),
                    'fivefold': np.mean([predict_head(args.out / f'fold_{i}_head.npz', test)
                                         for i in range(5)], axis=0)}
        for name, probs in variants.items():
            np.save(args.out / f'{name}_test_probabilities.npy', probs)
            path = args.out / f'{name}_submission.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Category'])
                writer.writerows((image_id, manifest['classes'][int(p)])
                                 for image_id, p in zip(manifest['test_ids'], probs.argmax(1)))
            print('Saved', path, flush=True)
        # Export the fitted scaling + linear classifier as one equivalent PyTorch layer.
        import torch
        with np.load(args.out / 'full_head.npz') as head:
            weight = head['coef'] / head['scale']
            bias = head['bias'] - (weight * head['mean']).sum(1)
            torch.save({'linear_state_dict': {
                'weight': torch.tensor(weight, dtype=torch.float32),
                'bias': torch.tensor(bias, dtype=torch.float32)},
                'classes': manifest['classes'], 'backbone': manifest['backbone'],
                'image_size': manifest['size'], 'rotations': manifest['rotations'],
                'foreground': manifest.get('foreground', False),
                'feature_slice': [int(head['start']), int(head['stop'])],
                'preprocessing_mean': [.485, .456, .406],
                'preprocessing_std': [.229, .224, .225]}, args.out / 'trained_head.pt')
        return

    # Match the original fivefold experiment exactly, for paired comparisons.
    fold_ids = np.zeros(n, dtype=int)
    rng = np.random.default_rng(42)
    for label in range(5):
        indices = np.where(y == label)[0].tolist()
        rng.shuffle(indices)
        for fold, chunk in enumerate(np.array_split(indices, 5)):
            fold_ids[chunk] = fold
    if manifest['backbone'].startswith('dinov2'):
        width = 768 if manifest['backbone'] == 'dinov2_vitb14' else 384
        reps = [('last_cls_patch', 3 * width, 5 * width), ('last4_cls_patch', 0, 5 * width)]
        if manifest.get('foreground'):
            reps += [('foreground_patch', 5 * width, 6 * width),
                     ('cls_global_foreground', 3 * width, 6 * width)]
    else:
        reps = [('pooled', 0, x.shape[1])]
    configs = [{'representation': name, 'start': start, 'stop': stop, 'C': c,
                'training_views': args.training_views}
               for name, start, stop in reps for c in (0.01, 0.1, 1.0)]

    def fit(indices, cfg):
        head = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['C'], max_iter=600, tol=1e-4))
        if cfg['training_views'] == 'mean':
            xx, yy = x[indices, cfg['start']:cfg['stop']], y[indices]
        else:
            xx = x_views[indices, :, cfg['start']:cfg['stop']].reshape(-1, cfg['stop'] - cfg['start'])
            yy = np.repeat(y[indices], x_views.shape[1])
        head.fit(xx, yy)
        return head

    def predict(pipe, indices, cfg):
        if cfg['training_views'] == 'mean':
            return pipe.predict_proba(x[indices, cfg['start']:cfg['stop']])
        xx = x_views[indices, :, cfg['start']:cfg['stop']].reshape(-1, cfg['stop'] - cfg['start'])
        return pipe.predict_proba(xx).reshape(len(indices), x_views.shape[1], 5).mean(1)

    oof = np.full((n, 5), np.nan)
    folds = []
    all_inner_scores = []
    with threadpool_limits(limits=1):
        for fold in range(5):
            train, val = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
            inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=142 + fold).split(train, y[train]))
            scores = []
            for cfg in configs:
                inner_pred = np.empty(len(train), dtype=int)
                for tr, va in inner:
                    head = fit(train[tr], cfg)
                    inner_pred[va] = predict(head, train[va], cfg).argmax(1)
                scores.append(macro_f1(y[train], inner_pred)[0])
            selected = int(np.argmax(scores))
            cfg = configs[selected]
            head = fit(train, cfg)
            oof[val] = predict(head, val, cfg)
            save_head(head, cfg, args.out / f'fold_{fold}_head.npz')
            f1, per_class, confusion = macro_f1(y[val], oof[val].argmax(1))
            folds.append({'fold': fold, 'macro_f1': f1, 'config': cfg, 'inner_scores': scores,
                          'per_class_f1': per_class, 'confusion_matrix_rows_true': confusion})
            all_inner_scores.append(scores)
            print(f'fold={fold + 1} heldout_f1={f1:.4f} selected={cfg}', flush=True)
        assert np.isfinite(oof).all()
        overall, per_class, confusion = macro_f1(y, oof.argmax(1))
        best = configs[int(np.argmax(np.mean(all_inner_scores, axis=0)))]
        full_head = fit(np.arange(n), best)
        save_head(full_head, best, args.out / 'full_head.npz')
    report = {'backbone': manifest['backbone'], 'image_size': manifest['size'],
              'method': 'frozen pretrained deep features + learned linear softmax head',
              'outer_folds': 5, 'inner_folds': 3, 'configs': configs, 'folds': folds,
              'nested_oof_macro_f1': overall, 'per_class_f1': dict(zip(manifest['classes'], per_class)),
              'confusion_matrix_rows_true': confusion, 'full_head_config': best,
              'classes': manifest['classes'], 'training_labels': 'provided 500 images only'}
    (args.out / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.save(args.out / 'oof_probabilities.npy', oof)
    np.save(args.out / 'fold_ids.npy', fold_ids)
    print(f'NESTED OOF macro F1={overall:.6f}', flush=True)


if __name__ == '__main__':
    main()
