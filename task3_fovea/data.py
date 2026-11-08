"""Read official images and preserve the transform back to original pixels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from contracts import load_labels, load_template


def retinal_crop(image: Image.Image) -> tuple[int, int, int, int]:
    """Find the illuminated image region; uses pixels only, never landmarks."""
    small = image.copy()
    small.thumbnail((512, 512))
    a = np.asarray(small, dtype=np.int16)
    mask = a.sum(2) > 60
    cols = np.flatnonzero(mask.mean(0) > 0.05)
    rows = np.flatnonzero(mask.mean(1) > 0.05)
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, image.width, image.height
    sx, sy = image.width / small.width, image.height / small.height
    margin = int(0.015 * max(image.size))
    return (max(0, int(cols[0] * sx) - margin),
            max(0, int(rows[0] * sy) - margin),
            min(image.width, int((cols[-1] + 1) * sx) + margin),
            min(image.height, int((rows[-1] + 1) * sy) + margin))


def prepare(data_dir: Path, output: Path, size: int = 320, folds: int = 4, seed: int = 20261024):
    output.mkdir(parents=True, exist_ok=True)
    labels = load_labels(data_dir / 'fovea_localization_train_GT.csv')
    template = load_template(data_dir / 'sample_submission.csv')
    records, arrays = [], []
    for split in ('train', 'test'):
        for p in sorted((data_dir / split).glob('*.jpg')):
            with Image.open(p) as im:
                im = im.convert('RGB')
                crop = retinal_crop(im)
                item = {'id': int(p.stem), 'split': split, 'file': str(p.resolve()),
                        'width': im.width, 'height': im.height, 'crop': list(crop),
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'pixel_sha256': hashlib.sha256(im.tobytes()).hexdigest()}
                if split == 'train':
                    xy = labels[item['id']]
                    if not (0 < xy[0] < im.width and 0 < xy[1] < im.height):
                        raise ValueError('This localization pipeline needs visible in-frame training points')
                    item['xy'] = [float(v) for v in xy]
                    item['xy_crop'] = [(float(xy[0]) - crop[0]) / (crop[2] - crop[0]),
                                       (float(xy[1]) - crop[1]) / (crop[3] - crop[1])]
                    if not all(0 < v < 1 for v in item['xy_crop']):
                        raise ValueError(f'Crop excludes label: {p}')
                arrays.append(np.asarray(im.crop(crop).resize((size, size), Image.Resampling.BILINEAR)))
                records.append(item)
    if {r['id'] for r in records if r['split'] == 'train'} != set(labels):
        raise ValueError('Image/label IDs differ')
    test_ids = {r['id'] for r in records if r['split'] == 'test'}
    if test_ids != {int(s.split('_')[0]) for s in template}:
        raise ValueError('Test/template IDs differ')
    # Group exact duplicate decoded images; balance camera size and coarse X location.
    groups = {}
    for r in records:
        if r['split'] == 'train':
            groups.setdefault(r['pixel_sha256'], []).append(r)
    strata = {}
    for group in groups.values():
        r = group[0]
        key = (r['width'], int(r['xy_crop'][0] >= 0.5))
        strata.setdefault(key, []).append(group)
    rng = np.random.default_rng(seed)
    totals = np.zeros(folds, dtype=int)
    for key in sorted(strata):
        group_list = strata[key]
        rng.shuffle(group_list)
        group_list.sort(key=len, reverse=True)
        counts = np.zeros(folds, dtype=int)
        for group in group_list:
            candidates = np.flatnonzero(counts == counts.min())
            candidates = candidates[totals[candidates] == totals[candidates].min()]
            fold = int(rng.choice(candidates))
            for r in group:
                r['fold'] = fold
            counts[fold] += len(group)
            totals[fold] += len(group)
    train_hashes = {r['pixel_sha256'] for r in records if r['split'] == 'train'}
    cross = [r['id'] for r in records if r['split'] == 'test' and r['pixel_sha256'] in train_hashes]
    metadata = {'size': size, 'folds': folds, 'seed': seed, 'records': records,
                'exact_duplicate_train_groups': [[r['id'] for r in g] for g in groups.values() if len(g) > 1],
                'cross_split_duplicate_test_ids': cross,
                'fold_counts': totals.tolist(), 'invisible_train_count': 0,
                'limitations': ['No patient identifiers supplied; exact-duplicate grouping is not patient grouping.',
                                'No invisible training examples: absence detection is not trained or validated.']}
    (output / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'images.npz', images=np.stack(arrays))
    return metadata


def original_coordinates(normalized: np.ndarray, records: list[dict]) -> np.ndarray:
    crops = np.array([r['crop'] for r in records], dtype=np.float64)
    return normalized * (crops[:, 2:] - crops[:, :2]) + crops[:, :2]
