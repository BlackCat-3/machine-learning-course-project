"""Extract rotation-averaged pretrained deep features, without using labels.

Example: python plant_frozen_features.py --backbone dinov2_vits14 --size 336
The companion plant_feature_classifier.py fits and evaluates the task-specific head.
"""
import argparse
import json
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
for _name in ('deps', 'probe_deps'):
    _path = _root / 'work' / _name
    if _path.is_dir():
        sys.path.insert(0, str(_path))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from plant_baseline import read_layout, ZipImages


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--zip', type=Path, default=Path(r'D:\Downloads\2026task1.zip'))
    parser.add_argument('--backbone', choices=['dinov2_vits14', 'dinov2_vitb14', 'convnext_tiny'], default='dinov2_vits14')
    parser.add_argument('--size', type=int, default=336)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--foreground', action='store_true', help='Also pool patch features over green pixels')
    parser.add_argument('--views', type=int, choices=[1, 2, 4], default=4)
    args = parser.parse_args()
    if args.backbone.startswith('dinov2') and args.size % 14:
        parser.error('DINOv2 image size must be divisible by 14')
    if args.foreground and not args.backbone.startswith('dinov2'):
        parser.error('Foreground patch pooling is supported for DINOv2 only')
    root = Path(__file__).resolve().parents[1]
    cache = root / 'work' / 'model_cache'
    suffix = ('_fg' if args.foreground else '') + (f'_v{args.views}' if args.views != 4 else '')
    out = root / 'work' / f'features_{args.backbone}_{args.size}{suffix}'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.hub.set_dir(str(cache))
    classes, train, test, ids = read_layout(args.zip, None)
    records = train + test
    dino = args.backbone.startswith('dinov2')
    width = 768 if args.backbone == 'dinov2_vitb14' else 384
    dim = width * (6 if args.foreground else 5) if dino else 768
    rotations = [0, 2] if args.views == 2 else list(range(args.views))
    manifest = {'backbone': args.backbone, 'size': args.size, 'classes': classes,
                'records': records, 'test_ids': ids, 'train_count': len(train),
                'rotations': [90 * r for r in rotations], 'features': dim,
                'normalization': 'imagenet', 'zip_bytes': args.zip.stat().st_size}
    if args.foreground:
        manifest['foreground'] = True
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == json.loads(json.dumps(manifest)), 'Cache differs'
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    progress = out / 'progress.json'
    completed = json.loads(progress.read_text())['completed'] if progress.exists() else 0
    feature_path = out / 'features.npy'
    features = np.lib.format.open_memmap(feature_path, mode='r+' if feature_path.exists() else 'w+',
                                        dtype=np.float32, shape=(len(records), len(rotations), dim))
    if completed == len(records):
        print('Features already complete', flush=True)
        return
    if dino:
        repo = cache / 'facebookresearch_dinov2_main'
        model = (torch.hub.load(str(repo), args.backbone, source='local') if repo.exists()
                 else torch.hub.load('facebookresearch/dinov2', args.backbone, trust_repo=True))
    else:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    dataset = ZipImages(args.zip, records, transform)
    loader = DataLoader(Subset(dataset, list(range(completed, len(records)))),
                        batch_size=args.batch_size, num_workers=0)
    with torch.inference_mode():
        for images, _ in loader:
            for view, rotation in enumerate(rotations):
                rotated = torch.rot90(images, rotation, dims=(2, 3)).contiguous()
                if dino:
                    layers = model.get_intermediate_layers(rotated, n=4, return_class_token=True)
                    parts = [cls for _, cls in layers] + [layers[-1][0].mean(1)]
                    if args.foreground:
                        rgb = rotated * torch.tensor([.229, .224, .225])[None, :, None, None] + torch.tensor([.485, .456, .406])[None, :, None, None]
                        red, green, blue = rgb.unbind(1)
                        mask = ((green > 1.05 * red) & (green > 1.05 * blue) & (green > .1)).float()
                        weights = torch.nn.functional.avg_pool2d(mask[:, None], 14, 14).flatten(1)
                        weights[weights.sum(1) < .01] = 1
                        pooled = (layers[-1][0] * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True)
                        parts.append(pooled)
                    embedding = torch.cat(parts, dim=1)
                else:
                    embedding = model.classifier[0](model.avgpool(model.features(rotated))).flatten(1)
                features[completed:completed + len(images), view] = embedding.numpy()
            completed += len(images)
            features.flush()
            progress.write_text(json.dumps({'completed': completed, 'total': len(records)}))
            if completed % 32 == 0 or completed >= len(records):
                print(f'{args.backbone} {args.size}: {completed}/{len(records)} images, {len(rotations)} rotations', flush=True)
    dataset.close()
    print(f'Features saved: {feature_path}', flush=True)


if __name__ == '__main__':
    main()
