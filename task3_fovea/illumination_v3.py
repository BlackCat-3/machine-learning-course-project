"""Bounded illumination-normalized ResNet34 route using the shared-epoch engine.

This wrapper changes only in-process engine configuration. It never edits the
fine_heatmap_v3 source or previous results. Each image is corrected independently
before augmentation; no labels or cross-image statistics enter preprocessing.
Use --stage self-check for CPU-only checks; training is explicitly scheduled.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import PIL
from PIL import Image, ImageFilter
import torch
from torch import nn
from torchvision.models import resnet34

import fine_heatmap_v3 as engine
from model import FoveaNet
from run import ROOT

SIZE = 448
BLUR_RADIUS = SIZE / 30.0
WEIGHTS = ROOT.parent / '.cache/torch/hub/checkpoints/resnet34-b627a593.pth'
DEFAULT_OUTPUT = ROOT / 'outputs/iteration3_20261024/illumination'
_base_plan = engine.experiment_plan
_base_load_inputs = engine.load_inputs
_base_final = engine.final
_configured = False


def preprocessing_spec():
    return {
        'name': 'per-image background illumination correction',
        'order': 'RGB -> unchanged manifest retina crop -> bilinear resize448 -> correction -> training augmentation',
        'blur': 'PIL.ImageFilter.GaussianBlur', 'blur_radius': BLUR_RADIUS,
        'corrected': 'clip(4 * (RGB - blurred_RGB) + 128, 0, 255)',
        'blend': '0.5 * original_RGB + 0.5 * corrected_RGB',
        'storage': 'np.rint then clip to [0,255], uint8',
        'uses_labels': False, 'uses_other_images': False, 'pil_version': PIL.__version__,
    }


def correct_illumination(image):
    """Correct one already-resized 448x448 RGB image without changing geometry."""
    if image.mode != 'RGB' or image.size != (SIZE, SIZE):
        raise ValueError('Illumination correction expects a 448x448 RGB image')
    original = np.asarray(image, dtype=np.float32)
    background = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)), dtype=np.float32)
    corrected = np.clip(4.0 * (original - background) + 128.0, 0.0, 255.0)
    return np.rint(0.5 * original + 0.5 * corrected).clip(0, 255).astype(np.uint8)


class IlluminationHeatmap34(FoveaNet):
    """ResNet34+64-channel FPN with inherited frozen encoder BN statistics."""

    def __init__(self, pretrained=True):
        super().__init__(None)
        self.encoder = resnet34(weights=None)
        if pretrained:
            self.encoder.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self._freeze_encoder_batchnorm()


def experiment_plan():
    plan = _base_plan()
    plan.update({
        'experiment': 'illumination_v3', 'architecture': 'ResNet34 FPN',
        'model_class': 'illumination_v3.IlluminationHeatmap34',
        'encoder_channels': [64, 128, 256, 512],
        'preprocessing': preprocessing_spec(),
        'engine': 'fine_heatmap_v3 shared-epoch training and selection functions',
        'pretrained_path': str(WEIGHTS.resolve()),
        'bounded_route': 'four folds, two seeds, 40 epochs; fixed shared candidates 6/10/16/24/32/40 only',
    })
    return plan


def load_inputs():
    records, train, test, truth, baseline, baseline_test, sources = _base_load_inputs()
    # The base loader already hashes the actual engine, model, augmentation and
    # contract sources; add this wrapper containing the true architecture and transform.
    sources = {**sources, str(Path(__file__).resolve()): engine.file_sha256(__file__)}
    if {records[index]['id'] for index in train} != set(range(1, 81)):
        raise ValueError('All 80 original training records must remain in this route')
    return records, train, test, truth, baseline, baseline_test, sources


def load_images(records, output, manifest_hash, device):
    """Cache only the isolated corrected appearance; original image files stay intact."""
    for record in records:
        if engine.file_sha256(record['file']) != record['sha256']:
            raise ValueError(f'Original image differs from manifest: {record["file"]}')
    specification = preprocessing_spec()
    transform_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode('utf-8')).hexdigest()
    cache = output / 'images448_illumination.npz'
    ids = np.asarray([record['id'] for record in records])
    if not cache.exists():
        arrays = []
        for record in records:
            with Image.open(record['file']) as image:
                cropped = image.convert('RGB').crop(record['crop']).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                arrays.append(correct_illumination(cropped))
        engine.save_npz(cache, images=np.stack(arrays), ids=ids, manifest_sha256=manifest_hash,
                        preprocessing_sha256=transform_hash)
    with np.load(cache, allow_pickle=False) as stored:
        if (str(stored['manifest_sha256'].item()) != manifest_hash
                or str(stored['preprocessing_sha256'].item()) != transform_hash
                or not np.array_equal(stored['ids'], ids)):
            raise ValueError('Illumination cache has different source/order/preprocessing')
        images = stored['images']
    if images.shape != (len(records), SIZE, SIZE, 3) or images.dtype != np.uint8:
        raise ValueError('Unexpected illumination cache shape or dtype')
    return torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device), engine.file_sha256(cache)


def configure_engine():
    """Bind the existing shared engine to this explicit, independently hashed recipe."""
    global _configured
    if _configured:
        return
    engine.SIZE, engine.BATCH_SIZE, engine.SCHEDULE_EPOCHS = SIZE, 8, 40
    engine.SEEDS, engine.CANDIDATE_EPOCHS = (43, 44), (6, 10, 16, 24, 32, 40)
    engine.EMA_DECAY, engine.SIGMA, engine.COORDINATE_WEIGHT = 0.95, 1.5, 100.0
    engine.ENCODER_LR, engine.HEAD_LR, engine.WEIGHT_DECAY = 5e-5, 5e-4, 2e-4
    engine.ETA_MIN, engine.WEIGHTS, engine.DEFAULT_OUTPUT = 1e-6, WEIGHTS, DEFAULT_OUTPUT
    # The engine factory name is historical; the config records the real class.
    engine.FineHeatmap50 = IlluminationHeatmap34
    engine.experiment_plan, engine.load_inputs, engine.load_images = experiment_plan, load_inputs, load_images
    _configured = True


def self_check():
    configure_engine()
    torch.set_num_threads(2)
    torch.manual_seed(43)
    constant = Image.new('RGB', (SIZE, SIZE), color=(64, 128, 192))
    result = correct_illumination(constant)
    assert result.shape == (SIZE, SIZE, 3) and result.dtype == np.uint8
    assert np.array_equal(result[SIZE // 2, SIZE // 2], np.array([96, 128, 160], dtype=np.uint8))
    gradient = np.tile(np.linspace(25, 230, SIZE, dtype=np.uint8), (SIZE, 1))
    original = np.stack((gradient, np.flip(gradient, axis=1), gradient), axis=-1)
    original[180:220, 190:230] = 220
    corrected = correct_illumination(Image.fromarray(original))
    assert np.array_equal(corrected, correct_illumination(Image.fromarray(original)))
    assert not np.array_equal(corrected, original)
    assert corrected.min() >= 0 and corrected.max() <= 255
    model = IlluminationHeatmap34(pretrained=True).cpu().train()
    assert [layer.in_channels for layer in model.lateral] == [64, 128, 256, 512]
    batch_norms = [module for module in model.encoder.modules() if isinstance(module, nn.BatchNorm2d)]
    stats = [(module.running_mean.clone(), module.running_var.clone()) for module in batch_norms]
    with torch.no_grad():
        full_shape = model(torch.randn(1, 3, SIZE, SIZE)).shape
    assert full_shape == (1, 1, 112, 112)
    logits = model(torch.randn(2, 3, 64, 64))
    loss = engine.fine_loss(logits, torch.tensor([[0.3, 0.6], [0.65, 0.4]]))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (model.encoder.conv1.weight, model.lateral[3].weight, model.head[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    for module, (mean, variance) in zip(batch_norms, stats):
        assert not module.training and torch.equal(module.running_mean, mean) and torch.equal(module.running_var, variance)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    previous = ema.head[-1].bias.clone()
    with torch.no_grad():
        model.head[-1].bias.add_(1)
    engine.update_ema(ema, model)
    assert torch.allclose(ema.head[-1].bias, previous + 0.05, atol=1e-6)
    plan = engine.experiment_plan()
    assert plan['architecture'] == 'ResNet34 FPN' and plan['size'] == 448
    assert plan['encoder_lr'] == 5e-5 and plan['gaussian_sigma_cells'] == 1.5
    assert plan['candidate_epochs'] == [6, 10, 16, 24, 32, 40]
    assert plan['gate'] == {'relative_rmse_reduction': 0.05, 'minimum_improved_folds': 3}
    print(json.dumps({'cpu_self_check': 'passed', 'preprocessing_deterministic': True,
        'constant_RGB_before': [64, 128, 192], 'constant_RGB_after': [96, 128, 160],
        'output_shape': list(full_shape), 'loss': float(loss.detach()),
        'frozen_batchnorm_layers': len(batch_norms), 'gpu_used': False}), flush=True)


def final(images, records, train, test, baseline_test, output, provenance, selection):
    destination = (output / 'submission_illumination_v3.csv').resolve()
    engine_destination = (output / 'submission_fine_v3.csv').resolve()
    if destination.exists() or engine_destination.exists():
        raise FileExistsError('Refusing to overwrite an existing illumination submission')
    _base_final(images, records, train, test, baseline_test, output, provenance, selection)
    if not selection['gate_passed']:
        return
    # The shared engine uses a fixed basename. Rename only within this isolated output.
    if not all(path.parent == output.resolve() for path in (destination, engine_destination)):
        raise ValueError('Unexpected final artifact path')
    engine_destination.rename(destination)
    result = engine.read_json(output / 'result.json')
    if result['submission_sha256'] != engine.file_sha256(destination):
        raise ValueError('Final illumination CSV hash changed')
    result.update({'experiment': 'illumination_v3', 'architecture': 'ResNet34 FPN',
                   'preprocessing': preprocessing_spec(), 'submission_file': str(destination)})
    engine.write_json(output / 'result.json', result)
    print(f'Illumination submission: {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'validate', 'final', 'all', 'self-check'), default='self-check')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    configure_engine()
    if args.stage == 'self-check':
        self_check()
        return
    output = args.output.resolve()
    if not output.is_relative_to(DEFAULT_OUTPUT.resolve()):
        raise ValueError('Output must remain inside outputs/iteration3_20261024/illumination')
    output.mkdir(parents=True, exist_ok=True)
    records, train, test, truth, baseline, baseline_test, sources = load_inputs()
    engine.require_same_json(output / 'plan.json', experiment_plan())
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    images, cache_hash = load_images(records, output, sources[str(engine.MANIFEST.resolve())], device)
    provenance = {'source_sha256': sources, 'cache_sha256': cache_hash, 'preprocessing': preprocessing_spec(),
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchvision': str(engine.torchvision.__version__), 'numpy': str(np.__version__),
            'pillow': PIL.__version__, 'cuda_build': torch.version.cuda,
            'device_type': device.type, 'cudnn_benchmark': True}}
    engine.require_same_json(output / 'provenance.json', provenance)
    if args.stage == 'prepare':
        print(json.dumps({'state': 'prepared', 'train_count': len(train), 'test_count': len(test),
                          'architecture': 'ResNet34 FPN', 'preprocessing': preprocessing_spec()}), flush=True)
        return
    if args.stage in ('validate', 'all'):
        selection = engine.validate(images, records, train, truth, baseline, output, provenance)
    else:
        selection = engine.read_json(output / 'selection.json')
    if args.stage in ('final', 'all'):
        final(images, records, train, test, baseline_test, output, provenance, selection)


if __name__ == '__main__':
    main()
