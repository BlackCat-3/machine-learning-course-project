"""CPU checks for random erasing and exact compatibility with the saved baseline.

Run: python verify_erasing.py
The legacy model file is the immutable source snapshot from the previous study.
No model training, GPU computation, or output/checkpoint mutation is performed.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from model import BatchPreprocessor
from train import parse_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-model", type=Path,
                        default=Path("outputs/improvement_20261016/source_snapshot/model.py"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("legacy_fer_model", args.legacy_model)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy model from {args.legacy_model}")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    baseline = BatchPreprocessor()
    previous = legacy.BatchPreprocessor()
    erased = BatchPreprocessor(erasing_prob=0.25)
    torch.manual_seed(101)
    images = torch.randint(0, 256, (12, 1, 48, 48), dtype=torch.uint8)
    for augment in (False, True):
        for seed in (43, 44, 100):
            torch.manual_seed(seed)
            expected = previous(images, augment=augment)
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(seed)
            actual = baseline(images, augment=augment)
            assert torch.equal(actual, expected), "Disabled erasing changed legacy preprocessing"
            assert torch.equal(torch.random.get_rng_state(), expected_rng), "Disabled erasing changed RNG consumption"

    torch.manual_seed(43)
    evaluation_rng = torch.random.get_rng_state().clone()
    expected = baseline(images, augment=False)
    actual = erased(images, augment=False)
    assert torch.equal(actual, expected), "Erasing leaked into evaluation"
    assert torch.equal(torch.random.get_rng_state(), evaluation_rng), "Evaluation consumed random numbers"

    # An even-sized linear ramp has no pixel exactly equal to its mean, so every
    # changed pixel reveals the erasure mask without depending on implementation.
    size, count = 112, 512
    ramps = (torch.arange(size * size, dtype=torch.float32) / (size * size)).reshape(1, 1, size, size)
    ramps = ramps.repeat(count, 1, 1, 1)
    # Different image means catch accidental global-batch filling as well.
    ramps = ramps * 0.5 + ((torch.arange(count, dtype=torch.float32) % 19) / 38).view(-1, 1, 1, 1)
    means = ramps.mean(dim=(2, 3), keepdim=True).expand_as(ramps)
    torch.manual_seed(713)
    result = erased._erase_images(ramps)
    changed = (result != ramps)[:, 0]
    applied = changed.flatten(1).any(dim=1)
    applied_count = int(applied.sum())
    assert 0.18 < applied_count / count < 0.32, "Per-image application probability is implausible"
    assert torch.equal(result[:, 0][changed], means[:, 0][changed]), "Fill is not the individual grayscale image mean"
    rectangles = []
    areas = []
    for mask in changed[applied]:
        rows, columns = torch.where(mask)
        top, bottom = int(rows.min()), int(rows.max()) + 1
        left, right = int(columns.min()), int(columns.max()) + 1
        height, width = bottom - top, right - left
        assert int(mask.sum()) == height * width, "Erasure is not one filled rectangle"
        assert 0.018 <= height * width / (size * size) <= 0.085, "Erasure area exceeds rounding tolerance"
        assert 0.45 <= height / width <= 2.2, "Erasure aspect ratio exceeds rounding tolerance"
        rectangles.append((top, left, height, width))
        areas.append(height * width / (size * size))
    assert len(set(rectangles)) > applied_count * 0.9, "Images did not receive independent rectangle samples"
    torch.manual_seed(713)
    assert torch.equal(erased._erase_images(ramps), result), "Fixed-seed erasing is not reproducible"
    all_erased = BatchPreprocessor(erasing_prob=1.0)._erase_images(ramps[:32])
    assert ((all_erased != ramps[:32]).flatten(1).any(dim=1)).all(), "p=1 failed to apply a fitting rectangle"

    # Validate rejection at both the reusable component and command-line layer.
    invalid_options = [
        {"erasing_prob": -0.1}, {"erasing_prob": 1.1}, {"erasing_prob": float("nan")},
        {"erasing_scale_min": 0}, {"erasing_scale_min": 0.09},
        {"erasing_scale_max": 1.01}, {"erasing_scale_max": float("nan")},
    ]
    for option in invalid_options:
        try:
            BatchPreprocessor(**option)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid component option accepted: {option}")
        key, value = next(iter(option.items()))
        with patch.object(sys, "argv", ["train.py", "--" + key.replace("_", "-"), str(value)]):
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    parse_args()
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError(f"Invalid CLI option accepted: {option}")
    with patch.object(sys, "argv", ["train.py"]):
        defaults = parse_args()
    assert (defaults.erasing_prob, defaults.erasing_scale_min, defaults.erasing_scale_max) == (0.0, 0.02, 0.08)
    print(json.dumps({
        "passed": True, "device": "cpu", "legacy_exact_output_and_rng_checks": 6,
        "evaluation_unchanged": True, "erasing_applied_images": applied_count,
        "sample_images": count, "unique_rectangles": len(set(rectangles)),
        "observed_area_range": [min(areas), max(areas)],
        "mean_fill_verified": True, "fixed_seed_reproducible": True,
        "invalid_options_rejected_at_both_interfaces": len(invalid_options),
    }, indent=2))


if __name__ == "__main__":
    main()
