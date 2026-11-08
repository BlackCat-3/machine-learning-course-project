"""Fixed, deterministic crop TTA study using the four previously selected models.

Validation evaluates three predeclared schemes. Test inference uses one explicit
scheme; it never chooses a scheme using test images or labels. No retraining.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import BatchPreprocessor, CLASS_TO_INDEX, NUM_CLASSES, build_model, classification_metrics
from seed_erasing_study import DEFAULT_OUTPUT, paths

ROOT = Path(__file__).resolve().parent
MODELS = ("old_mixup", "detail_seed43", "resnet34", "detail_seed44")
# Each raw view is followed by its horizontal flip. Coordinates are top, left,
# height, width in the original 48 x 48 image; resize matches predict.py exactly.
VIEWS = ((0, 0, 48, 48), (2, 2, 44, 44), (0, 0, 44, 44),
         (0, 4, 44, 44), (4, 0, 44, 44), (4, 4, 44, 44))
SCHEMES = {"full": 1, "center": 2, "five": 6}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def raw_view(images, index):
    top, left, height, width = VIEWS[index]
    return images[:, :, top:top + height, left:left + width]


def check_probabilities(probabilities, rows):
    if (probabilities.shape != (rows, NUM_CLASSES) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid probability output")


def save_predictions(path, ids, labels, probabilities):
    check_probabilities(probabilities, len(ids))
    prediction = probabilities.argmax(1)
    np.savez_compressed(path.with_suffix(".npz"), ids=ids, probabilities=probabilities,
                        predictions=prediction, **({"labels": labels} if labels is not None else {}))
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Emotion"])
        writer.writerows(zip(ids.tolist(), prediction.tolist()))
    return {"npz": str(path.with_suffix(".npz").resolve()),
            "csv": str(path.with_suffix(".csv").resolve()),
            "csv_sha256": sha(path.with_suffix(".csv")),
            **({"correct": int((prediction == labels).sum()),
                "metrics": classification_metrics(labels, prediction)} if labels is not None else {})}


def load_data(data, split):
    with np.load(data, allow_pickle=False) as cache:
        all_labels = cache["train_labels"].astype(np.int64)
        all_ids = cache["train_ids"].astype(str)
        train_indices = cache["train_indices"].astype(np.int64)
        val_indices = cache["val_indices"].astype(np.int64)
        prefix = all_labels.tobytes() + all_ids.tobytes()
        fingerprints = {
            "val": hashlib.sha256(prefix + train_indices.tobytes() + val_indices.tobytes()).hexdigest(),
            "test": hashlib.sha256(prefix + np.arange(len(all_labels), dtype=np.int64).tobytes()
                                   + np.array([], dtype=np.int64).tobytes()).hexdigest(),
        }
        if split == "val":
            images, ids, labels = cache["train_images"][val_indices], all_ids[val_indices], all_labels[val_indices]
        else:
            images, ids, labels = cache["test_images"], cache["test_ids"].astype(str), None
    if images.dtype != np.uint8 or images.shape != (len(ids), 48, 48) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid image cache or repeated output IDs")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Overlapping fit/validation indices")
    return images, ids, labels, fingerprints, len(all_labels), len(train_indices), len(val_indices)


def validate_checkpoint(checkpoint, split, fingerprint, n_all, n_train, n_val):
    config = checkpoint["config"]
    if config.get("class_to_index") != CLASS_TO_INDEX or checkpoint.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Checkpoint class order mismatch")
    if bool(config.get("full_train")) != (split == "test"):
        raise ValueError("Only heldout models may validate; only full-refit models may predict test")
    if config.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Checkpoint dataset/split fingerprint mismatch")
    if (config.get("training_is_partial") or config.get("validation_is_partial")
            or config.get("max_train_batches", 0) or config.get("max_val_batches", 0)):
        raise ValueError("Partial/smoke checkpoint rejected")
    if config.get("train_samples") != (n_all if split == "test" else n_train):
        raise ValueError("Training sample count mismatch")
    if config.get("validation_samples") != (0 if split == "test" else n_val):
        raise ValueError("Validation sample count mismatch")
    if checkpoint.get("selected_source") not in ("raw", "ema"):
        raise ValueError("Unknown selected weights")
    if split == "test" and checkpoint.get("validation_accuracy") is not None:
        raise ValueError("Full-refit checkpoint must not claim independent validation")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--scheme", choices=tuple(SCHEMES), default="five",
                        help="Test scheme. Validation always evaluates all three fixed schemes.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/cache.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/late_20261018/tta")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) <= 0:
        parser.error("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    amp = args.device == "cuda" and not args.no_amp
    schemes = tuple(SCHEMES) if args.split == "val" else (args.scheme,)
    max_views = max(SCHEMES[name] for name in schemes)
    images, ids, labels, fingerprints, n_all, n_train, n_val = load_data(args.data, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"split": args.split, "schemes": {k: 2 * SCHEMES[k] for k in schemes},
                "crop_coordinates_top_left_height_width": VIEWS, "crop_order": "full, center, top-left, top-right, bottom-left, bottom-right",
                "averaging": "equal probabilities across views including flips, then equal across four models",
                "class_to_index": CLASS_TO_INDEX, "rows": len(ids), "dataset_fingerprint": fingerprints[args.split],
                "script_sha256": sha(__file__), "model_source_sha256": sha(ROOT / "model.py"),
                "batch_size": args.batch_size, "amp": amp, "models": {}, "ensembles": {},
                "scope": "Repeated model-selection holdout; no test labels or duplicate-label matching."}
    write(args.output_dir / f"{args.split}_plan.json", metadata)
    ensembles = {name: np.zeros((len(ids), NUM_CLASSES), np.float64) for name in schemes}
    source_paths = paths(DEFAULT_OUTPUT)
    started = time.monotonic()
    for name in MODELS:
        checkpoint_path = source_paths[name][0] / "best.pt" if args.split == "val" else source_paths[name][2] / "final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, args.split, fingerprints[args.split], n_all, n_train, n_val)
        expected_metrics = json.loads((source_paths[name][0] / "best_metrics.json").read_text(encoding="utf-8"))
        if (checkpoint["epoch"] != expected_metrics["epoch"]
                or checkpoint["selected_source"] != expected_metrics["selected_source"]):
            raise ValueError("Checkpoint epoch/raw-or-EMA source differs from heldout selection")
        config = checkpoint["config"]
        model = build_model(pretrained=False, dropout=config.get("dropout", 0.2),
                            architecture=config.get("architecture", "resnet18"), no_maxpool=config.get("no_maxpool", False))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model = model.to(device, memory_format=torch.channels_last).eval()
        preprocessor = BatchPreprocessor(config["image_size"]).to(device)
        arrays = {scheme: np.zeros((len(ids), NUM_CLASSES), np.float32) for scheme in schemes}
        print(f"{args.split} {name}: {max_views * 2} views, {len(ids)} images", flush=True)
        for offset in range(0, len(ids), args.batch_size):
            raw = torch.from_numpy(images[offset:offset + args.batch_size]).unsqueeze(1).to(device)
            running = torch.zeros((len(raw), NUM_CLASSES), device=device, dtype=torch.float32)
            for view_index in range(max_views):
                batch = preprocessor(raw_view(raw, view_index), augment=False)
                with torch.autocast(device_type=device.type, enabled=amp):
                    pair = (model(batch).float().softmax(1) + model(batch.flip(-1)).float().softmax(1)) / 2
                running += pair
                for scheme in schemes:
                    if view_index + 1 == SCHEMES[scheme]:
                        arrays[scheme][offset:offset + len(raw)] = (running / (view_index + 1)).cpu().numpy()
        model_meta = {"checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha(checkpoint_path),
                      "epoch": checkpoint["epoch"], "source": checkpoint["selected_source"], "schemes": {}}
        for scheme in schemes:
            model_meta["schemes"][scheme] = save_predictions(args.output_dir / f"{args.split}_{name}_{scheme}", ids, labels, arrays[scheme])
            ensembles[scheme] += arrays[scheme]
        metadata["models"][name] = model_meta
        write(args.output_dir / f"{args.split}_progress.json", metadata)
        print(f"Finished {name}; elapsed {time.monotonic() - started:.1f}s", flush=True)
        del checkpoint, model, preprocessor, arrays
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for scheme in schemes:
        probabilities = (ensembles[scheme] / len(MODELS)).astype(np.float32)
        metadata["ensembles"][scheme] = save_predictions(args.output_dir / f"{args.split}_ensemble_{scheme}", ids, labels, probabilities)
        if labels is not None:
            result = metadata["ensembles"][scheme]
            result["gain_correct_vs_previous"] = result["correct"] - 2973
            result["gain_percentage_points_vs_previous"] = 100 * (result["metrics"]["accuracy"] - 2973 / len(labels))
            print(f"{scheme}: {result['correct']}/{len(labels)} = {100 * result['metrics']['accuracy']:.4f}%", flush=True)
    if args.split == "val":
        # An independently saved previous result anchors the unchanged full+flip
        # preprocessing, IDs, and labels before interpreting new crop results.
        with np.load(DEFAULT_OUTPUT / "selected_validation.npz", allow_pickle=False) as old:
            if not np.array_equal(old["ids"], ids) or not np.array_equal(old["labels"], labels):
                raise ValueError("Previous validation prediction ordering differs")
            if int((old["probabilities"].argmax(1) == labels).sum()) != 2973:
                raise ValueError("Previous baseline result differs")
            baseline = (ensembles["full"] / len(MODELS)).astype(np.float32)
            metadata["baseline_max_probability_difference"] = float(np.abs(baseline - old["probabilities"]).max())
            metadata["baseline_changed_predictions"] = int((baseline.argmax(1) != old["probabilities"].argmax(1)).sum())
            if metadata["baseline_max_probability_difference"] > 0.002:
                raise ValueError("Baseline TTA unexpectedly differs from previous predictions")
        best = max(schemes, key=lambda scheme: (metadata["ensembles"][scheme]["correct"], -SCHEMES[scheme]))
        for scheme in schemes:
            metadata["ensembles"][scheme]["gain_correct_vs_recomputed_full"] = (
                metadata["ensembles"][scheme]["correct"] - metadata["ensembles"]["full"]["correct"])
        metadata["best_scheme"] = best
        metadata["improved_vs_previous"] = metadata["ensembles"][best]["correct"] > 2973
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["status"] = "complete"
    write(args.output_dir / f"{args.split}_summary.json", metadata)
    print(f"Complete: {args.output_dir / (args.split + '_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
