"""Predict competition IDs in template order; optionally average models and flip TTA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import BatchPreprocessor, CLASS_TO_INDEX, NUM_CLASSES, SUPPORTED_ARCHITECTURES, build_model, classification_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/cache.npz"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, help="Default: outputs/submission.csv for test; outputs/validation_predictions.csv for val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta", action="store_true", help="Average original and horizontally flipped probabilities")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="val evaluates the SAME ensemble/TTA used for submission")
    parser.add_argument("--allow-smoke-checkpoint", action="store_true", help="Explicitly permit a checkpoint produced with batch limits; testing only")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("outputs/submission.csv" if args.split == "test" else "outputs/validation_predictions.csv")
    if min(args.batch_size, args.threads) <= 0 or args.workers < 0:
        parser.error("batch-size/threads must be positive and workers non-negative")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    amp_enabled = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True
    with np.load(args.data, allow_pickle=False) as cache:
        cache_labels = cache["train_labels"].astype(np.int64)
        cache_ids = cache["train_ids"].astype(str)
        cache_train_indices = cache["train_indices"].astype(np.int64)
        cache_val_indices = cache["val_indices"].astype(np.int64)
        validation_fingerprint = hashlib.sha256(cache_labels.tobytes() + cache_ids.tobytes() + cache_train_indices.tobytes() + cache_val_indices.tobytes()).hexdigest()
        if args.split == "test":
            images, ids, labels = cache["test_images"], cache["test_ids"].astype(str), None
        else:
            indices = cache["val_indices"]
            images, ids, labels = cache["train_images"][indices], cache["train_ids"][indices].astype(str), cache["train_labels"][indices]
    if images.dtype != np.uint8 or images.shape != (len(ids), 48, 48) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid image cache or duplicate output IDs")
    loader = DataLoader(TensorDataset(torch.from_numpy(images).unsqueeze(1)), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    ensemble_probabilities = np.zeros((len(ids), NUM_CLASSES), dtype=np.float64)
    checkpoint_metadata = []
    for path in args.checkpoints:
        # Only load trusted checkpoints produced by this project.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config.get("class_to_index") != CLASS_TO_INDEX:
            raise ValueError(f"Class mapping mismatch: {path}")
        architecture = config.get("architecture", "resnet18")
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}")
        if (config.get("training_is_partial") or config.get("validation_is_partial")) and not args.allow_smoke_checkpoint:
            raise ValueError(f"Smoke-test checkpoint rejected: {path}. Only use --allow-smoke-checkpoint for testing.")
        if args.split == "val" and config.get("full_train"):
            raise ValueError("A full-train checkpoint has seen validation images; it cannot provide independent validation metrics")
        if args.split == "val" and config.get("dataset_fingerprint") != validation_fingerprint:
            raise ValueError("Validation cache/split differs from the checkpoint; independent validation cannot be established")
        model = build_model(pretrained=False, dropout=config.get("dropout", 0.2),
                            architecture=architecture, no_maxpool=config.get("no_maxpool", False))
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device, memory_format=torch.channels_last).eval()
        preprocessor = BatchPreprocessor(config["image_size"]).to(device)
        offset = 0
        print(f"Predicting {args.split}: {path}, image_size={config['image_size']}, flip_TTA={args.tta}", flush=True)
        for (batch_images,) in loader:
            batch_images = preprocessor(batch_images.to(device, non_blocking=True), augment=False)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                probabilities = model(batch_images).float().softmax(dim=1)
                if args.tta:
                    probabilities = (probabilities + model(batch_images.flip(-1)).float().softmax(dim=1)) / 2
            size = len(batch_images)
            ensemble_probabilities[offset:offset + size] += probabilities.cpu().numpy()
            offset += size
        checkpoint_metadata.append({"path": str(path.resolve()), "epoch": checkpoint["epoch"], "source": checkpoint.get("selected_source", "raw")})
        del checkpoint, model, preprocessor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ensemble_probabilities = (ensemble_probabilities / len(args.checkpoints)).astype(np.float32)
    if not np.isfinite(ensemble_probabilities).all() or not np.allclose(ensemble_probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Invalid model probabilities")
    predictions = ensemble_probabilities.argmax(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Emotion"])
        writer.writerows(zip(ids.tolist(), predictions.tolist()))
    probability_path = args.output.with_suffix(".probabilities.npz")
    np.savez_compressed(probability_path, ids=ids, probabilities=ensemble_probabilities, predictions=predictions,
                        **({"labels": labels} if labels is not None else {}))
    metadata = {"split": args.split, "rows": len(ids), "class_to_index": CLASS_TO_INDEX, "flip_tta": args.tta,
                "checkpoints": checkpoint_metadata, "prediction_counts": {name: int((predictions == value).sum()) for name, value in CLASS_TO_INDEX.items()},
                "output_csv": str(args.output.resolve()), "probabilities": str(probability_path.resolve())}
    if labels is not None:
        metadata["validation_metrics"] = classification_metrics(labels, predictions)
        print(f"Validation accuracy: {metadata['validation_metrics']['accuracy']:.6f}", flush=True)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(ids)} predictions: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
