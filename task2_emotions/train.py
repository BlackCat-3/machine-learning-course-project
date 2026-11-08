"""Fine-tune an image classifier locally and keep a validation-selected best.

Example:
  python train.py --data data/cache.npz --output-dir outputs/resnet18_main
  python train.py --resume outputs/resnet18_main/last.pt --output-dir outputs/resnet18_main
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import (BatchPreprocessor, CLASS_TO_INDEX, NUM_CLASSES, SUPPORTED_ARCHITECTURES,
                   build_model, classification_metrics, classification_head_parameters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/cache.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resnet18_main"))
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, default="resnet18")
    parser.add_argument("--no-maxpool", action="store_true", help="Remove the initial max-pooling layer to retain more spatial detail")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--head-lr-multiplier", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Beta distribution parameter; 0 disables training-only mixup")
    parser.add_argument("--mixup-prob", type=float, default=0.5, help="Probability of applying mixup to a training batch")
    parser.add_argument("--erasing-prob", type=float, default=0.0, help="Per-image probability of training-only random erasing; 0 preserves the original augmentation")
    parser.add_argument("--erasing-scale-min", type=float, default=0.02, help="Minimum erased fraction of image area, before integer rounding")
    parser.add_argument("--erasing-scale-max", type=float, default=0.08, help="Maximum erased fraction of image area, before integer rounding")
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995, help="0 disables EMA; both raw/EMA models are validated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Start a new fine-tuning run from a trusted best.pt/final.pt; reset optimizer, schedule and epoch counters")
    parser.add_argument("--full-train", action="store_true", help="Train on all labeled data; no validation or best-by-validation claims")
    parser.add_argument("--no-pretrained", action="store_true", help="Random initialization, useful for offline smoke tests")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Slower deterministic algorithms where supported; warn for unsupported operations")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Smoke-test limit; 0 processes all batches")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Smoke-test limit; 0 processes all batches")
    parser.add_argument("--stop-after-epochs", type=int, default=0, help="Stop after this absolute completed epoch, preserving the --epochs LR plan; 0 runs the full plan")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.architecture == "efficientnet_b0" and args.no_maxpool:
        parser.error("--no-maxpool is only valid for ResNet architectures")
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint is not None and args.init_checkpoint.resolve().parent == args.output_dir.resolve():
        parser.error("--init-checkpoint requires a different output directory to preserve its source checkpoint")
    if min(args.epochs, args.batch_size, args.image_size, args.threads) <= 0:
        parser.error("epochs, batch-size, image-size and threads must be positive")
    if args.workers < 0 or min(args.max_train_batches, args.max_val_batches) < 0:
        parser.error("workers and batch limits must be non-negative")
    if not 0 <= args.ema_decay < 1 or not 0 <= args.label_smoothing < 1 or not 0 <= args.dropout < 1:
        parser.error("ema-decay, label-smoothing and dropout must be in [0, 1)")
    if args.lr <= 0 or args.head_lr_multiplier <= 0 or args.warmup_epochs < 0:
        parser.error("learning rates must be positive; warmup must be non-negative")
    if not math.isfinite(args.mixup_alpha) or args.mixup_alpha < 0 or not 0 <= args.mixup_prob <= 1:
        parser.error("mixup-alpha must be finite/non-negative and mixup-prob must be in [0,1]")
    if not 0 <= args.erasing_prob <= 1:
        parser.error("erasing-prob must be in [0,1]")
    if not 0 < args.erasing_scale_min <= args.erasing_scale_max <= 1:
        parser.error("erasing scales must satisfy 0 < min <= max <= 1")
    if args.stop_after_epochs < 0:
        parser.error("stop-after-epochs must be non-negative")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_initialization_checkpoint(path: Path, config: dict) -> tuple[dict, dict]:
    """Validate a trusted inference checkpoint before initializing a new run.

    The data split and model topology must match. Training hyperparameters and
    input resolution may deliberately differ; no optimizer/history is inherited.
    """
    path = path.resolve()
    if path.name not in ("best.pt", "final.pt"):
        raise ValueError("Initialization requires an inference best.pt or final.pt, not last.pt")
    # Only load trusted local checkpoints produced by this project.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config", {})
    if "optimizer" in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Initialization requires an inference-only checkpoint with model_state")
    if checkpoint.get("class_to_index") != CLASS_TO_INDEX or saved.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Initialization class mapping mismatch")
    defaults = {"architecture": "resnet18", "no_maxpool": False}
    for key in ("architecture", "no_maxpool", "dataset_fingerprint", "full_train", "train_samples", "validation_samples"):
        value = saved.get(key, defaults.get(key))
        if value is None or value != config.get(key):
            raise ValueError(f"Initialization mismatch for {key}: saved={value!r}, requested={config.get(key)!r}")
    for key in ("training_is_partial", "validation_is_partial", "max_train_batches", "max_val_batches"):
        if saved.get(key, False):
            raise ValueError(f"Cannot initialize from a partial run: {key}")
    source = checkpoint.get("selected_source")
    if source not in ("raw", "ema"):
        raise ValueError("Initialization checkpoint must identify its selected raw/ema source")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise ValueError("Initialization checkpoint has no valid completed epoch")
    image_size = saved.get("image_size")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Initialization checkpoint has no valid source image_size")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "checkpoint": str(path), "sha256": digest.hexdigest(), "source_epoch": epoch,
        "selected_source": source, "source_image_size": image_size,
        "source_full_train": saved["full_train"],
        "source_dataset_fingerprint": saved["dataset_fingerprint"],
        "optimizer_and_schedule_reset": True,
    }
    return checkpoint, metadata


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float, steps: int) -> None:
    decay = min(decay, (1 + steps) / (10 + steps))
    source = model.state_dict()
    for key, value in ema.state_dict().items():
        if value.is_floating_point():
            value.mul_(decay).add_(source[key], alpha=1 - decay)
        else:
            value.copy_(source[key])


@torch.inference_mode()
def evaluate(model, loader, preprocessor, device, amp_enabled: bool, criterion, max_batches=0):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    probability_parts, label_parts = [], []
    num_samples = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        labels = labels.to(device, non_blocking=True)
        images = preprocessor(images.to(device, non_blocking=True), augment=False)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_sum += loss.detach().float() * len(labels)
        probability_parts.append(logits.float().softmax(dim=1).cpu().numpy())
        label_parts.append(labels.cpu().numpy())
        num_samples += len(labels)
    if num_samples == 0:
        raise ValueError("Validation set is empty")
    probabilities = np.concatenate(probability_parts)
    labels = np.concatenate(label_parts)
    metrics = classification_metrics(labels, probabilities.argmax(axis=1))
    metrics["loss"] = float(loss_sum.item() / num_samples)
    return metrics, probabilities, labels


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output_dir / "last.pt").exists():
        raise FileExistsError(f"Existing run: {args.output_dir}. Use --resume or a new --output-dir.")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = not args.deterministic
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install the CUDA-enabled PyTorch build or use --device cpu.")
    device = torch.device(device_name)
    amp_enabled = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'} | AMP={amp_enabled}", flush=True)
    with np.load(args.data, allow_pickle=False) as cache:
        images = cache["train_images"]
        labels = cache["train_labels"].astype(np.int64)
        ids = cache["train_ids"].astype(str)
        train_indices = np.arange(len(labels), dtype=np.int64) if args.full_train else cache["train_indices"].astype(np.int64)
        val_indices = np.array([], dtype=np.int64) if args.full_train else cache["val_indices"].astype(np.int64)
    if images.dtype != np.uint8 or images.shape != (len(labels), 48, 48):
        raise ValueError("Expected uint8 train_images shaped [N,48,48]")
    if len(ids) != len(labels) or labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError("Invalid train IDs or labels; labels must follow the six-class assignment mapping")
    for name, indices in (("train", train_indices), ("validation", val_indices)):
        if len(indices) != len(np.unique(indices)) or (len(indices) and (indices.min() < 0 or indices.max() >= len(labels))):
            raise ValueError(f"Invalid {name} indices")
    if not len(train_indices) or (not args.full_train and not len(val_indices)):
        raise ValueError("Training/validation split must not be empty")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Training and validation indices overlap")
    fingerprint = hashlib.sha256(labels.tobytes() + ids.tobytes() + train_indices.tobytes() + val_indices.tobytes()).hexdigest()
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(class_to_index=CLASS_TO_INDEX, dataset_fingerprint=fingerprint,
                  train_samples=int(len(train_indices)), validation_samples=int(len(val_indices)),
                  torch_version=str(torch.__version__), device_name=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                  validation_is_partial=bool(args.max_val_batches), training_is_partial=bool(args.max_train_batches))
    config["training_accuracy_definition"] = (
        "On mixup batches: lambda * correct(original_label) + (1-lambda) * correct(permuted_label); "
        "on other batches: ordinary hard-label accuracy. Validation always uses original hard labels."
    )
    initialization_checkpoint = None
    if args.init_checkpoint is not None:
        initialization_checkpoint, config["initialization"] = load_initialization_checkpoint(args.init_checkpoint, config)
    print(f"Images: train={len(train_indices)}, validation={len(val_indices)}, full_train={args.full_train}", flush=True)
    if args.max_train_batches or args.max_val_batches:
        print("SMOKE TEST: batch limits enabled; resulting metrics are not full-run validation results.", flush=True)
    tensor_images = torch.from_numpy(images).unsqueeze(1)
    tensor_labels = torch.from_numpy(labels)
    train_data = TensorDataset(tensor_images[train_indices], tensor_labels[train_indices])
    generator = torch.Generator()
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    if args.workers:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(TensorDataset(tensor_images[val_indices], tensor_labels[val_indices]), shuffle=False, **loader_kwargs) if len(val_indices) else None
    steps_per_epoch = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
    model = build_model(pretrained=not args.no_pretrained and args.resume is None and args.init_checkpoint is None, dropout=args.dropout,
                        architecture=args.architecture, no_maxpool=args.no_maxpool).to(device, memory_format=torch.channels_last)
    if initialization_checkpoint is not None:
        model.load_state_dict(initialization_checkpoint["model_state"], strict=True)
        print(f"Initialize new run from {args.init_checkpoint}; optimizer, schedule and epoch counters reset", flush=True)
        del initialization_checkpoint
    preprocessor = BatchPreprocessor(args.image_size, erasing_prob=args.erasing_prob,
                                    erasing_scale_min=args.erasing_scale_min,
                                    erasing_scale_max=args.erasing_scale_max).to(device)
    head_parameters = classification_head_parameters(model)
    head_ids = {id(parameter) for parameter in head_parameters}
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in model.parameters() if id(parameter) not in head_ids], "lr": args.lr},
        {"params": head_parameters, "lr": args.lr * args.head_lr_multiplier},
    ], weight_decay=args.weight_decay)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_factor(step):
        if warmup_steps and step < warmup_steps:
            return max(0.1, (step + 1) / warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=8192.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False) if args.ema_decay else None
    start_epoch, global_step, best_acc, best_epoch = 0, 0, -1.0, None
    history = []
    if args.resume:
        if args.resume.name != "last.pt":
            raise ValueError("Resume requires the original last.pt checkpoint")
        # Only load trusted checkpoints produced by this project.
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_output_dir = checkpoint["config"].get("output_dir")
        if saved_output_dir is None or Path(saved_output_dir).resolve() != args.output_dir.resolve():
            raise ValueError(
                f"Resume must use the original output directory: {saved_output_dir!r}. "
                "The existing best.pt and validation artifacts must remain with the continued run."
            )
        # A resume continues the same experiment and LR schedule. In particular,
        # a partial smoke run must never inherit its best score into a full run.
        resume_keys = (
            "architecture", "no_maxpool", "image_size", "dropout", "class_to_index", "dataset_fingerprint",
            "full_train", "ema_decay", "seed", "epochs", "batch_size", "lr",
            "head_lr_multiplier", "warmup_epochs", "weight_decay", "label_smoothing",
            "max_train_batches", "max_val_batches", "training_is_partial", "validation_is_partial",
            "deterministic", "no_amp", "mixup_alpha", "mixup_prob",
            "erasing_prob", "erasing_scale_min", "erasing_scale_max",
        )
        # Older checkpoints use ResNet18 with its original max-pooling layer;
        # Checkpoints predating mixup/random erasing use their disabled defaults.
        resume_defaults = {"architecture": "resnet18", "no_maxpool": False, "mixup_alpha": 0.0, "mixup_prob": 0.5,
                           "erasing_prob": 0.0, "erasing_scale_min": 0.02, "erasing_scale_max": 0.08}
        for key in resume_keys:
            saved_value = checkpoint["config"].get(key, resume_defaults.get(key))
            if saved_value != config.get(key):
                raise ValueError(
                    f"Resume configuration mismatch for {key}: saved={saved_value!r}, "
                    f"requested={config.get(key)!r}. Resume with the original training arguments; "
                    "use a new output directory without --resume for a different experiment."
                )
        if "optimizer" not in checkpoint:
            raise ValueError("Resume requires last.pt (including optimizer state), not the inference-only best.pt")
        # This flag records how the original run was initialized; resuming loads
        # saved weights regardless of the current CLI initialization flag.
        config["no_pretrained"] = checkpoint["config"].get("no_pretrained", False)
        # Resuming an initialized run keeps its original provenance; the source
        # file is not loaded again and need not be supplied on the resume CLI.
        config["init_checkpoint"] = checkpoint["config"].get("init_checkpoint")
        if "initialization" in checkpoint["config"]:
            config["initialization"] = copy.deepcopy(checkpoint["config"]["initialization"])
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema_model is not None:
            ema_model.load_state_dict(checkpoint["ema_state"])
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
        best_acc, best_epoch = checkpoint["best_accuracy"], checkpoint["best_epoch"]
        history = checkpoint.get("history", [])
        print(f"Resume after epoch {start_epoch}; best validation accuracy={best_acc if best_acc >= 0 else 'not applicable'}", flush=True)
    write_json(args.output_dir / "config.json", config)
    print("Class mapping: " + json.dumps(CLASS_TO_INDEX), flush=True)
    run_start = time.perf_counter()
    stop_epoch = min(args.epochs, args.stop_after_epochs) if args.stop_after_epochs else args.epochs
    if stop_epoch < args.epochs:
        print(f"Stopping after completed epoch {stop_epoch}; cosine schedule still spans {args.epochs} epochs.", flush=True)
    for epoch in range(start_epoch, stop_epoch):
        epoch_start = time.perf_counter()
        # Epoch-addressable seeds make interruption/resume reproducible at epoch boundaries.
        set_seed(args.seed + epoch)
        generator.manual_seed(args.seed + epoch)
        model.train()
        running_loss = torch.zeros((), device=device)
        running_correct = torch.zeros((), dtype=torch.float32, device=device)
        seen = 0
        skipped_optimizer_steps = 0
        mixup_batches = 0
        for batch_index, (batch_images, batch_labels) in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            batch_images = preprocessor(batch_images.to(device, non_blocking=True), augment=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            mixup_lambda = 1.0
            permuted_labels = None
            if args.mixup_alpha > 0 and np.random.random() < args.mixup_prob:
                mixup_lambda = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                permutation = torch.randperm(len(batch_labels), device=device)
                batch_images = batch_images * mixup_lambda + batch_images[permutation] * (1 - mixup_lambda)
                permuted_labels = batch_labels[permutation]
                mixup_batches += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch_images)
                loss = criterion(logits, batch_labels)
                if permuted_labels is not None:
                    loss = mixup_lambda * loss + (1 - mixup_lambda) * criterion(logits, permuted_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler reduces its scale when nonfinite gradients caused the
            # optimizer step to be skipped. Keep the LR schedule, EMA and step
            # counter aligned with actual parameter updates, including the first
            # iteration (where overflow otherwise triggers a scheduler warning).
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()
                global_step += 1
                if ema_model is not None:
                    update_ema(ema_model, model, args.ema_decay, global_step)
            else:
                skipped_optimizer_steps += 1
            running_loss += loss.detach().float() * len(batch_labels)
            batch_predictions = logits.detach().argmax(dim=1)
            if permuted_labels is None:
                running_correct += (batch_predictions == batch_labels).sum()
            else:
                running_correct += mixup_lambda * (batch_predictions == batch_labels).sum()
                running_correct += (1 - mixup_lambda) * (batch_predictions == permuted_labels).sum()
            seen += len(batch_labels)
            if args.log_every and ((batch_index + 1) % args.log_every == 0 or batch_index + 1 == steps_per_epoch):
                print(f"Epoch {epoch + 1}/{args.epochs} batch {batch_index + 1}/{steps_per_epoch} loss={running_loss.item()/seen:.4f} acc={running_correct.item()/seen:.4f}", flush=True)
        selected_model = ema_model if ema_model is not None else model
        selected_source = "ema" if ema_model is not None else "raw"
        selected_metrics = raw_metrics = ema_metrics = None
        selected_probabilities = selected_labels = None
        if val_loader is not None:
            raw_metrics, raw_probabilities, raw_labels = evaluate(model, val_loader, preprocessor, device, amp_enabled, criterion, args.max_val_batches)
            selected_model, selected_source = model, "raw"
            selected_metrics, selected_probabilities, selected_labels = raw_metrics, raw_probabilities, raw_labels
            if ema_model is not None:
                ema_metrics, ema_probabilities, ema_labels = evaluate(ema_model, val_loader, preprocessor, device, amp_enabled, criterion, args.max_val_batches)
                if ema_metrics["accuracy"] > raw_metrics["accuracy"]:
                    selected_model, selected_source = ema_model, "ema"
                    selected_metrics, selected_probabilities, selected_labels = ema_metrics, ema_probabilities, ema_labels
        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch + 1, "train_loss": float(running_loss.item() / seen), "train_accuracy": float(running_correct.item() / seen),
            "train_samples_seen": seen, "skipped_optimizer_steps": skipped_optimizer_steps,
            "mixup_batches": mixup_batches, "train_accuracy_kind": "mixup_weighted" if mixup_batches else "hard_label",
            "val_loss": selected_metrics["loss"] if selected_metrics else None,
            "val_accuracy": selected_metrics["accuracy"] if selected_metrics else None,
            "raw_val_accuracy": raw_metrics["accuracy"] if raw_metrics else None,
            "ema_val_accuracy": ema_metrics["accuracy"] if ema_metrics else None,
            "selected_source": selected_source, "lr": optimizer.param_groups[0]["lr"], "seconds": round(epoch_seconds, 3),
        }
        history.append(record)
        inference_state = {
            "model_state": selected_model.state_dict(), "config": config, "epoch": epoch + 1,
            "class_to_index": CLASS_TO_INDEX, "selected_source": selected_source,
            "validation_accuracy": selected_metrics["accuracy"] if selected_metrics else None,
        }
        if selected_metrics is not None and selected_metrics["accuracy"] > best_acc:
            best_acc, best_epoch = selected_metrics["accuracy"], epoch + 1
            save_checkpoint(args.output_dir / "best.pt", inference_state)
            selected_metrics.update(epoch=epoch + 1, selected_source=selected_source, validation_is_partial=bool(args.max_val_batches))
            write_json(args.output_dir / "best_metrics.json", selected_metrics)
            np.savez_compressed(args.output_dir / "validation_predictions.npz", ids=ids[val_indices[:len(selected_labels)]], labels=selected_labels,
                                probabilities=selected_probabilities, predictions=selected_probabilities.argmax(axis=1))
        if args.full_train:
            save_checkpoint(args.output_dir / "final.pt", inference_state)
        save_checkpoint(args.output_dir / "last.pt", {
            "model_state": model.state_dict(), "ema_state": ema_model.state_dict() if ema_model is not None else None,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "config": config, "class_to_index": CLASS_TO_INDEX, "epoch": epoch + 1, "global_step": global_step,
            "best_accuracy": best_acc, "best_epoch": best_epoch, "history": history,
        })
        write_json(args.output_dir / "history.json", history)
        with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        val_text = f"val_acc={record['val_accuracy']:.4f} ({selected_source}) best={best_acc:.4f}@{best_epoch}" if selected_metrics else "full training: no validation metrics"
        print(f"EPOCH {epoch + 1}: train_loss={record['train_loss']:.4f} train_acc={record['train_accuracy']:.4f} {val_text} seconds={epoch_seconds:.1f}", flush=True)
    write_json(args.output_dir / "summary.json", {
        "epochs_completed": history[-1]["epoch"] if history else start_epoch,
        "planned_epochs": args.epochs, "stop_after_epochs": args.stop_after_epochs,
        "best_epoch": best_epoch, "best_validation_accuracy": best_acc if best_acc >= 0 else None,
        "full_train": args.full_train, "validation_is_partial": bool(args.max_val_batches), "training_is_partial": bool(args.max_train_batches),
        "seconds_this_invocation": round(time.perf_counter() - run_start, 3), "output_dir": str(args.output_dir.resolve()),
    })
    print(f"Finished. Artifacts: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
