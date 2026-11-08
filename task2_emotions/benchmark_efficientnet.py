"""Disposable CUDA throughput benchmark; never saves or supplies model weights."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from model import BatchPreprocessor, build_model


def timed_steps(operation, count):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    return [start.elapsed_time(end) / 1000 for start, end in zip(starts, ends)], wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data", type=Path, default=Path("data/cache.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/late_20261018/efficientnet_benchmark.json"))
    args = parser.parse_args()
    if min(args.image_size, args.batch_size, args.warmup_steps, args.timed_steps, args.threads) <= 0:
        parser.error("sizes and step counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = True
    with np.load(args.data, allow_pickle=False) as cache:
        indices = cache["train_indices"][:args.batch_size]
        images = torch.from_numpy(cache["train_images"][indices]).unsqueeze(1).pin_memory()
        labels = torch.from_numpy(cache["train_labels"][indices].astype(np.int64)).pin_memory()
        n_train, n_val, n_full = len(cache["train_indices"]), len(cache["val_indices"]), len(cache["train_labels"])
    assert len(images) == args.batch_size and images.dtype == torch.uint8
    model = build_model(architecture="efficientnet_b0", pretrained=False, dropout=0.3).to("cuda", memory_format=torch.channels_last).train()
    preprocessor = BatchPreprocessor(args.image_size).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
    scaler = torch.amp.GradScaler("cuda", init_scale=8192.0)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    losses = []
    scales = []

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        batch = preprocessor(images.to("cuda", non_blocking=True), augment=True)
        target = labels.to("cuda", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(batch), target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach())
        scales.append(scaler.get_scale())

    print(f"Benchmark EfficientNet-B0: batch={args.batch_size}, image={args.image_size}, warmup={args.warmup_steps}, timed={args.timed_steps}", flush=True)
    for _ in range(args.warmup_steps):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    seconds, wall = timed_steps(train_step, args.timed_steps)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    optimizer.zero_grad(set_to_none=True)
    model.eval()

    @torch.inference_mode()
    def validation_step():
        batch = preprocessor(images.to("cuda", non_blocking=True), augment=False)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model(batch).float().softmax(1)

    for _ in range(2):
        validation_step()
    val_seconds, val_wall = timed_steps(validation_step, 5)
    step_mean = statistics.mean(seconds)
    val_mean = statistics.mean(val_seconds)
    train_batches = math.ceil(n_train / args.batch_size)
    val_batches = math.ceil(n_val / args.batch_size)
    full_batches = math.ceil(n_full / args.batch_size)
    losses_float = [float(loss.cpu()) for loss in losses]
    assert all(math.isfinite(loss) for loss in losses_float)
    props = torch.cuda.get_device_properties(0)
    result = {
        "status": "complete_benchmark_only", "architecture": "efficientnet_b0", "pretrained": False,
        "weights_saved": False, "image_size": args.image_size, "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps, "timed_steps": args.timed_steps, "amp": True, "channels_last": True,
        "optimizer": "AdamW", "gpu": props.name, "gpu_total_memory_bytes": props.total_memory,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "train_step_seconds": seconds, "mean_train_step_seconds": step_mean,
        "median_train_step_seconds": statistics.median(seconds), "train_wall_seconds": wall,
        "mean_train_wall_seconds_per_step": wall / args.timed_steps,
        "validation_step_seconds": val_seconds, "mean_validation_step_seconds": val_mean,
        "validation_wall_seconds": val_wall, "last_loss": losses_float[-1],
        "grad_scaler_scales": scales, "optimizer_skips_detected": sum(b < a for a, b in zip(scales, scales[1:])),
        "fit_samples": n_train, "validation_samples": n_val, "full_samples": n_full,
        "fit_batches_per_epoch": train_batches, "validation_batches_per_model": val_batches,
        "full_batches_per_epoch": full_batches,
        "estimated_fit_train_seconds_per_epoch": step_mean * train_batches,
        "estimated_validation_raw_and_ema_seconds_per_epoch": 2 * val_mean * val_batches,
        "estimated_fit_epoch_seconds_including_raw_and_ema_validation": step_mean * train_batches + 2 * val_mean * val_batches,
        "estimated_full_epoch_seconds": step_mean * full_batches,
        "scope": "Disposable random-weight throughput test on first real fit batch. Estimates omit checkpoint I/O, EMA updates, loader/scheduler/logging overhead and first-epoch autotuning. No benchmark weights retained or used for submission.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
