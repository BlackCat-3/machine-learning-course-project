"""Plant classification baseline for the original 2026task1.zip.

Example:
    python plant_baseline.py --zip D:\\Downloads\\2026task1.zip --normalization train
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


SAMPLE_NAME = "submission-for-task2.csv"


class ZipImages(Dataset):
    def __init__(self, zip_path: Path, records: list[tuple[str, int]], transform):
        self.archive = zipfile.ZipFile(zip_path)
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(io.BytesIO(self.archive.read(path))) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label

    def close(self):
        self.archive.close()


def read_layout(zip_path: Path, limit_per_class: int | None):
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        train_paths = sorted(name for name in names if "/train/" in name and name.lower().endswith(".png"))
        test_paths = sorted(name for name in names if "/test/" in name and name.lower().endswith(".png"))
        if not train_paths or not test_paths:
            raise ValueError("ZIP must contain train/<class>/*.png and test/*.png")
        classes = sorted({Path(name).parent.name for name in train_paths})
        class_to_id = {name: i for i, name in enumerate(classes)}
        records = [(name, class_to_id[Path(name).parent.name]) for name in train_paths]
        if limit_per_class is not None:
            counts = Counter()
            limited = []
            for record in records:
                if counts[record[1]] < limit_per_class:
                    limited.append(record)
                    counts[record[1]] += 1
            records = limited
        if SAMPLE_NAME not in names:
            raise ValueError(f"Missing {SAMPLE_NAME} in ZIP")
        sample = list(csv.DictReader(io.StringIO(archive.read(SAMPLE_NAME).decode("utf-8-sig"))))
        if not sample or list(sample[0]) != ["ID", "Category"]:
            raise ValueError("Unexpected sample CSV columns")
        test_by_id = {Path(name).name: name for name in test_paths}
        ids = [row["ID"] for row in sample]
        if len(test_by_id) != len(test_paths) or len(ids) != len(set(ids)) or set(ids) != set(test_by_id):
            raise ValueError("Sample CSV IDs do not match test images")
        ordered_test = [(test_by_id[image_id], -1) for image_id in ids]
    return classes, records, ordered_test, ids


def split_stratified(records, n_classes: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(n_classes):
        group = [record for record in records if record[1] == label]
        if len(group) < 2:
            raise ValueError(f"Class {label} has fewer than two images")
        rng.shuffle(group)
        n_val = max(1, min(len(group) - 1, round(len(group) * val_fraction)))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    return train, val


def compute_train_rgb_stats(zip_path: Path, train_records):
    """Compute per-channel pixel statistics from unaugmented training images only."""
    prepare = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for path, _ in train_records:
            with Image.open(io.BytesIO(archive.read(path))) as image:
                tensor = prepare(image.convert("RGB")).to(torch.float64)
            channel_sum += tensor.sum(dim=(1, 2))
            channel_sq_sum += tensor.square().sum(dim=(1, 2))
            pixel_count += tensor.shape[1] * tensor.shape[2]
    mean = channel_sum / pixel_count
    std = (channel_sq_sum / pixel_count - mean.square()).clamp_min(0).sqrt()
    if (std == 0).any():
        raise ValueError("A training image channel has zero standard deviation")
    return mean.tolist(), std.tolist()


def macro_f1(actual, predicted, n_classes: int):
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(actual, predicted):
        confusion[int(true), int(pred)] += 1
    per_class = []
    for label in range(n_classes):
        tp = confusion[label, label]
        fp = confusion[:, label].sum() - tp
        fn = confusion[label, :].sum() - tp
        denominator = 2 * tp + fp + fn
        per_class.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(per_class)), per_class, confusion.tolist()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, total_correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    actual, predicted = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        predicted.extend(logits.argmax(1).cpu().tolist())
        actual.extend(labels.tolist())
    return actual, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("plant_run"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int,
                        help="Seed for train/validation split; defaults to --seed")
    parser.add_argument("--normalization", choices=["imagenet", "train"], default="imagenet")
    parser.add_argument("--cache-dir", type=Path, help="Directory containing downloaded pretrained weights")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--limit-per-class", type=int, help="Quick code check only")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("epochs must be >=1 and warmup-epochs must be between 0 and epochs")
    if not 0 < args.val_fraction < 1 or args.batch_size < 1:
        parser.error("val-fraction must be in (0,1) and batch-size must be positive")
    if args.limit_per_class is not None and args.limit_per_class < 2:
        parser.error("limit-per-class must be at least 2")
    if not args.zip.is_file():
        parser.error(f"ZIP not found: {args.zip}")
    args.out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, records, test_records, test_ids = read_layout(args.zip, args.limit_per_class)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_records, val_records = split_stratified(records, len(classes), args.val_fraction, split_seed)
    print(f"device={device}; classes={classes}; train={len(train_records)}; val={len(val_records)}; test={len(test_records)}", flush=True)

    if args.normalization == "train":
        rgb_mean, rgb_std = compute_train_rgb_stats(args.zip, train_records)
    else:
        rgb_mean, rgb_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    print(f"normalization={args.normalization}; RGB mean={rgb_mean}; std={rgb_std}", flush=True)
    normalize = transforms.Normalize(rgb_mean, rgb_std)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), normalize,
    ])
    train_set = ZipImages(args.zip, train_records, train_transform)
    val_set = ZipImages(args.zip, val_records, eval_transform)
    test_set = ZipImages(args.zip, test_records, eval_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

    torch.hub.set_dir(str((args.cache_dir or args.out / ".torch_cache").resolve()))
    weights = None if args.random_init else ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_epoch, best_state, best_details = -1.0, 0, None, None
    optimizer = None
    for epoch in range(args.epochs):
        if epoch == 0 and args.warmup_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
        if epoch == args.warmup_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss, accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        actual, predicted = predict(model, val_loader, device)
        score, class_f1, confusion = macro_f1(actual, predicted, len(classes))
        print(f"epoch {epoch + 1}/{args.epochs}: loss={loss:.4f} train_acc={accuracy:.3f} val_macro_f1={score:.4f}", flush=True)
        if score > best_f1:
            best_f1, best_epoch = score, epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_details = {"best_epoch": best_epoch, "val_macro_f1": score,
                            "per_class_f1": dict(zip(classes, class_f1)),
                            "confusion_matrix_rows_true": confusion, "classes": classes,
                            "train_count": len(train_records), "val_count": len(val_records),
                            "seed": args.seed, "split_seed": split_seed,
                            "normalization": args.normalization,
                            "rgb_mean": rgb_mean, "rgb_std": rgb_std}

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": classes, "epoch": best_epoch,
                "val_macro_f1": best_f1}, args.out / "best_model.pt")
    (args.out / "validation.json").write_text(json.dumps(best_details, ensure_ascii=False, indent=2), encoding="utf-8")
    _, test_predictions = predict(model, test_loader, device)
    submission = args.out / "submission.csv"
    with submission.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["ID", "Category"])
        writer.writerows((image_id, classes[label]) for image_id, label in zip(test_ids, test_predictions))
    print(f"best epoch={best_epoch}, val_macro_f1={best_f1:.4f}; saved {submission}", flush=True)
    train_set.close(); val_set.close(); test_set.close()


if __name__ == "__main__":
    main()
