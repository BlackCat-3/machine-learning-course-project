"""Experiment with alternative pretrained models and plant augmentations."""

import argparse
import copy
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
    resnet50,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
spec = importlib.util.spec_from_file_location("plant_baseline", OUTPUTS / "plant_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path(r"D:\Downloads\2026task1.zip"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"], required=True)
    parser.add_argument("--augment", choices=["base", "rotate", "random_crop"], default="base")
    parser.add_argument("--normalization", choices=["train", "imagenet"], default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.label_smoothing < 1:
        parser.error("label-smoothing must be in [0, 1)")
    args.out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, records, test_records, test_ids = baseline.read_layout(args.zip, None)
    train_records, val_records = baseline.split_stratified(records, len(classes), 0.2, args.split_seed)
    if args.normalization == "train":
        mean, std = baseline.compute_train_rgb_stats(args.zip, train_records)
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    print(f"model={args.model} augment={args.augment} image_size={args.image_size} seed={args.seed} train={len(train_records)} val={len(val_records)} device={device}", flush=True)

    augmentation = ([transforms.RandomResizedCrop((args.image_size, args.image_size),
                                                   scale=(0.7, 1.0), ratio=(0.9, 1.1))]
                    if args.augment == "random_crop" else
                    [transforms.Resize((args.image_size, args.image_size))])
    if args.augment == "rotate":
        fill = tuple(round(value * 255) for value in mean)
        augmentation.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, fill=fill),
        ])
    else:
        augmentation.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])
    augmentation.extend([
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_transform = transforms.Compose(augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = baseline.ZipImages(args.zip, train_records, train_transform)
    val_set = baseline.ZipImages(args.zip, val_records, eval_transform)
    test_set = baseline.ZipImages(args.zip, test_records, eval_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    torch.hub.set_dir(str((ROOT / "work" / "model_cache").resolve()))
    if args.model == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(classes))
        head = model.classifier
    elif args.model == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
        head = model.classifier
    elif args.model == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        head = model.fc
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        head = model.fc
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = None
    best_f1, best_epoch, best_state, best_details = -1.0, 0, None, None
    history = []
    for epoch in range(args.epochs):
        if epoch == 0 and args.warmup_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in head.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        if epoch == args.warmup_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=2e-5 if args.model == "convnext_tiny" else 1e-4)
        loss, acc = baseline.train_epoch(model, train_loader, optimizer, criterion, device)
        actual, predicted = baseline.predict(model, val_loader, device)
        f1, per_class, confusion = baseline.macro_f1(actual, predicted, len(classes))
        history.append({"epoch": epoch + 1, "train_loss": loss, "train_accuracy": acc, "val_macro_f1": f1})
        print(f"epoch {epoch+1}/{args.epochs} loss={loss:.4f} train_acc={acc:.3f} val_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            best_state = copy.deepcopy({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()})
            best_details = {"best_epoch": best_epoch, "val_macro_f1": f1,
                            "per_class_f1": dict(zip(classes, per_class)),
                            "confusion_matrix_rows_true": confusion, "classes": classes,
                            "model": args.model, "augment": args.augment,
                            "label_smoothing": args.label_smoothing,
                            "normalization": args.normalization, "mean": mean, "std": std,
                            "seed": args.seed, "split_seed": args.split_seed,
                            "image_size": args.image_size,
                            "train_count": len(train_records), "val_count": len(val_records)}
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": classes, "model": args.model,
                "epoch": best_epoch, "val_macro_f1": best_f1}, args.out / "best_model.pt")
    (args.out / "validation.json").write_text(json.dumps(best_details, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _, test_preds = baseline.predict(model, test_loader, device)
    with (args.out / "submission.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Category"])
        writer.writerows((image_id, classes[pred]) for image_id, pred in zip(test_ids, test_preds))
    print(f"best epoch {best_epoch} val_f1={best_f1:.4f}", flush=True)
    train_set.close(); val_set.close(); test_set.close()


if __name__ == "__main__":
    main()
