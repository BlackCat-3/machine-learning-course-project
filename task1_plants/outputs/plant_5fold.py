"""Stratified five-fold ResNet18 training and test-probability ensemble."""

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
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(r"D:\Downloads\2026task1.zip")
OUT = ROOT / "work" / "fivefold"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("baseline", ROOT / "outputs" / "plant_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

SPLIT_SEED = 42
EPOCHS = 10
WARMUP = 2
BATCH_SIZE = 16
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes, records, test_records, test_ids = baseline.read_layout(ZIP, None)
torch.hub.set_dir(str((ROOT / "work" / "model_cache").resolve()))

# The first fold matches the original 400/100 split, so its existing checkpoint is reusable.
rng = np.random.default_rng(SPLIT_SEED)
fold_records = [[] for _ in range(5)]
for label in range(len(classes)):
    group = [record for record in records if record[1] == label]
    rng.shuffle(group)
    for fold, chunk in enumerate(np.array_split(np.arange(len(group)), 5)):
        fold_records[fold].extend(group[int(i)] for i in chunk)
assert len(fold_records) == 5 and all(len(part) == 100 for part in fold_records)
assert {record for part in fold_records for record in part} == set(records)
_, original_val = baseline.split_stratified(records, len(classes), 0.2, SPLIT_SEED)
assert set(fold_records[0]) == set(original_val)


@torch.inference_mode()
def probabilities(model, loader):
    model.eval()
    chunks = []
    for images, _ in loader:
        chunks.append(model(images.to(device)).softmax(dim=1).cpu().numpy())
    return np.concatenate(chunks)


oof = np.full((len(records), len(classes)), np.nan, dtype=np.float32)
record_index = {record[0]: i for i, record in enumerate(records)}
test_sum = np.zeros((len(test_records), len(classes)), dtype=np.float64)
fold_results = []
for fold in range(5):
    val_records = fold_records[fold]
    val_set_records = set(val_records)
    train_records = [record for record in records if record not in val_set_records]
    fold_path = OUT / f"fold_{fold}"
    fold_path.mkdir(exist_ok=True)
    mean, std = baseline.compute_train_rgb_stats(ZIP, train_records)
    normalize = transforms.Normalize(mean, std)
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
    train_set = baseline.ZipImages(ZIP, train_records, train_transform)
    val_set = baseline.ZipImages(ZIP, val_records, eval_transform)
    test_set = baseline.ZipImages(ZIP, test_records, eval_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

    prior_checkpoint = ROOT / "outputs" / "plant_run_train_norm" / "best_model.pt"
    if fold == 0 and prior_checkpoint.is_file():
        checkpoint = torch.load(prior_checkpoint,
                                map_location="cpu", weights_only=False)
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        best_epoch = checkpoint["epoch"]
        print(f"fold={fold} reused original checkpoint epoch={best_epoch}", flush=True)
    else:
        seed = 42 + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = None
        best_f1, best_epoch, best_state = -1.0, None, None
        for epoch in range(EPOCHS):
            if epoch == 0:
                for parameter in model.parameters():
                    parameter.requires_grad = False
                for parameter in model.fc.parameters():
                    parameter.requires_grad = True
                optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
            if epoch == WARMUP:
                for parameter in model.parameters():
                    parameter.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loss, accuracy = baseline.train_epoch(model, train_loader, optimizer, criterion, device)
            actual, predicted = baseline.predict(model, val_loader, device)
            f1, _, _ = baseline.macro_f1(actual, predicted, len(classes))
            print(f"fold={fold} epoch={epoch+1}/{EPOCHS} loss={loss:.4f} "
                  f"train_acc={accuracy:.3f} val_f1={f1:.4f}", flush=True)
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch + 1
                best_state = copy.deepcopy({key: value.detach().cpu()
                                            for key, value in model.state_dict().items()})
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "classes": classes,
                    "epoch": best_epoch, "val_macro_f1": best_f1}, fold_path / "best_model.pt")

    val_prob = probabilities(model, val_loader)
    test_prob = probabilities(model, test_loader)
    val_actual = [label for _, label in val_records]
    val_pred = val_prob.argmax(axis=1)
    fold_f1, class_f1, confusion = baseline.macro_f1(val_actual, val_pred, len(classes))
    for (path, _), prob in zip(val_records, val_prob):
        oof[record_index[path]] = prob
    test_sum += test_prob
    fold_results.append({"fold": fold, "best_epoch": best_epoch,
                         "val_macro_f1": fold_f1, "per_class_f1": dict(zip(classes, class_f1)),
                         "confusion_matrix_rows_true": confusion,
                         "train_count": len(train_records), "val_count": len(val_records),
                         "rgb_mean": mean, "rgb_std": std})
    np.save(fold_path / "validation_probabilities.npy", val_prob)
    np.save(fold_path / "test_probabilities.npy", test_prob)
    print(f"fold={fold} best_epoch={best_epoch} heldout_f1={fold_f1:.4f}", flush=True)
    train_set.close(); val_set.close(); test_set.close()

assert np.isfinite(oof).all()
oof_actual = [label for _, label in records]
oof_f1, oof_class_f1, oof_confusion = baseline.macro_f1(
    oof_actual, oof.argmax(axis=1), len(classes))
ensemble = test_sum / 5
submission = ROOT / "outputs" / "fivefold_submission.csv"
with submission.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(["ID", "Category"])
    writer.writerows((image_id, classes[int(pred)])
                     for image_id, pred in zip(test_ids, ensemble.argmax(axis=1)))
results = {"split_seed": SPLIT_SEED, "epochs": EPOCHS, "warmup_epochs": WARMUP,
           "batch_size": BATCH_SIZE, "classes": classes, "folds": fold_results,
           "oof_macro_f1": oof_f1,
           "oof_per_class_f1": dict(zip(classes, oof_class_f1)),
           "oof_confusion_matrix_rows_true": oof_confusion,
           "submission": str(submission)}
(ROOT / "outputs" / "fivefold_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
np.save(OUT / "oof_probabilities.npy", oof)
np.save(OUT / "test_ensemble_probabilities.npy", ensemble)
print(f"OOF macro F1={oof_f1:.4f}; submission={submission}", flush=True)
