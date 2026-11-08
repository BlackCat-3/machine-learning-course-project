"""Evaluate flip test-time augmentation on the five held-out folds."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "fivefold"
ZIP = Path(r"D:\Downloads\2026task1.zip")
spec = importlib.util.spec_from_file_location("baseline", ROOT / "outputs" / "plant_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
classes, records, test_records, test_ids = baseline.read_layout(ZIP, None)
fold_info = json.loads((ROOT / "outputs" / "fivefold_results.json").read_text(encoding="utf-8"))["folds"]
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng(42)
fold_records = [[] for _ in range(5)]
for label in range(len(classes)):
    group = [record for record in records if record[1] == label]
    rng.shuffle(group)
    for fold, chunk in enumerate(np.array_split(np.arange(len(group)), 5)):
        fold_records[fold].extend(group[int(i)] for i in chunk)


@torch.inference_mode()
def flip_probabilities(model, loader):
    model.eval()
    result = {key: [] for key in ("h", "v", "hv")}
    for images, _ in loader:
        images = images.to(device)
        for key, dims in (("h", (3,)), ("v", (2,)), ("hv", (2, 3))):
            result[key].append(model(torch.flip(images, dims=dims)).softmax(1).cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


record_index = {record[0]: i for i, record in enumerate(records)}
oof = {key: np.full((len(records), len(classes)), np.nan, dtype=np.float32)
       for key in ("original", "h", "v", "hv")}
test_sum = {key: np.zeros((len(test_records), len(classes)), dtype=np.float64)
            for key in oof}
for fold, val_records in enumerate(fold_records):
    cache_path = OUT / f"fold_{fold}" / "tta_flip_probabilities.npz"
    norm = transforms.Normalize(fold_info[fold]["rgb_mean"], fold_info[fold]["rgb_std"])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    val_set = baseline.ZipImages(ZIP, val_records, transform)
    test_set = baseline.ZipImages(ZIP, test_records, transform)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=0)
    fold_val = {"original": np.load(OUT / f"fold_{fold}" / "validation_probabilities.npy")}
    fold_test = {"original": np.load(OUT / f"fold_{fold}" / "test_probabilities.npy")}
    if cache_path.is_file():
        with np.load(cache_path) as cached:
            for key in ("h", "v", "hv"):
                fold_val[key] = cached[f"val_{key}"]
                fold_test[key] = cached[f"test_{key}"]
    else:
        checkpoint_path = (ROOT / "outputs" / "plant_run_train_norm" / "best_model.pt"
                           if fold == 0 else OUT / f"fold_{fold}" / "best_model.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        fold_val.update(flip_probabilities(model, val_loader))
        fold_test.update(flip_probabilities(model, test_loader))
        np.savez_compressed(cache_path,
                            **{f"val_{key}": fold_val[key] for key in ("h", "v", "hv")},
                            **{f"test_{key}": fold_test[key] for key in ("h", "v", "hv")})
    for key in oof:
        for (path, _), prob in zip(val_records, fold_val[key]):
            oof[key][record_index[path]] = prob
        test_sum[key] += fold_test[key]
    val_set.close(); test_set.close()
    print(f"fold {fold + 1}/5 TTA complete", flush=True)

actual = [label for _, label in records]
variants = {
    "original": ["original"],
    "horizontal": ["original", "h"],
    "vertical": ["original", "v"],
    "both_flips": ["original", "h", "v", "hv"],
    "original_h_v": ["original", "h", "v"],
}
scores = {}
for name, keys in variants.items():
    probs = np.mean([oof[key] for key in keys], axis=0)
    score, per_class, confusion = baseline.macro_f1(actual, probs.argmax(1), len(classes))
    scores[name] = {"oof_macro_f1": score, "per_class_f1": dict(zip(classes, per_class)),
                    "confusion_matrix_rows_true": confusion}
    print(f"{name}: OOF macro F1={score:.4f}", flush=True)

best = max(variants, key=lambda name: scores[name]["oof_macro_f1"])
if best != "original":
    submission = ROOT / "outputs" / f"fivefold_tta_{best}.csv"
    test_prob = np.mean([test_sum[key] / 5 for key in variants[best]], axis=0)
    with submission.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["ID", "Category"])
        writer.writerows((image_id, classes[int(pred)])
                         for image_id, pred in zip(test_ids, test_prob.argmax(1)))
    print(f"best={best}; submission={submission}", flush=True)
(ROOT / "outputs" / "tta_results.json").write_text(
    json.dumps({"scores": scores, "selected_variant": best}, ensure_ascii=False, indent=2),
    encoding="utf-8")
