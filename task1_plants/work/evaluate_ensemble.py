"""Evaluate existing plant checkpoints on the shared validation set."""

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
OUTPUTS = ROOT / "outputs"
ZIP = Path(r"D:\Downloads\2026task1.zip")

spec = importlib.util.spec_from_file_location("plant_baseline", OUTPUTS / "plant_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

torch.set_num_threads(8)
classes, records, _, _ = baseline.read_layout(ZIP, None)
_, val_records = baseline.split_stratified(records, len(classes), 0.2, 42)
labels = np.array([label for _, label in val_records])

runs = {
    "seed42": OUTPUTS / "plant_run_train_norm",
    "seed21": OUTPUTS / "plant_fixed_seed_21",
    "seed2026": OUTPUTS / "plant_fixed_seed_2026",
    "seed7": OUTPUTS / "plant_fixed_seed_7",
    "seed123": OUTPUTS / "plant_fixed_seed_123",
    "imagenet": OUTPUTS / "plant_run_imagenet_norm",
}


def validation_inputs(mean, std):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = baseline.ZipImages(ZIP, val_records, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    batches = [images for images, _ in loader]
    dataset.close()
    return batches


inputs_by_norm = {}
probs = {}
probs_tta = {}
for name, folder in runs.items():
    details = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
    norm_key = details["normalization"]
    if norm_key not in inputs_by_norm:
        inputs_by_norm[norm_key] = validation_inputs(details["rgb_mean"], details["rgb_std"])
    checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions = []
    flip_predictions = []
    with torch.inference_mode():
        for images in inputs_by_norm[norm_key]:
            logits = model(images)
            flip_logits = model(torch.flip(images, dims=[3]))
            predictions.append(torch.softmax(logits, dim=1).numpy())
            flip_predictions.append(torch.softmax(flip_logits, dim=1).numpy())
    probs[name] = np.concatenate(predictions)
    probs_tta[name] = (probs[name] + np.concatenate(flip_predictions)) / 2
    print("loaded", name, flush=True)


def report(name, probabilities):
    predicted = probabilities.argmax(axis=1)
    f1, per_class, confusion = baseline.macro_f1(labels, predicted, len(classes))
    print(f"{name:30s} F1={f1:.6f}  class_f1={[round(v, 3) for v in per_class]}", flush=True)
    return f1


results = {}
for name in runs:
    results[name] = report(name, probs[name])
    results[name + "_tta"] = report(name + "_tta", probs_tta[name])

groups = {
    "five_seed_mean": ["seed42", "seed21", "seed2026", "seed7", "seed123"],
    "top3_seed_mean": ["seed42", "seed21", "seed2026"],
    "seed42_21_mean": ["seed42", "seed21"],
    "seed42_imagenet_mean": ["seed42", "imagenet"],
    "all_six_mean": list(runs),
}
for name, members in groups.items():
    results[name] = report(name, np.mean([probs[m] for m in members], axis=0))
    results[name + "_tta"] = report(name + "_tta", np.mean([probs_tta[m] for m in members], axis=0))

print("BEST", sorted(results.items(), key=lambda item: item[1], reverse=True)[:8])
(ROOT / "work" / "ensemble_validation.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
