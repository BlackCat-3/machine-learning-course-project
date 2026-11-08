"""Compare architecture diversity on the same held-out plant images."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18, resnet50

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", ROOT / "outputs" / "plant_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
torch.set_num_threads(8)
classes, records, _, _ = baseline.read_layout(Path(r"D:\Downloads\2026task1.zip"), None)
_, val_records = baseline.split_stratified(records, len(classes), 0.2, 42)
mean, std = baseline.compute_train_rgb_stats(Path(r"D:\Downloads\2026task1.zip"),
    baseline.split_stratified(records, len(classes), 0.2, 42)[0])
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)
])
dataset = baseline.ZipImages(Path(r"D:\Downloads\2026task1.zip"), val_records, transform)
loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
images = [batch for batch, _ in loader]
dataset.close()
actual = np.array([label for _, label in val_records])

models = {
    "resnet18": (ROOT / "outputs" / "plant_run_train_norm" / "best_model.pt", resnet18),
    "resnet50": (ROOT / "work" / "resnet50_base" / "best_model.pt", resnet50),
    "efficientnet_b0": (ROOT / "work" / "efficientnet_b0_base" / "best_model.pt", efficientnet_b0),
}
probs = {}
for name, (path, constructor) in models.items():
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = constructor(weights=None)
    if name == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    else:
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        probs[name] = np.concatenate([
            torch.softmax(model(batch), dim=1).numpy() for batch in images
        ])
    print("inferred", name, flush=True)


def report(name, values):
    score, per_class, _ = baseline.macro_f1(actual, values.argmax(axis=1), len(classes))
    print(name, score, per_class)
    return score


scores = {name: report(name, values) for name, values in probs.items()}
scores["resnet18_resnet50_mean"] = report("resnet18_resnet50_mean",
    (probs["resnet18"] + probs["resnet50"]) / 2)
scores["resnet18_2_resnet50_1"] = report("resnet18_2_resnet50_1",
    (2 * probs["resnet18"] + probs["resnet50"]) / 3)
scores["three_arch_mean"] = report("three_arch_mean", np.mean(list(probs.values()), axis=0))
(ROOT / "work" / "arch_ensemble_validation.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
