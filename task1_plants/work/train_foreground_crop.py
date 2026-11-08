"""Pilot: train ResNet18 with deterministic green-foreground cropping."""

import copy
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
OUT = ROOT / "work" / "foreground_crop_pilot"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("baseline", ROOT / "outputs" / "plant_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class ForegroundCrop:
    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        red = rgb[:, :, 0].astype(np.float32)
        green = rgb[:, :, 1].astype(np.float32)
        blue = rgb[:, :, 2].astype(np.float32)
        mask = (green > 1.06 * red) & (green > 1.06 * blue) & (green > 35)
        if mask.sum() < 0.001 * height * width:
            return image
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        side = int(min(min(width, height), max(0.55 * min(width, height),
                                             1.25 * max(x1 - x0, y1 - y0))))
        left = int(np.clip((x0 + x1 - side) / 2, 0, width - side))
        top = int(np.clip((y0 + y1 - side) / 2, 0, height - side))
        return image.crop((left, top, left + side, top + side))


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes, records, _, _ = baseline.read_layout(ZIP, None)
train_records, val_records = baseline.split_stratified(records, len(classes), 0.2, seed)
mean, std = baseline.compute_train_rgb_stats(ZIP, train_records)
norm = transforms.Normalize(mean, std)
crop = ForegroundCrop()
train_transform = transforms.Compose([
    crop, transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(), norm,
])
val_transform = transforms.Compose([crop, transforms.Resize((224, 224)),
                                    transforms.ToTensor(), norm])
train_set = baseline.ZipImages(ZIP, train_records, train_transform)
val_set = baseline.ZipImages(ZIP, val_records, val_transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=16, num_workers=0)
torch.hub.set_dir(str((ROOT / "work" / "model_cache").resolve()))
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = None
best_f1, best_epoch, best_state, best_details = -1, None, None, None
history = []
for epoch in range(10):
    if epoch == 0:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    if epoch == 2:
        for parameter in model.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, acc = baseline.train_epoch(model, train_loader, optimizer, criterion, device)
    actual, pred = baseline.predict(model, val_loader, device)
    score, class_f1, confusion = baseline.macro_f1(actual, pred, len(classes))
    history.append({"epoch": epoch + 1, "loss": loss, "train_accuracy": acc,
                    "val_macro_f1": score})
    print(f"epoch={epoch+1} loss={loss:.4f} train_acc={acc:.3f} val_f1={score:.4f}",
          flush=True)
    if score > best_f1:
        best_f1, best_epoch = score, epoch + 1
        best_state = copy.deepcopy({key: value.detach().cpu()
                                    for key, value in model.state_dict().items()})
        best_details = {"per_class_f1": dict(zip(classes, class_f1)),
                        "confusion_matrix_rows_true": confusion}
torch.save({"state_dict": best_state, "classes": classes, "epoch": best_epoch},
           OUT / "best_model.pt")
(OUT / "validation.json").write_text(json.dumps({"best_epoch": best_epoch,
    "best_val_macro_f1": best_f1, "history": history, **best_details}, indent=2),
    encoding="utf-8")
train_set.close(); val_set.close()
print(f"best_epoch={best_epoch} best_val_macro_f1={best_f1:.4f}", flush=True)
