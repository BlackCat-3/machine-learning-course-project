"""Train final ResNet18 candidate with all 500 labeled images."""

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
OUT = ROOT / "work" / "full500_resnet18"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("baseline", ROOT / "outputs" / "plant_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

seed = 42
epochs = 7
warmup_epochs = 2
batch_size = 32
torch.set_num_threads(8)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes, train_records, test_records, test_ids = baseline.read_layout(ZIP, None)
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
test_set = baseline.ZipImages(ZIP, test_records, eval_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

torch.hub.set_dir(str((ROOT / "work" / "model_cache").resolve()))
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = None
history = []
for epoch in range(epochs):
    if epoch == 0:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    if epoch == warmup_epochs:
        for parameter in model.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, accuracy = baseline.train_epoch(model, train_loader, optimizer, criterion, device)
    history.append({"epoch": epoch + 1, "train_loss": loss, "train_accuracy": accuracy})
    print(f"epoch {epoch+1}/{epochs} train_loss={loss:.4f} train_accuracy={accuracy:.4f}", flush=True)
    if epoch + 1 == 6:
        epoch_six_state = copy.deepcopy({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()})
        torch.save({"state_dict": epoch_six_state, "classes": classes, "epoch": 6},
                   OUT / "model_epoch6.pt")
        _, epoch_six_predictions = baseline.predict(model, test_loader, device)
        with (OUT / "submission_epoch6.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Category"])
            writer.writerows((image_id, classes[label])
                            for image_id, label in zip(test_ids, epoch_six_predictions))

state = copy.deepcopy({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()})
torch.save({"state_dict": state, "classes": classes, "epoch": epochs}, OUT / "model.pt")
_, predictions = baseline.predict(model, test_loader, device)
with (OUT / "submission.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Category"])
    writer.writerows((image_id, classes[label]) for image_id, label in zip(test_ids, predictions))
(OUT / "training.json").write_text(json.dumps({"train_count": len(train_records),
    "mean": mean, "std": std, "seed": seed, "epochs": epochs,
    "warmup_epochs": warmup_epochs, "history": history}, indent=2), encoding="utf-8")
train_set.close(); test_set.close()
print("saved", OUT / "submission.csv", flush=True)
