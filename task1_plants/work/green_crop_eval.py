"""Check whether a green-pixel crop improves existing model validation F1."""

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(r"D:\Downloads\2026task1.zip")
spec = importlib.util.spec_from_file_location("baseline", ROOT / "outputs" / "plant_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
torch.set_num_threads(8)
classes, records, _, _ = baseline.read_layout(ZIP, None)
train_records, val_records = baseline.split_stratified(records, len(classes), 0.2, 42)
mean, std = baseline.compute_train_rgb_stats(ZIP, train_records)
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)
])


def green_crop(image):
    pixels = np.asarray(image.convert("RGB")).astype(np.float32)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    mask = (green > red * 1.06) & (green > blue * 1.10) & (green > 38)
    ys, xs = np.where(mask)
    if len(xs) < 25:
        return image
    height, width = pixels.shape[:2]
    x0, x1 = np.quantile(xs, [0.01, 0.99])
    y0, y1 = np.quantile(ys, [0.01, 0.99])
    crop_width = min(width, max((x1 - x0) * 1.4, width * 0.5))
    crop_height = min(height, max((y1 - y0) * 1.4, height * 0.5))
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    left = max(0, min(width - crop_width, center_x - crop_width / 2))
    top = max(0, min(height - crop_height, center_y - crop_height / 2))
    return image.crop((round(left), round(top), round(left + crop_width), round(top + crop_height)))


checkpoint = torch.load(ROOT / "outputs" / "plant_run_train_norm" / "best_model.pt",
    map_location="cpu", weights_only=True)
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
full_probs, crop_probs = [], []
actual = []
with zipfile.ZipFile(ZIP) as archive, torch.inference_mode():
    for path, label in val_records:
        with Image.open(io.BytesIO(archive.read(path))) as opened:
            image = opened.convert("RGB")
        full = transform(image).unsqueeze(0)
        crop = transform(green_crop(image)).unsqueeze(0)
        full_probs.append(torch.softmax(model(full), dim=1).numpy()[0])
        crop_probs.append(torch.softmax(model(crop), dim=1).numpy()[0])
        actual.append(label)
full_probs, crop_probs = np.array(full_probs), np.array(crop_probs)
for name, values in [
    ("full", full_probs),
    ("green_crop", crop_probs),
    ("full_crop_mean", (full_probs + crop_probs) / 2),
    ("full_2_crop_1", (2 * full_probs + crop_probs) / 3),
]:
    score, per_class, _ = baseline.macro_f1(actual, values.argmax(axis=1), len(classes))
    print(name, score, per_class)
