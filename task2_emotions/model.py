"""Model and shared image preprocessing for the six-class FER assignment."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50,
                                EfficientNet_B0_Weights, efficientnet_b0)


CLASS_TO_INDEX = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}
CLASS_NAMES = list(CLASS_TO_INDEX)
NUM_CLASSES = len(CLASS_NAMES)
SUPPORTED_ARCHITECTURES = ("resnet18", "resnet34", "efficientnet_b0", "resnet50")


def build_model(*, pretrained: bool = True, dropout: float = 0.2,
                architecture: str = "resnet18", no_maxpool: bool = False) -> nn.Module:
    """Use official ImageNet weights; always replace the ImageNet classifier."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if architecture == "efficientnet_b0" and no_maxpool:
        raise ValueError("no_maxpool is a ResNet-only option; EfficientNet-B0 has no initial max-pooling layer")
    torch.hub.set_dir(str(Path(__file__).resolve().parent / ".cache" / "torch" / "hub"))
    if architecture == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, NUM_CLASSES))
        return model
    if architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif architecture == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    # Retain finer spatial detail for small face images without changing any
    # pretrained parameter shapes. The original ResNet stem remains the default.
    if no_maxpool:
        model.maxpool = nn.Identity()
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, NUM_CLASSES))
    return model


def classification_head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only classifier parameters without aliasing/changing state keys."""
    head = getattr(model, "fc", None)
    if head is None:
        head = getattr(model, "classifier", None)
    if not isinstance(head, nn.Module):
        raise ValueError("Model does not expose a supported fc/classifier head")
    return list(head.parameters())


class BatchPreprocessor(nn.Module):
    """Per-image augmentation on the training device, with no PIL/worker overhead.

    Input is a uint8 tensor [B, 1, 48, 48]. Each image receives independent
    random parameters. Validation/test never use random transforms.
    """

    def __init__(self, image_size: int = 112, *, erasing_prob: float = 0.0,
                 erasing_scale_min: float = 0.02, erasing_scale_max: float = 0.08) -> None:
        super().__init__()
        if not 0 <= erasing_prob <= 1:
            raise ValueError("erasing_prob must be in [0, 1]")
        if not 0 < erasing_scale_min <= erasing_scale_max <= 1:
            raise ValueError("erasing scales must satisfy 0 < min <= max <= 1")
        self.image_size = image_size
        self.erasing_prob = erasing_prob
        self.erasing_scale_min = erasing_scale_min
        self.erasing_scale_max = erasing_scale_max
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _erase_images(self, images: torch.Tensor) -> torch.Tensor:
        """Erase independent rectangles with each augmented grayscale image's mean.

        Sampling and masking stay on the input device, with no per-image Python
        loops or device synchronization. Integer rounding slightly changes the
        sampled area/aspect ratio. Rectangles that do not fit are skipped.
        """
        batch_size, _, height, width = images.shape
        draws = torch.rand((batch_size, 5), device=images.device)
        areas = (self.erasing_scale_min + draws[:, 1] *
                 (self.erasing_scale_max - self.erasing_scale_min)) * (height * width)
        # Log-uniform height/width ratio in [0.5, 2.0].
        ratios = (math.log(0.5) + draws[:, 2] * math.log(4.0)).exp()
        erase_heights = (areas * ratios).sqrt().round().long().clamp_min(1)
        erase_widths = (areas / ratios).sqrt().round().long().clamp_min(1)
        apply = ((draws[:, 0] < self.erasing_prob) &
                 (erase_heights <= height) & (erase_widths <= width))
        tops = (draws[:, 3] * (height - erase_heights + 1).clamp_min(1)).floor().long()
        lefts = (draws[:, 4] * (width - erase_widths + 1).clamp_min(1)).floor().long()
        rows = torch.arange(height, device=images.device).view(1, height, 1)
        columns = torch.arange(width, device=images.device).view(1, 1, width)
        mask = (apply[:, None, None] &
                (rows >= tops[:, None, None]) &
                (rows < (tops + erase_heights)[:, None, None]) &
                (columns >= lefts[:, None, None]) &
                (columns < (lefts + erase_widths)[:, None, None]))
        image_means = images.mean(dim=(2, 3), keepdim=True)
        return torch.where(mask[:, None], image_means, images)

    def forward(self, images: torch.Tensor, *, augment: bool = False) -> torch.Tensor:
        images = images.float().div_(255.0)
        if augment:
            batch_size = len(images)
            device = images.device
            angles = (torch.rand(batch_size, device=device) * 2 - 1) * (8 * math.pi / 180)
            scales = torch.empty(batch_size, device=device).uniform_(0.90, 1.08)
            flips = torch.where(torch.rand(batch_size, device=device) < 0.5, -1.0, 1.0)
            cosine, sine = angles.cos() * scales, angles.sin() * scales
            theta = torch.zeros((batch_size, 2, 3), device=device)
            theta[:, 0, 0] = cosine * flips
            theta[:, 0, 1] = -sine
            theta[:, 1, 0] = sine * flips
            theta[:, 1, 1] = cosine
            # Normalized coordinates range [-1,1], so 0.12 means 6% of an image.
            theta[:, :, 2].uniform_(-0.12, 0.12)
            grid = F.affine_grid(theta, (batch_size, 1, self.image_size, self.image_size), align_corners=False)
            images = F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
            brightness = torch.empty((batch_size, 1, 1, 1), device=device).uniform_(0.9, 1.1)
            contrast = torch.empty((batch_size, 1, 1, 1), device=device).uniform_(0.9, 1.1)
            image_means = images.mean(dim=(2, 3), keepdim=True)
            images = ((images - image_means) * contrast + image_means) * brightness
            images = images.clamp_(0, 1)
            # Disabled by default; this branch consumes no new RNG draws at p=0,
            # preserving legacy training and all validation/test preprocessing.
            if self.erasing_prob > 0:
                images = self._erase_images(images)
        else:
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return ((images.expand(-1, 3, -1, -1) - self.mean) / self.std).contiguous(memory_format=torch.channels_last)


def classification_metrics(labels, predictions) -> dict:
    """Rows are true classes; columns are predicted classes."""
    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    matrix = np.bincount(labels * NUM_CLASSES + predictions, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    support = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(NUM_CLASSES, dtype=float), where=support > 0)
    return {
        "accuracy": float((labels == predictions).mean()) if len(labels) else None,
        "num_samples": int(len(labels)),
        "class_to_index": CLASS_TO_INDEX,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {"rows": "true", "columns": "predicted"},
        "per_class": {name: {"support": int(support[i]), "recall": float(recall[i])} for i, name in enumerate(CLASS_NAMES)},
    }
