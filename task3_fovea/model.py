"""ResNet18 feature-pyramid heatmaps for fovea localization.

Inputs are RGB tensors normalized with the ImageNet mean and standard deviation.
Coordinates are normalized (x, y); heatmap pixels represent cell centers, so
pixel (column, row) has coordinate ((column + .5) / width, (row + .5) / height).
No weights are downloaded: pass an existing local ResNet18 checkpoint to opt
into ImageNet initialization, or pass None for random initialization.
"""

from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _refine(channels: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, channels),
        nn.ReLU(inplace=True),
    )


class FoveaNet(nn.Module):
    """Trainable ResNet18 encoder and a 64-channel top-down FPN.

    The output has stride four for inputs with dimensions divisible by four.
    Encoder BatchNorm uses fixed running statistics, including after train().
    Its affine parameters and all convolution parameters remain trainable.
    """

    def __init__(self, pretrained_path: str | Path | None = None) -> None:
        super().__init__()
        self.encoder = resnet18(weights=None)
        if pretrained_path is not None:
            checkpoint_path = Path(pretrained_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Local ResNet18 weights not found: {checkpoint_path}")
            weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(weights, strict=True)
        # The classifier is not used for spatial prediction.
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.lateral = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in (64, 128, 256, 512))
        self.refine = nn.ModuleList(_refine() for _ in range(4))
        self.head = nn.Sequential(_refine(), nn.Conv2d(64, 1, 1))
        self._freeze_encoder_batchnorm()

    def _freeze_encoder_batchnorm(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True) -> "FoveaNet":
        super().train(mode)
        self._freeze_encoder_batchnorm()
        return self

    def forward(self, x: Tensor) -> Tensor:
        encoder = self.encoder
        x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(x))))
        c1 = encoder.layer1(x)
        c2 = encoder.layer2(c1)
        c3 = encoder.layer3(c2)
        c4 = encoder.layer4(c3)
        features = (c1, c2, c3, c4)
        pyramid = self.refine[3](self.lateral[3](c4))
        for level in (2, 1, 0):
            lateral = self.lateral[level](features[level])
            pyramid = F.interpolate(
                pyramid, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            pyramid = self.refine[level](lateral + pyramid)
        return self.head(pyramid)


def _flat_logits(logits: Tensor) -> tuple[Tensor, int, int]:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"Expected heatmap logits (B, 1, H, W), got {tuple(logits.shape)}")
    height, width = logits.shape[-2:]
    if height < 1 or width < 1:
        raise ValueError("Heatmap dimensions must be positive")
    # Keep spatial softmax, expected coordinates, and the loss stable under AMP.
    return logits.float().flatten(1), height, width


def heatmap_coordinates(logits: Tensor) -> Tensor:
    """Return differentiable softmax expected (x, y) coordinates, shape (B, 2)."""
    flat, height, width = _flat_logits(logits)
    probability = flat.softmax(dim=1).reshape(-1, height, width)
    grid_x = (torch.arange(width, device=logits.device, dtype=flat.dtype) + 0.5) / width
    grid_y = (torch.arange(height, device=logits.device, dtype=flat.dtype) + 0.5) / height
    x = (probability.sum(dim=1) * grid_x).sum(dim=1)
    y = (probability.sum(dim=2) * grid_y).sum(dim=1)
    return torch.stack((x, y), dim=1)


def heatmap_loss(logits: Tensor, target_xy: Tensor, sigma: float = 1.5) -> Tensor:
    """Gaussian target cross entropy plus five times normalized coordinate MSE.

    sigma is measured in heatmap cells. target_xy must contain one normalized
    (x, y) position per image, in the same order as the model outputs.
    """
    flat, height, width = _flat_logits(logits)
    if target_xy.shape != (logits.shape[0], 2):
        raise ValueError(f"Expected target coordinates (B, 2), got {tuple(target_xy.shape)}")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    targets = target_xy.to(device=logits.device, dtype=flat.dtype)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=logits.device, dtype=flat.dtype),
        torch.arange(width, device=logits.device, dtype=flat.dtype),
        indexing="ij",
    )
    target_x = targets[:, 0, None, None] * width - 0.5
    target_y = targets[:, 1, None, None] * height - 0.5
    distance_squared = (grid_x - target_x).square() + (grid_y - target_y).square()
    gaussian = (-distance_squared.flatten(1) / (2.0 * sigma**2)).softmax(dim=1)
    cross_entropy = -(gaussian * flat.log_softmax(dim=1)).sum(dim=1).mean()
    coordinate_mse = F.mse_loss(heatmap_coordinates(logits), targets)
    return cross_entropy + 5.0 * coordinate_mse


def _self_check() -> None:
    """Small CPU-only shape, gradient, coordinate, and frozen-statistics check."""
    torch.set_num_threads(2)
    torch.manual_seed(23)
    local_weights = (
        Path(__file__).resolve().parent.parent
        / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    )
    model = FoveaNet(local_weights if local_weights.is_file() else None).cpu().train()
    batch_norms = [
        module for module in model.encoder.modules() if isinstance(module, nn.BatchNorm2d)
    ]
    running_means = [module.running_mean.clone() for module in batch_norms]
    output = model(torch.randn(2, 3, 64, 96))
    assert output.shape == (2, 1, 16, 24), output.shape
    loss = heatmap_loss(output, torch.tensor([[0.25, 0.75], [0.65, 0.40]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.conv1.weight.grad is not None
    assert torch.isfinite(model.encoder.conv1.weight.grad).all()
    assert model.encoder.bn1.weight.grad is not None
    assert all(not module.training for module in batch_norms)
    assert all(torch.equal(before, module.running_mean) for before, module in zip(running_means, batch_norms))
    centered = heatmap_coordinates(torch.zeros(1, 1, 2, 3))
    assert torch.allclose(centered, torch.tensor([[0.5, 0.5]]))
    peaked = torch.full((1, 1, 2, 3), -80.0)
    peaked[0, 0, 1, 2] = 80.0
    corner_cell = heatmap_coordinates(peaked)
    assert torch.allclose(corner_cell, torch.tensor([[2.5 / 3.0, 1.5 / 2.0]]))
    print(f"CPU self-check passed: output={tuple(output.shape)}, loss={loss.item():.6f}")
    print(f"uniform={centered.tolist()}, peak_cell={corner_cell.tolist()}")
    print(f"encoder_batchnorm_frozen={len(batch_norms)}, pretrained={local_weights.is_file()}")


if __name__ == "__main__":
    _self_check()
