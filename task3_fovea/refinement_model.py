"""Two-head fovea refinement within automatically selected local image crops.

Inputs are ImageNet-normalized RGB patches, usually 256 or 320 pixels square.
Both coordinate heads return normalized (x, y) positions within the patch.
This module only defines a model and loss; crop selection belongs to the data
pipeline, where validation crops must use out-of-fold coarse predictions.
"""

from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if __package__:
    from .model import FoveaNet, heatmap_coordinates, heatmap_loss
else:
    from model import FoveaNet, heatmap_coordinates, heatmap_loss


class LocalRefiner(FoveaNet):
    """Shared ResNet18/FPN heatmap plus a spatially pooled regression head.

    Inherited train() freezes encoder BatchNorm running statistics while keeping
    all encoder parameters trainable. An explicit local checkpoint initializes
    the encoder; None uses random initialization and never downloads weights.
    """

    def __init__(self, pretrained_path: str | Path | None = None) -> None:
        super().__init__(pretrained_path)
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.reg_head[-2].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.reg_head[-2].bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
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
        logits = self.head(pyramid)
        return {
            "logits": logits,
            "xy": heatmap_coordinates(logits),
            "reg_xy": self.reg_head(c4),
        }


def refinement_loss(output: dict[str, Tensor], target_xy: Tensor) -> Tensor:
    """Gaussian CE (sigma=1.5) + 25*heatmap MSE + 50*regression MSE.

    target_xy contains normalized patch coordinates, shaped (B, 2). MSE is
    averaged over both coordinates and the batch. For inference, callers may
    use either coordinate head or their fixed arithmetic average.
    """
    logits = output["logits"]
    target = target_xy.to(device=logits.device, dtype=torch.float32)
    expected_shape = (logits.shape[0], 2)
    for key in ("xy", "reg_xy"):
        if output[key].shape != expected_shape:
            raise ValueError(f"Expected {key} shape {expected_shape}, got {tuple(output[key].shape)}")
    # The shared heatmap loss already includes 5 * coordinate MSE.
    gaussian_and_coordinate = heatmap_loss(logits, target, sigma=1.5)
    heatmap_mse = F.mse_loss(output["xy"].float(), target)
    regression_mse = F.mse_loss(output["reg_xy"].float(), target)
    return gaussian_and_coordinate + 20.0 * heatmap_mse + 50.0 * regression_mse


def _self_check() -> None:
    """Check shapes, gradients, loss weighting, and frozen BN using CPU input."""
    torch.set_num_threads(2)
    torch.manual_seed(23)
    local_weights = (
        Path(__file__).resolve().parent.parent
        / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    )
    model = LocalRefiner(local_weights if local_weights.is_file() else None).cpu().train()
    batch_norms = [
        module for module in model.encoder.modules() if isinstance(module, nn.BatchNorm2d)
    ]
    running_stats = [
        (module.running_mean.clone(), module.running_var.clone()) for module in batch_norms
    ]
    output = model(torch.randn(2, 3, 256, 256))
    assert output["logits"].shape == (2, 1, 64, 64)
    for key in ("xy", "reg_xy"):
        assert output[key].shape == (2, 2)
        assert torch.isfinite(output[key]).all()
        assert ((output[key] >= 0) & (output[key] <= 1)).all()
    target = torch.tensor([[0.25, 0.75], [0.65, 0.40]])
    loss = refinement_loss(output, target)
    assert torch.isfinite(loss)
    # Independently reconstruct the stated loss to catch coefficient mistakes.
    grid_y, grid_x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    squared_distance = (
        (grid_x - (target[:, 0, None, None] * 64 - 0.5)).square()
        + (grid_y - (target[:, 1, None, None] * 64 - 0.5)).square()
    )
    gaussian = (-squared_distance.flatten(1) / (2 * 1.5**2)).softmax(dim=1)
    cross_entropy = -(gaussian * output["logits"].flatten(1).log_softmax(dim=1)).sum(dim=1).mean()
    expected_loss = (
        cross_entropy
        + 25 * F.mse_loss(output["xy"], target)
        + 50 * F.mse_loss(output["reg_xy"], target)
    )
    assert torch.allclose(loss, expected_loss)
    loss.backward()
    for parameter in (
        model.encoder.conv1.weight,
        model.encoder.bn1.weight,
        model.head[-1].weight,
        model.reg_head[-2].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(not module.training for module in batch_norms)
    for (mean_before, variance_before), module in zip(running_stats, batch_norms):
        assert torch.equal(mean_before, module.running_mean)
        assert torch.equal(variance_before, module.running_var)
    model.eval()
    with torch.no_grad():
        larger_output = model(torch.randn(1, 3, 320, 320))
    assert larger_output["logits"].shape == (1, 1, 80, 80)
    model.train()
    assert all(not module.training for module in batch_norms)
    print(f"CPU self-check passed: 256 -> 64x64, 320 -> 80x80; loss={loss.item():.6f}")
    print(f"xy={output['xy'].detach().tolist()}, reg_xy={output['reg_xy'].detach().tolist()}")
    print(f"encoder_batchnorm_frozen={len(batch_norms)}, pretrained={local_weights.is_file()}")


if __name__ == "__main__":
    _self_check()
