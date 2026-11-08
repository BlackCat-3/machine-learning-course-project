"""Global fovea coordinate regression with a spatial ResNet18 representation.

Input: ImageNet-normalized RGB tensors (B, 3, H, W).
Output: normalized (x, y) coordinates (B, 2), constrained to [0, 1].
Pretrained weights are loaded only from an explicit local checkpoint path.
"""

from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


class FoveaRegressor(nn.Module):
    """ResNet18 features pooled to 4x4 cells, then global coordinate regression.

    BatchNorm running statistics are fixed even in training mode. All encoder
    parameters, including BatchNorm affine parameters, remain trainable.
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
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
        # Begin close to the image center without suppressing encoder gradients.
        nn.init.normal_(self.head[-2].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.head[-2].bias)
        self._freeze_encoder_batchnorm()

    def _freeze_encoder_batchnorm(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True) -> "FoveaRegressor":
        super().train(mode)
        self._freeze_encoder_batchnorm()
        return self

    def forward(self, x: Tensor) -> Tensor:
        encoder = self.encoder
        x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(x))))
        x = encoder.layer1(x)
        x = encoder.layer2(x)
        x = encoder.layer3(x)
        x = encoder.layer4(x)
        return self.head(x)


def _self_check() -> None:
    """CPU-only forward/backward and frozen-statistics checks on synthetic input."""
    torch.set_num_threads(2)
    torch.manual_seed(23)
    local_weights = (
        Path(__file__).resolve().parent.parent
        / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    )
    model = FoveaRegressor(local_weights if local_weights.is_file() else None).cpu().train()
    batch_norms = [
        module for module in model.encoder.modules() if isinstance(module, nn.BatchNorm2d)
    ]
    running_stats = [
        (module.running_mean.clone(), module.running_var.clone()) for module in batch_norms
    ]
    coordinates = model(torch.randn(2, 3, 128, 160))
    assert coordinates.shape == (2, 2), coordinates.shape
    assert torch.isfinite(coordinates).all()
    assert ((coordinates >= 0) & (coordinates <= 1)).all()
    targets = torch.tensor([[0.25, 0.75], [0.65, 0.40]])
    loss = nn.functional.mse_loss(coordinates, targets)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in (model.encoder.conv1.weight, model.encoder.bn1.weight, model.head[-2].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(not module.training for module in batch_norms)
    assert model.head[4].training
    for (mean_before, variance_before), module in zip(running_stats, batch_norms):
        assert torch.equal(mean_before, module.running_mean)
        assert torch.equal(variance_before, module.running_var)
    model.eval().train()
    assert all(not module.training for module in batch_norms)
    print(f"CPU self-check passed: output={tuple(coordinates.shape)}, loss={loss.item():.6f}")
    print(f"initial_coordinates={coordinates.detach().tolist()}")
    print(f"encoder_batchnorm_frozen={len(batch_norms)}, pretrained={local_weights.is_file()}")


if __name__ == "__main__":
    _self_check()
