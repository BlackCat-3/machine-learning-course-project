"""Native-pixel residual refinement conditioned on frozen global features."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model import FoveaNet, heatmap_coordinates
from run import normalize

PATCH = 640
CACHE = 768


class ResidualRefiner(FoveaNet):
    def __init__(self, pretrained_path=None):
        super().__init__(pretrained_path)
        # Features come from the frozen coarse encoder, including spatial context.
        self.context = nn.Sequential(nn.Flatten(), nn.Linear(2048, 64),
                                     nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.15))
        self.condition = nn.Sequential(nn.Linear(66, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Conv2d(96, 64, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
                                  nn.Dropout2d(0.1), nn.Conv2d(64, 1, 1))
        # A symmetric prior and zero final layer initially give zero correction.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, context, center_xy):
        e = self.encoder
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        c1 = e.layer1(x)
        c2 = e.layer2(c1)
        c3 = e.layer3(c2)
        c4 = e.layer4(c3)
        features = (c1, c2, c3, c4)
        p = self.refine[3](self.lateral[3](c4))
        for level in (2, 1, 0):
            lateral = self.lateral[level](features[level])
            p = self.refine[level](lateral + F.interpolate(
                p, size=lateral.shape[-2:], mode='bilinear', align_corners=False))
        conditioning = self.condition(torch.cat((self.context(context), center_xy), 1))
        conditioning = conditioning[:, :, None, None].expand(-1, -1, *p.shape[-2:])
        evidence = self.head(torch.cat((p, conditioning), 1)).float()
        h, w = evidence.shape[-2:]
        gy = (torch.arange(h, device=x.device, dtype=torch.float32) + .5) / h - .5
        gx = (torch.arange(w, device=x.device, dtype=torch.float32) + .5) / w - .5
        # Prior standard deviation = 64 original pixels, independent of crop size.
        prior = -(gy[:, None].square() + gx[None, :].square()) / (2 * (64 / PATCH)**2)
        logits = evidence + prior[None, None]
        return logits, heatmap_coordinates(logits)


def extract_native(cache, centers, origins, flip):
    """Sample PATCH original pixels from unresized 768-pixel caches.

    Centers/origins are original-image edge coordinates. Pixel-center sampling
    and align_corners=False match PIL's integer crop geometry.
    """
    n = len(cache)
    theta = torch.zeros(n, 2, 3, device=centers.device, dtype=torch.float32)
    theta[:, 0, 0] = PATCH / CACHE * flip
    theta[:, 1, 1] = PATCH / CACHE
    theta[:, :, 2] = 2 * (centers - origins) / CACHE - 1
    grid = F.affine_grid(theta, (n, 3, PATCH, PATCH), align_corners=False)
    return F.grid_sample(cache.float() / 255, grid, align_corners=False,
                         mode='bilinear', padding_mode='zeros')


def to_original(local_xy, centers, flip):
    scale = torch.stack((flip, torch.ones_like(flip)), 1) * PATCH
    return centers + (local_xy.float() - .5) * scale


def loss_original(logits, local_xy, centers, flip, target_original):
    """Original-pixel coordinate MSE / 32^2 + 0.2 Gaussian heatmap CE."""
    prediction = to_original(local_xy, centers, flip)
    pixel_mse = F.mse_loss(prediction, target_original)
    h, w = logits.shape[-2:]
    scale = torch.stack((flip, torch.ones_like(flip)), 1) * PATCH
    target_local = .5 + (target_original - centers) / scale
    if not ((target_local > 0) & (target_local < 1)).all():
        raise ValueError('Training target outside native patch; no samples may be dropped')
    yy, xx = torch.meshgrid(torch.arange(h, device=logits.device),
                            torch.arange(w, device=logits.device), indexing='ij')
    tx = target_local[:, 0, None, None] * w - .5
    ty = target_local[:, 1, None, None] * h - .5
    target = (-((xx - tx).square() + (yy - ty).square()).flatten(1) / (2 * 2.**2)).softmax(1)
    ce = -(target * logits.float().flatten(1).log_softmax(1)).sum(1).mean()
    return pixel_mse / (32**2) + .2 * ce


def patch_batch(data, indices, training=False, flip_value=None):
    device = data['patches'].device
    centers = data['centers'][indices].clone()
    if training:
        # Small jitter supplements actual inner-OOF errors; it does not replace them.
        centers += torch.randn_like(centers).clamp(-2, 2) * 8
    if flip_value is None:
        flip = torch.where(torch.rand(len(indices), device=device) < .5, -1., 1.)
    else:
        flip = torch.full((len(indices),), float(flip_value), device=device)
    context = data['context'][indices, (flip < 0).long()].float()
    cxy = (centers - data['retina_origin'][indices]) / data['retina_size'][indices]
    cxy[:, 0] = torch.where(flip < 0, 1 - cxy[:, 0], cxy[:, 0])
    x = extract_native(data['patches'][indices], centers, data['origins'][indices], flip)
    if training:
        rand = lambda *shape: torch.rand(*shape, device=device)
        brightness = .85 + .30 * rand(len(indices), 1, 1, 1)
        contrast = .85 + .30 * rand(len(indices), 1, 1, 1)
        color = .9 + .2 * rand(len(indices), 3, 1, 1)
        gamma = .8 + .45 * rand(len(indices), 1, 1, 1)
        mean = x.mean((2, 3), keepdim=True)
        x = (((x - mean) * contrast + mean) * brightness * color).clamp(0, 1).pow(gamma)
    return normalize(x), context, cxy, centers, flip
