"""Coordinate and zero-correction checks for native residual refinement."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from residual_refinement import (CACHE, PATCH, ResidualRefiner, extract_native,
                                 to_original, loss_original)


class ResidualGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_units_and_horizontal_flip(self):
        center = torch.tensor([[1300.2, 900.4], [700.3, 1000.1]])
        flip = torch.tensor([1., -1.])
        target = center + torch.tensor([[12., -27.], [12., -27.]])
        local = torch.tensor([[.5 + 12/PATCH, .5 - 27/PATCH],
                              [.5 - 12/PATCH, .5 - 27/PATCH]])
        torch.testing.assert_close(to_original(local, center, flip), target, atol=1e-4, rtol=0)

    def test_native_pixels_no_resize(self):
        # A continuous coordinate ramp detects half-pixel and scaling mistakes.
        xx = torch.arange(CACHE).float() + .5
        image = xx[None, None, None, :].expand(2, 3, CACHE, CACHE)
        origins = torch.tensor([[800., 400.], [800., 400.]])
        centers = origins + torch.tensor([[384.25, 384.], [384.25, 384.]])
        out = extract_native(image, centers, origins, torch.tensor([1., -1.])) * 255
        expected = 384.25 + torch.arange(PATCH).float() + .5 - PATCH/2
        torch.testing.assert_close(out[0, 0, PATCH//2], expected, atol=2e-4, rtol=0)
        torch.testing.assert_close(out[1, 0, PATCH//2], expected.flip(0), atol=2e-4, rtol=0)

    def test_zero_initialized_residual_and_trainable_head(self):
        torch.manual_seed(4)
        model = ResidualRefiner().train()
        logits, local = model(torch.randn(2, 3, 64, 64), torch.randn(2, 512, 2, 2),
                              torch.tensor([[.3, .6], [.7, .6]]))
        torch.testing.assert_close(local, torch.full((2, 2), .5), atol=1e-6, rtol=0)
        centers = torch.tensor([[1400., 1000.], [1500., 900.]])
        target = centers + torch.tensor([[12., -8.], [-10., 6.]])
        loss = loss_original(logits, local, centers, torch.tensor([1., -1.]), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(not m.training for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)))


if __name__ == '__main__':
    unittest.main()
