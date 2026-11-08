"""Verify crop geometry against visible synthetic image landmarks."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refinement_study import extract_patches, make_theta, patch_to_retina


class RefinementGeometryTests(unittest.TestCase):
    def test_raster_landmark_returns_to_retina_after_rotation_and_flip(self):
        torch.set_num_threads(2)
        size = 1024
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        signal = torch.exp(-((xx - 451).float().square() + (yy - 534).float().square()) / 50) * 255
        image = signal.round().to(torch.uint8)[None, None].expand(1, 3, size, size).clone()
        centers = torch.tensor([[0.42, 0.50]]).repeat(4, 1)
        theta = make_theta(centers, torch.tensor([0.34, 0.42, 0.34, 0.42]),
                           torch.tensor([0., 0.2, -0.3, 0.1]), torch.tensor([1., -1., 1., -1.]))
        patches = extract_patches(image, [0, 0, 0, 0], theta).mean(1)
        py, px = torch.meshgrid((torch.arange(256) + 0.5) / 256,
                               (torch.arange(256) + 0.5) / 256, indexing='ij')
        mass = patches.sum((1, 2))
        local = torch.stack(((patches * px).sum((1, 2)) / mass,
                             (patches * py).sum((1, 2)) / mass), 1)
        recovered = patch_to_retina(local, theta)
        truth = torch.tensor([(451 + 0.5) / size, (534 + 0.5) / size])
        self.assertLess(float((recovered - truth).abs().max() * size), 0.3)

    def test_patch_center_is_supplied_prediction_center(self):
        centers = torch.tensor([[0.4, 0.55], [0.62, 0.49]])
        theta = make_theta(centers, torch.tensor([0.34, 0.42]), flips=torch.tensor([-1., 1.]))
        torch.testing.assert_close(patch_to_retina(torch.full((2, 2), 0.5), theta), centers)


if __name__ == '__main__':
    unittest.main()
