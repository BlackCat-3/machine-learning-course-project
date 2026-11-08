"""Check that visible raster landmarks follow the same augmentation as labels."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import original_coordinates
from run import augment


class GeometryTests(unittest.TestCase):
    def test_augmented_raster_landmark_agrees_with_target(self):
        torch.set_num_threads(2)
        torch.manual_seed(912)
        size = 128
        x = torch.zeros(32, 3, size, size)
        # Symmetric disk centered at a pixel center, away from all boundaries.
        center_x, center_y = 51, 72
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        disk = ((xx - center_x).square() + (yy - center_y).square() <= 9).float()
        x[:] = disk
        target = torch.tensor([[(center_x + 0.5) / size, (center_y + 0.5) / size]]).repeat(len(x), 1)
        transformed, points = augment(x, target)
        # Remove the uniform photometric offset before computing the landmark centroid.
        mass = transformed.mean(1)
        mass = (mass - mass.amin((1, 2), keepdim=True)).clamp_min(0)
        cx = (mass * (xx + 0.5)).sum((1, 2)) / mass.sum((1, 2)) / size
        cy = (mass * (yy + 0.5)).sum((1, 2)) / mass.sum((1, 2)) / size
        error_pixels = (torch.stack((cx, cy), 1) - points).abs() * size
        self.assertLess(float(error_pixels.max()), 0.6)

    def test_original_pixel_frame_for_different_crop_shapes(self):
        records = [{'crop': [20, 30, 220, 130]}, {'crop': [400, 100, 2400, 1900]}]
        actual = original_coordinates(np.array([[0.25, 0.75], [0.4, 0.5]]), records)
        np.testing.assert_allclose(actual, [[70, 105], [1200, 1000]])


if __name__ == '__main__':
    unittest.main()
