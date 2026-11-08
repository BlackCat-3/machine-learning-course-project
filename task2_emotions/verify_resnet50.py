"""CPU regression for ResNet50 V2 support and existing architecture behavior."""
import unittest

import torch
from torch import nn
from torchvision.models import efficientnet_b0, resnet50, ResNet50_Weights

from model import build_model, classification_head_parameters
from verify_efficientnet import ArchitectureTests


class ResNet50Tests(unittest.TestCase):
    def assert_states_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_efficientnet_builder_state_and_rng_unchanged(self):
        torch.manual_seed(44)
        old = efficientnet_b0(weights=None)
        old.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(old.classifier[1].in_features, 6))
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(44)
        new = build_model(architecture="efficientnet_b0", pretrained=False, dropout=0.3)
        self.assert_states_equal(old.state_dict(), new.state_dict())
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_resnet50_roundtrip_head_and_pooling(self):
        model = build_model(architecture="resnet50", pretrained=False, dropout=0.3).eval()
        self.assertIsInstance(model.maxpool, nn.MaxPool2d)
        self.assertEqual(model.fc[1].in_features, 2048)
        self.assertEqual(model.fc[1].out_features, 6)
        head = classification_head_parameters(model)
        ids = {id(p) for p in head}
        self.assertEqual(ids, {id(p) for p in model.fc.parameters()})
        self.assertEqual(len(head), 2)
        backbone = [p for p in model.parameters() if id(p) not in ids]
        self.assertEqual({id(p) for p in backbone} & ids, set())
        self.assertEqual({id(p) for p in backbone} | ids, {id(p) for p in model.parameters()})
        restored = build_model(architecture="resnet50", pretrained=False, dropout=0.3).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        images = torch.randn(1, 3, 160, 160)
        with torch.inference_mode():
            first, second = model(images), restored(images)
        self.assertEqual(tuple(first.shape), (1, 6))
        self.assertTrue(torch.isfinite(first).all().item())
        self.assertTrue(torch.equal(first, second))
        without_pool = build_model(architecture="resnet50", pretrained=False, no_maxpool=True)
        self.assertIsInstance(without_pool.maxpool, nn.Identity)
        without_pool.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(ResNet50_Weights.DEFAULT, ResNet50_Weights.IMAGENET1K_V2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
