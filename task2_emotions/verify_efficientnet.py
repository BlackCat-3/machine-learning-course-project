"""CPU regressions for EfficientNet support and unchanged legacy ResNets."""
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import resnet18, resnet34

from model import build_model, classification_head_parameters
from train import parse_args


class ArchitectureTests(unittest.TestCase):
    def assert_states_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]), name)

    def test_legacy_builds_preserve_rng_and_state_names(self):
        for architecture, builder in (("resnet18", resnet18), ("resnet34", resnet34)):
            for no_maxpool in (False, True):
                with self.subTest(architecture=architecture, no_maxpool=no_maxpool):
                    torch.manual_seed(44)
                    old = builder(weights=None)
                    if no_maxpool:
                        old.maxpool = nn.Identity()
                    old.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(old.fc.in_features, 6))
                    old_rng = torch.get_rng_state().clone()
                    torch.manual_seed(44)
                    new = build_model(pretrained=False, architecture=architecture, no_maxpool=no_maxpool, dropout=0.3)
                    self.assert_states_equal(old.state_dict(), new.state_dict())
                    self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
                    self.assertEqual([id(p) for p in new.fc.parameters()], [id(p) for p in classification_head_parameters(new)])

    def test_real_legacy_checkpoints_strict_load(self):
        paths = [Path("outputs/resnet18_mixup_s43/best.pt"), Path("outputs/improvement_20261016/resnet34/best.pt")]
        for path in paths:
            with self.subTest(path=str(path)):
                state = torch.load(path, map_location="cpu", weights_only=False)
                config = state["config"]
                model = build_model(pretrained=False, architecture=config.get("architecture", "resnet18"),
                                    no_maxpool=config.get("no_maxpool", False), dropout=config["dropout"])
                model.load_state_dict(state["model_state"], strict=True)
                self.assert_states_equal(model.state_dict(), state["model_state"])

    def test_efficientnet_output_roundtrip_and_optimizer_partition(self):
        model = build_model(pretrained=False, architecture="efficientnet_b0", dropout=0.3).eval()
        self.assertFalse(hasattr(model, "fc"))
        head = classification_head_parameters(model)
        head_ids = {id(p) for p in head}
        self.assertEqual(head_ids, {id(p) for p in model.classifier.parameters()})
        self.assertEqual(len(head), 2)
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        self.assertEqual({id(p) for p in backbone} & head_ids, set())
        self.assertEqual({id(p) for p in backbone} | head_ids, {id(p) for p in model.parameters()})
        self.assertEqual(model.classifier[1].out_features, 6)
        other = build_model(pretrained=False, architecture="efficientnet_b0", dropout=0.3).eval()
        other.load_state_dict(model.state_dict(), strict=True)
        images = torch.randn(2, 3, 128, 128)
        with torch.inference_mode():
            actual, restored = model(images), other(images)
        self.assertEqual(tuple(actual.shape), (2, 6))
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(torch.equal(actual, restored))

    def test_efficientnet_rejects_resnet_only_flag(self):
        with self.assertRaisesRegex(ValueError, "ResNet-only"):
            build_model(pretrained=False, architecture="efficientnet_b0", no_maxpool=True)
        with patch.object(sys, "argv", ["train.py", "--architecture", "efficientnet_b0", "--no-maxpool"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                parse_args()
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
