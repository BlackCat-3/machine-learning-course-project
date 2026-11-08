"""CPU checks for inference-only warm starts; no training or source modification."""
from __future__ import annotations

import copy
import hashlib
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

from model import CLASS_TO_INDEX, build_model
from train import load_initialization_checkpoint, parse_args


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        # Keep small reproducible fixtures inside the writable workspace; the
        # Windows sandbox's default temporary directory may deny torch.save.
        directory = Path(__file__).resolve().parent / "outputs" / "warmstart_checks" / self._testMethodName
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "best.pt"
        self.source_config = {
            "architecture": "resnet18", "no_maxpool": True,
            "dataset_fingerprint": "verified-fixed-split", "full_train": False,
            "train_samples": 24034, "validation_samples": 4241,
            "class_to_index": CLASS_TO_INDEX, "image_size": 112,
            "epochs": 18, "lr": 0.0003,
        }
        self.checkpoint = {
            "config": self.source_config, "class_to_index": CLASS_TO_INDEX,
            "model_state": {"weight": torch.tensor([[1., 2.]])},
            "epoch": 13, "selected_source": "ema",
        }
        self.target = dict(self.source_config, image_size=160, epochs=6, lr=3e-5)

    def save(self, checkpoint=None):
        torch.save(self.checkpoint if checkpoint is None else checkpoint, self.path)

    def test_resolution_and_training_parameters_can_change(self):
        self.save()
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        checkpoint, metadata = load_initialization_checkpoint(self.path, self.target)
        model = torch.nn.Linear(2, 1, bias=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        self.assertTrue(torch.equal(model.weight, self.checkpoint["model_state"]["weight"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.target["lr"])
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(metadata["sha256"], before)
        self.assertEqual(metadata["source_epoch"], 13)
        self.assertEqual(metadata["source_image_size"], 112)
        self.assertEqual(metadata["selected_source"], "ema")
        self.assertTrue(metadata["optimizer_and_schedule_reset"])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_split_and_topology_mismatches_rejected(self):
        self.save()
        for key, value in (("architecture", "resnet34"), ("no_maxpool", False),
                           ("dataset_fingerprint", "different-split"), ("full_train", True),
                           ("train_samples", 28275), ("validation_samples", 0)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                load_initialization_checkpoint(self.path, dict(self.target, **{key: value}))

    def test_bad_sources_rejected(self):
        for key in ("training_is_partial", "validation_is_partial", "max_train_batches", "max_val_batches"):
            bad = copy.deepcopy(self.checkpoint)
            bad["config"][key] = 1
            self.save(bad)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "partial"):
                load_initialization_checkpoint(self.path, self.target)
        for mutation in (
            lambda x: x.update(class_to_index={"wrong": 0}),
            lambda x: x["config"].update(class_to_index={"wrong": 0}),
            lambda x: x.update(selected_source="unknown"),
            lambda x: x.update(epoch=0),
            lambda x: x.update(optimizer={}),
            lambda x: x["config"].update(image_size=None),
        ):
            bad = copy.deepcopy(self.checkpoint)
            mutation(bad)
            self.save(bad)
            with self.assertRaises(ValueError):
                load_initialization_checkpoint(self.path, self.target)
        with self.assertRaisesRegex(ValueError, "inference"):
            load_initialization_checkpoint(self.path.with_name("last.pt"), self.target)

    def test_resume_and_initialization_are_exclusive(self):
        with patch.object(sys, "argv", ["train.py", "--resume", "last.pt", "--init-checkpoint", "best.pt"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                parse_args()
            self.assertEqual(caught.exception.code, 2)

    def test_source_directory_cannot_be_overwritten(self):
        arguments = ["train.py", "--init-checkpoint", str(self.path), "--output-dir", str(self.path.parent)]
        with patch.object(sys, "argv", arguments), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                parse_args()
            self.assertEqual(caught.exception.code, 2)

    def test_actual_seed44_checkpoint_state_equality(self):
        path = Path("outputs/seed_erasing_20261016/detail_seed44/best.pt")
        if not path.exists():
            self.skipTest("Actual seed44 checkpoint unavailable at conventional path")
        original = torch.load(path, map_location="cpu", weights_only=False)
        config = dict(original["config"], image_size=160, epochs=6, lr=3e-5)
        checkpoint, metadata = load_initialization_checkpoint(path, config)
        model = build_model(pretrained=False, dropout=config["dropout"],
                            architecture=config["architecture"], no_maxpool=config["no_maxpool"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original["model_state"][key]), key)
        self.assertEqual(metadata["source_epoch"], original["epoch"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
