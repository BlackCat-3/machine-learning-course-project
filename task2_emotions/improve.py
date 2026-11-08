"""Run a small, controlled improvement study and refit its selected recipe.

Examples:
  python improve.py --stage validation
  python improve.py --stage final
Uses the same held-out split as the original study; never uses test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from model import CLASS_TO_INDEX, NUM_CLASSES, classification_metrics

ROOT = Path(__file__).resolve().parent
RECIPES = {
    "resnet18_detail": ["--architecture", "resnet18", "--no-maxpool"],
    "resnet34": ["--architecture", "resnet34"],
}
COMMON = ["--epochs", "18", "--batch-size", "128", "--image-size", "112",
          "--lr", "0.0003", "--weight-decay", "0.001", "--label-smoothing", "0.05",
          "--dropout", "0.3", "--seed", "43", "--mixup-alpha", "0.2", "--mixup-prob", "1",
          "--device", "cuda", "--log-every", "0"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("RUN " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def probabilities(path):
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key] for key in a.files}


def dataset_contract():
    """Use the exact fingerprint contract used by train.py and predict.py."""
    with np.load(ROOT / "data/cache.npz", allow_pickle=False) as cache:
        labels = cache["train_labels"].astype(np.int64)
        ids = cache["train_ids"].astype(str)
        train_indices = cache["train_indices"].astype(np.int64)
        val_indices = cache["val_indices"].astype(np.int64)
    prefix = labels.tobytes() + ids.tobytes()
    return {
        "holdout_fingerprint": hashlib.sha256(prefix + train_indices.tobytes() + val_indices.tobytes()).hexdigest(),
        "full_fingerprint": hashlib.sha256(prefix + np.arange(len(labels), dtype=np.int64).tobytes() + np.array([], dtype=np.int64).tobytes()).hexdigest(),
        "train_samples": len(train_indices), "validation_samples": len(val_indices),
        "full_samples": len(labels), "validation_ids": ids[val_indices], "validation_labels": labels[val_indices],
    }


def expected_config(name, *, full, stop_epoch, data):
    # Include defaults that materially affect training even when COMMON omits
    # their CLI arguments. Missing no_maxpool in original checkpoints means False.
    expected = {"architecture": "resnet18", "no_maxpool": False,
                "head_lr_multiplier": 3.0, "warmup_epochs": 1.0, "ema_decay": 0.995,
                "no_pretrained": False, "no_amp": False, "deterministic": False,
                "max_train_batches": 0, "max_val_batches": 0,
                "full_train": full, "stop_after_epochs": stop_epoch if full else 0,
                "training_is_partial": False, "validation_is_partial": False,
                "class_to_index": CLASS_TO_INDEX,
                "dataset_fingerprint": data["full_fingerprint" if full else "holdout_fingerprint"],
                "train_samples": data["full_samples" if full else "train_samples"],
                "validation_samples": 0 if full else data["validation_samples"]}
    if name != "old_mixup" and name not in RECIPES:
        raise ValueError(f"Unknown recipe: {name}")
    options = iter(COMMON + RECIPES.get(name, []))
    integer_keys = {"epochs", "batch_size", "image_size", "seed"}
    text_keys = {"architecture", "device"}
    for flag in options:
        key = flag.removeprefix("--").replace("-", "_")
        if key == "no_maxpool":
            expected[key] = True
            continue
        value = next(options)
        if key == "log_every":
            continue  # Logging frequency does not change the trained model.
        expected[key] = int(value) if key in integer_keys else value if key in text_keys else float(value)
    return expected


def check_config(config, expected, context):
    for key, value in expected.items():
        actual = config.get(key, False if key == "no_maxpool" else None)
        if actual != value:
            raise ValueError(f"Stale/incompatible run {context}: {key}={actual!r}, expected {value!r}")


def checked_run(run_dir, name, *, full, stop_epoch, data):
    expected = expected_config(name, full=full, stop_epoch=stop_epoch, data=data)
    check_config(read(run_dir / "config.json"), expected, run_dir)
    summary = read(run_dir / "summary.json")
    completed = stop_epoch if full else expected["epochs"]
    for key, value in {"epochs_completed": completed, "planned_epochs": expected["epochs"],
                       "stop_after_epochs": stop_epoch if full else 0,
                       "full_train": full, "training_is_partial": False, "validation_is_partial": False}.items():
        if summary.get(key) != value:
            raise ValueError(f"Incomplete/invalid run {run_dir}: {key}={summary.get(key)!r}, expected {value!r}")
    if full and (summary.get("best_epoch") is not None or summary.get("best_validation_accuracy") is not None):
        raise ValueError(f"Full refit must not claim independent validation: {run_dir}")
    return expected, summary


def checked_checkpoint(path, expected, *, epoch, source=None, accuracy=None):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    check_config(checkpoint["config"], expected, path)
    if checkpoint.get("class_to_index") != CLASS_TO_INDEX or checkpoint.get("epoch") != epoch:
        raise ValueError(f"Checkpoint class mapping/epoch mismatch: {path}")
    if checkpoint.get("selected_source") not in {"raw", "ema"}:
        raise ValueError(f"Missing/invalid checkpoint source: {path}")
    if source is not None and checkpoint["selected_source"] != source:
        raise ValueError(f"Checkpoint raw/EMA source differs from model selection: {path}")
    if checkpoint.get("validation_accuracy") != accuracy:
        raise ValueError(f"Checkpoint validation provenance mismatch: {path}")
    return checkpoint


def checked_predictions(pred, name, data):
    if not np.array_equal(pred["ids"], data["validation_ids"]) or not np.array_equal(pred["labels"], data["validation_labels"]):
        raise ValueError(f"Validation order/labels differ from the current cache: {name}")
    p = pred["probabilities"]
    if (p.shape != (data["validation_samples"], NUM_CLASSES) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-4)):
        raise ValueError(f"Invalid probabilities: {name}")
    if not np.array_equal(pred["predictions"], p.argmax(1)):
        raise ValueError(f"Saved predictions do not match probabilities: {name}")


def validation(output):
    data = dataset_contract()
    for name, extra in RECIPES.items():
        run_dir = output / name
        if not (run_dir / "summary.json").exists():
            run("-u", "train.py", "--output-dir", run_dir, *COMMON, *extra)
        expected, summary = checked_run(run_dir, name, full=False, stop_epoch=0, data=data)
        metrics = read(run_dir / "best_metrics.json")
        if metrics["epoch"] != summary["best_epoch"] or metrics["accuracy"] != summary["best_validation_accuracy"]:
            raise ValueError(f"Best metrics/summary mismatch: {run_dir}")
        checked_checkpoint(run_dir / "best.pt", expected, epoch=metrics["epoch"],
                           source=metrics["selected_source"], accuracy=metrics["accuracy"])
        run("-u", "predict.py", "--checkpoints", run_dir / "best.pt", "--split", "val", "--tta",
            "--device", "cuda", "--output", output / f"{name}_tta.csv")
        run("plot_results.py", "--run-dir", run_dir)

    original = probabilities(ROOT / "outputs/selected_validation_predictions.npz")
    old_mixup = probabilities(ROOT / "outputs/validation_mixup_tta.probabilities.npz")
    old_expected, _ = checked_run(ROOT / "outputs/resnet18_mixup_s43", "old_mixup", full=False, stop_epoch=0, data=data)
    old_metrics = read(ROOT / "outputs/resnet18_mixup_s43/best_metrics.json")
    checked_checkpoint(ROOT / "outputs/resnet18_mixup_s43/best.pt", old_expected,
                       epoch=old_metrics["epoch"], source=old_metrics["selected_source"], accuracy=old_metrics["accuracy"])
    checked_predictions(original, "original ensemble", data)
    predictions = {"old_mixup": old_mixup}
    predictions.update({name: probabilities(output / f"{name}_tta.probabilities.npz") for name in RECIPES})
    for name, pred in predictions.items():
        checked_predictions(pred, name, data)
    # Fixed, small candidate set: single models and simple equal-weight averages.
    # No continuous weight search or class-specific leaderboard tuning.
    choices = []
    for size in range(1, len(predictions) + 1):
        for names in itertools.combinations(predictions, size):
            p = np.mean([predictions[name]["probabilities"] for name in names], axis=0)
            metrics = classification_metrics(original["labels"], p.argmax(1))
            choices.append({"models": list(names), "accuracy": metrics["accuracy"], "correct": int((p.argmax(1) == original["labels"]).sum())})
    choices.sort(key=lambda x: (-x["correct"], len(x["models"]), x["models"]))
    selected = choices[0]
    p = np.mean([predictions[name]["probabilities"] for name in selected["models"]], axis=0)
    np.savez_compressed(output / "selected_validation.npz", ids=original["ids"], labels=original["labels"],
                        probabilities=p, predictions=p.argmax(1))
    baseline_accuracy = float((original["predictions"] == original["labels"]).mean())
    details = {name: read(output / name / "best_metrics.json") for name in RECIPES}
    result = {"original_validation_accuracy": baseline_accuracy,
              "original_kaggle_score": {"value": 0.70, "source": "user reported on 2026-10-16; not independently retrieved"},
              "validation_samples": len(original["labels"]), "candidate_scores": choices, "selected": selected,
              "improvement_percentage_points": 100 * (selected["accuracy"] - baseline_accuracy),
              "selected_metrics": classification_metrics(original["labels"], p.argmax(1)),
              "new_model_selection": details,
              "checkpoint_selection": "Each run selects its best raw/EMA checkpoint by validation accuracy without TTA; fixed candidate comparison uses horizontal flip TTA.",
              "scope": "Repeated model-selection holdout; not an untouched final test, and not the Kaggle score.",
              "common_arguments": COMMON, "recipe_arguments": RECIPES,
              "dataset_fingerprint": data["holdout_fingerprint"]}
    write(output / "selection.json", result)
    print(json.dumps({"selected": selected, "gain_pp": result["improvement_percentage_points"]}), flush=True)


def final(output):
    import torch

    selection = read(output / "selection.json")
    data = dataset_contract()
    if (selection.get("common_arguments") != COMMON or selection.get("recipe_arguments") != RECIPES
            or selection.get("dataset_fingerprint") != data["holdout_fingerprint"]):
        raise ValueError("Selection belongs to a different recipe or dataset; rerun --stage validation.")
    names = selection["selected"]["models"]
    if not names or len(set(names)) != len(names) or not set(names).issubset({"old_mixup", *RECIPES}):
        raise ValueError("Invalid selected model names")
    selected_predictions = probabilities(output / "selected_validation.npz")
    checked_predictions(selected_predictions, "selected ensemble", data)
    actual_accuracy = float((selected_predictions["predictions"] == selected_predictions["labels"]).mean())
    if actual_accuracy != selection["selected"]["accuracy"]:
        raise ValueError("Selection accuracy differs from its saved validation predictions")
    components = []
    for name in names:
        path = (ROOT / "outputs/validation_mixup_tta.probabilities.npz" if name == "old_mixup"
                else output / f"{name}_tta.probabilities.npz")
        pred = probabilities(path)
        checked_predictions(pred, name, data)
        components.append(pred["probabilities"])
    if not np.allclose(np.mean(components, axis=0), selected_predictions["probabilities"], rtol=0, atol=1e-7):
        raise ValueError("Saved selected probabilities differ from the selected model average")
    original = probabilities(ROOT / "outputs/selected_validation_predictions.npz")
    checked_predictions(original, "original ensemble", data)
    original_accuracy = float((original["predictions"] == original["labels"]).mean())
    expected_gain = 100 * (actual_accuracy - original_accuracy)
    if (selection["original_validation_accuracy"] != original_accuracy
            or not np.isclose(selection["improvement_percentage_points"], expected_gain, rtol=0, atol=1e-10)):
        raise ValueError("Selection improvement does not match its validation evidence")
    if selection["improvement_percentage_points"] <= 0:
        raise ValueError("No measured improvement; keep the original submission and inspect the study.")
    checkpoints = []
    for name in selection["selected"]["models"]:
        if name == "old_mixup":
            run_dir = ROOT / "outputs/final_models/mixup"
            metrics = read(ROOT / "outputs/resnet18_mixup_s43/best_metrics.json")
            expected, _ = checked_run(run_dir, name, full=True, stop_epoch=metrics["epoch"], data=data)
            checked_checkpoint(run_dir / "final.pt", expected, epoch=metrics["epoch"], source=metrics["selected_source"])
            checkpoints.append(run_dir / "final.pt")
            continue
        metrics = selection["new_model_selection"][name]
        holdout_dir = output / name
        holdout_expected, _ = checked_run(holdout_dir, name, full=False, stop_epoch=0, data=data)
        if metrics != read(holdout_dir / "best_metrics.json"):
            raise ValueError(f"Holdout selection has changed since selection.json was written: {name}")
        checked_checkpoint(holdout_dir / "best.pt", holdout_expected, epoch=metrics["epoch"],
                           source=metrics["selected_source"], accuracy=metrics["accuracy"])
        run_dir = output / "full" / name
        if not (run_dir / "summary.json").exists():
            run("-u", "train.py", "--output-dir", run_dir, *COMMON, *RECIPES[name],
                "--full-train", "--stop-after-epochs", metrics["epoch"])
        expected, _ = checked_run(run_dir, name, full=True, stop_epoch=metrics["epoch"], data=data)
        final_state = checked_checkpoint(run_dir / "final.pt", expected, epoch=metrics["epoch"])
        # Match the raw-versus-EMA source selected on holdout. Full training's
        # default final.pt is EMA; last.pt also retains the raw parameters.
        if metrics["selected_source"] == "raw":
            last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            check_config(last["config"], expected, run_dir / "last.pt")
            if last["epoch"] != metrics["epoch"] or last.get("class_to_index") != CLASS_TO_INDEX:
                raise ValueError(f"Raw refit checkpoint epoch/mapping mismatch: {run_dir}")
            final_state.update(model_state=last["model_state"], selected_source="raw")
            temporary = run_dir / "final.pt.tmp"
            torch.save(final_state, temporary)
            temporary.replace(run_dir / "final.pt")
        if final_state["selected_source"] != metrics["selected_source"]:
            raise ValueError(f"Full refit raw/EMA source differs from selection: {run_dir}")
        checkpoints.append(run_dir / "final.pt")
        run("plot_results.py", "--run-dir", run_dir)
    submission = output / "submission_improved.csv"
    run("-u", "predict.py", "--checkpoints", *checkpoints, "--tta", "--device", "cuda", "--output", submission)
    run("verify_submission.py", submission, "--json", output / "submission_improved.check.json")
    write(output / "completion.json", {
        "status": "complete", "submission": str(submission),
        "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "holdout_selection_accuracy": selection["selected"]["accuracy"],
        "selected_models": selection["selected"]["models"],
        "full_checkpoints": list(map(str, checkpoints)),
        "kaggle_status": "not_uploaded", "new_kaggle_score": None,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation", "final", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/improvement_20261016")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("validation", "all"):
        validation(output)
    if args.stage in ("final", "all"):
        final(output)


if __name__ == "__main__":
    main()
