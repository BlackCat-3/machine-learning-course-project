"""Independently audit improved predictions and training provenance on CPU.

Run after improve.py finishes. No training, model forward pass or upload occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        require(next(reader, None) == ["ID", "Emotion"], f"Wrong CSV columns: {path}")
        rows = list(reader)
    require(bool(rows) and all(len(row) == 2 and row[0] for row in rows), f"Malformed CSV rows: {path}")
    ids = np.array([row[0] for row in rows])
    require(len(np.unique(ids)) == len(ids), f"Duplicate CSV IDs: {path}")
    require(all(row[1] in {"0", "1", "2", "3", "4", "5"} for row in rows), f"Invalid CSV labels: {path}")
    return ids, np.array([int(row[1]) for row in rows], dtype=np.int64)


def read_predictions(path, expected_ids, expected_labels=None):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    require(np.array_equal(values["ids"], expected_ids), f"NPZ ID order mismatch: {path}")
    probabilities = values["probabilities"]
    require(probabilities.shape == (len(expected_ids), 6), f"Probability shape mismatch: {path}")
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all(),
            f"Invalid probability values: {path}")
    require(np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-4), f"Probability row sums differ from one: {path}")
    require(np.array_equal(values["predictions"], probabilities.argmax(axis=1)), f"Saved predictions differ from argmax: {path}")
    if expected_labels is not None:
        require(np.array_equal(values["labels"], expected_labels), f"Validation labels differ from cache: {path}")
    return values


def metrics(labels, predictions):
    matrix = np.bincount(labels * 6 + predictions, minlength=36).reshape(6, 6)
    result = {"correct": int((labels == predictions).sum()), "samples": len(labels),
              "accuracy": float((labels == predictions).mean()), "confusion_matrix": matrix.tolist(),
              "confusion_matrix_axes": {"rows": "true", "columns": "predicted"}, "per_class": {}}
    for name, label in LABELS.items():
        support = int(matrix[label].sum())
        result["per_class"][name] = {"support": support, "correct": int(matrix[label, label]),
                                      "recall": float(matrix[label, label] / support) if support else None}
    return result


def check_history(directory, expected_epoch, samples, *, full):
    history = read_json(directory / "history.json")
    require([row["epoch"] for row in history] == list(range(1, expected_epoch + 1)), f"Incomplete epoch history: {directory}")
    require(all(row["train_samples_seen"] == samples for row in history), f"Incomplete sample processing: {directory}")
    if full:
        require(all(row["val_accuracy"] is None and row["val_loss"] is None for row in history),
                f"Full-training history claims validation metrics: {directory}")
    return {"epochs": len(history), "samples_per_epoch": samples,
            "skipped_optimizer_steps": sum(row.get("skipped_optimizer_steps", 0) for row in history)}


def check_saved_weights(final, final_path):
    """Check the actual saved source tensors and architecture, without inference."""
    import torch
    from model import build_model

    last_path = final_path.parent / "last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    require(last["epoch"] == final["epoch"] and last["config"] == final["config"],
            f"Final/last checkpoint training provenance differs: {final_path}")
    require(last["class_to_index"] == LABELS, f"Last checkpoint label mapping differs: {last_path}")
    source = final["selected_source"]
    require(source in {"raw", "ema"}, f"Unknown final weight source: {source}")
    source_key = "ema_state" if source == "ema" else "model_state"
    expected = last[source_key]
    actual = final["model_state"]
    require(isinstance(expected, dict) and expected.keys() == actual.keys(),
            f"Final/source weight keys differ: {final_path}")
    for key, value in actual.items():
        reference = expected[key]
        require(isinstance(value, torch.Tensor) and isinstance(reference, torch.Tensor)
                and value.device.type == "cpu" and reference.device.type == "cpu"
                and value.dtype == reference.dtype and value.shape == reference.shape
                and torch.equal(value, reference), f"Final weight differs from last.pt {source_key}: {key}")
    config = final["config"]
    model = build_model(pretrained=False, architecture=config.get("architecture", "resnet18"),
                        no_maxpool=config.get("no_maxpool", False), dropout=config.get("dropout", 0.2))
    model.load_state_dict(actual, strict=True)
    result = {"last_checkpoint": str(last_path.resolve()), "source_state": source_key,
              "all_tensors_exactly_equal": True, "matched_tensors": len(actual), "strict_architecture_load": True}
    del last, model
    return result


def audit(output):
    import torch

    torch.set_num_threads(4)
    selection = read_json(output / "selection.json")
    completion = read_json(output / "completion.json")
    require(completion["status"] == "complete", "Improvement run has not completed")
    original_completion = read_json(ROOT / "outputs/completion.json")
    old_submission = ROOT / "outputs/submission.csv"
    old_hash = sha256(old_submission)
    require(old_hash == original_completion["artifacts"]["outputs/submission.csv"]["sha256"], "Original submission was overwritten")

    with np.load(ROOT / "data/cache.npz", allow_pickle=False) as cache:
        ids = cache["train_ids"].astype(str)
        labels = cache["train_labels"].astype(np.int64)
        train_indices = cache["train_indices"].astype(np.int64)
        val_indices = cache["val_indices"].astype(np.int64)
        test_ids = cache["test_ids"].astype(str)
    require(len(np.intersect1d(train_indices, val_indices)) == 0, "Cache training/validation indices overlap")
    val_ids, val_labels = ids[val_indices], labels[val_indices]
    prefix = labels.tobytes() + ids.tobytes()
    holdout_fingerprint = hashlib.sha256(prefix + train_indices.tobytes() + val_indices.tobytes()).hexdigest()
    full_fingerprint = hashlib.sha256(prefix + np.arange(len(labels), dtype=np.int64).tobytes() + np.array([], dtype=np.int64).tobytes()).hexdigest()
    require(selection["dataset_fingerprint"] == holdout_fingerprint, "Selection has a different holdout split")

    template_ids, _ = read_csv(ROOT / "data/submission_template.csv")
    require(np.array_equal(template_ids, test_ids), "Cache test IDs differ from supplied template")
    submission = resolved(completion["submission"])
    require(submission == (output / "submission_improved.csv").resolve(), "Completion points to an unexpected submission")
    csv_ids, csv_labels = read_csv(submission)
    require(np.array_equal(csv_ids, template_ids), "Submission ID order differs from template")
    submission_hash = sha256(submission)
    require(submission_hash == completion["sha256"], "Submission hash differs from completion record")
    test = read_predictions(submission.with_suffix(".probabilities.npz"), template_ids)
    require(np.array_equal(csv_labels, test["predictions"]), "CSV labels differ from saved probability argmax")

    old = read_predictions(ROOT / "outputs/selected_validation_predictions.npz", val_ids, val_labels)
    new = read_predictions(output / "selected_validation.npz", val_ids, val_labels)
    old_metrics, new_metrics = metrics(val_labels, old["predictions"]), metrics(val_labels, new["predictions"])
    require(selection["validation_samples"] == len(val_labels), "Selection validation sample count differs")
    require(selection["original_validation_accuracy"] == old_metrics["accuracy"], "Original accuracy differs from saved predictions")
    require(selection["selected"]["accuracy"] == new_metrics["accuracy"] == completion["holdout_selection_accuracy"], "Selected accuracy differs from saved predictions")
    require(selection["selected"]["correct"] == new_metrics["correct"], "Selected correct count differs")
    gain = 100 * (new_metrics["accuracy"] - old_metrics["accuracy"])
    require(np.isclose(selection["improvement_percentage_points"], gain, rtol=0, atol=1e-10), "Recorded improvement differs from predictions")
    names = selection["selected"]["models"]
    require(names == completion["selected_models"] and len(set(names)) == len(names) > 0, "Selected model lists differ or contain duplicates")
    require(set(names).issubset({"old_mixup", "resnet18_detail", "resnet34"}), "Unknown selected recipe")
    final_paths = [resolved(path) for path in completion["full_checkpoints"]]
    require(len(final_paths) == len(names), "Selected model/checkpoint counts differ")
    components, checkpoint_audits = [], []
    comparable = ("architecture", "no_maxpool", "image_size", "dropout", "epochs", "batch_size", "seed", "lr",
                  "head_lr_multiplier", "warmup_epochs", "weight_decay", "label_smoothing", "ema_decay",
                  "mixup_alpha", "mixup_prob", "no_pretrained", "no_amp", "deterministic")
    defaults = {"architecture": "resnet18", "no_maxpool": False, "mixup_alpha": 0.0, "mixup_prob": 0.5}
    for name, path in zip(names, final_paths):
        holdout_dir = ROOT / "outputs/resnet18_mixup_s43" if name == "old_mixup" else output / name
        full_dir = ROOT / "outputs/final_models/mixup" if name == "old_mixup" else output / "full" / name
        require(path == (full_dir / "final.pt").resolve(), f"Unexpected final checkpoint for {name}")
        best_metrics = read_json(holdout_dir / "best_metrics.json")
        if name != "old_mixup":
            require(best_metrics == selection["new_model_selection"][name], f"Holdout selection changed: {name}")
        holdout = torch.load(holdout_dir / "best.pt", map_location="cpu", weights_only=False)
        final = torch.load(path, map_location="cpu", weights_only=False)
        hconfig, fconfig = holdout["config"], final["config"]
        epoch, source = best_metrics["epoch"], best_metrics["selected_source"]
        for state, config, full in ((holdout, hconfig, False), (final, fconfig, True)):
            require(state["class_to_index"] == LABELS == config["class_to_index"], f"Label mapping mismatch: {name}")
            require(state["epoch"] == epoch and state["selected_source"] == source, f"Selected epoch/source mismatch: {name}")
            require(config["full_train"] is full, f"Wrong training scope: {name}")
            require(config["training_is_partial"] is False and config["validation_is_partial"] is False
                    and config["max_train_batches"] == 0 and config["max_val_batches"] == 0, f"Partial checkpoint: {name}")
            require(config["train_samples"] == (len(labels) if full else len(train_indices))
                    and config["validation_samples"] == (0 if full else len(val_indices)), f"Wrong sample counts: {name}")
            require(config["dataset_fingerprint"] == (full_fingerprint if full else holdout_fingerprint), f"Dataset fingerprint mismatch: {name}")
            require(state["validation_accuracy"] == (None if full else best_metrics["accuracy"]), f"Validation provenance mismatch: {name}")
        for key in comparable:
            require(hconfig.get(key, defaults.get(key)) == fconfig.get(key, defaults.get(key)), f"Refit changed {key}: {name}")
        require(fconfig["stop_after_epochs"] == epoch, f"Wrong refit stopping epoch: {name}")
        summary = read_json(full_dir / "summary.json")
        require(summary["epochs_completed"] == epoch and summary["planned_epochs"] == hconfig["epochs"], f"Final summary epoch mismatch: {name}")
        require(summary["full_train"] is True and summary["best_epoch"] is None and summary["best_validation_accuracy"] is None,
                f"Final summary claims independent validation: {name}")
        history = check_history(full_dir, epoch, len(labels), full=True)
        holdout_summary = read_json(holdout_dir / "summary.json")
        require(holdout_summary["best_epoch"] == epoch and holdout_summary["best_validation_accuracy"] == best_metrics["accuracy"],
                f"Holdout best checkpoint differs from summary: {name}")
        check_history(holdout_dir, hconfig["epochs"], len(train_indices), full=False)
        probability_path = ROOT / "outputs/validation_mixup_tta.probabilities.npz" if name == "old_mixup" else output / f"{name}_tta.probabilities.npz"
        components.append(read_predictions(probability_path, val_ids, val_labels)["probabilities"])
        weights_audit = check_saved_weights(final, path)
        checkpoint_audits.append({"model": name, "checkpoint": str(path), "sha256": sha256(path), "epoch": epoch,
                                  "source": source, "architecture": fconfig["architecture"], "no_maxpool": fconfig.get("no_maxpool", False),
                                  "train_samples": len(labels), "validation_samples": 0, "history": history, "weights": weights_audit})
        del holdout, final
    require(np.allclose(np.mean(components, axis=0), new["probabilities"], rtol=0, atol=1e-7), "Selected probabilities differ from equal-weight component average")
    metadata = read_json(submission.with_suffix(".json"))
    require(metadata["split"] == "test" and metadata["flip_tta"] is True and metadata["class_to_index"] == LABELS,
            "Submission metadata scope/TTA/mapping differs")
    require([resolved(item["path"]) for item in metadata["checkpoints"]] == final_paths, "Inference checkpoint list differs from selected refits")
    require(metadata["rows"] == len(test_ids), "Submission metadata row count differs")
    for name, label in LABELS.items():
        require(metadata["prediction_counts"][name] == int((csv_labels == label).sum()), f"Prediction count mismatch: {name}")

    old_right, new_right = old["predictions"] == val_labels, new["predictions"] == val_labels
    paired = {"both_correct": int((old_right & new_right).sum()), "both_wrong": int((~old_right & ~new_right).sum()),
              "old_wrong_new_correct": int((~old_right & new_right).sum()), "old_correct_new_wrong": int((old_right & ~new_right).sum()),
              "net_additional_correct": new_metrics["correct"] - old_metrics["correct"], "gain_percentage_points": gain,
              "prediction_changes": int((old["predictions"] != new["predictions"]).sum())}
    return {"passed": True, "submission": str(submission), "submission_sha256": submission_hash, "rows": len(test_ids),
            "ids_match_template_in_order": True, "csv_matches_probability_argmax": True,
            "original_submission_preserved": {"path": str(old_submission), "sha256": old_hash},
            "selected_models": checkpoint_audits, "validation": {"original": old_metrics, "improved": new_metrics, "paired": paired},
            "validation_scope": "Repeated model-selection holdout; full-data checkpoints have no independent validation score.",
            "new_kaggle_score": None, "audit_source_sha256": sha256(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/improvement_20261016")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = audit(output)
    except Exception as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "rows": result["rows"], "paired": result["validation"]["paired"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
