"""Explicit deadline fallback: predict with the selected heldout-trained model.

This does not train, cancel jobs, change selection, or pretend the new model was
refit on all labeled images. Existing selected ResNets use their full-fit weights;
EfficientNet uses its validated best.pt trained on 24,034 labeled images.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from efficientnet_study import (BASELINE, CANDIDATES, DEFAULT_OUTPUT, MINIMUM_GAIN_CORRECT,
                               NEW, check_full, check_holdout, paths, require, sha)
from improve import ROOT, checked_predictions, dataset_contract, probabilities, read, run, write
from model import CLASS_TO_INDEX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, help="Default: <study-dir>/submission_deadline_holdout.csv")
    args = parser.parse_args()
    study = args.study_dir.resolve()
    output = args.output.resolve() if args.output else study / "submission_deadline_holdout.csv"
    require(output.suffix.lower() == ".csv", "Output must be a CSV path")
    for suffix in (".csv", ".json", ".probabilities.npz", ".check.json", ".audit.json"):
        require(not output.with_suffix(suffix).exists(), f"Refusing to overwrite an existing artifact: {output.with_suffix(suffix)}")
    plan, selection, data = read(study / "plan.json"), read(study / "selection.json"), dataset_contract()
    require(selection["plan_sha256"] == sha(study / "plan.json"), "Selection plan changed")
    require(plan["dataset_fingerprint"] == data["holdout_fingerprint"], "Data split differs from the frozen plan")
    require(plan["candidate_models"] == CANDIDATES and plan["baseline_models"] == BASELINE, "Candidate plan mismatch")
    require(plan["minimum_gain_correct_for_submission"] == MINIMUM_GAIN_CORRECT == 43, "Submission threshold changed")
    require(selection["meets_minimum_gain"] and selection["improved"], "No qualifying validation improvement")
    selected = selection["selected"]
    require(selected == selection["best_observed"] and selected in selection["candidate_scores"], "Selected candidate mismatch")
    require(selected["models"] in CANDIDATES and NEW in selected["models"], "Undeclared/unchanged candidate")
    require(selected["correct"] - plan["baseline_validation_correct"] >= MINIMUM_GAIN_CORRECT,
            "Selected gain is below the predeclared 43-image threshold")
    protected = plan["protected_submission_hashes"]
    for path, digest in protected.items():
        require(sha(path) == digest, "An existing submission changed")
    source_paths, components, checkpoints, source_records = paths(study), [], [], []
    import torch
    for name in selected["models"]:
        holdout_dir, p_path, full_dir = source_paths[name]
        metrics = check_holdout(holdout_dir, name, plan, data)
        require(metrics == selection["checkpoint_selection"][name], "Checkpoint selection metadata changed")
        require(sha(holdout_dir / "best.pt") == selection["holdout_checkpoint_hashes"][name], "Heldout checkpoint changed")
        pred = probabilities(p_path)
        checked_predictions(pred, name, data)
        components.append(pred["probabilities"])
        if name == NEW:
            checkpoint_path = holdout_dir / "best.pt"
            training_scope = "heldout_fit_only; not refit on all labeled data"
        else:
            checkpoint_path = check_full(full_dir, holdout_dir, name, metrics, plan, data)
            training_scope = "existing full-data refit"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        require(checkpoint["class_to_index"] == CLASS_TO_INDEX and config["class_to_index"] == CLASS_TO_INDEX,
                "Prediction checkpoint class mapping mismatch")
        require(checkpoint["epoch"] == metrics["epoch"] and checkpoint["selected_source"] == metrics["selected_source"],
                "Prediction checkpoint epoch/raw-or-EMA source differs from selection")
        expected_full = name != NEW
        require(config["full_train"] == expected_full, "Unexpected training scope")
        require(config["train_samples"] == data["full_samples" if expected_full else "train_samples"], "Training sample count mismatch")
        require(config["validation_samples"] == (0 if expected_full else data["validation_samples"]), "Validation sample count mismatch")
        source_records.append({"name": name, "checkpoint": str(checkpoint_path), "sha256": sha(checkpoint_path),
                               "epoch": checkpoint["epoch"], "selected_source": checkpoint["selected_source"],
                               "training_scope": training_scope, "full_train": config["full_train"],
                               "train_samples": config["train_samples"], "validation_samples": config["validation_samples"],
                               "dataset_fingerprint": config["dataset_fingerprint"], "image_size": config["image_size"],
                               "architecture": config.get("architecture", "resnet18")})
        checkpoints.append(checkpoint_path)
        del checkpoint
    selected_p = np.mean(components, axis=0)
    saved = probabilities(study / "selected_validation.npz")
    checked_predictions(saved, "selected validation ensemble", data)
    require(np.allclose(selected_p, saved["probabilities"], atol=1e-7, rtol=0), "Validation probability reconstruction differs")
    correct = int((selected_p.argmax(1) == data["validation_labels"]).sum())
    require(correct == selected["correct"] and correct - plan["baseline_validation_correct"] >= MINIMUM_GAIN_CORRECT,
            "Validation improvement did not reproduce")
    require(abs(selected["accuracy"] - correct / data["validation_samples"]) < 1e-12, "Selected accuracy is inconsistent")
    output.parent.mkdir(parents=True, exist_ok=True)
    print("DEADLINE FALLBACK: EfficientNet uses heldout-trained best.pt; no full-data refit is claimed.", flush=True)
    run("-u", "predict.py", "--checkpoints", *checkpoints, "--tta", "--device", "cuda", "--output", output)
    check_path = output.with_suffix(".check.json")
    run("verify_submission.py", output, "--json", check_path)
    prediction = probabilities(output.with_suffix(".probabilities.npz"))
    with np.load(ROOT / "data/cache.npz", allow_pickle=False) as cache:
        expected_ids = cache["test_ids"].astype(str)
    p = prediction["probabilities"]
    require(np.array_equal(prediction["ids"], expected_ids) and p.shape == (len(expected_ids), len(CLASS_TO_INDEX)),
            "Test prediction IDs/shape mismatch")
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            and np.allclose(p.sum(1), 1, atol=1e-4), "Invalid test probabilities")
    require(np.array_equal(prediction["predictions"], p.argmax(1)), "Predictions are not probability argmax")
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["ID", "Emotion"], "Wrong CSV columns")
        csv_rows = list(reader)
    require([row["ID"] for row in csv_rows] == expected_ids.tolist(), "CSV IDs differ from template order")
    require(np.array_equal(np.array([int(row["Emotion"]) for row in csv_rows]), p.argmax(1)), "CSV labels differ from probabilities")
    inference_metadata = read(output.with_suffix(".json"))
    require(inference_metadata["split"] == "test" and inference_metadata["flip_tta"], "Wrong inference mode")
    require([record["path"] for record in inference_metadata["checkpoints"]] == [str(path.resolve()) for path in checkpoints],
            "Inference model list/order differs from selected fallback")
    for source in source_records:
        require(sha(source["checkpoint"]) == source["sha256"], "Prediction source weights changed during inference")
    for path, digest in protected.items():
        require(sha(path) == digest, "An old submission changed during fallback prediction")
    audit = {"status": "complete_deadline_submission", "passed": True,
             "submission": str(output), "sha256": sha(output), "rows": len(expected_ids),
             "class_to_index": CLASS_TO_INDEX, "selected_models": selected["models"], "sources": source_records,
             "fallback": {"new_model_full_refit": False, "new_model_training_samples": data["train_samples"],
                          "all_available_training_samples": data["full_samples"],
                          "reason": "Explicit deadline fallback; use the validated new checkpoint directly for test inference",
                          "training_jobs_cancelled_by_this_script": False},
             "reference_model_selection": {"correct": correct, "samples": data["validation_samples"],
                                           "accuracy": correct / data["validation_samples"], "gain_correct": correct - plan["baseline_validation_correct"],
                                           "scope": "Computed with heldout counterparts; not validation of the mixed full-fit/heldout prediction weights, and not a Kaggle score"},
             "plan_sha256": sha(study / "plan.json"), "selection_sha256": sha(study / "selection.json"),
             "format_check": str(check_path), "old_submission_hashes_preserved": True,
             "kaggle_status": "not_uploaded", "new_kaggle_score": None}
    write(output.with_suffix(".audit.json"), audit)
    print(f"Fallback submission and explicit training-scope audit complete: {output}", flush=True)


if __name__ == "__main__":
    main()
