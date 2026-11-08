"""Independent CPU audit of the paired resolution study and final submission.

No forward passes, training, GPU allocation, checkpoint modification or uploads.
Validation-only mode checks selection before full refits are available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_improvement import (read_json, read_csv, read_predictions, metrics,
                               require, resolved, sha256, check_saved_weights)
from audit_seed_erasing import (data_contract, normalized, differences,
                                check_history, check_inference_metadata,
                                strict_cpu_load, check_plan_arguments)

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT / "outputs/seed_erasing_20261016"
STRUCTURE = ROOT / "outputs/improvement_20261016"
LABELS = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}
OLD3 = ["old_mixup", "detail_seed43", "resnet34"]
BASELINE = [*OLD3, "detail_seed44"]
NEW = {"tune112": 112, "tune160": 160}
CANDIDATES = [BASELINE, ["tune112"], ["tune160"],
              [*OLD3, "tune112"], [*OLD3, "tune160"],
              [*BASELINE, "tune112"], [*BASELINE, "tune160"],
              ["detail_seed44", "tune160"]]
RUN_FIELDS = {"output_dir", "resume", "log_every"}
FULL_FIELDS = {"full_train", "stop_after_epochs", "dataset_fingerprint", "train_samples", "validation_samples",
               "init_checkpoint", "initialization"}
OLD_CSV_HASHES = {
    ROOT / "outputs/submission.csv": "4414f84a494aedb722eafaa3f9ce75ef2991286a8e869792d1cb7974d2bc8f09",
    STRUCTURE / "submission_improved.csv": "1be9b64328089eb25de1959d79c5a6c6d1e4eeea50838f358a53f21d7ba47bfa",
    PRIOR / "submission_improved.csv": "b889290cc001ca7e5c3d8c375693b85e183656f3f81c63ac3cc843a1788ce80a",
}


def sources(output):
    return {
        "old_mixup": (ROOT / "outputs/resnet18_mixup_s43", ROOT / "outputs/validation_mixup_tta.probabilities.npz", ROOT / "outputs/final_models/mixup"),
        "detail_seed43": (STRUCTURE / "resnet18_detail", STRUCTURE / "resnet18_detail_tta.probabilities.npz", STRUCTURE / "full/resnet18_detail"),
        "resnet34": (STRUCTURE / "resnet34", STRUCTURE / "resnet34_tta.probabilities.npz", STRUCTURE / "full/resnet34"),
        "detail_seed44": (PRIOR / "detail_seed44", PRIOR / "detail_seed44_tta.probabilities.npz", PRIOR / "full/detail_seed44"),
        **{name: (output / name, output / f"{name}_tta.probabilities.npz", output / "full" / name) for name in NEW},
    }


def check_scope(config, data, full, planned_epochs):
    expected = {
        "full_train": full, "class_to_index": LABELS,
        "dataset_fingerprint": data["full_fingerprint" if full else "holdout_fingerprint"],
        "train_samples": len(data["ids"] if full else data["training"]),
        "validation_samples": 0 if full else len(data["validation"]),
        "training_is_partial": False, "validation_is_partial": False,
        "max_train_batches": 0, "max_val_batches": 0, "epochs": planned_epochs,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"Scope mismatch: {key}")
    require(resolved(config["data"]) == ROOT / "data/cache.npz", "Wrong training cache")


def check_init(config, plan, data, full):
    source = plan["initialization"]["full" if full else "holdout"]
    expected_path = PRIOR / ("full/detail_seed44/final.pt" if full else "detail_seed44/best.pt")
    require(resolved(source["path"]) == expected_path, "Wrong initialization source")
    require(resolved(config["init_checkpoint"]) == expected_path, "Config initialization source differs")
    expected = {
        "checkpoint": str(expected_path), "sha256": source["sha256"],
        "source_epoch": 13, "selected_source": "ema", "source_image_size": 112,
        "source_full_train": full,
        "source_dataset_fingerprint": data["full_fingerprint" if full else "holdout_fingerprint"],
        "optimizer_and_schedule_reset": True,
    }
    require(config.get("initialization") == expected, "Warm-start metadata mismatch")


def audit(output, validation_only):
    import torch

    torch.set_num_threads(2)
    data = data_contract()
    plan, selection = read_json(output / "plan.json"), read_json(output / "selection.json")
    require(plan["candidate_models"] == CANDIDATES and plan["baseline_models"] == BASELINE, "Declared candidates changed")
    require(plan["image_sizes"] == NEW and plan["baseline_validation_correct"] == 2973, "Wrong resolution/baseline plan")
    require(plan["dataset_fingerprint"] == data["holdout_fingerprint"], "Plan split differs")
    require(selection["plan_sha256"] == sha256(output / "plan.json"), "Plan changed after selection")
    for path, digest in OLD_CSV_HASHES.items():
        require(sha256(path) == digest == plan["protected_submission_hashes"][str(path)], f"Existing CSV changed: {path}")

    prior_audit = read_json(PRIOR / "audit.json")
    require(prior_audit["passed"] is True, "Prior audit did not pass")
    old_holdout_hashes = {row["model"]: row["checkpoint_sha256"] for row in prior_audit["holdout_models"]}
    old_full_hashes = {row["model"]: row["sha256"] for row in prior_audit["full_models"]}
    paths = sources(output)
    for full in (False, True):
        source = plan["initialization"]["full" if full else "holdout"]
        source_path = PRIOR / ("full/detail_seed44/final.pt" if full else "detail_seed44/best.pt")
        require(resolved(source["path"]) == source_path, "Initialization path mismatch")
        digest = sha256(source_path)
        expected_hash = (old_full_hashes if full else old_holdout_hashes)["detail_seed44"]
        require(digest == source["sha256"] == expected_hash, "Seed44 initialization weights changed")
        state = torch.load(source_path, map_location="cpu", weights_only=False)
        check_scope(state["config"], data, full, 18)
        require(state["epoch"] == 13 and state["selected_source"] == "ema" and state["class_to_index"] == LABELS,
                "Initialization is not selected seed44 EMA13")
        require(state["config"]["image_size"] == 112 and state["config"]["seed"] == 44, "Wrong seed44 source settings")
        if full:
            require(state["validation_accuracy"] is None, "Full source claims validation")
        del state

    configs, all_probabilities, holdout_records = {}, {}, []
    for name, (directory, probability_path, _) in paths.items():
        config = read_json(directory / "config.json")
        configs[name] = config
        check_scope(config, data, False, 6 if name in NEW else 18)
        require(config["stop_after_epochs"] == 0, f"Incomplete holdout plan: {name}")
        if name in NEW:
            check_init(config, plan, data, False)
            check_plan_arguments(plan["common_arguments"] + ["--image-size", str(NEW[name])], config)
            require(config["lr"] == 3e-5 and config["seed"] == 44 and config["architecture"] == "resnet18"
                    and config["no_maxpool"] is True, "Wrong low-LR adaptation recipe")
        checkpoint_path = directory / "best.pt"
        digest = sha256(checkpoint_path)
        require(digest == selection["holdout_checkpoint_hashes"][name], f"Heldout checkpoint changed: {name}")
        if name in BASELINE:
            require(digest == old_holdout_hashes[name] == plan["old_holdout_checkpoint_hashes"][name], f"Old heldout weights changed: {name}")
        summary, best = read_json(directory / "summary.json"), read_json(directory / "best_metrics.json")
        require(summary["epochs_completed"] == summary["planned_epochs"] == config["epochs"], f"Incomplete study: {name}")
        require(summary["full_train"] is False and not summary["training_is_partial"] and not summary["validation_is_partial"], f"Invalid holdout summary: {name}")
        history = check_history(directory, config, full=False)
        best_row = max(history, key=lambda row: row["val_accuracy"])
        require(best["epoch"] == summary["best_epoch"] == best_row["epoch"], f"Best epoch mismatch: {name}")
        require(best["accuracy"] == summary["best_validation_accuracy"] == best_row["val_accuracy"], f"Best accuracy mismatch: {name}")
        require(best["selected_source"] == best_row["selected_source"] and best["num_samples"] == len(data["validation"]), f"Selected source/sample count mismatch: {name}")
        require(best == selection["checkpoint_selection"][name], f"Selection metrics changed: {name}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(state["config"] == config and state["class_to_index"] == LABELS, f"Checkpoint config/map mismatch: {name}")
        require(state["epoch"] == best["epoch"] and state["selected_source"] == best["selected_source"]
                and state["validation_accuracy"] == best["accuracy"], f"Checkpoint best metadata mismatch: {name}")
        tensors = strict_cpu_load(state)
        no_tta = read_predictions(directory / "validation_predictions.npz", data["val_ids"], data["val_labels"])
        computed = metrics(data["val_labels"], no_tta["predictions"])
        require(computed["accuracy"] == best["accuracy"] and computed["confusion_matrix"] == best["confusion_matrix"], f"Best saved predictions mismatch: {name}")
        prediction = read_predictions(probability_path, data["val_ids"], data["val_labels"])
        metadata_path = probability_path.with_name(probability_path.name.replace(".probabilities.npz", ".json"))
        metadata = check_inference_metadata(metadata_path, [checkpoint_path], [state], prediction, "val")
        computed = metrics(data["val_labels"], prediction["predictions"])
        require(computed["accuracy"] == metadata["validation_metrics"]["accuracy"]
                and computed["confusion_matrix"] == metadata["validation_metrics"]["confusion_matrix"], f"TTA metrics mismatch: {name}")
        all_probabilities[name] = prediction["probabilities"]
        holdout_records.append({"model": name, "sha256": digest, "epoch": best["epoch"], "source": best["selected_source"],
                                "tta_accuracy": computed["accuracy"], "strict_cpu_loaded_tensors": tensors})
        del state
    diff = differences(configs["tune112"], configs["tune160"], RUN_FIELDS)
    require(diff == {"image_size": [112, 160]}, f"Paired runs changed more than image size: {diff}")

    choices, candidate_arrays = [], []
    for names in CANDIDATES:
        averaged = np.mean([all_probabilities[name] for name in names], axis=0)
        correct = int((averaged.argmax(1) == data["val_labels"]).sum())
        candidate_arrays.append(averaged)
        choices.append({"models": names, "correct": correct, "accuracy": correct / len(data["val_labels"])})
    require(choices == selection["candidate_scores"], "Candidate means/scores differ")
    require(choices[0]["correct"] == 2973 and choices[0] == selection["baseline"], "Baseline did not reproduce")
    best_index = min(range(len(choices)), key=lambda index: (-choices[index]["correct"], len(choices[index]["models"]), index))
    if choices[best_index]["correct"] <= 2973:
        best_index = 0
    selected = choices[best_index]
    improved = selected["correct"] > 2973
    require(selection["selected"] == selected and selection["improved"] is improved, "Selection/tie rule differs")
    require(selection["validation_samples"] == len(data["val_labels"]), "Wrong validation size")
    previous = read_predictions(PRIOR / "selected_validation.npz", data["val_ids"], data["val_labels"])
    require(np.allclose(candidate_arrays[0], previous["probabilities"], atol=1e-7, rtol=0), "Baseline probabilities changed")
    selected_saved = read_predictions(output / "selected_validation.npz", data["val_ids"], data["val_labels"])
    require(np.allclose(candidate_arrays[best_index], selected_saved["probabilities"], atol=1e-7, rtol=0), "Selected mean probabilities differ")
    computed = metrics(data["val_labels"], selected_saved["predictions"])
    require(computed["accuracy"] == selection["selected_metrics"]["accuracy"]
            and computed["confusion_matrix"] == selection["selected_metrics"]["confusion_matrix"], "Selected metrics differ")
    old_ok = previous["predictions"] == data["val_labels"]
    new_ok = selected_saved["predictions"] == data["val_labels"]
    paired = {"old_wrong_new_correct": int((~old_ok & new_ok).sum()), "old_correct_new_wrong": int((old_ok & ~new_ok).sum())}
    require(paired == selection["paired_counts"], "Paired comparison differs")
    gain = 100 * (selected["accuracy"] - choices[0]["accuracy"])
    require(np.isclose(selection["gain_percentage_points"], gain, rtol=0, atol=1e-10), "Gain differs")
    result = {"passed": True, "stage": "validation_only" if validation_only else "complete", "improved": improved,
              "single_factor_changes": diff, "selected": selected, "candidate_scores": choices,
              "gain_percentage_points": gain, "paired_counts": paired, "holdout_models": holdout_records,
              "full_models": [], "new_submission": None, "previous_submissions_preserved": True,
              "scope": "CPU provenance and saved-probability audit; repeated model-selection holdout, no independent test-score claim.",
              "audit_source_sha256": sha256(__file__)}
    if validation_only:
        return result

    completion = read_json(output / "completion.json")
    if not improved:
        require(completion["status"] == "complete_no_improvement", "Wrong no-improvement completion")
        require(resolved(completion["keep_submission"]) == PRIOR / "submission_improved.csv", "Wrong preserved submission")
        require(not (output / "submission_improved.csv").exists(), "Unexpected submission without improvement")
        return result
    require(completion["status"] == "complete" and completion["selected_models"] == selected["models"], "Wrong completion/models")
    require(completion["holdout_selection_accuracy"] == selected["accuracy"], "Wrong completion score")
    final_paths = [paths[name][2] / "final.pt" for name in selected["models"]]
    require([resolved(path) for path in completion["full_checkpoints"]] == final_paths, "Wrong final checkpoints")
    states_metadata = []
    for name, final_path in zip(selected["models"], final_paths):
        directory = final_path.parent
        config, summary = read_json(directory / "config.json"), read_json(directory / "summary.json")
        check_scope(config, data, True, 6 if name in NEW else 18)
        require(not differences(configs[name], config, RUN_FIELDS | FULL_FIELDS), f"Full refit changed recipe: {name}")
        if name in NEW:
            check_init(config, plan, data, True)
        digest = sha256(final_path)
        require(digest == completion["full_checkpoint_hashes"][str(final_path)], f"Final checkpoint hash changed: {name}")
        if name in BASELINE:
            require(digest == old_full_hashes[name], f"Reused full weights changed: {name}")
        best = selection["checkpoint_selection"][name]
        epoch, source = best["epoch"], best["selected_source"]
        require(config["stop_after_epochs"] == summary["stop_after_epochs"] == summary["epochs_completed"] == epoch,
                f"Full refit epoch mismatch: {name}")
        require(summary["planned_epochs"] == config["epochs"] and summary["full_train"] is True
                and summary["best_epoch"] is None and summary["best_validation_accuracy"] is None
                and not summary["training_is_partial"] and not summary["validation_is_partial"], f"Invalid full summary: {name}")
        history = check_history(directory, config, full=True)
        state = torch.load(final_path, map_location="cpu", weights_only=False)
        require(state["config"] == config and state["class_to_index"] == LABELS, f"Full checkpoint config/map differs: {name}")
        require(state["epoch"] == epoch and state["selected_source"] == source and state["validation_accuracy"] is None,
                f"Full checkpoint source/epoch/score mismatch: {name}")
        weight_audit = check_saved_weights(state, final_path)
        tensors = strict_cpu_load(state)
        states_metadata.append({"epoch": epoch, "selected_source": source})
        result["full_models"].append({"model": name, "checkpoint": str(final_path), "sha256": digest,
                                      "epoch": epoch, "source": source, "source_weight_audit": weight_audit,
                                      "strict_cpu_loaded_tensors": tensors,
                                      "skipped_optimizer_steps": sum(row.get("skipped_optimizer_steps", 0) for row in history)})
        del state
    submission = output / "submission_improved.csv"
    require(resolved(completion["submission"]) == submission and completion["sha256"] == sha256(submission), "Submission path/hash mismatch")
    ids, labels = read_csv(submission)
    require(np.array_equal(ids, data["test_ids"]), "CSV does not match template IDs/order")
    prediction = read_predictions(submission.with_suffix(".probabilities.npz"), data["test_ids"])
    require(np.array_equal(labels, prediction["predictions"]), "CSV differs from probability argmax")
    check_inference_metadata(submission.with_suffix(".json"), final_paths, states_metadata, prediction, "test")
    result["new_submission"] = {"path": str(submission), "sha256": sha256(submission), "rows": len(ids),
                                "ids_match_template_in_order": True, "csv_matches_probability_argmax": True,
                                "label_counts": {name: int((labels == label).sum()) for name, label in LABELS.items()}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/late_20261018/resolution")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    target = output / ("validation_audit.json" if args.validation_only else "audit.json")
    try:
        result = audit(output, args.validation_only)
    except Exception as error:
        target.write_text(json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "stage", "improved", "selected", "gain_percentage_points", "new_submission")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
