"""Independent CPU audit of EfficientNet selection, provenance and submission.

No model forward passes, training, GPU allocation, uploads or weight edits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_improvement import (read_json, read_csv, read_predictions, metrics,
                               require, resolved, sha256, check_saved_weights)
from audit_seed_erasing import (data_contract, differences, check_history,
                                check_inference_metadata, strict_cpu_load,
                                check_plan_arguments)
from audit_late_resolution import check_scope, OLD_CSV_HASHES

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT / "outputs/seed_erasing_20261016"
STRUCTURE = ROOT / "outputs/improvement_20261016"
LABELS = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}
NEW = "efficientnet_b0"
OLD3 = ["old_mixup", "detail_seed43", "resnet34"]
BASELINE = [*OLD3, "detail_seed44"]
CANDIDATES = [BASELINE, [NEW], [*BASELINE, NEW], [*OLD3, NEW], ["detail_seed44", NEW]]
MINIMUM_GAIN = 43
WEIGHTS = ROOT / ".cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth"
WEIGHTS_SHA = "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934"
RUN_FIELDS = {"output_dir", "resume", "log_every"}
FULL_FIELDS = {"full_train", "stop_after_epochs", "dataset_fingerprint", "train_samples", "validation_samples"}


def sources(output):
    return {
        "old_mixup": (ROOT / "outputs/resnet18_mixup_s43", ROOT / "outputs/validation_mixup_tta.probabilities.npz", ROOT / "outputs/final_models/mixup"),
        "detail_seed43": (STRUCTURE / "resnet18_detail", STRUCTURE / "resnet18_detail_tta.probabilities.npz", STRUCTURE / "full/resnet18_detail"),
        "resnet34": (STRUCTURE / "resnet34", STRUCTURE / "resnet34_tta.probabilities.npz", STRUCTURE / "full/resnet34"),
        "detail_seed44": (PRIOR / "detail_seed44", PRIOR / "detail_seed44_tta.probabilities.npz", PRIOR / "full/detail_seed44"),
        NEW: (output / NEW, output / f"{NEW}_tta.probabilities.npz", output / "full" / NEW),
    }


def check_recipe(config, plan):
    expected = {"architecture": NEW, "no_maxpool": False, "epochs": plan["epochs"],
                "image_size": plan["image_size"], "batch_size": 128, "lr": 0.0003,
                "head_lr_multiplier": 3.0, "weight_decay": 0.001, "label_smoothing": 0.05,
                "dropout": 0.3, "seed": 44, "mixup_alpha": 0.2, "mixup_prob": 1.0,
                "ema_decay": 0.995, "warmup_epochs": 1.0, "erasing_prob": 0.0,
                "no_pretrained": False, "no_amp": False, "init_checkpoint": None}
    for key, value in expected.items():
        require(config.get(key) == value, f"EfficientNet recipe mismatch: {key}")
    require(not config.get("initialization"), "Unexpected trained-checkpoint initialization")
    check_plan_arguments(list(map(str, plan["common_arguments"])), config)


def audit_holdout(name, directory, probability_path, plan, selection, data, prior_hash):
    import torch

    config = read_json(directory / "config.json")
    check_scope(config, data, False, plan["epochs"] if name == NEW else 18)
    require(config["stop_after_epochs"] == 0, f"Incomplete holdout plan: {name}")
    if name == NEW:
        check_recipe(config, plan)
    path = directory / "best.pt"
    digest = sha256(path)
    require(digest == selection["holdout_checkpoint_hashes"][name], f"Heldout weights changed: {name}")
    if name != NEW:
        require(digest == prior_hash == plan["old_holdout_checkpoint_hashes"][name], f"Reused heldout model changed: {name}")
    summary, best = read_json(directory / "summary.json"), read_json(directory / "best_metrics.json")
    require(summary["epochs_completed"] == summary["planned_epochs"] == config["epochs"], f"Incomplete holdout: {name}")
    require(summary["full_train"] is False and not summary["training_is_partial"] and not summary["validation_is_partial"], f"Invalid heldout summary: {name}")
    history = check_history(directory, config, full=False)
    best_row = max(history, key=lambda row: row["val_accuracy"])
    require(best["epoch"] == summary["best_epoch"] == best_row["epoch"], f"Wrong first-best epoch: {name}")
    require(best["accuracy"] == summary["best_validation_accuracy"] == best_row["val_accuracy"], f"Wrong best accuracy: {name}")
    require(best["selected_source"] == best_row["selected_source"] and best["num_samples"] == len(data["val_labels"]), f"Wrong best source/sample count: {name}")
    require(best == selection["checkpoint_selection"][name], f"Selected metadata changed: {name}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(state["config"] == config and state["class_to_index"] == LABELS, f"Checkpoint config/map mismatch: {name}")
    require(state["epoch"] == best["epoch"] and state["selected_source"] == best["selected_source"]
            and state["validation_accuracy"] == best["accuracy"], f"Checkpoint best metadata mismatch: {name}")
    tensors = strict_cpu_load(state)
    no_tta = read_predictions(directory / "validation_predictions.npz", data["val_ids"], data["val_labels"])
    computed = metrics(data["val_labels"], no_tta["predictions"])
    require(computed["accuracy"] == best["accuracy"] and computed["confusion_matrix"] == best["confusion_matrix"], f"No-TTA predictions mismatch: {name}")
    prediction = read_predictions(probability_path, data["val_ids"], data["val_labels"])
    metadata_path = probability_path.with_name(probability_path.name.replace(".probabilities.npz", ".json"))
    metadata = check_inference_metadata(metadata_path, [path], [state], prediction, "val")
    computed = metrics(data["val_labels"], prediction["predictions"])
    require(computed["accuracy"] == metadata["validation_metrics"]["accuracy"]
            and computed["confusion_matrix"] == metadata["validation_metrics"]["confusion_matrix"], f"TTA metrics mismatch: {name}")
    record = {"model": name, "sha256": digest, "epoch": best["epoch"], "source": best["selected_source"],
              "tta_accuracy": computed["accuracy"], "strict_cpu_loaded_tensors": tensors,
              "skipped_optimizer_steps": sum(row.get("skipped_optimizer_steps", 0) for row in history)}
    return config, prediction["probabilities"], record


def audit(output, validation_only):
    import torch

    torch.set_num_threads(2)
    data = data_contract()
    plan, selection = read_json(output / "plan.json"), read_json(output / "selection.json")
    require(plan["architecture"] == NEW and plan["epochs"] == 12 and plan["image_size"] == 160, "Plan differs from predeclared 12-epoch/160 recipe")
    require(plan["candidate_models"] == CANDIDATES and plan["baseline_models"] == BASELINE, "Candidate plan changed")
    require(plan["baseline_validation_correct"] == 2973 and plan["validation_samples"] == 4241, "Baseline differs")
    require(plan["minimum_gain_correct_for_submission"] == MINIMUM_GAIN, "Gain gate changed")
    require(np.isclose(plan["minimum_gain_percentage_points"], 100 * MINIMUM_GAIN / 4241, atol=1e-12, rtol=0), "Wrong gain threshold conversion")
    require(plan["dataset_fingerprint"] == data["holdout_fingerprint"], "Plan split differs")
    require(selection["plan_sha256"] == sha256(output / "plan.json"), "Plan changed after selection")
    require(sha256(WEIGHTS) == WEIGHTS_SHA, "Official ImageNet checkpoint hash differs")
    for path, digest in OLD_CSV_HASHES.items():
        require(sha256(path) == digest == plan["protected_submission_hashes"][str(path)], f"Prior submission changed: {path}")
    prior_audit = read_json(PRIOR / "audit.json")
    require(prior_audit["passed"] is True, "Prior audit did not pass")
    old_holdout_hashes = {row["model"]: row["checkpoint_sha256"] for row in prior_audit["holdout_models"]}
    old_full_hashes = {row["model"]: row["sha256"] for row in prior_audit["full_models"]}
    paths = sources(output)
    configs, probabilities, records = {}, {}, []
    for name, (directory, prediction_path, _) in paths.items():
        config, array, record = audit_holdout(name, directory, prediction_path, plan, selection, data, old_holdout_hashes.get(name))
        configs[name], probabilities[name] = config, array
        records.append(record)

    choices, candidate_arrays = [], []
    for names in CANDIDATES:
        averaged = np.mean([probabilities[name] for name in names], axis=0)
        correct = int((averaged.argmax(1) == data["val_labels"]).sum())
        choices.append({"models": names, "correct": correct, "accuracy": correct / len(data["val_labels"])})
        candidate_arrays.append(averaged)
    require(choices == selection["candidate_scores"], "Candidate means/scores differ")
    require(choices[0]["correct"] == 2973 and choices[0] == selection["baseline"], "Baseline did not reproduce")
    best_index = min(range(len(choices)), key=lambda index: (-choices[index]["correct"], len(choices[index]["models"]), index))
    best = choices[best_index]
    gain_correct = best["correct"] - 2973
    qualifies = gain_correct >= MINIMUM_GAIN
    selected_index = best_index if qualifies else 0
    selected = choices[selected_index]
    require(selection["best_observed"] == best and selection["selected"] == selected, "Selection/tie rule differs")
    require(selection["improved"] is qualifies and selection["meets_minimum_gain"] is qualifies, "Qualification flag differs")
    require(selection["minimum_gain_correct_for_submission"] == MINIMUM_GAIN and selection["observed_gain_correct"] == gain_correct, "Observed gain/gate differs")
    require(np.isclose(selection["observed_gain_percentage_points"], 100 * (best["accuracy"] - choices[0]["accuracy"]), atol=1e-10, rtol=0), "Observed percentage-point gain differs")
    require(selection["validation_samples"] == 4241, "Wrong validation count")
    previous = read_predictions(PRIOR / "selected_validation.npz", data["val_ids"], data["val_labels"])
    require(np.allclose(candidate_arrays[0], previous["probabilities"], atol=1e-7, rtol=0), "Baseline probabilities changed")
    saved = read_predictions(output / "selected_validation.npz", data["val_ids"], data["val_labels"])
    require(np.allclose(candidate_arrays[selected_index], saved["probabilities"], atol=1e-7, rtol=0), "Selected probabilities differ from equal mean")
    computed = metrics(data["val_labels"], saved["predictions"])
    require(computed["accuracy"] == selection["selected_metrics"]["accuracy"]
            and computed["confusion_matrix"] == selection["selected_metrics"]["confusion_matrix"], "Selected metrics differ")
    old_ok, new_ok = previous["predictions"] == data["val_labels"], saved["predictions"] == data["val_labels"]
    paired = {"old_wrong_new_correct": int((~old_ok & new_ok).sum()), "old_correct_new_wrong": int((old_ok & ~new_ok).sum())}
    require(paired == selection["paired_counts"], "Paired comparison differs")
    selected_gain = 100 * (selected["accuracy"] - choices[0]["accuracy"])
    require(np.isclose(selection["gain_percentage_points"], selected_gain, atol=1e-10, rtol=0), "Selected gain differs")
    result = {"passed": True, "stage": "validation_only" if validation_only else "complete", "meets_minimum_gain": qualifies,
              "minimum_gain_correct": MINIMUM_GAIN, "best_observed": best, "selected": selected,
              "observed_gain_correct": gain_correct, "candidate_scores": choices, "paired_counts": paired,
              "holdout_models": records, "full_models": [], "new_submission": None,
              "official_imagenet_checkpoint": {"path": str(WEIGHTS), "sha256": WEIGHTS_SHA},
              "previous_submissions_preserved": True,
              "scope": "CPU provenance/saved-probability audit of a practical new recipe; repeated selection holdout, not an untouched test.",
              "audit_source_sha256": sha256(__file__)}
    if validation_only:
        return result
    completion = read_json(output / "completion.json")
    if not qualifies:
        require(completion["status"] == "complete_no_qualifying_improvement", "Wrong no-qualification completion")
        require(completion["observed_gain_correct"] == gain_correct and completion["minimum_gain_correct_for_submission"] == MINIMUM_GAIN, "Wrong completion gate/result")
        require(resolved(completion["keep_submission"]) == PRIOR / "submission_improved.csv", "Wrong preserved submission")
        require(not (output / "submission_improved.csv").exists(), "Unexpected submission below gain gate")
        return result
    require(completion["status"] == "complete" and completion["selected_models"] == selected["models"], "Completion/model list mismatch")
    require(completion["holdout_selection_accuracy"] == selected["accuracy"], "Completion accuracy mismatch")
    final_paths = [paths[name][2] / "final.pt" for name in selected["models"]]
    require([resolved(path) for path in completion["full_checkpoints"]] == final_paths, "Wrong full checkpoint paths")
    metadata_states = []
    for name, path in zip(selected["models"], final_paths):
        config, summary = read_json(path.parent / "config.json"), read_json(path.parent / "summary.json")
        check_scope(config, data, True, plan["epochs"] if name == NEW else 18)
        require(not differences(configs[name], config, RUN_FIELDS | FULL_FIELDS), f"Full recipe differs: {name}")
        if name == NEW:
            check_recipe(config, plan)
        digest = sha256(path)
        require(digest == completion["full_checkpoint_hashes"][str(path)], f"Final hash changed: {name}")
        if name != NEW:
            require(digest == old_full_hashes[name], f"Old full model changed: {name}")
        selected_checkpoint = selection["checkpoint_selection"][name]
        epoch, source = selected_checkpoint["epoch"], selected_checkpoint["selected_source"]
        require(config["stop_after_epochs"] == summary["stop_after_epochs"] == summary["epochs_completed"] == epoch, f"Full epoch mismatch: {name}")
        require(summary["planned_epochs"] == config["epochs"] and summary["full_train"] is True
                and summary["best_epoch"] is None and summary["best_validation_accuracy"] is None
                and not summary["training_is_partial"] and not summary["validation_is_partial"], f"Wrong full summary: {name}")
        history = check_history(path.parent, config, full=True)
        state = torch.load(path, map_location="cpu", weights_only=False)
        require(state["config"] == config and state["class_to_index"] == LABELS, f"Full config/map differs: {name}")
        require(state["epoch"] == epoch and state["selected_source"] == source and state["validation_accuracy"] is None, f"Full epoch/source/score differs: {name}")
        weight_audit = check_saved_weights(state, path)
        tensors = strict_cpu_load(state)
        metadata_states.append({"epoch": epoch, "selected_source": source})
        result["full_models"].append({"model": name, "checkpoint": str(path), "sha256": digest, "epoch": epoch,
                                      "source": source, "source_weight_audit": weight_audit, "strict_cpu_loaded_tensors": tensors,
                                      "skipped_optimizer_steps": sum(row.get("skipped_optimizer_steps", 0) for row in history)})
        del state
    submission = output / "submission_improved.csv"
    require(resolved(completion["submission"]) == submission and completion["sha256"] == sha256(submission), "Submission path/hash mismatch")
    ids, labels = read_csv(submission)
    require(np.array_equal(ids, data["test_ids"]), "CSV IDs differ from template")
    test = read_predictions(submission.with_suffix(".probabilities.npz"), data["test_ids"])
    require(np.array_equal(labels, test["predictions"]), "CSV labels differ from probability argmax")
    check_inference_metadata(submission.with_suffix(".json"), final_paths, metadata_states, test, "test")
    result["new_submission"] = {"path": str(submission), "sha256": sha256(submission), "rows": len(ids),
                                "ids_match_template_in_order": True, "csv_matches_probability_argmax": True,
                                "label_counts": {name: int((labels == label).sum()) for name, label in LABELS.items()}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/late_20261018/efficientnet")
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
    print(json.dumps({key: result[key] for key in ("passed", "stage", "meets_minimum_gain", "best_observed", "selected", "new_submission")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
