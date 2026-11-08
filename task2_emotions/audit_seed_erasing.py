"""Independently audit the two controlled experiments and any improved CSV on CPU.

Run after seed_erasing_study.py completes. --validation-only checks the finished
holdout study before full refits and writes validation_audit.json instead.
No training, GPU use, model forward pass, or upload occurs. Saved probabilities
are checked independently; their scores are not independent test-set results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_improvement import (metrics, read_csv, read_json, read_predictions,
                               require, resolved, sha256)

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "outputs/improvement_20261016"
LABELS = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}
BASELINE = ["old_mixup", "detail_seed43", "resnet34"]
NEW = ["detail_seed44", "detail_erasing"]
# Kept here, independently of the orchestration module, to detect plan drift.
CANDIDATES = [BASELINE, ["detail_seed43"], ["detail_seed44"], ["detail_erasing"],
              ["detail_seed43", "detail_seed44"], ["detail_seed43", "detail_erasing"],
              [*BASELINE, "detail_seed44"], [*BASELINE, "detail_erasing"], [*BASELINE, *NEW]]
DEFAULTS = {"architecture": "resnet18", "no_maxpool": False, "erasing_prob": 0.0,
            "erasing_scale_min": 0.02, "erasing_scale_max": 0.08}
OLD_HASH = "4414f84a494aedb722eafaa3f9ce75ef2991286a8e869792d1cb7974d2bc8f09"
PREVIOUS_HASH = "1be9b64328089eb25de1959d79c5a6c6d1e4eeea50838f358a53f21d7ba47bfa"
RUN_METADATA = {"output_dir", "resume", "log_every"}
FULL_METADATA = {"full_train", "stop_after_epochs", "dataset_fingerprint", "train_samples", "validation_samples"}


def normalized(config):
    return {**DEFAULTS, **config}


def differences(first, second, ignored=RUN_METADATA):
    first, second = normalized(first), normalized(second)
    return {key: [first.get(key), second.get(key)]
            for key in sorted(first.keys() | second.keys())
            if key not in ignored and first.get(key) != second.get(key)}


def check_plan_arguments(arguments, config):
    """Compare declared CLI values with recorded values without using the runner."""
    config = normalized(config)
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        require(argument.startswith("--"), f"Malformed planned argument: {argument}")
        key = argument[2:].replace("-", "_")
        if key == "no_maxpool":
            expected = True
            index += 1
        else:
            require(index + 1 < len(arguments), f"Missing planned argument value: {argument}")
            raw = arguments[index + 1]
            try:
                expected = json.loads(raw)
            except json.JSONDecodeError:
                expected = raw
            index += 2
        require(key in config and config[key] == expected, f"Planned {argument} differs from actual config")


def sources(output):
    return {
        "old_mixup": (ROOT / "outputs/resnet18_mixup_s43", ROOT / "outputs/validation_mixup_tta.probabilities.npz", ROOT / "outputs/final_models/mixup"),
        "detail_seed43": (PREVIOUS / "resnet18_detail", PREVIOUS / "resnet18_detail_tta.probabilities.npz", PREVIOUS / "full/resnet18_detail"),
        "resnet34": (PREVIOUS / "resnet34", PREVIOUS / "resnet34_tta.probabilities.npz", PREVIOUS / "full/resnet34"),
        **{name: (output / name, output / f"{name}_tta.probabilities.npz", output / "full" / name) for name in NEW},
    }


def data_contract():
    with np.load(ROOT / "data/cache.npz", allow_pickle=False) as cache:
        ids = cache["train_ids"].astype(str)
        labels = cache["train_labels"].astype(np.int64)
        training = cache["train_indices"].astype(np.int64)
        validation = cache["val_indices"].astype(np.int64)
        test_ids = cache["test_ids"].astype(str)
    require((len(ids), len(training), len(validation), len(test_ids)) == (28275, 24034, 4241, 7065), "Unexpected dataset sizes")
    require(len(np.unique(ids)) == len(ids) and len(labels) == len(ids), "Invalid training IDs/labels")
    require(np.isin(labels, list(LABELS.values())).all(), "Invalid training labels")
    require(np.array_equal(np.sort(np.concatenate([training, validation])), np.arange(len(ids))), "Holdout indices overlap, repeat, or omit samples")
    prefix = labels.tobytes() + ids.tobytes()
    full_indices = np.arange(len(ids), dtype=np.int64)
    contract = {"ids": ids, "labels": labels, "training": training, "validation": validation,
                "val_ids": ids[validation], "val_labels": labels[validation], "test_ids": test_ids,
                "holdout_fingerprint": hashlib.sha256(prefix + training.tobytes() + validation.tobytes()).hexdigest(),
                "full_fingerprint": hashlib.sha256(prefix + full_indices.tobytes() + np.array([], dtype=np.int64).tobytes()).hexdigest()}
    template_ids, _ = read_csv(ROOT / "data/submission_template.csv")
    require(np.array_equal(template_ids, test_ids), "Cache and template test IDs differ")
    return contract


def check_scope(config, data, full):
    expected = {"full_train": full, "class_to_index": LABELS,
                "dataset_fingerprint": data["full_fingerprint" if full else "holdout_fingerprint"],
                "train_samples": len(data["ids"]) if full else len(data["training"]),
                "validation_samples": 0 if full else len(data["validation"]),
                "training_is_partial": False, "validation_is_partial": False,
                "max_train_batches": 0, "max_val_batches": 0, "epochs": 18}
    for key, value in expected.items():
        require(config.get(key) == value, f"Incorrect config {key}: {config.get(key)!r}, expected {value!r}")
    require(resolved(config["data"]) == ROOT / "data/cache.npz", "Config uses an unexpected data cache")


def check_history(directory, config, *, full):
    history = read_json(directory / "history.json")
    epochs = config["stop_after_epochs"] if full else config["epochs"]
    require([row["epoch"] for row in history] == list(range(1, epochs + 1)), f"Incomplete epoch sequence: {directory}")
    require(all(row["train_samples_seen"] == config["train_samples"] for row in history), f"Partial epoch: {directory}")
    for row in history:
        if full:
            require(all(row[key] is None for key in ("val_loss", "val_accuracy", "raw_val_accuracy", "ema_val_accuracy")), f"Full refit claims validation: {directory}")
        else:
            raw, ema = row["raw_val_accuracy"], row["ema_val_accuracy"]
            source = "ema" if ema is not None and ema > raw else "raw"
            require(row["selected_source"] == source and row["val_accuracy"] == (ema if source == "ema" else raw), f"Raw/EMA selection differs from history: {directory}")
    return history


def check_inference_metadata(path, checkpoint_paths, states, predictions, split):
    metadata = read_json(path)
    require(metadata["split"] == split and metadata["flip_tta"] is True and metadata["class_to_index"] == LABELS, f"Inference scope/TTA/mapping differs: {path}")
    require(metadata["rows"] == len(predictions["ids"]), f"Inference row count differs: {path}")
    require([resolved(item["path"]) for item in metadata["checkpoints"]] == checkpoint_paths, f"Inference checkpoints differ: {path}")
    for item, state in zip(metadata["checkpoints"], states):
        require(item["epoch"] == state["epoch"] and item["source"] == state["selected_source"], f"Inference epoch/source differs: {path}")
    for name, label in LABELS.items():
        require(metadata["prediction_counts"][name] == int((predictions["predictions"] == label).sum()), f"Inference label counts differ: {path}")
    return metadata


def strict_cpu_load(state):
    import torch
    from model import build_model

    config = normalized(state["config"])
    model = build_model(pretrained=False, architecture=config["architecture"], no_maxpool=config["no_maxpool"], dropout=config["dropout"])
    model.load_state_dict(state["model_state"], strict=True)
    for name, tensor in state["model_state"].items():
        require(tensor.device.type == "cpu", f"Non-CPU audit tensor: {name}")
        require(not tensor.is_floating_point() or torch.isfinite(tensor).all().item(), f"Nonfinite model tensor: {name}")
    return len(state["model_state"])


def audit(output, validation_only=False):
    import torch

    torch.set_num_threads(4)
    data, plan, selection = data_contract(), read_json(output / "plan.json"), read_json(output / "selection.json")
    require(plan["candidate_models"] == CANDIDATES and plan["baseline_models"] == BASELINE, "Predeclared candidate set differs")
    require(plan["baseline_validation_correct"] == 2953, "Baseline is not the 2953/4241 reference")
    expected_changes = {"detail_seed44": {"seed": [43, 44]}, "detail_erasing": {"erasing_prob": [0.0, 0.25]}}
    require(plan["single_factor_changes"] == expected_changes, "Plan does not describe the agreed one-factor studies")
    require(plan["erasing"] == {"area_fraction": [0.02, 0.08], "aspect_ratio": [0.5, 2.0], "fill": "per-image grayscale mean", "training_only": True}, "Erasing plan differs")
    require(plan["dataset_fingerprint"] == data["holdout_fingerprint"], "Plan uses a different holdout")
    require(selection["plan_sha256"] == sha256(output / "plan.json"), "Plan changed after selection")
    originals = []
    for path, key, expected_hash in ((ROOT / "outputs/submission.csv", "original_submission_sha256", OLD_HASH), (PREVIOUS / "submission_improved.csv", "previous_submission_sha256", PREVIOUS_HASH)):
        actual_hash = sha256(path)
        require(actual_hash == plan[key] == expected_hash, f"Previous submission changed: {path}")
        originals.append({"path": str(path), "sha256": actual_hash})

    paths = sources(output)
    reference = read_json(paths["detail_seed43"][0] / "config.json")
    require(reference["seed"] == 43 and reference["no_maxpool"] is True and reference["architecture"] == "resnet18", "Wrong controlled reference architecture/seed")
    configurations, probabilities, holdout_records, actual_changes = {}, {}, [], {}
    for name, (directory, p_path, _) in paths.items():
        config = read_json(directory / "config.json")
        configurations[name] = config
        check_scope(config, data, False)
        require(config["stop_after_epochs"] == 0, f"Holdout early termination requested: {name}")
        if name in NEW:
            diff = differences(reference, config)
            require(diff == expected_changes[name], f"Not a one-factor experiment {name}: {diff}")
            check_plan_arguments(plan["common_arguments"] + plan["experiment_arguments"][name], config)
            actual_changes[name] = diff
        summary, best = read_json(directory / "summary.json"), read_json(directory / "best_metrics.json")
        require(summary["epochs_completed"] == summary["planned_epochs"] == 18, f"Incomplete holdout run: {name}")
        require(summary["full_train"] is False and summary["training_is_partial"] is False and summary["validation_is_partial"] is False, f"Invalid holdout summary: {name}")
        history = check_history(directory, config, full=False)
        best_row = max(history, key=lambda row: row["val_accuracy"])
        require(best["epoch"] == summary["best_epoch"] == best_row["epoch"], f"Selected checkpoint is not first best epoch: {name}")
        require(best["accuracy"] == summary["best_validation_accuracy"] == best_row["val_accuracy"], f"Selected no-TTA accuracy differs: {name}")
        require(best["selected_source"] == best_row["selected_source"] and best["num_samples"] == 4241 and best["class_to_index"] == LABELS and best["validation_is_partial"] is False, f"Selected metrics provenance differs: {name}")
        require(selection["checkpoint_selection"][name] == best, f"Selection checkpoint record changed: {name}")
        checkpoint_path = (directory / "best.pt").resolve()
        checkpoint_hash = sha256(checkpoint_path)
        require(checkpoint_hash == selection["holdout_checkpoint_hashes"][name], f"Holdout checkpoint changed: {name}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(state["config"] == config and state["class_to_index"] == LABELS, f"Holdout checkpoint config/mapping differs: {name}")
        require(state["epoch"] == best["epoch"] and state["selected_source"] == best["selected_source"] and state["validation_accuracy"] == best["accuracy"], f"Holdout checkpoint selection differs: {name}")
        tensors = strict_cpu_load(state)
        no_tta = read_predictions(directory / "validation_predictions.npz", data["val_ids"], data["val_labels"])
        computed_no_tta = metrics(data["val_labels"], no_tta["predictions"])
        require(computed_no_tta["accuracy"] == best["accuracy"] and computed_no_tta["confusion_matrix"] == best["confusion_matrix"], f"No-TTA saved predictions differ from selected metrics: {name}")
        predicted = read_predictions(p_path, data["val_ids"], data["val_labels"])
        probabilities[name] = predicted["probabilities"]
        metadata = check_inference_metadata(p_path.with_name(p_path.name.replace(".probabilities.npz", ".json")), [checkpoint_path], [state], predicted, "val")
        computed_tta = metrics(data["val_labels"], predicted["predictions"])
        require(metadata["validation_metrics"]["accuracy"] == computed_tta["accuracy"] and metadata["validation_metrics"]["confusion_matrix"] == computed_tta["confusion_matrix"], f"TTA metadata metrics differ: {name}")
        holdout_records.append({"model": name, "checkpoint_sha256": checkpoint_hash, "epoch": best["epoch"], "source": best["selected_source"], "tta_metrics": computed_tta,
                                "strict_cpu_loaded_tensors": tensors, "skipped_optimizer_steps": sum(row.get("skipped_optimizer_steps", 0) for row in history)})
        del state

    recomputed, candidate_probabilities = [], []
    for names in CANDIDATES:
        averaged = np.mean([probabilities[name] for name in names], axis=0)
        candidate_probabilities.append(averaged)
        correct = int((averaged.argmax(1) == data["val_labels"]).sum())
        recomputed.append({"models": names, "correct": correct, "accuracy": correct / len(data["val_labels"])})
    require(selection["candidate_scores"] == recomputed, "Candidate scores differ from independently averaged probabilities")
    require(recomputed[0]["correct"] == 2953 and selection["baseline"] == recomputed[0], "Baseline differs from previous selected ensemble")
    previous = read_predictions(PREVIOUS / "selected_validation.npz", data["val_ids"], data["val_labels"])
    require(np.allclose(candidate_probabilities[0], previous["probabilities"], rtol=0, atol=1e-7), "Baseline component probabilities differ from previous study")
    best = sorted(recomputed, key=lambda item: (-item["correct"], len(item["models"]), item["models"]))[0]
    expected_selection = best if best["correct"] > 2953 else recomputed[0]
    require(selection["selected"] == expected_selection, "Selection does not follow predeclared best-count/fewer-models/tie policy")
    improved = expected_selection["correct"] > 2953
    require(selection["improved"] is improved and selection["validation_samples"] == 4241, "Improvement flag/validation size differs")
    selected_p = candidate_probabilities[CANDIDATES.index(expected_selection["models"])]
    saved = read_predictions(output / "selected_validation.npz", data["val_ids"], data["val_labels"])
    require(np.allclose(selected_p, saved["probabilities"], rtol=0, atol=1e-7), "Saved selected probabilities differ")
    selected_metrics = metrics(data["val_labels"], saved["predictions"])
    require(selection["selected_metrics"]["accuracy"] == selected_metrics["accuracy"] and selection["selected_metrics"]["confusion_matrix"] == selected_metrics["confusion_matrix"], "Selected classification metrics differ")
    gain = 100 * (expected_selection["accuracy"] - recomputed[0]["accuracy"])
    require(np.isclose(selection["gain_percentage_points"], gain, rtol=0, atol=1e-10), "Recorded gain differs")
    old_right, new_right = previous["predictions"] == data["val_labels"], saved["predictions"] == data["val_labels"]
    paired = {"old_wrong_new_correct": int((~old_right & new_right).sum()), "old_correct_new_wrong": int((old_right & ~new_right).sum())}
    require(selection["paired_counts"] == paired, "Paired comparisons differ")
    result = {"passed": True, "stage": "validation_only" if validation_only else "complete", "improved": improved,
              "single_factor_changes": actual_changes, "previous_submissions_preserved": originals,
              "dataset_fingerprint": data["holdout_fingerprint"], "candidate_scores": recomputed,
              "selected": expected_selection, "gain_percentage_points": gain, "paired_counts": paired,
              "holdout_models": holdout_records, "full_models": [], "new_submission": None,
              "scope": "CPU audit of saved predictions and checkpoint provenance; repeated model-selection holdout, not an untouched test.",
              "new_kaggle_score": None, "audit_source_sha256": sha256(__file__)}
    if validation_only:
        return result
    completion = read_json(output / "completion.json")
    submission = output / "submission_improved.csv"
    if not improved:
        require(completion["status"] == "complete_no_improvement", "No-improvement completion status differs")
        require(resolved(completion["keep_submission"]) == PREVIOUS / "submission_improved.csv", "No-improvement branch points to wrong previous submission")
        require(not submission.exists(), "Unexpected new submission for a study with no improvement")
        return result

    names = expected_selection["models"]
    require(completion["status"] == "complete" and completion["selected_models"] == names, "Final completion/model list differs")
    require(completion["holdout_selection_accuracy"] == expected_selection["accuracy"], "Completion holdout score differs")
    require(resolved(completion["submission"]) == submission and completion["sha256"] == sha256(submission), "Final submission location/hash differs")
    final_paths = [(paths[name][2] / "final.pt").resolve() for name in names]
    require([resolved(path) for path in completion["full_checkpoints"]] == final_paths, "Full checkpoint paths differ")
    final_state_metadata = []
    for name, final_path in zip(names, final_paths):
        config, summary = read_json(final_path.parent / "config.json"), read_json(final_path.parent / "summary.json")
        check_scope(config, data, True)
        require(not differences(configurations[name], config, RUN_METADATA | FULL_METADATA), f"Full refit changed recipe: {name}")
        best = selection["checkpoint_selection"][name]
        epoch, source = best["epoch"], best["selected_source"]
        require(config["stop_after_epochs"] == summary["stop_after_epochs"] == summary["epochs_completed"] == epoch and summary["planned_epochs"] == 18, f"Full refit epochs differ: {name}")
        require(summary["full_train"] is True and summary["best_epoch"] is None and summary["best_validation_accuracy"] is None and summary["training_is_partial"] is False and summary["validation_is_partial"] is False, f"Full refit summary scope differs: {name}")
        history = check_history(final_path.parent, config, full=True)
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        last = torch.load(final_path.parent / "last.pt", map_location="cpu", weights_only=False)
        require(final["config"] == last["config"] == config and final["class_to_index"] == last["class_to_index"] == LABELS, f"Full checkpoint config/mapping differs: {name}")
        require(final["epoch"] == last["epoch"] == epoch and final["selected_source"] == source and final["validation_accuracy"] is None, f"Full checkpoint epoch/source/score differs: {name}")
        require(last["history"] == history, f"Last checkpoint history differs: {name}")
        source_key = "ema_state" if source == "ema" else "model_state"
        require(source in {"raw", "ema"}, f"Unknown selected source: {name}")
        actual, expected = final["model_state"], last[source_key]
        require(isinstance(expected, dict) and actual.keys() == expected.keys(), f"Full weight keys differ: {name}")
        for key, value in actual.items():
            require(isinstance(value, torch.Tensor) and isinstance(expected[key], torch.Tensor) and value.dtype == expected[key].dtype and value.shape == expected[key].shape and torch.equal(value, expected[key]), f"Full weights do not exactly match last.pt/{source_key}: {name}/{key}")
        tensor_count = strict_cpu_load(final)
        final_state_metadata.append({"epoch": epoch, "selected_source": source})
        result["full_models"].append({"model": name, "checkpoint": str(final_path), "sha256": sha256(final_path), "epoch": epoch, "source": source,
                                      "train_samples": 28275, "validation_samples": 0, "all_source_tensors_equal": True,
                                      "strict_cpu_loaded_tensors": tensor_count, "skipped_optimizer_steps": sum(row.get("skipped_optimizer_steps", 0) for row in history)})
        del final, last
    csv_ids, csv_labels = read_csv(submission)
    require(np.array_equal(csv_ids, data["test_ids"]), "Final CSV ID order differs from template")
    test = read_predictions(submission.with_suffix(".probabilities.npz"), data["test_ids"])
    require(np.array_equal(csv_labels, test["predictions"]), "Final CSV labels differ from probability argmax")
    check_inference_metadata(submission.with_suffix(".json"), final_paths, final_state_metadata, test, "test")
    result["new_submission"] = {"path": str(submission), "sha256": sha256(submission), "rows": len(csv_ids),
                                "ids_match_template_in_order": True, "csv_matches_probability_argmax": True,
                                "label_counts": {name: int((csv_labels == label).sum()) for name, label in LABELS.items()}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/seed_erasing_20261016")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / ("validation_audit.json" if args.validation_only else "audit.json")
    try:
        result = audit(output, args.validation_only)
    except Exception as error:
        audit_path.write_text(json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "stage": result["stage"], "improved": result["improved"], "selected": result["selected"], "gain_pp": result["gain_percentage_points"], "submission": result["new_submission"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
