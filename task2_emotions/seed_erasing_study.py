"""Two predeclared one-factor experiments; refit only a measured improvement.

Run with the project Python: seed_erasing_study.py --stage all
The original fixed holdout is reused for model selection, never test labels.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from improve import (ROOT, checked_predictions, dataset_contract, probabilities,
                     read, run, write)
from model import CLASS_TO_INDEX, classification_metrics

PREVIOUS = ROOT / "outputs/improvement_20261016"
DEFAULT_OUTPUT = ROOT / "outputs/seed_erasing_20261016"
NEW = ("detail_seed44", "detail_erasing")
BASELINE = ("old_mixup", "detail_seed43", "resnet34")
# Declare a small candidate set before either new experiment is evaluated.
CANDIDATES = [
    list(BASELINE), ["detail_seed43"], ["detail_seed44"], ["detail_erasing"],
    ["detail_seed43", "detail_seed44"], ["detail_seed43", "detail_erasing"],
    [*BASELINE, "detail_seed44"], [*BASELINE, "detail_erasing"],
    [*BASELINE, *NEW],
]
COMMON = ["--architecture", "resnet18", "--no-maxpool", "--epochs", "18",
          "--batch-size", "128", "--image-size", "112", "--lr", "0.0003",
          "--weight-decay", "0.001", "--label-smoothing", "0.05", "--dropout", "0.3",
          "--mixup-alpha", "0.2", "--mixup-prob", "1", "--device", "cuda", "--log-every", "0"]
EXTRA = {"detail_seed44": ["--seed", "44"],
         "detail_erasing": ["--seed", "43", "--erasing-prob", "0.25",
                            "--erasing-scale-min", "0.02", "--erasing-scale-max", "0.08"]}
DEFAULTS = {"no_maxpool": False, "architecture": "resnet18", "erasing_prob": 0.0,
            "erasing_scale_min": 0.02, "erasing_scale_max": 0.08}
MATERIAL = ("architecture", "no_maxpool", "epochs", "batch_size", "image_size", "lr",
            "head_lr_multiplier", "weight_decay", "label_smoothing", "dropout",
            "mixup_alpha", "mixup_prob", "warmup_epochs", "ema_decay", "seed",
            "workers", "threads", "no_pretrained", "no_amp", "deterministic",
            "max_train_batches", "max_val_batches", "erasing_prob",
            "erasing_scale_min", "erasing_scale_max")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paths(output):
    return {
        "old_mixup": (ROOT / "outputs/resnet18_mixup_s43", ROOT / "outputs/validation_mixup_tta.probabilities.npz", ROOT / "outputs/final_models/mixup"),
        "detail_seed43": (PREVIOUS / "resnet18_detail", PREVIOUS / "resnet18_detail_tta.probabilities.npz", PREVIOUS / "full/resnet18_detail"),
        "resnet34": (PREVIOUS / "resnet34", PREVIOUS / "resnet34_tta.probabilities.npz", PREVIOUS / "full/resnet34"),
        **{name: (output / name, output / f"{name}_tta.probabilities.npz", output / "full" / name) for name in NEW},
    }


def check_config(config, reference, data, full=False):
    for key in MATERIAL:
        if config.get(key, DEFAULTS.get(key)) != reference.get(key, DEFAULTS.get(key)):
            raise ValueError(f"Training configuration mismatch: {key}")
    expected = {"full_train": full, "class_to_index": CLASS_TO_INDEX,
                "dataset_fingerprint": data["full_fingerprint" if full else "holdout_fingerprint"],
                "train_samples": data["full_samples" if full else "train_samples"],
                "validation_samples": 0 if full else data["validation_samples"],
                "training_is_partial": False, "validation_is_partial": False}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Dataset/provenance mismatch: {key}")


def expected_new(name):
    config = read(PREVIOUS / "resnet18_detail/config.json")
    config.update(erasing_prob=0.0, erasing_scale_min=0.02, erasing_scale_max=0.08)
    config.update(seed=44) if name == "detail_seed44" else config.update(erasing_prob=0.25)
    return config


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    plan = {"common_arguments": COMMON, "experiment_arguments": EXTRA, "candidate_models": CANDIDATES,
            "baseline_models": list(BASELINE), "baseline_validation_correct": 2953,
            "reported_kaggle_score": {"value": 0.7220, "source": "user reported; not independently retrieved"},
            "single_factor_changes": {"detail_seed44": {"seed": [43, 44]}, "detail_erasing": {"erasing_prob": [0.0, 0.25]}},
            "erasing": {"area_fraction": [0.02, 0.08], "aspect_ratio": [0.5, 2.0], "fill": "per-image grayscale mean", "training_only": True},
            "selection_rule": "Fixed equal-weight candidates; highest holdout correct count, then fewer models; keep baseline on ties with baseline.",
            "checkpoint_rule": "Best raw/EMA checkpoint by no-TTA validation accuracy, then evaluate fixed horizontal-flip TTA.",
            "scope": "Repeated selection holdout, not an untouched test. No guarantee of online improvement.",
            "original_submission_sha256": sha(ROOT / "outputs/submission.csv"),
            "previous_submission_sha256": sha(PREVIOUS / "submission_improved.csv"),
            "dataset_fingerprint": dataset_contract()["holdout_fingerprint"]}
    plan_path = output / "plan.json"
    if plan_path.exists():
        if read(plan_path) != plan:
            raise ValueError("Plan changed; use a fresh output directory")
    else:
        write(plan_path, plan)
    return plan


def check_holdout(run_dir, reference, data):
    import torch
    config, summary, metrics = (read(run_dir / f"{name}.json") for name in ("config", "summary", "best_metrics"))
    check_config(config, reference, data)
    assert summary["epochs_completed"] == summary["planned_epochs"] == 18
    assert not summary["full_train"] and not summary["training_is_partial"] and not summary["validation_is_partial"]
    assert metrics["num_samples"] == data["validation_samples"]
    assert metrics["epoch"] == summary["best_epoch"] and metrics["accuracy"] == summary["best_validation_accuracy"]
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    check_config(checkpoint["config"], config, data)
    assert checkpoint["epoch"] == metrics["epoch"] and checkpoint["validation_accuracy"] == metrics["accuracy"]
    assert checkpoint["selected_source"] == metrics["selected_source"] and checkpoint["class_to_index"] == CLASS_TO_INDEX
    return metrics


def validation(output):
    plan, data = prepare(output), dataset_contract()
    source_paths, metrics = paths(output), {}
    for name in NEW:
        run_dir, probability_path, _ = source_paths[name]
        if not (run_dir / "summary.json").exists():
            resume = ["--resume", run_dir / "last.pt"] if (run_dir / "last.pt").exists() else []
            run("-u", "train.py", "--output-dir", run_dir, *COMMON, *EXTRA[name], *resume)
        metrics[name] = check_holdout(run_dir, expected_new(name), data)
        run("-u", "predict.py", "--checkpoints", run_dir / "best.pt", "--split", "val", "--tta",
            "--device", "cuda", "--output", output / f"{name}_tta.csv")
        run("plot_results.py", "--run-dir", run_dir)
    predictions = {}
    for name, (run_dir, p_path, _) in source_paths.items():
        if name not in NEW:
            metrics[name] = check_holdout(run_dir, read(run_dir / "config.json"), data)
        predictions[name] = probabilities(p_path)
        checked_predictions(predictions[name], name, data)
    labels = data["validation_labels"]
    choices = []
    for names in CANDIDATES:
        p = np.mean([predictions[name]["probabilities"] for name in names], axis=0)
        choices.append({"models": names, "correct": int((p.argmax(1) == labels).sum()),
                        "accuracy": float((p.argmax(1) == labels).mean())})
    baseline = choices[0]
    assert baseline["correct"] == plan["baseline_validation_correct"]
    best = sorted(choices, key=lambda c: (-c["correct"], len(c["models"]), c["models"]))[0]
    selected = best if best["correct"] > baseline["correct"] else baseline
    selected_p = np.mean([predictions[name]["probabilities"] for name in selected["models"]], axis=0)
    base_p = np.mean([predictions[name]["probabilities"] for name in BASELINE], axis=0)
    previous = probabilities(PREVIOUS / "selected_validation.npz")
    checked_predictions(previous, "previous selected ensemble", data)
    assert np.allclose(base_p, previous["probabilities"], atol=1e-7, rtol=0)
    old_ok, new_ok = base_p.argmax(1) == labels, selected_p.argmax(1) == labels
    np.savez_compressed(output / "selected_validation.npz", ids=data["validation_ids"], labels=labels,
                        probabilities=selected_p, predictions=selected_p.argmax(1))
    result = {"plan_sha256": sha(output / "plan.json"), "validation_samples": len(labels),
              "candidate_scores": choices, "selected": selected, "baseline": baseline,
              "improved": selected["correct"] > baseline["correct"],
              "gain_percentage_points": 100 * (selected["accuracy"] - baseline["accuracy"]),
              "paired_counts": {"old_wrong_new_correct": int((~old_ok & new_ok).sum()),
                                "old_correct_new_wrong": int((old_ok & ~new_ok).sum())},
              "selected_metrics": classification_metrics(labels, selected_p.argmax(1)),
              "checkpoint_selection": metrics,
              "holdout_checkpoint_hashes": {name: sha(v[0] / "best.pt") for name, v in source_paths.items()},
              "scope": plan["scope"]}
    write(output / "selection.json", result)
    print({"selected": selected, "gain_pp": result["gain_percentage_points"]}, flush=True)


def final(output):
    import torch
    plan, data, selection = prepare(output), dataset_contract(), read(output / "selection.json")
    assert selection["plan_sha256"] == sha(output / "plan.json")
    if not selection["improved"]:
        write(output / "completion.json", {"status": "complete_no_improvement", "keep_submission": str(PREVIOUS / "submission_improved.csv")})
        print("No improvement over current validation baseline; retaining previous submission.", flush=True)
        return
    selected = selection["selected"]["models"]
    assert selected in CANDIDATES
    source_paths, components, checkpoints = paths(output), [], []
    for name in selected:
        holdout_dir, probability_path, run_dir = source_paths[name]
        reference = expected_new(name) if name in NEW else read(holdout_dir / "config.json")
        metrics = check_holdout(holdout_dir, reference, data)
        assert metrics == selection["checkpoint_selection"][name]
        assert sha(holdout_dir / "best.pt") == selection["holdout_checkpoint_hashes"][name]
        pred = probabilities(probability_path)
        checked_predictions(pred, name, data)
        components.append(pred["probabilities"])
    reconstructed = np.mean(components, axis=0)
    saved = probabilities(output / "selected_validation.npz")
    checked_predictions(saved, "selected", data)
    assert np.allclose(reconstructed, saved["probabilities"], atol=1e-7, rtol=0)
    assert int((reconstructed.argmax(1) == data["validation_labels"]).sum()) == selection["selected"]["correct"] > plan["baseline_validation_correct"]
    for name in selected:
        holdout_dir, _, run_dir = source_paths[name]
        metrics = selection["checkpoint_selection"][name]
        if name in NEW and not (run_dir / "summary.json").exists():
            resume = ["--resume", run_dir / "last.pt"] if (run_dir / "last.pt").exists() else []
            run("-u", "train.py", "--output-dir", run_dir, *COMMON, *EXTRA[name],
                "--full-train", "--stop-after-epochs", metrics["epoch"], *resume)
        config, summary = read(run_dir / "config.json"), read(run_dir / "summary.json")
        check_config(config, read(holdout_dir / "config.json"), data, full=True)
        assert config["stop_after_epochs"] == summary["epochs_completed"] == metrics["epoch"]
        assert summary["full_train"] and summary["best_validation_accuracy"] is None and summary["best_epoch"] is None
        final_path = run_dir / "final.pt"
        state = torch.load(final_path, map_location="cpu", weights_only=False)
        check_config(state["config"], config, data, full=True)
        assert state["epoch"] == metrics["epoch"] and state["validation_accuracy"] is None and state["class_to_index"] == CLASS_TO_INDEX
        if name in NEW and metrics["selected_source"] == "raw" and state["selected_source"] != "raw":
            last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            assert last["epoch"] == metrics["epoch"]
            state.update(model_state=last["model_state"], selected_source="raw")
            torch.save(state, final_path)
        assert state["selected_source"] == metrics["selected_source"]
        checkpoints.append(final_path)
        if name in NEW:
            run("plot_results.py", "--run-dir", run_dir)
    submission = output / "submission_improved.csv"
    run("-u", "predict.py", "--checkpoints", *checkpoints, "--tta", "--device", "cuda", "--output", submission)
    run("verify_submission.py", submission, "--json", output / "submission_improved.check.json")
    assert sha(PREVIOUS / "submission_improved.csv") == plan["previous_submission_sha256"]
    assert sha(ROOT / "outputs/submission.csv") == plan["original_submission_sha256"]
    write(output / "completion.json", {"status": "complete", "submission": str(submission), "sha256": sha(submission),
          "holdout_selection_accuracy": selection["selected"]["accuracy"], "selected_models": selected,
          "full_checkpoints": list(map(str, checkpoints)), "kaggle_status": "not_uploaded", "new_kaggle_score": None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "validation", "final", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    prepare(output)
    if args.stage in ("validation", "all"):
        validation(output)
    if args.stage in ("final", "all"):
        final(output)


if __name__ == "__main__":
    main()
