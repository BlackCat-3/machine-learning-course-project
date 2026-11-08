"""Evaluate a new EfficientNet-B0 recipe and refit only a meaningful improvement.

The architecture/pretraining and input resolution differ from prior ResNets;
this is a practical recipe comparison, not a single-factor ablation claim.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from improve import ROOT, checked_predictions, dataset_contract, probabilities, read, run, write
from model import CLASS_TO_INDEX, classification_metrics
from seed_erasing_study import check_config as check_material_config, check_holdout as check_old_holdout
from late_resolution_study import paths as prior_paths

PRIOR = ROOT / "outputs/seed_erasing_20261016"
DEFAULT_OUTPUT = ROOT / "outputs/late_20261018/efficientnet"
NEW = "efficientnet_b0"
OLD3 = ["old_mixup", "detail_seed43", "resnet34"]
BASELINE = [*OLD3, "detail_seed44"]
CANDIDATES = [BASELINE, [NEW], [*BASELINE, NEW], [*OLD3, NEW], ["detail_seed44", NEW]]
MINIMUM_GAIN_CORRECT = 43


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def paths(output):
    old = prior_paths(output)
    return {**{name: old[name] for name in BASELINE},
            NEW: (output / NEW, output / f"{NEW}_tta.probabilities.npz", output / "full" / NEW)}


def common(plan):
    return ["--architecture", NEW, "--epochs", plan["epochs"], "--image-size", plan["image_size"],
            "--batch-size", "128", "--lr", "0.0003", "--head-lr-multiplier", "3",
            "--warmup-epochs", "1", "--weight-decay", "0.001", "--label-smoothing", "0.05",
            "--dropout", "0.3", "--seed", "44", "--mixup-alpha", "0.2", "--mixup-prob", "1",
            "--ema-decay", "0.995", "--erasing-prob", "0", "--device", "cuda", "--log-every", "0"]


def expected_new(plan):
    config = read(PRIOR / "detail_seed44/config.json")
    config.update(architecture=NEW, no_maxpool=False, epochs=plan["epochs"], image_size=plan["image_size"],
                  init_checkpoint=None, no_pretrained=False)
    config.pop("initialization", None)
    return config


def prepare(output, *, epochs=None, image_size=None):
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    saved = read(plan_path) if plan_path.exists() else {}
    epochs = epochs if epochs is not None else saved.get("epochs", 12)
    image_size = image_size if image_size is not None else saved.get("image_size", 160)
    require(isinstance(epochs, int) and epochs > 0 and isinstance(image_size, int) and image_size > 0,
            "epochs and image_size must be positive integers")
    data, sources = dataset_contract(), paths(output)
    protected = [ROOT / "outputs/submission.csv", ROOT / "outputs/improvement_20261016/submission_improved.csv",
                 PRIOR / "submission_improved.csv"]
    plan = {
        "architecture": NEW, "epochs": epochs, "image_size": image_size,
        "candidate_models": CANDIDATES, "baseline_models": BASELINE,
        "baseline_validation_correct": 2973, "validation_samples": 4241,
        "minimum_gain_correct_for_submission": MINIMUM_GAIN_CORRECT,
        "minimum_gain_percentage_points": MINIMUM_GAIN_CORRECT / 4241 * 100,
        "dataset_fingerprint": data["holdout_fingerprint"],
        "old_holdout_checkpoint_hashes": {name: sha(sources[name][0] / "best.pt") for name in BASELINE},
        "protected_submission_hashes": {str(path): sha(path) for path in protected},
        "recipe_scope": "Practical new recipe: EfficientNet-B0 official ImageNet initialization and chosen input resolution; not a single-factor ablation.",
        "checkpoint_rule": "Best raw/EMA checkpoint by no-TTA heldout accuracy; fixed candidate comparison uses original plus horizontal-flip TTA.",
        "selection_rule": "Highest correct count, then fewer models, then declared order. Keep baseline unless the best candidate gains at least 43 correct images.",
        "reported_kaggle_score": {"value": 0.722, "source": "user reported; not independently retrieved"},
        "scope": "Repeated model-selection holdout, not an untouched test. No guarantee of online improvement.",
    }
    plan["common_arguments"] = common(plan)
    if plan_path.exists():
        require(saved == plan, "Plan/settings/source changed; choose a fresh output directory")
    else:
        write(plan_path, plan)
    return plan, data


def check_new_config(config, plan, data, *, full=False, stop_epoch=0):
    check_material_config(config, expected_new(plan), data, full=full)
    require(config.get("stop_after_epochs") == stop_epoch, "Stop epoch mismatch")
    require(config.get("init_checkpoint") is None and not config.get("initialization"),
            "This recipe must initialize from official ImageNet weights, not another trained checkpoint")


def check_holdout(run_dir, name, plan, data):
    import torch
    if name != NEW:
        require(sha(run_dir / "best.pt") == plan["old_holdout_checkpoint_hashes"][name], "Existing heldout weights changed")
        return check_old_holdout(run_dir, read(run_dir / "config.json"), data)
    config, summary, metrics = [read(run_dir / f"{part}.json") for part in ("config", "summary", "best_metrics")]
    check_new_config(config, plan, data)
    require(summary["epochs_completed"] == summary["planned_epochs"] == plan["epochs"], "Incomplete EfficientNet validation run")
    require(not summary["full_train"] and not summary["training_is_partial"] and not summary["validation_is_partial"], "Invalid validation provenance")
    require(metrics["num_samples"] == data["validation_samples"], "Incomplete heldout metrics")
    require(metrics["epoch"] == summary["best_epoch"] and metrics["accuracy"] == summary["best_validation_accuracy"], "Best metrics/summary mismatch")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    check_new_config(checkpoint["config"], plan, data)
    require(checkpoint["epoch"] == metrics["epoch"] and checkpoint["class_to_index"] == CLASS_TO_INDEX, "Best checkpoint metadata mismatch")
    require(checkpoint["validation_accuracy"] == metrics["accuracy"] and checkpoint["selected_source"] == metrics["selected_source"], "Best raw/EMA selection mismatch")
    return metrics


def train_new(output, plan, *, full=False, stop_epoch=0):
    run_dir = output / "full" / NEW if full else output / NEW
    if not (run_dir / "summary.json").exists():
        resume = ["--resume", run_dir / "last.pt"] if (run_dir / "last.pt").exists() else []
        extra = ["--full-train", "--stop-after-epochs", stop_epoch] if full else []
        run("-u", "train.py", "--output-dir", run_dir, *common(plan), *extra, *resume)
    return run_dir


def validation(output):
    plan, data = prepare(output)
    sources = paths(output)
    new_dir = train_new(output, plan)
    metrics = {NEW: check_holdout(new_dir, NEW, plan, data)}
    run("-u", "predict.py", "--checkpoints", new_dir / "best.pt", "--split", "val", "--tta",
        "--device", "cuda", "--output", output / f"{NEW}_tta.csv")
    predictions = {}
    for name, (run_dir, p_path, _) in sources.items():
        if name != NEW:
            metrics[name] = check_holdout(run_dir, name, plan, data)
        predictions[name] = probabilities(p_path)
        checked_predictions(predictions[name], name, data)
    labels = data["validation_labels"]
    choices = []
    for names in CANDIDATES:
        p = np.mean([predictions[name]["probabilities"] for name in names], axis=0)
        correct = int((p.argmax(1) == labels).sum())
        choices.append({"models": names, "correct": correct, "accuracy": correct / len(labels)})
    baseline = choices[0]
    require(baseline["correct"] == plan["baseline_validation_correct"], "Existing baseline did not reproduce")
    best_index = min(range(len(choices)), key=lambda i: (-choices[i]["correct"], len(choices[i]["models"]), i))
    best = choices[best_index]
    qualifies = best["correct"] - baseline["correct"] >= plan["minimum_gain_correct_for_submission"]
    selected = best if qualifies else baseline
    baseline_p = np.mean([predictions[name]["probabilities"] for name in BASELINE], axis=0)
    prior = probabilities(PRIOR / "selected_validation.npz")
    checked_predictions(prior, "prior ensemble", data)
    require(np.allclose(baseline_p, prior["probabilities"], atol=1e-7, rtol=0), "Existing baseline probabilities changed")
    selected_p = np.mean([predictions[name]["probabilities"] for name in selected["models"]], axis=0)
    old_ok, new_ok = baseline_p.argmax(1) == labels, selected_p.argmax(1) == labels
    np.savez_compressed(output / "selected_validation.npz", ids=data["validation_ids"], labels=labels,
                        probabilities=selected_p, predictions=selected_p.argmax(1))
    selection = {
        "plan_sha256": sha(output / "plan.json"), "validation_samples": len(labels),
        "candidate_scores": choices, "baseline": baseline, "best_observed": best, "selected": selected,
        "improved": qualifies, "meets_minimum_gain": qualifies,
        "minimum_gain_correct_for_submission": plan["minimum_gain_correct_for_submission"],
        "observed_gain_correct": best["correct"] - baseline["correct"],
        "observed_gain_percentage_points": 100 * (best["accuracy"] - baseline["accuracy"]),
        "gain_percentage_points": 100 * (selected["accuracy"] - baseline["accuracy"]),
        "paired_counts": {"old_wrong_new_correct": int((~old_ok & new_ok).sum()),
                          "old_correct_new_wrong": int((old_ok & ~new_ok).sum())},
        "checkpoint_selection": metrics,
        "holdout_checkpoint_hashes": {name: sha(values[0] / "best.pt") for name, values in sources.items()},
        "selected_metrics": classification_metrics(labels, selected_p.argmax(1)), "scope": plan["scope"],
    }
    write(output / "selection.json", selection)
    print({"best_observed": best, "qualifies_for_new_submission": qualifies,
           "gain_correct": selection["observed_gain_correct"], "minimum_required": MINIMUM_GAIN_CORRECT}, flush=True)


def check_full(run_dir, holdout_dir, name, metrics, plan, data):
    import torch
    config, summary = read(run_dir / "config.json"), read(run_dir / "summary.json")
    epoch = metrics["epoch"]
    if name == NEW:
        check_new_config(config, plan, data, full=True, stop_epoch=epoch)
    else:
        check_material_config(config, read(holdout_dir / "config.json"), data, full=True)
    require(config["stop_after_epochs"] == summary["epochs_completed"] == epoch, "Incomplete/wrong full training epoch")
    require(summary["planned_epochs"] == config["epochs"] and summary["full_train"], "Full training plan mismatch")
    require(summary["best_epoch"] is None and summary["best_validation_accuracy"] is None, "Full fit must not claim independent validation")
    require(not summary["training_is_partial"] and not summary["validation_is_partial"], "Partial full training rejected")
    target = run_dir / "final.pt"
    state = torch.load(target, map_location="cpu", weights_only=False)
    if name == NEW:
        check_new_config(state["config"], plan, data, full=True, stop_epoch=epoch)
    else:
        check_material_config(state["config"], config, data, full=True)
    require(state["epoch"] == epoch and state["validation_accuracy"] is None and state["class_to_index"] == CLASS_TO_INDEX,
            "Full checkpoint provenance mismatch")
    if name == NEW and metrics["selected_source"] == "raw" and state["selected_source"] != "raw":
        last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        check_new_config(last["config"], plan, data, full=True, stop_epoch=epoch)
        require(last["epoch"] == epoch, "Last checkpoint epoch mismatch")
        state.update(model_state=last["model_state"], selected_source="raw")
        temporary = target.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(target)
    require(state["selected_source"] == metrics["selected_source"], "Full raw/EMA source differs from heldout selection")
    return target


def final(output):
    plan, data = prepare(output)
    selection = read(output / "selection.json")
    require(selection["plan_sha256"] == sha(output / "plan.json"), "Selection plan changed")
    if not selection["meets_minimum_gain"]:
        write(output / "completion.json", {"status": "complete_no_qualifying_improvement",
              "observed_gain_correct": selection["observed_gain_correct"],
              "minimum_gain_correct_for_submission": MINIMUM_GAIN_CORRECT,
              "keep_submission": str(PRIOR / "submission_improved.csv")})
        print("No candidate meets the predeclared improvement threshold; no full retraining or new submission.", flush=True)
        return
    selected, sources = selection["selected"]["models"], paths(output)
    require(selected in CANDIDATES and NEW in selected, "Undeclared or unchanged selected candidate")
    components = []
    for name in selected:
        holdout_dir, p_path, _ = sources[name]
        metrics = check_holdout(holdout_dir, name, plan, data)
        require(metrics == selection["checkpoint_selection"][name], "Heldout selection metadata changed")
        require(sha(holdout_dir / "best.pt") == selection["holdout_checkpoint_hashes"][name], "Heldout weights changed")
        pred = probabilities(p_path)
        checked_predictions(pred, name, data)
        components.append(pred["probabilities"])
    reconstructed = np.mean(components, axis=0)
    saved = probabilities(output / "selected_validation.npz")
    checked_predictions(saved, "selected ensemble", data)
    require(np.allclose(reconstructed, saved["probabilities"], atol=1e-7, rtol=0), "Selected predictions changed")
    correct = int((reconstructed.argmax(1) == data["validation_labels"]).sum())
    require(correct == selection["selected"]["correct"] and correct - plan["baseline_validation_correct"] >= MINIMUM_GAIN_CORRECT,
            "Selected gain no longer meets threshold")
    checkpoints = []
    for name in selected:
        holdout_dir, _, run_dir = sources[name]
        metrics = selection["checkpoint_selection"][name]
        if name == NEW:
            train_new(output, plan, full=True, stop_epoch=metrics["epoch"])
        checkpoints.append(check_full(run_dir, holdout_dir, name, metrics, plan, data))
    submission = output / "submission_improved.csv"
    run("-u", "predict.py", "--checkpoints", *checkpoints, "--tta", "--device", "cuda", "--output", submission)
    run("verify_submission.py", submission, "--json", output / "submission_improved.check.json")
    for path, digest in plan["protected_submission_hashes"].items():
        require(sha(path) == digest, "An existing submission changed")
    write(output / "completion.json", {"status": "complete", "submission": str(submission), "sha256": sha(submission),
          "holdout_selection_accuracy": selection["selected"]["accuracy"], "selected_models": selected,
          "full_checkpoints": list(map(str, checkpoints)), "full_checkpoint_hashes": {str(path): sha(path) for path in checkpoints},
          "kaggle_status": "not_uploaded", "new_kaggle_score": None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "validation", "final", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, help="Freeze before training; defaults to 12 or the existing plan")
    parser.add_argument("--image-size", type=int, help="Freeze before training; defaults to 160 or the existing plan")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    prepare(output, epochs=args.epochs, image_size=args.image_size)
    if args.stage in ("validation", "all"):
        validation(output)
    if args.stage in ("final", "all"):
        final(output)


if __name__ == "__main__":
    main()
