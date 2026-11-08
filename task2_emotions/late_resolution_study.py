"""Paired six-epoch low-LR adaptation at 112/160, then refit an improvement.

All candidates are declared before evaluation. The existing grouped holdout is
reused for selection, not claimed to be a new independent test set.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from improve import ROOT, checked_predictions, dataset_contract, probabilities, read, run, write
from model import CLASS_TO_INDEX, classification_metrics
from seed_erasing_study import check_config as check_material_config, check_holdout as check_old_holdout

PRIOR = ROOT / "outputs/seed_erasing_20261016"
STRUCTURE = ROOT / "outputs/improvement_20261016"
DEFAULT_OUTPUT = ROOT / "outputs/late_20261018/resolution"
OLD3 = ["old_mixup", "detail_seed43", "resnet34"]
BASELINE = [*OLD3, "detail_seed44"]
NEW = {"tune112": 112, "tune160": 160}
CANDIDATES = [BASELINE, ["tune112"], ["tune160"],
              [*OLD3, "tune112"], [*OLD3, "tune160"],
              [*BASELINE, "tune112"], [*BASELINE, "tune160"],
              ["detail_seed44", "tune160"]]
COMMON = ["--architecture", "resnet18", "--no-maxpool", "--epochs", "6",
          "--batch-size", "128", "--lr", "0.00003", "--head-lr-multiplier", "3",
          "--warmup-epochs", "1", "--weight-decay", "0.001",
          "--label-smoothing", "0.05", "--dropout", "0.3", "--seed", "44",
          "--mixup-alpha", "0.2", "--mixup-prob", "1", "--ema-decay", "0.995",
          "--erasing-prob", "0", "--device", "cuda", "--log-every", "0"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths(output):
    return {
        "old_mixup": (ROOT / "outputs/resnet18_mixup_s43", ROOT / "outputs/validation_mixup_tta.probabilities.npz", ROOT / "outputs/final_models/mixup"),
        "detail_seed43": (STRUCTURE / "resnet18_detail", STRUCTURE / "resnet18_detail_tta.probabilities.npz", STRUCTURE / "full/resnet18_detail"),
        "resnet34": (STRUCTURE / "resnet34", STRUCTURE / "resnet34_tta.probabilities.npz", STRUCTURE / "full/resnet34"),
        "detail_seed44": (PRIOR / "detail_seed44", PRIOR / "detail_seed44_tta.probabilities.npz", PRIOR / "full/detail_seed44"),
        **{name: (output / name, output / f"{name}_tta.probabilities.npz", output / "full" / name) for name in NEW},
    }


def init_path(full=False):
    return PRIOR / ("full/detail_seed44/final.pt" if full else "detail_seed44/best.pt")


def expected_new(name):
    reference = read(PRIOR / "detail_seed44/config.json")
    reference.update(epochs=6, image_size=NEW[name], lr=0.00003)
    return reference


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    data = dataset_contract()
    source_paths = paths(output)
    protected = [ROOT / "outputs/submission.csv", STRUCTURE / "submission_improved.csv", PRIOR / "submission_improved.csv"]
    plan = {
        "common_arguments": COMMON, "image_sizes": NEW, "candidate_models": CANDIDATES,
        "baseline_models": BASELINE, "baseline_validation_correct": 2973,
        "dataset_fingerprint": data["holdout_fingerprint"],
        "initialization": {"holdout": {"path": str(init_path()), "sha256": sha(init_path())},
                           "full": {"path": str(init_path(True)), "sha256": sha(init_path(True))}},
        "old_holdout_checkpoint_hashes": {name: sha(source_paths[name][0] / "best.pt") for name in BASELINE},
        "protected_submission_hashes": {str(path): sha(path) for path in protected},
        "single_factor_comparison": "Both new runs start from the same selected seed44 weights and use the same fresh six-epoch low-LR schedule. Only image_size differs (112 versus 160).",
        "checkpoint_rule": "Best raw/EMA no-TTA validation accuracy, then horizontal-flip TTA.",
        "selection_rule": "Fixed equal-weight candidates; highest correct count, then fewer models, then declared order. Keep baseline on a tie with baseline.",
        "reported_kaggle_score": {"value": 0.722, "source": "user reported; not independently retrieved"},
        "scope": "Repeated model-selection holdout, not an untouched test. No guarantee of Kaggle improvement.",
    }
    target = output / "plan.json"
    if target.exists():
        require(read(target) == plan, "Plan/source data changed; use a fresh output directory")
    else:
        write(target, plan)
    return plan, data


def check_new_config(config, name, data, plan, *, full=False, stop_epoch=0):
    check_material_config(config, expected_new(name), data, full=full)
    require(config.get("stop_after_epochs") == stop_epoch, "Stop epoch mismatch")
    source = plan["initialization"]["full" if full else "holdout"]
    require(config.get("init_checkpoint") == source["path"], "Initialization checkpoint path mismatch")
    metadata = config.get("initialization", {})
    expected = {"checkpoint": source["path"], "sha256": source["sha256"],
                "source_epoch": 13, "selected_source": "ema", "source_image_size": 112,
                "source_full_train": full,
                "source_dataset_fingerprint": data["full_fingerprint" if full else "holdout_fingerprint"],
                "optimizer_and_schedule_reset": True}
    require(metadata == expected, "Initialization provenance mismatch")


def check_holdout(run_dir, name, data, plan):
    import torch
    if name not in NEW:
        require(sha(run_dir / "best.pt") == plan["old_holdout_checkpoint_hashes"][name], "Old checkpoint changed")
        return check_old_holdout(run_dir, read(run_dir / "config.json"), data)
    config, summary, metrics = [read(run_dir / f"{part}.json") for part in ("config", "summary", "best_metrics")]
    check_new_config(config, name, data, plan)
    require(summary["epochs_completed"] == summary["planned_epochs"] == 6, "Incomplete adaptation")
    require(not summary["full_train"] and not summary["training_is_partial"] and not summary["validation_is_partial"], "Invalid validation run")
    require(metrics["num_samples"] == data["validation_samples"], "Incomplete heldout metrics")
    require(metrics["epoch"] == summary["best_epoch"] and metrics["accuracy"] == summary["best_validation_accuracy"], "Metrics/summary mismatch")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    check_new_config(checkpoint["config"], name, data, plan)
    require(checkpoint["class_to_index"] == CLASS_TO_INDEX and checkpoint["epoch"] == metrics["epoch"], "Best checkpoint metadata mismatch")
    require(checkpoint["validation_accuracy"] == metrics["accuracy"] and checkpoint["selected_source"] == metrics["selected_source"], "Best checkpoint selection mismatch")
    return metrics


def train_new(output, name, *, full=False, stop_epoch=0):
    run_dir = output / "full" / name if full else output / name
    if not (run_dir / "summary.json").exists():
        # Resume only its own unchanged experiment; initialization and optimizer
        # reset happen exactly once at the beginning of the new run.
        start = ["--resume", run_dir / "last.pt"] if (run_dir / "last.pt").exists() else ["--init-checkpoint", init_path(full)]
        extra = ["--full-train", "--stop-after-epochs", stop_epoch] if full else []
        run("-u", "train.py", "--output-dir", run_dir, *COMMON, "--image-size", NEW[name], *extra, *start)
    return run_dir


def validation(output):
    plan, data = prepare(output)
    source_paths, metrics = paths(output), {}
    for name in NEW:
        run_dir = train_new(output, name)
        metrics[name] = check_holdout(run_dir, name, data, plan)
        run("-u", "predict.py", "--checkpoints", run_dir / "best.pt", "--split", "val", "--tta",
            "--device", "cuda", "--output", output / f"{name}_tta.csv")
    predictions = {}
    for name, (run_dir, probability_path, _) in source_paths.items():
        if name not in NEW:
            metrics[name] = check_holdout(run_dir, name, data, plan)
        predictions[name] = probabilities(probability_path)
        checked_predictions(predictions[name], name, data)
    labels = data["validation_labels"]
    choices = []
    for names in CANDIDATES:
        p = np.mean([predictions[name]["probabilities"] for name in names], axis=0)
        correct = int((p.argmax(1) == labels).sum())
        choices.append({"models": names, "correct": correct, "accuracy": correct / len(labels)})
    baseline = choices[0]
    require(baseline["correct"] == plan["baseline_validation_correct"], "Current baseline did not reproduce")
    best_index = min(range(len(choices)), key=lambda i: (-choices[i]["correct"], len(choices[i]["models"]), i))
    selected = choices[best_index] if choices[best_index]["correct"] > baseline["correct"] else baseline
    selected_p = np.mean([predictions[name]["probabilities"] for name in selected["models"]], axis=0)
    baseline_p = np.mean([predictions[name]["probabilities"] for name in BASELINE], axis=0)
    previous = probabilities(PRIOR / "selected_validation.npz")
    checked_predictions(previous, "previous selected", data)
    require(np.allclose(baseline_p, previous["probabilities"], atol=1e-7, rtol=0), "Prior ensemble probabilities changed")
    old_ok, new_ok = baseline_p.argmax(1) == labels, selected_p.argmax(1) == labels
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
              "holdout_checkpoint_hashes": {name: sha(value[0] / "best.pt") for name, value in source_paths.items()},
              "scope": plan["scope"]}
    write(output / "selection.json", result)
    print({"selected": selected, "gain_pp": result["gain_percentage_points"]}, flush=True)


def check_full(run_dir, holdout_dir, name, metrics, data, plan):
    import torch
    config, summary = read(run_dir / "config.json"), read(run_dir / "summary.json")
    epoch = metrics["epoch"]
    if name in NEW:
        check_new_config(config, name, data, plan, full=True, stop_epoch=epoch)
    else:
        check_material_config(config, read(holdout_dir / "config.json"), data, full=True)
    require(config["stop_after_epochs"] == summary["epochs_completed"] == epoch, "Incomplete/wrong full epoch")
    require(summary["planned_epochs"] == config["epochs"] and summary["full_train"], "Full plan mismatch")
    require(summary["best_epoch"] is None and summary["best_validation_accuracy"] is None, "Full fit claims independent validation")
    require(not summary["training_is_partial"] and not summary["validation_is_partial"], "Partial full fit")
    final_path = run_dir / "final.pt"
    state = torch.load(final_path, map_location="cpu", weights_only=False)
    if name in NEW:
        check_new_config(state["config"], name, data, plan, full=True, stop_epoch=epoch)
    else:
        check_material_config(state["config"], config, data, full=True)
    require(state["epoch"] == epoch and state["validation_accuracy"] is None and state["class_to_index"] == CLASS_TO_INDEX, "Full checkpoint metadata mismatch")
    if name in NEW and metrics["selected_source"] == "raw" and state["selected_source"] != "raw":
        last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        check_new_config(last["config"], name, data, plan, full=True, stop_epoch=epoch)
        require(last["epoch"] == epoch, "Last checkpoint epoch mismatch")
        state.update(model_state=last["model_state"], selected_source="raw")
        temporary = final_path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(final_path)
    require(state["selected_source"] == metrics["selected_source"], "Full raw/EMA differs from selection")
    return final_path


def final(output):
    plan, data = prepare(output)
    selection = read(output / "selection.json")
    require(selection["plan_sha256"] == sha(output / "plan.json"), "Selection plan changed")
    if not selection["improved"]:
        write(output / "completion.json", {"status": "complete_no_improvement", "keep_submission": str(PRIOR / "submission_improved.csv")})
        print("No measured improvement; retaining prior submission.", flush=True)
        return
    selected, source_paths = selection["selected"]["models"], paths(output)
    require(selected in CANDIDATES, "Undeclared candidate")
    components = []
    for name in selected:
        holdout_dir, probability_path, _ = source_paths[name]
        metrics = check_holdout(holdout_dir, name, data, plan)
        require(metrics == selection["checkpoint_selection"][name], "Heldout selection metadata changed")
        require(sha(holdout_dir / "best.pt") == selection["holdout_checkpoint_hashes"][name], "Heldout weights changed")
        pred = probabilities(probability_path)
        checked_predictions(pred, name, data)
        components.append(pred["probabilities"])
    reconstructed = np.mean(components, axis=0)
    saved = probabilities(output / "selected_validation.npz")
    checked_predictions(saved, "selected ensemble", data)
    require(np.allclose(reconstructed, saved["probabilities"], atol=1e-7, rtol=0), "Selected probabilities differ")
    correct = int((reconstructed.argmax(1) == data["validation_labels"]).sum())
    require(correct == selection["selected"]["correct"] > plan["baseline_validation_correct"], "Selected improvement did not reproduce")
    checkpoints = []
    for name in selected:
        holdout_dir, _, run_dir = source_paths[name]
        metrics = selection["checkpoint_selection"][name]
        if name in NEW:
            train_new(output, name, full=True, stop_epoch=metrics["epoch"])
        checkpoints.append(check_full(run_dir, holdout_dir, name, metrics, data, plan))
    submission = output / "submission_improved.csv"
    run("-u", "predict.py", "--checkpoints", *checkpoints, "--tta", "--device", "cuda", "--output", submission)
    run("verify_submission.py", submission, "--json", output / "submission_improved.check.json")
    for path, digest in plan["protected_submission_hashes"].items():
        require(sha(path) == digest, "An existing submission changed")
    write(output / "completion.json", {"status": "complete", "submission": str(submission), "sha256": sha(submission),
          "holdout_selection_accuracy": selection["selected"]["accuracy"], "selected_models": selected,
          "full_checkpoints": list(map(str, checkpoints)), "full_checkpoint_hashes": {str(p): sha(p) for p in checkpoints},
          "kaggle_status": "not_uploaded", "new_kaggle_score": None})


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
