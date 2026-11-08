"""Render completed ResNet50 validation evidence without model inference.

Online scores are recorded separately in final_summary.json, never on a chart
with validation accuracy. The unfinished resolution experiment is not compared.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N = 4241
OLD3 = ["old_mixup", "detail_seed43", "resnet34"]
OLD4 = [*OLD3, "detail_seed44"]
MODEL = "resnet50"
DISPLAY = [(OLD4, "Previous 4-model ensemble"),
           ([MODEL], "ResNet50 alone"),
           ([*OLD4, MODEL], "Previous 4 models + ResNet50"),
           ([*OLD3, MODEL], "Previous 3 models + ResNet50"),
           (["detail_seed44", MODEL], "Seed-44 ResNet18 + ResNet50")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_accuracy(row):
    count, accuracy = row["correct"], row["accuracy"]
    require(isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= N, "Invalid correct count")
    require(math.isfinite(accuracy) and math.isclose(accuracy, count / N, rel_tol=0, abs_tol=1e-12),
            "Accuracy/correct count mismatch")


def load_results(study):
    plan, selection = read(study / "plan.json"), read(study / "selection.json")
    require(selection["plan_sha256"] == sha(study / "plan.json"), "Selection plan digest mismatch")
    require(plan["validation_samples"] == selection["validation_samples"] == N, "Unexpected holdout size")
    require(plan["candidate_models"] == [models for models, _ in DISPLAY], "Expected five fixed candidates")
    require(plan["minimum_gain_correct_for_submission"] == selection["minimum_gain_correct_for_submission"] == 43,
            "The declared submission gate must be +43 correct")
    scores = selection["candidate_scores"]
    require([row["models"] for row in scores] == plan["candidate_models"], "Measured candidate order differs")
    for row in scores:
        checked_accuracy(row)
    for field in ("baseline", "best_observed", "selected"):
        require(selection[field] in scores, f"{field} is not a measured candidate")
    baseline = selection["baseline"]
    require(baseline["models"] == OLD4 and baseline["correct"] == plan["baseline_validation_correct"], "Baseline changed")
    best_index = min(range(len(scores)), key=lambda i: (-scores[i]["correct"], len(scores[i]["models"]), i))
    require(scores[best_index] == selection["best_observed"], "Best candidate/tie rule mismatch")
    qualifies = selection["best_observed"]["correct"] - baseline["correct"] >= 43
    require(selection["meets_minimum_gain"] == selection["improved"] == qualifies, "Qualification flag mismatch")
    require(selection["selected"] == (selection["best_observed"] if qualifies else baseline), "Selected candidate violates gate")
    rows = []
    for (models, label), score in zip(DISPLAY, scores):
        rows.append({"candidate": label, "models": "|".join(models), "model_count": len(models),
                     "validation_samples": N, "correct": score["correct"], "accuracy": score["accuracy"],
                     "accuracy_percent": score["accuracy"] * 100,
                     "gain_correct_vs_baseline": score["correct"] - baseline["correct"],
                     "gain_percentage_points_vs_baseline": 100 * (score["accuracy"] - baseline["accuracy"]),
                     "meets_43_correct_threshold": score["correct"] - baseline["correct"] >= 43,
                     "best_observed": score == selection["best_observed"], "selected": score == selection["selected"],
                     "inference_scheme": "original + horizontal flip; equal-weight probability average",
                     "scope": "repeated model-selection validation; not Kaggle accuracy"})
    directory = study / MODEL
    config, summary = read(directory / "config.json"), read(directory / "summary.json")
    completed = plan.get("stop_after_epochs", 0) or plan["epochs"]
    require(config["architecture"] == MODEL and config["epochs"] == summary["planned_epochs"] == plan["epochs"]
            and summary["epochs_completed"] == completed,
            "Expected a completed ResNet50 run")
    require(config["image_size"] == plan["image_size"] and config["no_maxpool"] == plan["no_maxpool"]
            and not config["full_train"], "Heldout recipe mismatch")
    require(not config["training_is_partial"] and not config["validation_is_partial"], "Partial training cannot be charted as complete")
    with (directory / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    require([int(row["epoch"]) for row in history] == list(range(1, completed + 1)), "Incomplete epoch history")
    for row in history:
        require(int(row["train_samples_seen"]) == config["train_samples"], "Partial training epoch")
        for key in ("train_loss", "raw_val_accuracy", "ema_val_accuracy"):
            require(math.isfinite(float(row[key])), f"Invalid history value: {key}")
    best = selection["checkpoint_selection"][MODEL]
    require(best["epoch"] == summary["best_epoch"] and best["accuracy"] == summary["best_validation_accuracy"],
            "Best checkpoint and run summary disagree")
    return plan, selection, rows, history, best


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C7D2DC")
    ax.tick_params(colors="#32495C")
    ax.grid(color="#AAB8C4", alpha=0.25)
    ax.set_axisbelow(True)


def draw_comparison(output, plan, selection, rows):
    fig, ax = plt.subplots(figsize=(13.8, 7.0))
    values = np.array([row["accuracy_percent"] for row in rows])
    baseline = selection["baseline"]["accuracy"] * 100
    threshold = (selection["baseline"]["correct"] + 43) / N * 100
    y = np.arange(len(rows))
    colors = ["#718190" if i == 0 else "#2F7B9C" if i == 1 else "#188678" for i in y]
    ax.axvline(baseline, color="#718190", linestyle=":", linewidth=1.2, label="Previous baseline")
    ax.axvline(threshold, color="#B26731", linestyle="--", linewidth=1.3,
               label=f"Submission gate: +43 correct ({threshold:.2f}%)")
    ax.scatter(values, y, c=colors, s=90, edgecolors="white", linewidth=0.7, zorder=3)
    low, high = min(float(values.min()), baseline), max(float(values.max()), threshold)
    span = max(high - low, 1.2)
    ax.set_xlim(low - 0.1 * span, high + 0.8 * span)
    for i, row in enumerate(rows):
        ax.annotate(f"{values[i]:.2f}%  ({row['correct']:,}/{N:,})   {row['gain_correct_vs_baseline']:+d} correct",
                    (values[i], i), xytext=(10, 0), textcoords="offset points", va="center", fontsize=10,
                    fontweight="bold" if row["best_observed"] else "normal", color="#283F52")
        if row["best_observed"]:
            ax.scatter(values[i], i, s=190, facecolors="none", edgecolors="#243B4E", linewidth=1.5, zorder=4)
    ax.set_yticks(y, [row["candidate"] for row in rows])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlabel("Validation accuracy (%) - zoomed horizontal axis", labelpad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.suptitle("ResNet50: fixed candidate comparison", x=0.025, ha="left", y=0.975,
                 fontsize=18, fontweight="bold", color="#203A4E")
    fig.text(0.025, 0.91, f"Same grouped holdout: {N:,} images. Original + horizontal-flip TTA; models average probabilities equally.",
             fontsize=10, color="#536A7D")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False, fontsize=9)
    gain = selection["best_observed"]["correct"] - selection["baseline"]["correct"]
    decision = "PASS: eligible for a deadline submission" if selection["meets_minimum_gain"] else "BELOW GATE: retain the previous submission"
    fig.text(0.025, 0.115, f"Best measured gain: {gain:+d} correct. Required gain: at least +43 correct. {decision}.",
             fontsize=10, fontweight="bold", color="#203A4E")
    fig.text(0.025, 0.07, "Circle outline = best observed candidate. Architecture, pretraining and resolution form a new recipe; this is not a one-factor ablation.",
             fontsize=9, color="#536A7D")
    fig.text(0.025, 0.03, "Repeated model-selection validation only. Kaggle scores are excluded; local improvement does not guarantee an online gain.",
             fontsize=9, color="#536A7D")
    fig.subplots_adjust(left=0.325, right=0.97, top=0.83, bottom=0.31)
    fig.savefig(output / "comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


def draw_curves(output, plan, history, best, *, partial=False):
    epochs = [int(row["epoch"]) for row in history]
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(13.8, 6.6))
    loss_ax.plot(epochs, [float(row["train_loss"]) for row in history], marker="o", color="#347BA2", linewidth=1.8)
    loss_ax.set_title("Training loss with Mixup", loc="left", pad=12)
    loss_ax.set_ylabel("Cross-entropy (Mixup + label smoothing)")
    for source, color, linestyle in (("raw", "#347BA2", "--"), ("ema", "#168677", "-")):
        accuracy_ax.plot(epochs, [float(row[f"{source}_val_accuracy"]) * 100 for row in history],
                         marker="o", color=color, linestyle=linestyle, label=source.upper())
    accuracy_ax.scatter(best["epoch"], best["accuracy"] * 100, marker="*", s=210,
                        facecolor="#C68137", edgecolor="#283E4E", linewidth=0.6, zorder=5,
                        label=f"{'Best observed' if partial else 'Selected'}: epoch {best['epoch']} {best['selected_source'].upper()}")
    accuracy_ax.set_title("Validation accuracy without TTA", loc="left", pad=12)
    accuracy_ax.set_ylabel("Accuracy on original hard labels (%)")
    accuracy_ax.legend(frameon=False, fontsize=9)
    for ax in (loss_ax, accuracy_ax):
        style_axis(ax)
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
    schedule = (f"{len(history)}/{plan['epochs']} epochs (stopped early)" if partial else
                f"{len(history)}/{plan['epochs']} epochs (deadline cap; LR plan retained)" if len(history) < plan["epochs"] else
                f"{plan['epochs']}-epoch schedule")
    fig.suptitle(f"ResNet50 training: {plan['image_size']}px, {schedule}",
                 x=0.025, ha="left", y=0.975, fontsize=18, fontweight="bold", color="#203A4E")
    subtitle = ("Resource-limited stop; the planned run and candidate comparison are unfinished. Only actually completed epochs are plotted."
                if partial else "Official ImageNet-v2 initialization; fixed grouped split. Model selection uses validation accuracy before TTA.")
    fig.text(0.025, 0.91, subtitle,
             fontsize=10, color="#536A7D")
    fig.text(0.025, 0.11, f"Best {'observed ' if partial else ''}no-TTA validation accuracy: {best['accuracy'] * 100:.2f}% at epoch {best['epoch']} ({best['selected_source'].upper()}).",
             fontsize=10, fontweight="bold", color="#203A4E")
    fig.text(0.025, 0.055, "Training uses augmented/mixed images; validation uses 4,241 original-label images. Mixup training accuracy is not plotted.",
             fontsize=9, color="#536A7D")
    fig.subplots_adjust(left=0.075, right=0.97, top=0.8, bottom=0.23, wspace=0.26)
    fig.savefig(output / ("training_curves_partial.png" if partial else "training_curves.png"), dpi=180, facecolor="white")
    plt.close(fig)


def resolution_status(late_root):
    runs = {}
    for name in ("tune112", "tune160"):
        directory = late_root / "resolution" / name
        history_path = directory / "history.csv"
        if history_path.exists():
            with history_path.open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            completed = int(history[-1]["epoch"]) if history else 0
        else:
            completed = 0
        runs[name] = {"completed_epochs_observed": completed, "summary_exists": (directory / "summary.json").exists()}
    return {"status": "interrupted_for_architecture_experiment",
            "status_source": "operator stopped the resolution experiment after the user requested a larger improvement",
            "runs": runs, "included_in_completed_comparison": False}


def efficientnet_status(late_root):
    path = late_root / "efficientnet/status.json"
    if not path.exists():
        return {"status": "unavailable"}
    status = read(path)
    return {"status": status["status"], "completed_epochs": status["completed_epochs"],
            "planned_epochs": status["planned_epochs"],
            "best_observed_checkpoint": status["best_observed_checkpoint"],
            "candidate_comparison_status": status["candidate_comparison_status"],
            "included_in_resnet50_candidate_comparison": False,
            "status_file": str(path), "status_file_sha256": sha(path)}


def plot_interrupted(study, output):
    """An interrupted training history is evidence, not a completed selection."""
    plan, status = read(study / "plan.json"), read(study / "status.json")
    require(status["status"] == "resource_early_stop", "No supported interrupted-study status")
    directory = study / MODEL
    config, best = read(directory / "config.json"), read(directory / "best_metrics.json")
    with (directory / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    completed = status["completed_epochs"]
    require(status["planned_epochs"] == config["epochs"] == plan["epochs"] and completed < plan["epochs"],
            "Interrupted epoch counts disagree")
    require([int(row["epoch"]) for row in history] == list(range(1, completed + 1)), "History does not match interruption record")
    require(config["architecture"] == MODEL and config["image_size"] == plan["image_size"] and not config["full_train"],
            "Interrupted model recipe mismatch")
    require(not config["training_is_partial"] and not config["validation_is_partial"], "Batch-limited smoke history rejected")
    for row in history:
        require(int(row["train_samples_seen"]) == config["train_samples"], "A completed epoch has incomplete samples")
        for key in ("train_loss", "raw_val_accuracy", "ema_val_accuracy"):
            require(math.isfinite(float(row[key])), "Non-finite interrupted history")
    require(best["num_samples"] == N and 1 <= best["epoch"] <= completed, "Invalid observed checkpoint epoch/sample count")
    observed = max(max(float(row["raw_val_accuracy"]), float(row["ema_val_accuracy"])) for row in history)
    require(abs(best["accuracy"] - observed) < 1e-12, "Best observed accuracy disagrees with history")
    require(abs(float(history[best["epoch"] - 1][f"{best['selected_source']}_val_accuracy"]) - best["accuracy"]) < 1e-12,
            "Best observed raw/EMA source disagrees with history")
    logged_best = status["best_observed_checkpoint"]
    require(logged_best["epoch"] == best["epoch"] and logged_best["selected_source"] == best["selected_source"]
            and abs(logged_best["validation_accuracy_without_tta"] - best["accuracy"]) < 1e-12,
            "Interruption status contains a different best checkpoint")
    draw_curves(output, plan, history, best, partial=True)
    with (output / "partial_history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["epoch", "train_loss", "raw_val_accuracy", "ema_val_accuracy", "train_samples_seen", "seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in history)
    sources = [study / "plan.json", study / "status.json", directory / "config.json",
               directory / "history.csv", directory / "best_metrics.json"]
    summary = {
        "status": "resource_early_stop", "completed_epochs": completed, "planned_epochs": plan["epochs"],
        "candidate_comparison_evaluated": False, "selected_candidate": None,
        "validation": {"samples": N, "best_observed_single_checkpoint_without_tta":
                       {key: best[key] for key in ("epoch", "selected_source", "accuracy")},
                       "scope": "Intermediate no-TTA validation results from an unfinished training plan; no fixed-candidate TTA/ensemble evaluation is claimed"},
        "online_user_reports": {
            "prior_four_model_submission": {"accuracy": 0.722, "source": "user reported", "independently_verified": False},
            "center_crop_tta_submission": {"accuracy": 0.719, "source": "user reported", "independently_verified": False},
            "new_resnet50_submission": {"accuracy": None, "status": "not generated by this unfinished study"}},
        "interrupted_resolution_experiment": resolution_status(study.parent),
        "interrupted_efficientnet_experiment": efficientnet_status(study.parent),
        "submission": None, "comparison_csv": None,
        "sources_sha256": {str(path): sha(path) for path in sources},
        "plots": [str(output / "training_curves_partial.png")],
        "partial_history_csv": str(output / "partial_history.csv")}
    (output / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved partial training curve/history and interrupted final_summary.json; no candidate comparison invented", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "outputs/late_20261018/resnet50")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else study
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    if not (study / "selection.json").exists():
        plot_interrupted(study, output)
        return
    plan, selection, rows, history, best = load_results(study)
    draw_comparison(output, plan, selection, rows)
    draw_curves(output, plan, history, best)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completion_path = study / "completion.json"
    completion = read(completion_path) if completion_path.exists() else None
    sources = [study / "plan.json", study / "selection.json", study / MODEL / "history.csv", study / MODEL / "summary.json"]
    if completion:
        sources.append(completion_path)
    summary = {
        "status": completion["status"] if completion else "validation_complete_submission_status_unavailable",
        "validation": {"samples": N, "baseline": selection["baseline"], "best_observed": selection["best_observed"],
                       "selected": selection["selected"], "minimum_gain_correct": 43,
                       "meets_minimum_gain": selection["meets_minimum_gain"],
                       "best_single_checkpoint_without_tta": {key: best[key] for key in ("epoch", "selected_source", "accuracy")},
                       "scope": "Repeated model-selection holdout; not Kaggle accuracy"},
        "online_user_reports": {
            "prior_four_model_submission": {"accuracy": 0.722, "source": "user reported", "independently_verified": False},
            "center_crop_tta_submission": {"accuracy": 0.719, "source": "user reported", "independently_verified": False},
            "new_resnet50_submission": {"accuracy": None, "status": "not independently retrieved"},
        },
        "interrupted_resolution_experiment": resolution_status(study.parent),
        "interrupted_efficientnet_experiment": efficientnet_status(study.parent),
        "recipe": {"architecture": plan["architecture"], "image_size": plan["image_size"],
                   "planned_epochs": plan["epochs"], "completed_epochs": len(history),
                   "execution_cap": plan.get("stop_after_epochs", 0), "no_maxpool": plan["no_maxpool"],
                   "official_initialization": plan["official_initialization"],
                   "default_submission_mode": plan["default_submission_mode"]},
        "actual_submission_training_scope": {
            "new_model_full_refit": completion.get("new_model_full_refit", True if completion.get("status") == "complete" else None) if completion else None,
            "new_model_training_samples": completion.get("new_model_training_samples") if completion else None,
            "model_sources": completion.get("sources") if completion else None,
            "note": "Validation candidate scores use heldout counterparts; they do not measure test accuracy of the final submitted weights"},
        "submission": completion.get("submission") if completion else None,
        "submission_sha256": completion.get("sha256") if completion else None,
        "sources_sha256": {str(path): sha(path) for path in sources},
        "plots": [str(output / "comparison.png"), str(output / "training_curves.png")],
        "comparison_csv": str(output / "comparison.csv"),
    }
    (output / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved comparison.png, training_curves.png, comparison.csv and final_summary.json", flush=True)


if __name__ == "__main__":
    main()
