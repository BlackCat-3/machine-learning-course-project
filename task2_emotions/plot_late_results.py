"""Plot completed late crop-TTA and paired-resolution validation experiments.

Reads saved metrics only; does not load models, train, or run inference.
Outputs comparison.png, training_curves.png, comparison.csv and a data manifest.
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

COUNT = 4241
OLD4 = ["old_mixup", "detail_seed43", "resnet34", "detail_seed44"]
COLORS = {"baseline": "#687887", "center": "#148578", "tune112": "#3476AD", "tune160": "#D0792D", "selected": "#784D9D"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validated_score(correct, accuracy):
    require(isinstance(correct, int) and not isinstance(correct, bool) and 0 <= correct <= COUNT,
            "Invalid correct count")
    require(math.isfinite(accuracy) and math.isclose(accuracy, correct / COUNT, rel_tol=0, abs_tol=1e-12),
            "Reported accuracy does not match the 4,241-image correct count")
    return correct, accuracy


def load_rows(root):
    selection = read(root / "resolution/selection.json")
    tta = read(root / "tta/val_summary.json")
    require(selection["validation_samples"] == COUNT, "Unexpected resolution-study holdout size")
    require(tta["split"] == "val" and tta["status"] == "complete" and tta["rows"] == COUNT,
            "Expected a complete validation TTA study")
    scores = {}
    for candidate in selection["candidate_scores"]:
        key = tuple(candidate["models"])
        require(key not in scores, "Repeated candidate in selection")
        validated_score(candidate["correct"], candidate["accuracy"])
        scores[key] = candidate
    for field in ("baseline", "selected"):
        value = selection[field]
        require(tuple(value["models"]) in scores and scores[tuple(value["models"])] == value,
                f"{field} does not match measured candidate scores")
    baseline = selection["baseline"]
    require(baseline["models"] == OLD4, "Unexpected four-model reference")
    center = tta["ensembles"]["center"]
    full = tta["ensembles"]["full"]
    validated_score(center["correct"], center["metrics"]["accuracy"])
    validated_score(full["correct"], full["metrics"]["accuracy"])
    require(center["metrics"]["num_samples"] == COUNT and full["metrics"]["num_samples"] == COUNT,
            "TTA metric sample count mismatch")
    require(full["correct"] == baseline["correct"], "Recomputed full+flip baseline differs; inspect before plotting")
    specifications = [
        ("baseline", "Previous 4 models + flip", baseline, "original + horizontal flip", 4),
        ("center", "Previous 4 models + center-crop TTA",
         {"models": OLD4, "correct": center["correct"], "accuracy": center["metrics"]["accuracy"]},
         "original and centered 44x44 crop, each with horizontal flip", 4),
        ("tune112", "112-pixel adaptation + flip", scores[("tune112",)], "original + horizontal flip", 1),
        ("tune160", "160-pixel adaptation + flip", scores[("tune160",)], "original + horizontal flip", 1),
        ("selected", "Resolution-study selection + flip", selection["selected"], "original + horizontal flip", len(selection["selected"]["models"])),
    ]
    rows = []
    for key, label, score, scheme, count_models in specifications:
        rows.append({"key": key, "candidate": label, "models": "|".join(score["models"]),
                     "model_count": count_models, "inference_scheme": scheme,
                     "correct": score["correct"], "validation_samples": COUNT,
                     "accuracy": score["accuracy"], "accuracy_percent": score["accuracy"] * 100,
                     "gain_correct_vs_baseline": score["correct"] - baseline["correct"],
                     "gain_percentage_points_vs_baseline": 100 * (score["accuracy"] - baseline["accuracy"]),
                     "scope": "repeated model-selection holdout; not Kaggle accuracy"})
    return selection, rows


def load_histories(root, selection):
    result = {}
    for name, image_size in (("tune112", 112), ("tune160", 160)):
        directory = root / "resolution" / name
        config, summary = read(directory / "config.json"), read(directory / "summary.json")
        require(config["image_size"] == image_size and config["epochs"] == summary["epochs_completed"] == 6,
                f"Expected completed six-epoch {name} adaptation")
        require(not config["full_train"] and not config["training_is_partial"] and not config["validation_is_partial"],
                f"Expected complete heldout training for {name}")
        with (directory / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        require([int(row["epoch"]) for row in history] == list(range(1, 7)), f"Incomplete {name} history")
        for row in history:
            for key in ("train_loss", "raw_val_accuracy", "ema_val_accuracy"):
                require(math.isfinite(float(row[key])), f"Non-finite {name} {key}")
            require(int(row["train_samples_seen"]) == config["train_samples"], f"Partial {name} epoch")
        metrics = selection["checkpoint_selection"][name]
        require(metrics["epoch"] == summary["best_epoch"] and metrics["accuracy"] == summary["best_validation_accuracy"],
                f"{name} best checkpoint/summary mismatch")
        result[name] = {"config": config, "history": history, "best": metrics}
    left, right = (result[key]["config"] for key in ("tune112", "tune160"))
    for key in ("architecture", "no_maxpool", "epochs", "batch_size", "lr", "seed", "mixup_alpha",
                "mixup_prob", "weight_decay", "label_smoothing", "dropout", "warmup_epochs", "ema_decay", "initialization"):
        require(left[key] == right[key], f"Paired configurations differ beyond resolution: {key}")
    return result


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#C8D1D8")
    ax.tick_params(colors="#34485A")
    ax.grid(alpha=0.25, color="#A6B4C0")
    ax.set_axisbelow(True)


def draw_comparison(output, rows):
    fig, ax = plt.subplots(figsize=(13.4, 6.6))
    values = np.array([row["accuracy_percent"] for row in rows])
    baseline = rows[0]["accuracy_percent"]
    y = np.arange(len(rows))
    ax.axvline(baseline, color=COLORS["baseline"], linestyle="--", linewidth=1.1)
    ax.scatter(values, y, c=[COLORS[row["key"]] for row in rows], s=100, edgecolors="white", linewidth=0.8, zorder=3)
    span = max(float(np.ptp(values)), 0.8)
    ax.set_xlim(values.min() - span * 0.13, values.max() + span * 0.77)
    for row, value, position in zip(rows, values, y):
        delta = row["gain_percentage_points_vs_baseline"]
        ax.annotate(f"{value:.2f}%  ({row['correct']:,}/{COUNT:,})   {delta:+.2f} pp",
                    (value, position), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=10, color="#263D50", fontweight="bold" if row["key"] == "selected" else "normal")
    ax.set_yticks(y, [row["candidate"] for row in rows])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlabel("Validation accuracy (%) - zoomed horizontal axis", labelpad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Late experiments: crop TTA and input resolution", x=0.025, ha="left", y=0.97,
                 fontsize=18, fontweight="bold", color="#21394C")
    fig.text(0.025, 0.90, f"Same grouped holdout: {COUNT:,} images. Every value is measured validation accuracy; no Kaggle scores are plotted.",
             fontsize=10, color="#536679")
    fig.text(0.025, 0.14, "Center-crop TTA averages original and centered 44x44 views, each with its horizontal flip.",
             fontsize=10, color="#536679")
    fig.text(0.025, 0.095, "Selected resolution candidate: " + rows[-1]["models"].replace("|", " + "),
             fontsize=10, color="#263D50")
    fig.text(0.025, 0.045, "Differences are percentage points (pp). The holdout was reused for model selection; improvements do not guarantee a higher online score.",
             fontsize=9, color="#536679")
    fig.subplots_adjust(left=0.315, right=0.97, top=0.82, bottom=0.25)
    fig.savefig(output / "comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


def draw_curves(output, histories):
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(13.4, 6.5))
    for name, data in histories.items():
        history, metrics = data["history"], data["best"]
        size, color = data["config"]["image_size"], COLORS[name]
        epochs = [int(row["epoch"]) for row in history]
        loss_ax.plot(epochs, [float(row["train_loss"]) for row in history], marker="o", color=color, label=f"{size}px")
        for source, linestyle in (("raw", "--"), ("ema", "-")):
            accuracy_ax.plot(epochs, [100 * float(row[f"{source}_val_accuracy"]) for row in history],
                             marker="o" if source == "ema" else None, linestyle=linestyle, color=color,
                             alpha=1 if source == "ema" else 0.65, label=f"{size}px {source.upper()}")
        accuracy_ax.scatter(metrics["epoch"], metrics["accuracy"] * 100, marker="*", s=185,
                            facecolor=color, edgecolor="#243B4D", linewidth=0.6, zorder=5)
    loss_ax.set_title("Training loss with Mixup", loc="left", pad=12)
    loss_ax.set_ylabel("Cross-entropy (Mixup + label smoothing)")
    accuracy_ax.set_title("Validation accuracy without TTA", loc="left", pad=12)
    accuracy_ax.set_ylabel("Accuracy on original hard labels (%)")
    for ax in (loss_ax, accuracy_ax):
        ax.set_xlabel("Second-stage epoch")
        ax.set_xticks(range(1, 7))
        style_axis(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Paired low-learning-rate adaptation: 112px vs 160px", x=0.025, ha="left", y=0.97,
                 fontsize=18, fontweight="bold", color="#21394C")
    fig.text(0.025, 0.90, "Both start from the same seed-44 heldout checkpoint. Six-epoch schedule, learning rate 0.00003; only input resolution differs.",
             fontsize=10, color="#536679")
    detail = "; ".join(f"{data['config']['image_size']}px: epoch {data['best']['epoch']} {data['best']['selected_source'].upper()}"
                       for data in histories.values())
    fig.text(0.025, 0.11, "Stars mark the selected checkpoint in each run. " + detail + ".", fontsize=10, color="#263D50")
    fig.text(0.025, 0.055, "Training uses augmented/mixed images; validation uses the fixed 4,241 original-label images. Training accuracy is not plotted.",
             fontsize=9, color="#536679")
    fig.subplots_adjust(left=0.075, right=0.97, top=0.79, bottom=0.23, wspace=0.28)
    fig.savefig(output / "training_curves.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "outputs/late_20261018")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.study_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else root
    selection, rows = load_rows(root)
    histories = load_histories(root, selection)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    draw_comparison(output, rows)
    draw_curves(output, histories)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [root / "tta/val_summary.json", root / "resolution/selection.json"]
    sources.extend(root / "resolution" / name / file for name in histories for file in ("config.json", "summary.json", "history.csv"))
    manifest = {"validation_samples": COUNT, "scope": "Repeated model-selection validation only; no Kaggle scores.",
                "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                "outputs": [str(output / file) for file in ("comparison.png", "training_curves.png", "comparison.csv")]}
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Saved comparison.png, training_curves.png, comparison.csv and plot_manifest.json", flush=True)


if __name__ == "__main__":
    main()
