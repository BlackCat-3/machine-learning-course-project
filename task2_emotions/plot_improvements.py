"""Plot the measured holdout comparison; no leaderboard scores are plotted."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/improvement_20261016")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    selection = read_json(output / "selection.json")
    original = read_json(ROOT / "outputs/selected_validation_metrics.json")
    selected = selection["selected_metrics"]
    if (original["class_to_index"] != selected["class_to_index"]
            or original["num_samples"] != selected["num_samples"]
            or original["num_samples"] != selection["validation_samples"]
            or original["accuracy"] != selection["original_validation_accuracy"]):
        raise ValueError("Original and selected metrics do not describe the same holdout")
    classes = sorted(original["class_to_index"], key=original["class_to_index"].get)
    if any(original["per_class"][c]["support"] != selected["per_class"][c]["support"] for c in classes):
        raise ValueError("Per-class holdout supports differ")
    scores = {frozenset(row["models"]): row["accuracy"] for row in selection["candidate_scores"]}
    rows = [
        ("Original two-model ensemble", original["accuracy"]),
        ("ResNet18 + Mixup", scores[frozenset(["old_mixup"])]),
        ("ResNet18 without maxpool", scores[frozenset(["resnet18_detail"])]),
        ("ResNet34", scores[frozenset(["resnet34"])]),
        ("Mixup + no-maxpool ensemble", scores[frozenset(["old_mixup", "resnet18_detail"])]),
        (f"Selected {len(selection['selected']['models'])}-model ensemble", selection["selected"]["accuracy"]),
    ]
    old_color, candidate_color, new_color = "#8996A8", "#4282B4", "#087F75"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.8), gridspec_kw={"width_ratios": [1, 1.22]})
    fig.patch.set_facecolor("white")
    y = np.arange(len(rows))
    accuracy = np.array([row[1] * 100 for row in rows])
    colors = [old_color] + [candidate_color] * (len(rows) - 2) + [new_color]
    ax1.scatter(accuracy, y, c=colors, s=90, zorder=3, edgecolors="white", linewidths=0.7)
    for i, value in enumerate(accuracy):
        ax1.annotate(f"{value:.2f}%", (value, i), xytext=(9, 0), textcoords="offset points",
                     va="center", fontsize=10, fontweight="bold" if i == len(rows) - 1 else "normal",
                     color=new_color if i == len(rows) - 1 else "#233447")
    ax1.set_yticks(y, [row[0] for row in rows])
    ax1.set_ylim(len(rows) - 0.45, -0.55)
    lower = np.floor(accuracy.min() * 2) / 2 - 0.25
    upper = np.ceil(accuracy.max() * 2) / 2 + 0.7
    ax1.set_xlim(lower, upper)
    ax1.set_xlabel("Accuracy (%) — zoomed horizontal axis", labelpad=9)
    ax1.set_title("Model comparison", loc="left", pad=14)
    ax1.grid(axis="x", alpha=0.18, zorder=0)
    ax1.tick_params(axis="y", length=0, pad=8)

    x = np.arange(len(classes))
    width = 0.36
    old_recall = np.array([original["per_class"][c]["recall"] * 100 for c in classes])
    new_recall = np.array([selected["per_class"][c]["recall"] * 100 for c in classes])
    bars_old = ax2.bar(x - width / 2, old_recall, width, color=old_color, label="Original ensemble", zorder=3)
    bars_new = ax2.bar(x + width / 2, new_recall, width, color=new_color, label="Selected ensemble", zorder=3)
    ax2.bar_label(bars_old, labels=[f"{v:.1f}" for v in old_recall], padding=3, fontsize=8)
    ax2.bar_label(bars_new, labels=[f"{v:.1f}" for v in new_recall], padding=3, fontsize=8)
    ax2.set_xticks(x, classes)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Recall (%)")
    ax2.set_title("Per-class recall", loc="left", pad=14)
    ax2.grid(axis="y", alpha=0.18, zorder=0)
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.10), ncol=2,
               frameon=False, fontsize=9, borderaxespad=0)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CAD2DB")
        ax.tick_params(colors="#3B4A59")

    gain = 100 * (selected["accuracy"] - original["accuracy"])
    extra_correct = int(round((selected["accuracy"] - original["accuracy"]) * original["num_samples"]))
    fig.suptitle("Validation improvement study", x=0.03, ha="left", y=0.98, fontsize=17, fontweight="bold")
    fig.text(0.03, 0.913,
             f"Same grouped holdout: {original['num_samples']:,} images. All models use original + horizontal-flip prediction averaging.",
             fontsize=10, color="#526273")
    fig.text(0.03, 0.061, f"Selected vs original: +{gain:.2f} percentage points; {extra_correct:+d} correct predictions.",
             fontsize=11, fontweight="bold", color=new_color)
    fig.text(0.03, 0.025, "Repeated model-selection holdout; these results are not an untouched test score or a Kaggle score.",
             fontsize=9, color="#526273")
    fig.subplots_adjust(left=0.245, right=0.985, bottom=0.22, top=0.80, wspace=0.30)
    path = output / "comparison.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
