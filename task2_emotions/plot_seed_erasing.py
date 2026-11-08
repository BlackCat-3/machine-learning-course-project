"""Plot the nine measured seed/erasing candidates on the shared holdout.

Run only after seed_erasing_study.py has written selection.json.
This chart describes validation model selection, never leaderboard performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BASELINE = ("old_mixup", "detail_seed43", "resnet34")
DISPLAY = [
    (("detail_seed43",), "Reference: seed 43, no erasing", "single"),
    (("detail_seed44",), "Change seed only: 44", "single"),
    (("detail_erasing",), "Add erasing only: seed 43", "single"),
    (BASELINE, "Previous three-model ensemble", "baseline"),
    (("detail_seed43", "detail_seed44"), "Reference + seed 44", "ensemble"),
    (("detail_seed43", "detail_erasing"), "Reference + erasing", "ensemble"),
    ((*BASELINE, "detail_seed44"), "Previous ensemble + seed 44", "ensemble"),
    ((*BASELINE, "detail_erasing"), "Previous ensemble + erasing", "ensemble"),
    ((*BASELINE, "detail_seed44", "detail_erasing"), "Previous ensemble + both", "ensemble"),
]


def candidate_key(models):
    if not isinstance(models, list) or not models or len(set(models)) != len(models):
        raise ValueError("Candidate must contain a nonempty list of distinct models")
    return frozenset(models)


def load_rows(output: Path):
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    count = selection["validation_samples"]
    if count != 4241:
        raise ValueError("Expected the study's fixed 4,241-image holdout")
    scores = {}
    for row in selection["candidate_scores"]:
        key = candidate_key(row["models"])
        accuracy, correct = row["accuracy"], row["correct"]
        if (not isinstance(correct, int) or not 0 <= correct <= count
                or not math.isfinite(accuracy)
                or not math.isclose(accuracy, correct / count, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Accuracy and correct count are inconsistent")
        if key in scores:
            raise ValueError("Duplicate candidate")
        scores[key] = row
    if set(scores) != {frozenset(models) for models, _, _ in DISPLAY}:
        raise ValueError("Expected exactly the nine predefined study candidates")
    for field in ("baseline", "selected"):
        row = selection[field]
        key = candidate_key(row["models"])
        if key not in scores or any(row[k] != scores[key][k] for k in ("accuracy", "correct")):
            raise ValueError(f"{field} does not match candidate_scores")
    if candidate_key(selection["baseline"]["models"]) != frozenset(BASELINE):
        raise ValueError("Unexpected previous ensemble")
    baseline = selection["baseline"]["accuracy"]
    selected_key = candidate_key(selection["selected"]["models"])
    rows = []
    for models, label, category in DISPLAY:
        score = scores[frozenset(models)]
        rows.append({
            "candidate": label,
            "model_names": "|".join(models),
            "category": category,
            "accuracy": score["accuracy"],
            "correct": score["correct"],
            "gain_vs_previous_ensemble_pp": 100 * (score["accuracy"] - baseline),
            "selected": frozenset(models) == selected_key,
        })
    return selection, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/seed_erasing_20261016")
    output = parser.parse_args().output_dir.resolve()
    selection, rows = load_rows(output)
    colors = {"single": "#3B7CA6", "baseline": "#788694", "ensemble": "#16877C"}
    y = np.arange(len(rows))
    accuracy = np.array([row["accuracy"] * 100 for row in rows])
    gain = np.array([row["gain_vs_previous_ensemble_pp"] for row in rows])
    row_colors = [colors[row["category"]] for row in rows]
    baseline = selection["baseline"]["accuracy"] * 100
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 8.0), sharey=True,
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")
    ax1.axvline(baseline, color=colors["baseline"], linestyle="--", linewidth=1, zorder=1)
    ax1.scatter(accuracy, y, c=row_colors, s=78, edgecolors="white", linewidths=0.7, zorder=3)
    span = max(float(np.ptp(accuracy)), 0.4)
    ax1.set_xlim(accuracy.min() - span * 0.10, accuracy.max() + span * 0.45)
    for i, row in enumerate(rows):
        ax1.annotate(f"{accuracy[i]:.2f}%  ({row['correct']:,})", (accuracy[i], i),
                     xytext=(9, 0), textcoords="offset points", va="center", fontsize=9,
                     fontweight="bold" if row["selected"] else "normal", color="#263B4C")
        if row["selected"]:
            ax1.scatter([accuracy[i]], [i], s=165, facecolors="none", edgecolors="#213B4A",
                        linewidths=1.5, zorder=4)
    ax1.set_yticks(y, [row["candidate"] for row in rows])
    ax1.set_ylim(len(rows) - 0.45, -0.55)
    ax1.set_title("Accuracy and number correct", loc="left", pad=14)
    ax1.set_xlabel("Accuracy (%) — zoomed horizontal axis", labelpad=10)
    ax1.tick_params(axis="y", length=0, pad=10)

    ax2.axvline(0, color=colors["baseline"], linewidth=1, zorder=1)
    ax2.hlines(y, np.minimum(gain, 0), np.maximum(gain, 0), colors=row_colors,
               linewidth=3, alpha=0.55, zorder=2)
    ax2.scatter(gain, y, c=row_colors, s=54, edgecolors="white", linewidths=0.6, zorder=3)
    gain_span = max(float(np.ptp(gain)), 0.4)
    ax2.set_xlim(min(float(gain.min()), 0) - gain_span * 0.26,
                 max(float(gain.max()), 0) + gain_span * 0.26)
    for i, row in enumerate(rows):
        offset = -8 if gain[i] < 0 else 8
        ax2.annotate(f"{gain[i]:+.2f}", (gain[i], i), xytext=(offset, 0),
                     textcoords="offset points", va="center",
                     ha="right" if gain[i] < 0 else "left", fontsize=9,
                     fontweight="bold" if row["selected"] else "normal", color="#263B4C")
    ax2.set_title("Difference from previous ensemble", loc="left", pad=14)
    ax2.set_xlabel("Accuracy change (percentage points)", labelpad=10)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    for ax in (ax1, ax2):
        ax.axhline(2.5, color="#D7DFE5", linewidth=1)
        ax.grid(axis="x", color="#E8ECF0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CAD2DB")
        ax.tick_params(colors="#3B4A59")
    fig.suptitle("Seed and random-erasing experiments", x=0.025, ha="left", y=0.974,
                 fontsize=18, fontweight="bold", color="#213B4A")
    fig.text(0.025, 0.917,
             f"Same grouped holdout: {selection['validation_samples']:,} images. "
             "All candidates use original + horizontal-flip TTA; ensembles average probabilities equally.",
             fontsize=10, color="#526273")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                       markeredgecolor="none", markersize=7, label=label)
               for color, label in [(colors["single"], "Single-model / one-factor trials"),
                                     (colors["baseline"], "Previous ensemble"),
                                     (colors["ensemble"], "Equal-weight ensembles")]]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                          markeredgecolor="#213B4A", markersize=9, label="Selected candidate"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.025, 0.104),
               ncol=4, frameon=False, fontsize=9, borderaxespad=0)
    extra_correct = selection["selected"]["correct"] - selection["baseline"]["correct"]
    selected_gain = 100 * (selection["selected"]["accuracy"] - selection["baseline"]["accuracy"])
    fig.text(0.025, 0.067,
             f"Selected vs previous ensemble: {selected_gain:+.2f} percentage points; "
             f"{extra_correct:+d} correct predictions. Count in parentheses = images classified correctly.",
             fontsize=10, fontweight="bold", color="#213B4A")
    fig.text(0.025, 0.029,
             "Repeated model-selection holdout; these are not untouched test results or Kaggle scores. "
             "Top three rows compare the same no-maxpool ResNet18 architecture.",
             fontsize=9, color="#526273")
    fig.subplots_adjust(left=0.245, right=0.975, bottom=0.215, top=0.83, wspace=0.22)
    png = output / "comparison.png"
    fig.savefig(png, dpi=180, facecolor="white")
    plt.close(fig)
    csv_path = output / "comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {png}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
