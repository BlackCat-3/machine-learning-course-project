"""Render training history and a held-out confusion matrix as standalone PNGs."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/resnet18_main"))
    args = parser.parse_args()
    history = json.loads((args.run_dir / "history.json").read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    uses_mixup = any(row.get("train_accuracy_kind") == "mixup_weighted" for row in history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Training (Mixup)" if uses_mixup else "Training (augmented)")
    axes[1].plot(epochs, [100 * row["train_accuracy"] for row in history], label="Training (Mixup weighted)" if uses_mixup else "Training (augmented)")
    if any(row["val_loss"] is not None for row in history):
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
        axes[1].plot(epochs, [100 * row["raw_val_accuracy"] for row in history], label="Validation: raw")
        if any(row["ema_val_accuracy"] is not None for row in history):
            axes[1].plot(epochs, [100 * row["ema_val_accuracy"] for row in history], label="Validation: EMA")
    for ax, ylabel in zip(axes, ["Cross-entropy loss", "Accuracy (%)"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.run_dir / "training_curves.png", dpi=160)
    plt.close(fig)
    metrics_path = args.run_dir / "best_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        matrix = np.asarray(metrics["confusion_matrix"])
        names = list(metrics["class_to_index"])
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xticks(range(6), names, rotation=30, ha="right")
        ax.set_yticks(range(6), names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Held-out validation: {metrics['accuracy']:.2%} (without flip TTA)")
        fig.colorbar(im, ax=ax, label="Number of images")
        fig.tight_layout()
        fig.savefig(args.run_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    print(f"Figures saved in {args.run_dir.resolve()}")


if __name__ == "__main__":
    main()
