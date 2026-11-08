"""Plot actual saved out-of-fold predictions and training histories."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
out = ROOT / 'outputs/study'
selection = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
oof = np.load(out / 'oof_predictions.npz')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
names = ['Mean position', 'Heatmap', 'Regression', 'Equal mean']
values = [selection['constant_baseline']['rmse']] + [selection['metrics'][n]['rmse'] for n in ('heatmap', 'regression', 'equal_mean')]
bars = axes[0, 0].bar(names, values, color=['#a9b2ba', '#297d9b', '#e09f3e', '#2c8c65'])
axes[0, 0].bar_label(bars, fmt='%.2f', padding=4)
axes[0, 0].set_ylim(0, max(values) * 1.18)
axes[0, 0].set_ylabel('Pooled coordinate RMSE (original pixels)')
axes[0, 0].set_title('Four-fold model-selection validation; 80 records')
for arch, color in [('heatmap', '#297d9b'), ('regression', '#e09f3e')]:
    fold_scores = [r['rmse'] for r in selection['fold_metrics'][arch]]
    axes[0, 1].plot(range(1, 5), fold_scores, '-o', label=arch.title(), color=color)
axes[0, 1].set(xticks=range(1, 5), xlabel='Validation fold', ylabel='Coordinate RMSE (pixels)', title='Variation between the four 20-image folds')
axes[0, 1].legend(frameon=False)
errors = np.sqrt(np.mean((oof[selection['selected']] - oof['truth']) ** 2, axis=1))
axes[1, 0].hist(errors, bins=[0, 10, 20, 30, 40, 60, 80, 100, 140], color='#2c8c65', edgecolor='white')
axes[1, 0].set(xlabel='Per-image coordinate RMSE (pixels)', ylabel='Number of images', title='Selected mean: distribution of validation errors')
for fold in range(4):
    path = ROOT / f'outputs/baseline/validation_fold{fold}_seed42/history.csv'
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    axes[1, 1].plot([int(r['epoch']) for r in rows], [float(r['rmse']) for r in rows], label=f'Fold {fold + 1}')
axes[1, 1].set(xlabel='Epoch', ylabel='Validation RMSE (pixels)', title='Heatmap validation curves: retain best checkpoint')
axes[1, 1].legend(frameon=False, ncol=2)
fig.suptitle('Fovea localization | Local validation, not a Kaggle score', fontsize=15, fontweight='bold')
fig.savefig(out / 'validation_summary.png', dpi=180)
plt.close(fig)
print(out / 'validation_summary.png')
