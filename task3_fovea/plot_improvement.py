"""Plot the completed fixed-fold improvement experiments."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/improvement_20261024'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

robust = json.loads((OUT / 'robust_heatmap/selection.json').read_text(encoding='utf8'))
refine = json.loads((OUT / 'refinement/selection.json').read_text(encoding='utf8'))
data = np.load(OUT / 'robust_heatmap/oof_predictions.npz')
old_error = np.linalg.norm(data['baseline'] - data['truth'], axis=1)
new_error = np.linalg.norm(data['robust_only'] - data['truth'], axis=1)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
names = ['Original', 'Local refinement', 'Robust ResNet34']
values = [robust['baseline_rmse'], refine['metrics']['local_heatmap']['rmse'], robust['candidates']['robust_only']['metrics']['rmse']]
bars = axes[0].bar(names, values, color=['#a5adb5', '#dcaa54', '#27876d'])
axes[0].bar_label(bars, fmt='%.2f', padding=4)
axes[0].set(ylim=(0, 34), ylabel='Pooled coordinate RMSE (pixels)', title='Same 80 records and four grouped folds')
axes[0].tick_params(axis='x', labelrotation=12)
for values, label, color in [(robust['baseline_fold_rmse'], 'Original', '#89939c'),
                             ([x['rmse'] for x in refine['fold_metrics']['local_heatmap']], 'Local refinement', '#dcaa54'),
                             (robust['candidates']['robust_only']['fold_rmse'], 'Robust ResNet34', '#27876d')]:
    axes[1].plot(range(1, 5), values, '-o', label=label, color=color)
axes[1].set(xticks=range(1, 5), xlabel='Fold', ylabel='Coordinate RMSE (pixels)', title='Improvement in all four folds')
axes[1].legend(frameon=False)
for error, name, color in [(old_error, 'Original', '#89939c'), (new_error, 'Robust ResNet34', '#27876d')]:
    axes[2].plot(np.sort(error), np.arange(1, 81) / 80, label=name, color=color, linewidth=2)
axes[2].set(xlabel='Euclidean landmark error (pixels)', ylabel='Fraction of validation images',
            title=f'Lower error on {(new_error < old_error).sum()}/80 images', ylim=(0, 1.02))
axes[2].legend(frameon=False)
fig.suptitle('Model-selection validation | Original Kaggle score: 14.73696 (user-reported); new score pending', fontsize=13)
fig.savefig(OUT / 'comparison.png', dpi=180)
plt.close(fig)
print(OUT / 'comparison.png')
