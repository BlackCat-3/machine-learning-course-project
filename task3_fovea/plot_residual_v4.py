"""Plot actual paired OOF results, including all 80 samples."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare_residual_v4 import OUT


def main():
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf8'))
    z = np.load(OUT / 'oof_predictions.npz')
    choice = selection['selected'] or selection['best_candidate']
    k = list(z['steps']).index(choice['steps'])
    refined = z['baseline'] + choice['residual_weight'] * (z['refined'][k] - z['baseline'])
    base_error = np.linalg.norm(z['baseline'] - z['truth'], axis=1)
    refined_error = np.linalg.norm(refined - z['truth'], axis=1)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for weight, color in [(1., '#e08032'), (.5, '#168b85')]:
        cs = [c for c in selection['candidates'] if c['residual_weight'] == weight]
        axs[0].plot([c['steps'] for c in cs], [c['metrics']['rmse'] for c in cs], 'o-',
                    color=color, label=f'Residual weight {weight:g}')
    axs[0].axhline(selection['baseline_rmse'], color='#555', ls='--', label='Current coarse model')
    axs[0].axhline(.95 * selection['baseline_rmse'], color='#aaa', ls=':', label='5% gain threshold')
    axs[0].set(xlabel='Shared optimizer updates', ylabel='Original-pixel RMSE', title='Four-fold pooled comparison')
    axs[0].legend(fontsize=8)
    x = np.arange(4)
    axs[1].bar(x - .18, selection['baseline_fold_rmse'], .36, label='Current', color='#8b9baa')
    axs[1].bar(x + .18, choice['fold_rmse'], .36, label='Candidate', color='#168b85')
    axs[1].set(xticks=x, xticklabels=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'],
               ylabel='Original-pixel RMSE', title=f'Candidate: {choice["steps"]} updates, weight {choice["residual_weight"]:g}')
    axs[1].legend(fontsize=8)
    axs[2].scatter(base_error, refined_error, s=20, alpha=.75, color='#168b85')
    maximum = max(base_error.max(), refined_error.max()) * 1.04
    axs[2].plot([0, maximum], [0, maximum], '--', color='#888', lw=1)
    for j in np.argsort(base_error)[-3:]:
        axs[2].annotate(str(z['ids'][j]), (base_error[j], refined_error[j]), xytext=(3, 3), textcoords='offset points')
    count = int(np.sum(refined_error < base_error))
    axs[2].set(xlabel='Current Euclidean error (pixels)', ylabel='Candidate Euclidean error (pixels)',
               title=f'All 80 training images; {count} improved')
    fig.suptitle('Native-pixel global-context residual refinement | model-selection OOF, not online score', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', dpi=160)
    squared_before = np.sum((z['baseline'] - z['truth'])**2, axis=1)
    squared_after = np.sum((refined - z['truth'])**2, axis=1)
    diagnostics = {
        'candidate': choice, 'improved_images': count,
        'baseline_median_euclidean': float(np.median(base_error)),
        'candidate_median_euclidean': float(np.median(refined_error)),
        'largest_baseline_errors': [{'id': int(z['ids'][j]), 'baseline_euclidean': float(base_error[j]),
                                    'candidate_euclidean': float(refined_error[j]),
                                    'squared_error_change': float(squared_after[j] - squared_before[j])}
                                   for j in np.argsort(base_error)[-5:][::-1]],
    }
    (OUT / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf8')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
