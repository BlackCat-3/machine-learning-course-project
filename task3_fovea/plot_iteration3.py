"""Plot measured third-round validation only; never label it as online scoring."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.mplconfig'))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=ROOT/'outputs/iteration3_20261024'
fine=json.loads((out/'fine_heatmap/selection.json').read_text())
source='illumination' if (out/'illumination/selection.json').exists() else 'fine_heatmap'
sel=json.loads((out/source/'selection.json').read_text())
chosen=sel['selected'] or min(sel['candidates'], key=lambda k:sel['candidates'][k]['metrics']['rmse'])
values=np.load(out/source/'oof_predictions.npz')
truth,baseline,prediction=values['truth'],values['baseline'],values[chosen]
old_error=np.linalg.norm(baseline-truth,axis=1)
new_error=np.linalg.norm(prediction-truth,axis=1)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
names=['Current\nResNet34','Peak\ndecode','Local\nrefiner','ResNet50\nblend']
scores=[sel['baseline']['rmse'],16.326937523163267,16.399281784057322,min(v['metrics']['rmse'] for v in fine['candidates'].values())]
colors=['#697A90','#BDC5D0','#BDC5D0','#BDC5D0']
if source=='illumination':
    names.append('Illumination\ncandidate');scores.append(sel['candidates'][chosen]['metrics']['rmse']);colors.append('#168B7A')
else: colors[-1]='#168B7A'
bars=axes[0].bar(names,scores,color=colors)
axes[0].bar_label(bars,fmt='%.2f',padding=4)
axes[0].set_ylim(0,max(scores)*1.18)
axes[0].set_ylabel('Original-pixel coordinate RMSE (lower is better)')
axes[0].set_title('Fixed candidate comparison')
x=np.arange(4)
axes[1].bar(x-.18,[v['rmse'] for v in sel['baseline_folds']],.36,color='#697A90',label='Current')
axes[1].bar(x+.18,[v['rmse'] for v in sel['candidates'][chosen]['fold_metrics']],.36,color='#168B7A',label='Candidate')
axes[1].set_xticks(x,[f'Fold {i+1}' for i in x]);axes[1].legend(frameon=False)
axes[1].set_title('All four folds retained')
for err,label,color in [(old_error,'Current','#697A90'),(new_error,'Candidate','#168B7A')]:
    axes[2].plot(np.sort(err),np.arange(1,len(err)+1)/len(err),label=label,color=color,lw=2)
axes[2].set_xlabel('Euclidean landmark error (pixels)')
axes[2].set_ylabel('Fraction of images')
axes[2].set_title(f'{int((new_error<old_error).sum())}/80 images improved')
axes[2].legend(frameon=False)
for ax in axes: ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
status='Qualified' if sel['selected'] else 'Not qualified'
fig.suptitle(f'Third-round model-selection validation | shared epoch {sel["shared_epoch"]} | {status}',fontsize=14)
fig.savefig(out/'comparison.png',dpi=160)
print(out/'comparison.png')
