"""Evaluate peak-local decoding with existing held-out models before test inference."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
from contracts import coordinate_metrics, write_submission
from data import original_coordinates
from model import heatmap_coordinates
from robust_heatmap import RobustHeatmap
from run import ROOT, normalize, save_json

BASE = ROOT / 'outputs/improvement_20261024/robust_heatmap'
OUT = ROOT / 'outputs/iteration3_20261024/decode'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def decode(logits):
    flat = logits.float().flatten(1)
    h, w = logits.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=logits.device), torch.arange(w, device=logits.device), indexing='ij')
    peak = flat.argmax(1)
    mask = (xx[None] - (peak % w)[:, None, None]).square() + (yy[None] - (peak // w)[:, None, None]).square() <= 36
    limited = logits.float().masked_fill(~mask[:, None], -torch.inf)
    return heatmap_coordinates(limited)

@torch.inference_mode()
def infer(model, images, indices, records):
    model.eval()
    result = {'original': [], 'peak_local': []}
    for start in range(0, len(indices), 8):
        idx = indices[start:start+8]
        x = images[idx].float() / 255
        variants = {k: [] for k in result}
        for flip in (False, True):
            with torch.autocast('cuda'):
                logits = model(normalize(x.flip(-1) if flip else x))
            values = {'original': heatmap_coordinates(logits), 'peak_local': decode(logits)}
            for name, xy in values.items():
                if flip: xy[:, 0] = 1 - xy[:, 0]
                variants[name].append(xy)
        for name in result:
            xy = torch.stack(variants[name]).mean(0).cpu().numpy()
            result[name].append(original_coordinates(xy, [records[i] for i in idx]))
    return {k: np.concatenate(v) for k, v in result.items()}

def load_checkpoint(directory, records, validation):
    ckpt = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    expected = [r['id'] for r in records if r['split']=='train' and (not validation or r['fold'] != validation[0])]
    assert ckpt['config']['train_ids'] == expected
    if validation:
        assert ckpt['config']['validation_ids'] == [r['id'] for r in records if r['split']=='train' and r['fold']==validation[0]]
    else:
        assert not ckpt['config']['validation_ids']
    model = RobustHeatmap(False).cuda()
    model.load_state_dict(ckpt['model'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['validate','final'], default='validate')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    records = json.loads((ROOT/'outputs/baseline/prepared/manifest.json').read_text())['records']
    images = torch.from_numpy(np.load(BASE/'images448.npz')['images'].transpose(0,3,1,2).copy()).cuda()
    train = [i for i,r in enumerate(records) if r['split']=='train']
    test = [i for i,r in enumerate(records) if r['split']=='test']
    assert train == list(range(80))
    sources = {'script':digest(__file__), 'baseline_oof':digest(BASE/'oof_predictions.npz'),
               'radius_cells':6, 'candidates':['peak_local','peak_half'],
               'gate':{'relative_reduction':.05,'improved_folds':3}}
    if args.stage == 'validate':
        preds = []
        old = np.load(BASE/'oof_predictions.npz')
        for seed in (43,44):
            oof = {name:np.full((80,2),np.nan) for name in ('original','peak_local')}
            for fold in range(4):
                directory = BASE/f'fold{fold}_seed{seed}'
                model = load_checkpoint(directory, records, [fold])
                indices = [i for i in train if records[i]['fold']==fold]
                values = infer(model,images,indices,records)
                for name in oof: oof[name][indices] = values[name]
                sources[directory.name] = digest(directory/'best.pt')
                del model
            delta = np.max(np.abs(oof['original']-old[f'seed{seed}']))
            np.savez(OUT/f'reinference_seed{seed}.npz', **oof)
            print(f'Baseline re-inference seed {seed}: max_delta={delta:.6f}px',flush=True)
            assert delta < .2, f'Baseline reproduction differs by {delta} pixels'
            preds.append(oof['peak_local'])
        peak = np.mean(preds,0)
        truth, baseline = old['truth'], old['robust_only']
        folds = np.array([records[i]['fold'] for i in train])
        candidates = {'peak_local':peak,'peak_half':(peak+baseline)/2}
        base_score = coordinate_metrics(truth,baseline)['rmse']
        base_folds = [coordinate_metrics(truth[folds==f],baseline[folds==f])['rmse'] for f in range(4)]
        details = {}
        for name, xy in candidates.items():
            metrics = coordinate_metrics(truth,xy)
            fm = [coordinate_metrics(truth[folds==f],xy[folds==f])['rmse'] for f in range(4)]
            passed = metrics['rmse'] <= base_score*.95 and sum(a<b for a,b in zip(fm,base_folds))>=3
            details[name] = {'metrics':metrics,'fold_rmse':fm,'passed_gate':passed}
        eligible = [k for k,v in details.items() if v['passed_gate']]
        selected = min(eligible,key=lambda k:details[k]['metrics']['rmse']) if eligible else None
        result = {'selected':selected,'candidates':details,'baseline_rmse':base_score,'baseline_folds':base_folds,'sources':sources,
                  'note':'Same folds used in prior model/epoch selection; not independent test performance.'}
        save_json(OUT/'selection.json',result)
        np.savez(OUT/'oof_predictions.npz',truth=truth,baseline=baseline,**candidates)
        print(json.dumps(result,indent=2),flush=True)
    else:
        result = json.loads((OUT/'selection.json').read_text())
        assert result['selected'] and result['sources']['script']==sources['script']
        values=[]
        for seed in (43,44):
            directory=BASE/f'full_seed{seed}'
            model=load_checkpoint(directory,records,[])
            values.append(infer(model,images,test,records))
            sources[directory.name]=digest(directory/'best.pt')
            del model
        peak=np.mean([v['peak_local'] for v in values],0)
        baseline=np.mean([v['original'] for v in values],0)
        final=peak if result['selected']=='peak_local' else (peak+baseline)/2
        np.savez(OUT/'test_predictions.npz',ids=[records[i]['id'] for i in test],peak=peak,baseline=baseline,final=final)
        path=OUT/'submission_decode.csv'
        write_submission(ROOT/'data/detection/sample_submission.csv',dict(zip([records[i]['id'] for i in test],final)),path,
                         {records[i]['id']:(records[i]['width'],records[i]['height']) for i in test})
        save_json(OUT/'result.json',{'selected':result['selected'],'sources':sources,'submission_sha256':digest(path),'online_score':None})
        print(path)

if __name__=='__main__': main()
