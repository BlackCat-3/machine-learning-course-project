"""Official FR-UNet with CHASEDB1 pretraining, adapted to this assignment.

Upstream: https://github.com/lseventeen/FR-UNet (MIT).
Only the supplied training labels are used for fine-tuning and selection.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

import pipeline as base
from third_party.fr_unet.models.fr_unet import FR_UNet

VENDOR = base.ROOT/'third_party/fr_unet'
WEIGHT = VENDOR/'pretrained_weights/CHASEDB1/checkpoint-epoch40.pth'


def create_model(device):
    # Ignore the old Bunch config object and load tensor weights with the restricted loader.
    with torch.serialization.safe_globals([(OrderedDict, 'bunch.Bunch')]):
        checkpoint = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    assert checkpoint['config']['data_set']['name'] == 'CHASEDB1'
    model = FR_UNet().to(device)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items()}, strict=True)
    return model


def preprocess(im):
    # Equivalent to upstream PIL Grayscale + standardize + per-image min-max:
    # the global mean and standard deviation cancel algebraically in min-max.
    a = np.asarray(Image.fromarray(im).convert('L')).astype(np.float32)/255
    return (a-a.min()) / max(float(a.max()-a.min()),1e-6)


@torch.inference_mode()
def predict(model, im, device, tta=False):
    model.eval()
    a = preprocess(im)
    x = torch.from_numpy(a[None,None]).to(device)
    h,w = a.shape
    # Same bottom/right zero padding convention as the official data processor.
    size = ((max(h,w)+15)//16)*16
    x = F.pad(x,(0,size-w,0,size-h))
    result = []
    for axes in ([(),(2,),(3,),(2,3)] if tta else [()]):
        xx = x.flip(axes) if axes else x
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True), torch.autocast('cuda'):
            p = model(xx).sigmoid().float()
        if axes:
            p = p.flip(axes)
        result.append(p[0,0,:h,:w].cpu().numpy())
    return np.mean(result,axis=0)


def evaluate(model, images, masks, ids, device, tta=False):
    probs = {i:predict(model,images[i],device,tta) for i in ids}
    scores=[]
    for threshold in np.arange(.2,.71,.05):
        ds=[base.dice(probs[i]>=threshold,masks[i]>.5) for i in ids]
        scores.append({'threshold':round(float(threshold),2),'dice':float(np.mean(ds)),'per_image':ds})
    return max(scores,key=lambda x:x['dice']),scores,probs


def batch(gray,masks,ids,args,device):
    xx,yy=[],[]
    size=args.patch
    for _ in range(args.batch):
        idx=random.choice(ids)
        h,w=masks[idx].shape
        y,x=random.randrange(h-size+1),random.randrange(w-size+1)
        a,b=gray[idx][y:y+size,x:x+size],masks[idx][y:y+size,x:x+size]
        k=random.randrange(4)
        a,b=np.rot90(a,k),np.rot90(b,k)
        if random.random()<.5:
            a,b=a[:,::-1],b[:,::-1]
        a=np.clip(a*random.uniform(.9,1.1)+random.uniform(-.025,.025),0,1)**random.uniform(.9,1.1)
        xx.append(a[None].copy());yy.append(b[None].copy())
    return torch.from_numpy(np.stack(xx)).to(device),torch.from_numpy(np.stack(yy)).to(device)


def train(images,masks,ids,val_ids,args,out,steps):
    base.seed_all(args.seed)
    device=torch.device('cuda')
    model=create_model(device)
    gray={i:preprocess(images[i]) for i in ids}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda')
    best={'dice':-1.0}
    start=time.time()
    if val_ids:
        best,_,_=evaluate(model,images,masks,val_ids,device)
        best['step']=0
        torch.save({'model':model.state_dict(),'selection':best,'train_ids':[],'val_ids':val_ids},out/'best.pt')
        print('PRETRAINED '+json.dumps(best),flush=True)
        base.write_json(out/'pretrained_validation.json',best)
    losses=[]
    for step in range(1,steps+1):
        model.train()
        x,y=batch(gray,masks,ids,args,device)
        lr=1e-4*(.2+.8*(1+np.cos(np.pi*step/args.steps))/2)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits=model(x)
            p=logits.sigmoid().float()
            dl=1-((2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean()
            loss=F.binary_cross_entropy_with_logits(logits,y)+.5*dl
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(opt);scaler.update()
        losses.append(loss.item())
        if step%args.eval_every==0 or step==steps:
            record={'step':step,'loss':float(np.mean(losses)),'seconds':round(time.time()-start,1)}
            losses=[]
            if val_ids:
                selected,_,_=evaluate(model,images,masks,val_ids,device)
                record.update(selected)
                if selected['dice']>best['dice']:
                    best={**selected,'step':step}
                    torch.save({'model':model.state_dict(),'selection':best,'train_ids':ids,'val_ids':val_ids},out/'best.pt')
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
    if not val_ids:
        torch.save({'model':model.state_dict(),'train_ids':ids,'steps':steps,'pretraining':'CHASEDB1'},out/'final.pt')
    return model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='fr_unet_chase_v1')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--eval-every',type=int,default=200)
    parser.add_argument('--patch',type=int,default=64)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20261101)
    parser.add_argument('--predict-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda')
    out=base.ROOT/'outputs'/args.run
    if args.predict_only:
        model=create_model(device)
        model.load_state_dict(torch.load(out/'full_train/final.pt',map_location=device,weights_only=True)['model'])
        selection=json.loads((out/'selection.json').read_text())
        base.export(model,selection['threshold'],out,device,predict_fn=predict)
        return
    out.mkdir(parents=True,exist_ok=False)
    for name in ['validation','full_train']:(out/name).mkdir()
    shutil.copy2(__file__,out/'fr_unet_transfer_snapshot.py')
    shutil.copy2(base.ROOT/'pipeline.py',out/'pipeline_snapshot.py')
    source_manifest={str(p.relative_to(VENDOR)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in VENDOR.rglob('*.py')}
    base.write_json(out/'source_manifest.json',source_manifest)
    base.write_json(out/'config.json',vars(args))
    base.write_json(out/'pretrained_source.json',{'repository':'https://github.com/lseventeen/FR-UNet',
                    'commit':json.loads((VENDOR/'provenance.json').read_text())['commit'],
                    'checkpoint':str(WEIGHT.relative_to(base.ROOT)),
                    'sha256':hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
                    'dataset_in_checkpoint_config':'CHASEDB1','license':'MIT'})
    images,masks=base.load_train()
    split=json.loads((base.ROOT/'outputs/baseline_v1/split.json').read_text())
    base.write_json(out/'split.json',split)
    shutil.copy2(base.ROOT/'outputs/baseline_v1/data_sha256.json',out/'data_sha256.json')
    train(images,masks,split['train'],split['validation'],args,out/'validation',args.steps)
    checkpoint=torch.load(out/'validation/best.pt',map_location=device,weights_only=True)
    model=create_model(device);model.load_state_dict(checkpoint['model'])
    selected,scores,probs=evaluate(model,images,masks,split['validation'],device,tta=True)
    base_score=json.loads((base.ROOT/'outputs/baseline_v1/selection.json').read_text())['dice']
    selection={**selected,'step':checkpoint['selection']['step'],'validation_ids':split['validation'],
               'threshold_search':scores,'tta':'four flips','baseline_dice':base_score,
               'delta_dice':selected['dice']-base_score,
               'note':'Same four-image development holdout; reused for model and threshold selection, not independent test performance.'}
    base.write_json(out/'selection.json',selection)
    np.savez_compressed(out/'validation_probabilities.npz',**{str(i):p for i,p in probs.items()})
    base.preview(out/'validation_preview.png',images,masks,probs,selected['threshold'],split['validation'])
    print('SELECTION '+json.dumps(selection),flush=True)
    del model
    torch.cuda.empty_cache()
    if selected['dice']<=base_score:
        print('NO IMPROVEMENT: baseline retained',flush=True)
        return
    model=train(images,masks,list(range(1,21)),[],args,out/'full_train',selection['step'])
    base.export(model,selection['threshold'],out,device,predict_fn=predict)
    print('COMPLETE '+str(out/'submission.csv'),flush=True)


if __name__=='__main__':
    main()
