"""V2: controlled contrast/patch/EMA study, image-level CV, and official export.

All labels come from the supplied training set. Official CHASEDB1 weights are
the only external learned initialization. The 512 nonzero-label metric is an
explicit local proxy for the organizer's conversion, not a verified server metric.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

import pipeline as base
import fr_unet_transfer as fr

ROOT = base.ROOT/'outputs/improve_v2_20261101'
THRESHOLDS = np.round(np.arange(.2,.71,.025),3)


def folds():
    first = [1,2,3,12]
    rest = np.random.default_rng(20261102).permutation([i for i in range(1,21) if i not in first])
    return [first]+[sorted(map(int,rest[i:i+4])) for i in range(0,16,4)]


def prep(im, mode):
    if mode == 'gray':
        return fr.preprocess(im)
    green = im[...,1].copy()
    a = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(green).astype(np.float32)/255
    a[im.max(axis=2)<10] = 0
    return a


@torch.inference_mode()
def predict(model, im, mode, tta=4):
    model.eval()
    a = prep(im,mode)
    h,w = a.shape
    x = torch.from_numpy(a[None,None]).cuda()
    size = ((max(h,w)+15)//16)*16
    x = F.pad(x,(0,size-w,0,size-h))
    predictions=[]
    for k,flip in ([(0,False)] if tta==1 else
                   [(0,False),(0,True),(2,False),(2,True)] if tta==4 else
                   [(k,f) for k in range(4) for f in [False,True]]):
        z=torch.rot90(x,k,(-2,-1))
        if flip:z=z.flip(-1)
        with torch.backends.cudnn.flags(benchmark=False,deterministic=True),torch.autocast('cuda'):
            p=model(z).sigmoid().float()
        if flip:p=p.flip(-1)
        p=torch.rot90(p,-k,(-2,-1))
        predictions.append(p[0,0,:h,:w].cpu().numpy())
    return np.mean(predictions,axis=0)


def official_resize(mask):
    return np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((512,512),Image.Resampling.BILINEAR))>0


def score(probs,masks,ids):
    target={i:official_resize(masks[i]>.5) for i in ids}
    rows=[]
    for t in THRESHOLDS:
        native=[base.dice(probs[i]>=t,masks[i]>.5) for i in ids]
        proxy=[base.dice(official_resize(probs[i]>=t),target[i]) for i in ids]
        rows.append({'threshold':float(t),'native_dice':float(np.mean(native)),
                     'official_proxy_dice':float(np.mean(proxy)),
                     'native_per_image':native,'proxy_per_image':proxy})
    return max(rows,key=lambda x:x['official_proxy_dice']),rows


def make_batch(arrays,masks,ids,args):
    xx,yy=[],[]
    for _ in range(args.batch):
        idx=random.choice(ids);a=arrays[idx];b=masks[idx];h,w=b.shape
        crop=round(args.patch*random.uniform(.85,1.15))
        for attempt in range(4):
            y,x=random.randrange(h-crop+1),random.randrange(w-crop+1)
            aa,bb=a[y:y+crop,x:x+crop],b[y:y+crop,x:x+crop]
            if aa.mean()>.08 or attempt==3:break
        aa=cv2.resize(aa,(args.patch,args.patch),interpolation=cv2.INTER_LINEAR)
        bb=cv2.resize(bb,(args.patch,args.patch),interpolation=cv2.INTER_NEAREST)
        if random.random()<.4:
            mat=cv2.getRotationMatrix2D((args.patch/2,args.patch/2),random.uniform(-180,180),1)
            aa=cv2.warpAffine(aa,mat,(args.patch,args.patch),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            bb=cv2.warpAffine(bb,mat,(args.patch,args.patch),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
        k=random.randrange(4);aa,bb=np.rot90(aa,k),np.rot90(bb,k)
        if random.random()<.5:aa,bb=aa[:,::-1],bb[:,::-1]
        aa=np.clip(aa*random.uniform(.9,1.1)+random.uniform(-.025,.025),0,1)**random.uniform(.9,1.1)
        xx.append(aa[None].copy());yy.append(bb[None].copy())
    return torch.from_numpy(np.stack(xx)).cuda(),torch.from_numpy(np.stack(yy)).cuda()


@torch.no_grad()
def update_ema(ema,model,step):
    decay=min(.99,(1+step)/(10+step))
    for dst,src in zip(ema.parameters(),model.parameters()):dst.lerp_(src,1-decay)
    for dst,src in zip(ema.buffers(),model.buffers()):
        if dst.is_floating_point():dst.lerp_(src,1-decay)
        else:dst.copy_(src)


def load_model(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    model=fr.create_model(torch.device('cuda'))
    model.load_state_dict(ckpt['model'])
    return model,ckpt


def train(args):
    name=args.name or f'{args.mode}_fold{args.fold}_seed{args.seed}'
    out=ROOT/name
    if (out/'completed.json').exists():
        print('ALREADY COMPLETE',name,flush=True);return
    if out.exists():raise FileExistsError('Incomplete run exists: use a new --name or investigate before rerunning')
    out.mkdir(parents=True)
    images,masks=base.load_train()
    val_ids=[] if args.fold==-1 else folds()[args.fold]
    ids=[i for i in range(1,21) if i not in val_ids]
    arrays={i:prep(images[i],args.mode) for i in ids}
    base.seed_all(args.seed)
    model=fr.create_model(torch.device('cuda'))
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda')
    base.write_json(out/'config.json',{**vars(args),'train_ids':ids,'validation_ids':val_ids})
    shutil.copy2(__file__,out/'source_snapshot.py')
    best=-1.;best_step=0;losses=[];start=time.time()
    for step in range(1,args.steps+1):
        model.train()
        lr=args.lr*(.1+.9*(1+np.cos(np.pi*min(step/args.schedule_steps,1)))/2)
        for group in optimizer.param_groups:group['lr']=lr
        x,y=make_batch(arrays,masks,ids,args)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits=model(x);p=logits.sigmoid().float()
            d=1-((2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean()
            loss=F.binary_cross_entropy_with_logits(logits,y)+.5*d
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()>=old_scale:update_ema(ema,model,step)
        losses.append(loss.item())
        if step%args.eval_every==0 or step==args.steps:
            row={'step':step,'loss':float(np.mean(losses)),'seconds':round(time.time()-start,1)};losses=[]
            keep=(step==args.steps) if args.fixed else False
            if val_ids:
                probs={i:predict(ema,images[i],args.mode,1) for i in val_ids}
                sel,_=score(probs,masks,val_ids);row.update(sel)
                if not args.fixed and sel['official_proxy_dice']>best:
                    keep=True;best=sel['official_proxy_dice']
            else:keep=(step==args.steps)
            if keep:
                best_step=step
                torch.save({'model':ema.state_dict(),'step':step,'mode':args.mode,'seed':args.seed,
                            'train_ids':ids,'val_ids':val_ids,'config':vars(args)},out/'best.pt')
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(name+' '+json.dumps(row),flush=True)
    del model,ema,optimizer
    torch.cuda.empty_cache()
    model,ckpt=load_model(out/'best.pt')
    result={'step':ckpt['step'],'train_ids':ids,'validation_ids':val_ids,'fixed':args.fixed}
    if val_ids:
        probs={i:predict(model,images[i],args.mode,4) for i in val_ids}
        np.savez_compressed(out/'validation_probs.npz',**{str(i):p for i,p in probs.items()})
        sel,grid=score(probs,masks,val_ids)
        result.update(sel);result['threshold_search']=grid
        base.preview(out/'validation_preview.png',images,masks,probs,sel['threshold'],val_ids)
    base.write_json(out/'completed.json',result)
    print('COMPLETED '+name+' '+json.dumps({k:v for k,v in result.items() if k!='threshold_search'}),flush=True)


def export_probs(probs,threshold,out,encoding='official'):
    out.mkdir(parents=True,exist_ok=True)
    native=out/'native_masks';native.mkdir(exist_ok=True)
    masks512=out/'masks_512';masks512.mkdir(exist_ok=True)
    rows=[]
    for i in range(1,21):
        p=probs[i]
        assert p.shape==(584,565) and np.isfinite(p).all() and 0<=p.min()<=p.max()<=1
        mask=p>=threshold
        Image.fromarray(mask.astype('uint8')*255).save(native/f'{i}.png')
        if encoding=='official':mask512=official_resize(mask)
        else:mask512=np.asarray(Image.fromarray(p).resize((512,512),Image.Resampling.BILINEAR))>=threshold
        Image.fromarray(mask512.astype('uint8')*255).save(masks512/f'{i}.png')
        rle=base.rle_encode(mask512)
        assert np.array_equal(base.rle_decode(rle),mask512)
        rows.append({'Id':str(i),'Predicted':rle})
    with (out/'submission.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['Id','Predicted']);writer.writeheader();writer.writerows(rows)
    spec=importlib.util.spec_from_file_location('official',base.DATA/'segmentation_to_csv.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    reference=official.turn_to_str([(native if encoding=='official' else masks512)/f'{i}.png' for i in range(1,21)])
    assert reference==[r['Predicted'] for r in rows]
    base.write_json(out/'audit.json',{'passed':True,'rows':20,'threshold':threshold,'encoding':encoding,
        'official_encoder_exact_match':True,'native_input_to_official_encoder':encoding=='official',
        'rle_roundtrip_all_pass':True,'sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()})


def infer(args):
    paths=[Path(x) for x in args.checkpoints]
    out=ROOT/(args.name or 'ensemble');out.mkdir(parents=True,exist_ok=True)
    images={i:base.read_image(base.DATA/'test/image'/f'{i}.jpg') for i in range(1,21)}
    predictions=[];metadata=[]
    for path in paths:
        model,ckpt=load_model(path)
        checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        cache=out/f'probs_{checkpoint_hash[:12]}_tta{args.tta}.npz'
        if cache.exists():
            z=np.load(cache);probs={i:z[str(i)] for i in range(1,21)}
        else:
            probs={i:predict(model,images[i],ckpt['mode'],args.tta) for i in range(1,21)}
            np.savez_compressed(cache,**{str(i):p for i,p in probs.items()})
        predictions.append(probs)
        metadata.append({'checkpoint':str(path.resolve()),'sha256':checkpoint_hash,
                         'train_ids':ckpt['train_ids'],'step':ckpt['step'],'mode':ckpt['mode'],'seed':ckpt['seed']})
        del model;torch.cuda.empty_cache()
        print('INFERRED',path,flush=True)
    mean={i:np.mean([p[i] for p in predictions],axis=0) for i in range(1,21)}
    np.savez_compressed(out/'test_probabilities.npz',**{str(i):p for i,p in mean.items()})
    base.write_json(out/'ensemble.json',{'models':metadata,'weights':[1/len(paths)]*len(paths),
                                      'tta':args.tta,'threshold':args.threshold,'encoding':args.encoding})
    export_probs(mean,args.threshold,out,args.encoding)
    print('SUBMISSION',out/'submission.csv',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','infer'])
    parser.add_argument('--mode',choices=['gray','clahe'],default='clahe')
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--name')
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--schedule-steps',type=int,default=4000)
    parser.add_argument('--eval-every',type=int,default=500)
    parser.add_argument('--patch',type=int,default=96)
    parser.add_argument('--batch',type=int,default=12)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--checkpoints',nargs='+',default=[])
    parser.add_argument('--threshold',type=float,default=.4)
    parser.add_argument('--encoding',choices=['official','probability'],default='official')
    parser.add_argument('--tta',type=int,choices=[1,4,8],default=4)
    args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=True
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.stage=='train':train(args)
    else:infer(args)


if __name__=='__main__':main()
