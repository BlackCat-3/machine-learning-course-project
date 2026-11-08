"""Reproducible patch U-Net baseline; only supplied competition labels are used."""
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'segmentation'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.GroupNorm(4, b), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.GroupNorm(4, b), nn.ReLU())


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([block(3, 16), block(16, 32), block(32, 64), block(64, 128)])
        self.dec = nn.ModuleList([block(192, 64), block(96, 32), block(48, 16)])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.enc):
            if i:
                x = F.max_pool2d(x, 2)
            x = layer(x)
            skips.append(x)
        for layer, skip in zip(self.dec, reversed(skips[:-1])):
            x = layer(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
        return self.head(x)


def read_image(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def load_train():
    images, masks = {}, {}
    for i in range(1, 21):
        images[i] = read_image(DATA / 'train/image' / f'{i}.jpg')
        masks[i] = (np.asarray(Image.open(DATA / 'train/label' / f'{i}.jpg')) >= 128).astype(np.float32)
    return images, masks


def make_batch(images, masks, ids, batch_size, patch, device):
    xx, yy = [], []
    for _ in range(batch_size):
        idx = random.choice(ids)
        im, mask = images[idx], masks[idx]
        h, w = mask.shape
        y, x = random.randrange(h-patch+1), random.randrange(w-patch+1)
        im, mask = im[y:y+patch, x:x+patch], mask[y:y+patch, x:x+patch]
        k = random.randrange(4)
        im, mask = np.rot90(im, k), np.rot90(mask, k)
        if random.random() < .5:
            im, mask = im[:, ::-1], mask[:, ::-1]
        im = im.astype(np.float32) / 255
        im = np.clip(im * random.uniform(.85, 1.15) + random.uniform(-.04, .04), 0, 1)
        im = im ** random.uniform(.85, 1.15)
        xx.append(im.transpose(2, 0, 1))
        yy.append(mask[None].copy())
    return (torch.from_numpy(np.stack(xx)).to(device) * 2 - 1,
            torch.from_numpy(np.stack(yy)).to(device))


@torch.inference_mode()
def predict(model, im, device, tta=False):
    model.eval()
    x = torch.from_numpy(im.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)/127.5-1
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode='reflect')
    predictions = []
    for axes in ([(), (2,), (3,), (2, 3)] if tta else [()]):
        xx = x.flip(axes) if axes else x
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            p = model(xx).sigmoid().float()
        predictions.append(p.flip(axes) if axes else p)
    return torch.stack(predictions).mean(0)[0, 0, :h, :w].cpu().numpy()


def dice(p, y):
    return float((2 * np.logical_and(p, y).sum() + 1e-7) / (p.sum()+y.sum()+1e-7))


def evaluate(model, images, masks, ids, device, tta=False):
    probs = {i: predict(model, images[i], device, tta) for i in ids}
    scores = []
    for threshold in np.arange(.2, .71, .05):
        per_image = [dice(probs[i] >= threshold, masks[i] > .5) for i in ids]
        scores.append({'threshold': round(float(threshold), 2), 'dice': float(np.mean(per_image)), 'per_image': per_image})
    return max(scores, key=lambda r: r['dice']), scores, probs


def train(images, masks, ids, val_ids, steps, args, out, device):
    seed_all(args.seed)
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best = {'dice': -1.0}
    started = time.time()
    losses = []
    for step in range(1, steps + 1):
        model.train()
        x, y = make_batch(images, masks, ids, args.batch, args.patch, device)
        # Fixed schedule permits full-data refit to follow exactly the selected trajectory.
        lr = .001 * (.2 + .8 * (1 + np.cos(np.pi * step / args.steps)) / 2)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            p = logits.sigmoid().float()
            soft_dice = ((2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean()
            loss = bce + 1-soft_dice
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if step % args.eval_every == 0 or step == steps:
            record = {'step': step, 'loss': float(np.mean(losses)), 'seconds': round(time.time()-started, 1)}
            losses.clear()
            if val_ids:
                selected, _, _ = evaluate(model, images, masks, val_ids, device)
                record.update(selected)
                if selected['dice'] > best['dice']:
                    best = {**selected, 'step': step}
                    torch.save({'model': model.state_dict(), 'selection': best, 'train_ids': ids,
                                'val_ids': val_ids, 'config': vars(args)}, out/'best.pt')
            with (out/'training.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
    if not val_ids:
        torch.save({'model': model.state_dict(), 'steps': steps, 'train_ids': ids,
                    'config': vars(args)}, out/'final.pt')
    return model


def rle_encode(mask):
    pixels = np.concatenate(([0], mask.astype(np.uint8).ravel(order='C'), [0]))
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    changes[1::2] -= changes[::2]
    return ' '.join(map(str, changes))


def rle_decode(text, shape=(512, 512)):
    values = np.array([int(v) for v in text.split()], dtype=np.int64).reshape(-1, 2)
    flat = np.zeros(np.prod(shape), dtype=np.uint8)
    last_end = 0
    for start, length in values:
        assert start >= 1 and length > 0 and start-1 >= last_end and start-1+length <= len(flat)
        flat[start-1:start-1+length] = 1
        last_end = start-1+length
    return flat.reshape(shape)


def preview(path, images, masks, probs, threshold, ids):
    canvas = Image.new('RGB', (4*292, len(ids)*324), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, i in enumerate(ids):
        truth = (masks[i]*255).astype('uint8')
        pred = ((probs[i] >= threshold)*255).astype('uint8')
        error = np.zeros((*truth.shape, 3), dtype='uint8')
        error[(pred>0)&(truth>0)] = (220,220,220)
        error[(pred>0)&(truth==0)] = (255,80,70)
        error[(pred==0)&(truth>0)] = (60,160,255)
        for col, (arr, title) in enumerate(zip([images[i],truth,pred,error], ['Input','Label','Prediction','Red=FP / Blue=FN'])):
            tile = ImageOps.contain(Image.fromarray(arr).convert('RGB'), (292,292))
            canvas.paste(tile,(col*292,row*324+24))
            draw.text((col*292+5,row*324+5),f'{i}: {title}',fill='black')
    canvas.save(path)


def export(model, threshold, out, device, predict_fn=None):
    maskdir = out/'test_masks_512'
    maskdir.mkdir(exist_ok=True)
    rows = []
    for i in range(1, 21):
        im = read_image(DATA/'test/image'/f'{i}.jpg')
        prob = (predict_fn or predict)(model, im, device, tta=True)
        prob512 = np.asarray(Image.fromarray(prob).resize((512,512), Image.Resampling.BILINEAR))
        mask = prob512 >= threshold
        Image.fromarray(mask.astype('uint8')*255).save(maskdir/f'{i}.png')
        encoded = rle_encode(mask)
        assert np.array_equal(rle_decode(encoded), mask)
        rows.append({'Id': str(i), 'Predicted': encoded})
    with (out/'submission.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Id','Predicted'])
        writer.writeheader()
        writer.writerows(rows)
    # Independently compare every row against the organizer's exact encoder.
    spec = importlib.util.spec_from_file_location('official_encoder', DATA/'segmentation_to_csv.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official = module.turn_to_str([maskdir/f'{i}.png' for i in range(1,21)])
    assert official == [row['Predicted'] for row in rows]
    with (DATA/'sample_submission.csv').open(newline='') as f:
        expected = [row['Id'] for row in csv.DictReader(f)]
    assert expected == [row['Id'] for row in rows]
    write_json(out/'submission_audit.json', {'rows':len(rows), 'ids_match_sample':True,
               'all_rle_roundtrips_pass':True, 'all_rows_equal_official_encoder':True,
               'shape':[512,512], 'foreground':'white vessels', 'order':'C (row-major)',
               'sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--patch', type=int, default=192)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--run', default='baseline_v1')
    parser.add_argument('--stage', choices=['all','validate','refit'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA required for this training run')
    torch.backends.cudnn.benchmark = True
    out = ROOT/'outputs'/args.run
    out.mkdir(parents=True, exist_ok=True)
    images, masks = load_train()
    ids = list(range(1,21))
    shuffled = np.random.default_rng(args.seed).permutation(ids).tolist()
    val_ids, train_ids = sorted(shuffled[:4]), sorted(shuffled[4:])
    if args.stage != 'refit':
        if (out/'validation/training.jsonl').exists():
            raise FileExistsError('Use a new --run name; existing experiments are preserved')
        shutil.copy2(__file__,out/'pipeline_snapshot.py')
        write_json(out/'config.json',vars(args))
        write_json(out/'split.json', {'train':train_ids,'validation':val_ids,'unit':'whole image'})
        write_json(out/'environment.json', {'torch':torch.__version__, 'numpy':np.__version__,
                   'gpu':torch.cuda.get_device_name(0), 'cuda':torch.version.cuda})
        manifest = {str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(DATA.rglob('*')) if p.is_file()}
        write_json(out/'data_sha256.json', manifest)
        valout = out/'validation'
        valout.mkdir(exist_ok=True)
        train(images,masks,train_ids,val_ids,args.steps,args,valout,device)
        checkpoint = torch.load(valout/'best.pt',map_location=device,weights_only=False)
        model = UNet().to(device)
        model.load_state_dict(checkpoint['model'])
        selected, scores, probs = evaluate(model,images,masks,val_ids,device,tta=True)
        selection = {**selected, 'step':checkpoint['selection']['step'], 'tta':'four flips',
                     'validation_ids':val_ids, 'threshold_search':scores,
                     'note':'Four-image development holdout used for checkpoint and threshold selection; not unbiased test performance.'}
        write_json(out/'selection.json',selection)
        np.savez_compressed(out/'validation_probabilities.npz', **{str(i):p for i,p in probs.items()})
        preview(out/'validation_preview.png',images,masks,probs,selected['threshold'],val_ids)
        print('SELECTION '+json.dumps(selection),flush=True)
    if args.stage in ['all','refit']:
        selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
        fullout = out/'full_train'
        fullout.mkdir(exist_ok=True)
        if (fullout/'training.jsonl').exists():
            raise FileExistsError('Full training already exists; use a new run')
        model = train(images,masks,ids,[],selection['step'],args,fullout,device)
        export(model,selection['threshold'],out,device)
        print('COMPLETE '+str(out/'submission.csv'),flush=True)


if __name__ == '__main__':
    main()
