"""Audit saved validation evidence and every serialized submission pixel."""
import argparse
import csv
import hashlib
import json
import numpy as np
from PIL import Image
import torch
from pipeline import ROOT, DATA, rle_decode, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='baseline_v1')
    args = parser.parse_args()
    out = ROOT/'outputs'/args.run
    selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    split = json.loads((out/'split.json').read_text(encoding='utf-8'))
    assert len(split['train']) == 16 and len(split['validation']) == 4
    assert not set(split['train']) & set(split['validation'])
    assert sorted(split['train']+split['validation']) == list(range(1,21))
    for name,digest in json.loads((out/'data_sha256.json').read_text(encoding='utf-8')).items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest
    checkpoint = torch.load(out/'validation/best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['train_ids'] == split['train']
    assert checkpoint['val_ids'] == split['validation']
    assert checkpoint['selection']['step'] == selection['step']
    final = torch.load(out/'full_train/final.pt',map_location='cpu',weights_only=False)
    assert final['train_ids'] == list(range(1,21)) and final['steps'] == selection['step']
    assert all(torch.isfinite(t).all() for t in final['model'].values())
    scores = []
    probs = np.load(out/'validation_probabilities.npz')
    for idx in split['validation']:
        truth = np.asarray(Image.open(DATA/'train/label'/f'{idx}.jpg')) >= 128
        prob = probs[str(idx)]
        assert prob.shape == truth.shape and np.isfinite(prob).all()
        assert prob.min()>=0 and prob.max()<=1
        pred = prob >= selection['threshold']
        tp = np.count_nonzero(pred & truth)
        fp = np.count_nonzero(pred & ~truth)
        fn = np.count_nonzero(~pred & truth)
        scores.append(2*tp/(2*tp+fp+fn))
    assert np.allclose(scores,selection['per_image'])
    assert np.isclose(np.mean(scores),selection['dice'])
    with (out/'submission.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['Id','Predicted']
        rows = list(reader)
    assert [r['Id'] for r in rows] == list(map(str,range(1,21)))
    fractions = []
    for row in rows:
        mask = np.asarray(Image.open(out/'test_masks_512'/f"{row['Id']}.png"))
        assert mask.shape == (512,512) and set(np.unique(mask)) <= {0,255}
        restored = rle_decode(row['Predicted'])
        assert np.array_equal(restored,mask==255)
        assert 0 < restored.mean() < 1
        fractions.append(float(restored.mean()))
    original = json.loads((out/'submission_audit.json').read_text(encoding='utf-8'))
    assert original['all_rows_equal_official_encoder']
    assert hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest() == original['sha256']
    result = {'passed':True, 'validation_mean_dice_recomputed':float(np.mean(scores)),
              'validation_per_image':dict(zip(map(str,split['validation']),scores)),
              'test_foreground_fraction_range':[min(fractions),max(fractions)],
              'submission_rows':len(rows), 'submission_sha256':original['sha256'],
              'checks':['image split isolation','source hashes unchanged','checkpoint training IDs',
                        'selected training duration','finite weights and probabilities','independent Dice recomputation',
                        'CSV schema and ID order','20 exact RLE roundtrips','binary PNG dimensions','official encoder comparison record']}
    write_json(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
