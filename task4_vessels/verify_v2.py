"""Verify the frozen V2 submission, then publish its convenient local copy."""
import csv
import hashlib
import importlib.util
import json
import shutil
import numpy as np
from PIL import Image
import pipeline as base
import improve_v2 as v2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=v2.ROOT
    run=root/'official_drive_d4'
    chosen=run/'hysteresis_official'
    meta=json.loads((run/'provenance.json').read_text())
    weight=base.ROOT/'third_party/fr_unet/pretrained_weights/DRIVE/checkpoint-epoch40.pth'
    assert sha(weight)==meta['source']['sha256']==meta['weight_sha256']
    assert meta['tta']==8 and meta['threshold']==.5
    assert not meta['development_score_claimed']
    old=base.ROOT/'outputs/fr_unet_chase_v1/submission.csv'
    assert sha(old)=='15590468d20e8d918b890eb89103535c03bcf654363809c8938e019557c6a223'
    # Saved before an independent --refresh process; matching this checks reproduction.
    expected='f7de24df3b7800aa91b98a461d7fb01e61cd5e21903e3fd41ac7e49f46e4f7da'
    assert sha(chosen/'submission.csv')==expected
    with (chosen/'submission.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['Id','Predicted']
        rows=list(reader)
    with (base.DATA/'sample_submission.csv').open(newline='') as f:
        ids=[r['Id'] for r in csv.DictReader(f)]
    assert [r['Id'] for r in rows]==ids and len(rows)==20
    spec=importlib.util.spec_from_file_location('official',base.DATA/'segmentation_to_csv.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    assert official.turn_to_str([chosen/'native_masks'/f'{i}.png' for i in ids])==[r['Predicted'] for r in rows]
    fractions=[]
    for row in rows:
        values=np.array([int(v) for v in row['Predicted'].split()],dtype=np.int64)
        assert len(values)%2==0
        starts,lengths=values[::2],values[1::2]
        assert (starts>=1).all() and (lengths>0).all()
        assert (starts+lengths-1<=512*512).all()
        assert (starts[1:]>starts[:-1]+lengths[:-1]-1).all()
        decoded=base.rle_decode(row['Predicted'])
        native=Image.open(chosen/'native_masks'/f"{row['Id']}.png").convert('L')
        assert native.size==(565,584) and set(np.unique(native)).issubset({0,255})
        converted=np.asarray(native.resize((512,512),Image.Resampling.BILINEAR))>0
        assert np.array_equal(decoded,converted)
        fractions.append(float(decoded.mean()))
    target=root/'submission_v2.csv'
    shutil.copy2(chosen/'submission.csv',target)
    assert sha(target)==expected
    sources=['drive_pretrained_infer.py','improve_v2.py','fr_unet_transfer.py','pipeline.py','verify_v2.py']
    base.write_json(root/'verification.json',{
        'passed':True,'rows':20,'old_submission_unchanged':True,
        'checkpoint_sha256_verified':True,'independent_refresh_matches_first_export':True,
        'official_conversion_exact_all_20':True,'rle_structure_and_roundtrip_pass':True,
        'submission_sha256':expected,'recommended_file':str(target.resolve()),
        'source_candidate':str((chosen/'submission.csv').resolve()),
        'foreground_fraction_range':[min(fractions),max(fractions)],
        'source_sha256':{name:sha(base.ROOT/name) for name in sources},
        'kaggle_score':None,'note':'No Kaggle submission performed. Official DRIVE model training fit is not validation.'})
    print(json.dumps({'passed':True,'submission':str(target),'sha256':expected},ensure_ascii=False))


if __name__=='__main__':main()
