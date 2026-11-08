"""Use the official DRIVE model after verifying its training-set identity.

No external test masks are requested or used. This model already saw all local
training images, so its training diagnostics are NOT holdout validation scores.
"""
import argparse
from collections import OrderedDict
import csv
import hashlib
import json
import shutil

import cv2
import numpy as np
from PIL import Image
import torch

import pipeline as base
import fr_unet_transfer as fr
import improve_v2 as v2
from third_party.fr_unet.models.fr_unet import FR_UNet

WEIGHT=fr.VENDOR/'pretrained_weights/DRIVE/checkpoint-epoch40.pth'


def hysteresis(prob,low=.4,high=.5):
    """Keep each 8-connected low-threshold component containing a high seed."""
    assert 0<=low<=high<=1
    n,labels=cv2.connectedComponents((prob>=low).astype(np.uint8),connectivity=8)
    keep=np.zeros(n,dtype=bool)
    keep[np.unique(labels[prob>=high])]=True
    keep[0]=False
    return keep[labels]


def load():
    with torch.serialization.safe_globals([(OrderedDict,'bunch.Bunch')]):
        checkpoint=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    assert checkpoint['config']['data_set']['name']=='DRIVE'
    model=FR_UNet().cuda()
    model.load_state_dict({k.removeprefix('module.'):val for k,val in checkpoint['state_dict'].items()},strict=True)
    return model,checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='official_drive_d4')
    parser.add_argument('--tta',type=int,choices=[1,4,8],default=8)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    out=v2.ROOT/args.name;out.mkdir(parents=True,exist_ok=True)
    source=v2.ROOT/'source_identity'
    identity=json.loads((source/'training_identity.json').read_text())
    labels=json.loads((source/'training_label_identity.json').read_text())
    assert identity['all_20_exactly_match_local_training']
    assert identity['no_original_training_image_matches_local_test']
    assert labels['all_20_training_labels_match']
    model,checkpoint=load()
    config=dict(checkpoint['config'])
    threshold=float(config['threshold'])
    assert threshold==.5
    metadata={'model':'Official FR-UNet DRIVE checkpoint epoch 40','weight_sha256':hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
              'source':json.loads((fr.VENDOR/'drive_weight_provenance.json').read_text()),
              'checkpoint_config':config,'tta':args.tta,'threshold':threshold,
              'training_images_exactly_match_local_train':True,'training_labels_exactly_match_local_train':True,
              'no_external_test_labels_used':True,'development_score_claimed':False,
              'warning':'All 20 local labeled images were in upstream training. Diagnostics on them are training fit only.',
              'cv2':cv2.__version__,'torch':torch.__version__}
    previous_metadata=json.loads((out/'provenance.json').read_text()) if (out/'provenance.json').exists() else {}
    cache_compatible=all(previous_metadata.get(k)==metadata[k] for k in ['tta','weight_sha256'])
    base.write_json(out/'provenance.json',metadata)
    shutil.copy2(__file__,out/'inference_snapshot.py')
    cache=out/'test_probabilities.npz'
    if cache.exists() and not args.refresh and cache_compatible:
        z=np.load(cache);probs={i:z[str(i)] for i in range(1,21)}
    else:
        probs={}
        for i in range(1,21):
            image=base.read_image(base.DATA/'test/image'/f'{i}.jpg')
            probs[i]=v2.predict(model,image,'gray',args.tta)
            print('PREDICTED',i,flush=True)
        np.savez_compressed(cache,**{str(i):p for i,p in probs.items()})
    v2.export_probs(probs,threshold,out/'standard_official',encoding='official')
    v2.export_probs(probs,threshold,out/'standard_probability',encoding='probability')
    # Optional connectivity candidate, using the thresholds embedded in upstream checkpoint.
    masks={i:hysteresis(p,low=float(config['threshold_low']),high=threshold).astype(np.float32) for i,p in probs.items()}
    v2.export_probs(masks,.5,out/'hysteresis_official',encoding='official')
    base.write_json(out/'hysteresis_official/postprocess.json',{'low':float(config['threshold_low']),'high':threshold,
                    'connectivity':8,'note':'Connected-component hysteresis implementation; not the buggy upstream iteration routine.'})
    # Diagnostics use only already-provided training labels. These cannot establish generalization.
    examples=[1,2,3,12]
    images,masks_train=base.load_train()
    train_probs={i:v2.predict(model,images[i],'gray',args.tta) for i in examples}
    np.savez_compressed(out/'training_diagnostic_probabilities.npz',**{str(i):p for i,p in train_probs.items()})
    base.preview(out/'training_fit_NOT_validation.png',images,masks_train,train_probs,threshold,examples)
    base.write_json(out/'training_fit_NOT_validation.json',{'ids':examples,'native_dice':[base.dice(train_probs[i]>=threshold,masks_train[i]>.5) for i in examples],
                    'note':'Training fit only: upstream model saw all these images. Do not compare as holdout validation.'})
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
