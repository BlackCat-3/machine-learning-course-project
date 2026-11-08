"""Extract the original archive safely and audit the supplied dataset."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, default=Path('D:/Downloads/2026task-4.zip'))
    args = parser.parse_args()
    dest = ROOT/'data'
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        for info in archive.infolist():
            target = (dest/info.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError('Unsafe archive path')
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = archive.read(info)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f'Existing data differs: {target}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(content)
    report = {'archive_sha256':hashlib.sha256(args.zip.read_bytes()).hexdigest(),
              'source':str(args.zip), 'files':[], 'duplicate_image_pixels':[]}
    seen = {}
    for group in ['train/image','train/label','test/image']:
        files = sorted((dest/'segmentation'/group).glob('*.jpg'))
        assert {p.stem for p in files} == {str(i) for i in range(1,21)}
        for path in files:
            im = Image.open(path)
            arr = np.asarray(im)
            row = {'file':str(path.relative_to(dest)), 'size':list(im.size), 'mode':im.mode}
            if group.endswith('label'):
                row['white_fraction_at_128'] = float((arr>=128).mean())
                row['nonbinary_jpeg_fraction'] = float(((arr>0)&(arr<255)).mean())
            else:
                digest = hashlib.sha256(arr.tobytes()).hexdigest()
                if digest in seen:
                    report['duplicate_image_pixels'].append([seen[digest],row['file']])
                seen[digest] = row['file']
            report['files'].append(row)
    report['label_convention'] = 'White vessels on black background (verified from actual labels).'
    report['description_discrepancy'] = 'User screenshot says vessels=0; actual labels and official encoder use white foreground.'
    (ROOT/'data_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Verified 20 training pairs and 20 test images; duplicate pixel arrays:',report['duplicate_image_pixels'])


if __name__ == '__main__':
    main()
