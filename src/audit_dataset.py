"""Inventory original NIfTI files without changing them. Requires numpy/nibabel.

This generates candidates for review, never an approved training manifest.
Matching geometry is necessary but does not prove anatomical registration.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    import nibabel as nib
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--archive', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.archive:
        observed = digest(a.archive, 'md5')
        result = dict(md5=observed, expected='7ace6e1b08efa10d0a1967073b0ba41c',
                      match=observed=='7ace6e1b08efa10d0a1967073b0ba41c',
                      sha256=digest(a.archive))
        (a.output/'archive_checksum.json').write_text(json.dumps(result,indent=2))
        if not result['match']:
            raise RuntimeError('Archive checksum mismatch; stop before use')
    rows = []
    for path in sorted(a.root.rglob('*.nii.gz')):
        img = nib.load(path)
        data = np.asarray(img.dataobj)
        rel = path.relative_to(a.root)
        rows.append(dict(path=str(rel), participant=path.name.split('_')[0],
                         sha256=digest(path),shape=list(img.shape),
                         spacing=list(map(float,img.header.get_zooms())),
                         orientation=''.join(nib.aff2axcodes(img.affine)),
                         affine=img.affine.tolist(),
                         qform_code=int(img.header['qform_code']),
                         sform_code=int(img.header['sform_code']),
                         finite=bool(np.isfinite(data).all()),
                         nonzero=int(np.count_nonzero(data)),
                         min=float(np.nanmin(data)),max=float(np.nanmax(data))))
    (a.output/'nifti_inventory.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    ids = sorted({r['participant'] for r in rows})
    (a.output/'participant_ids.json').write_text(json.dumps(ids,indent=2))
    # Duplicate byte hashes require adjudication; never auto-exclude.
    groups = {}
    for row in rows:
        groups.setdefault(row['sha256'],[]).append(row['path'])
    (a.output/'duplicate_files.json').write_text(json.dumps(
        [v for v in groups.values() if len(v)>1],indent=2))
    with open(a.output/'manual_pair_review.csv','x',newline='') as f:
        csv.writer(f).writerow(['participant','image','mask','rater','structure',
                               'same_physical_grid','overlay_review','decision','reason'])


if __name__ == '__main__':
    main()
