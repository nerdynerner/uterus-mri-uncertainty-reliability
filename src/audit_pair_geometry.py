"""Header and mask audit. Geometry matches remain candidates for visual review."""
import argparse
import collections
import csv
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from study_manifest import sha256, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory, participants, pairs, errors = [], [], [], []
    for cohort, folder in [('D1', 'D1_MHS'), ('D2', 'D2_TCPW')]:
        for directory in sorted((args.root/folder).iterdir()):
            if not directory.is_dir():
                continue
            files = sorted(directory.glob('*.nii.gz'))
            sequence_records, mask_records = [], []
            for path in files:
                try:
                    image = nib.load(path)
                    q, qc = image.get_qform(coded=True)
                    s, sc = image.get_sform(coded=True)
                    suffix = path.name.removeprefix(directory.name+'_').removesuffix('.nii.gz').strip()
                    row = {'participant': directory.name, 'cohort': cohort,
                           'path': str(path.relative_to(args.root)), 'suffix': suffix,
                           'sha256': sha256(path), 'shape': list(image.shape),
                           'spacing': list(map(float, image.header.get_zooms())),
                           'affine': image.affine.tolist(), 'orientation': ''.join(nib.aff2axcodes(image.affine)),
                           'qform_code': int(qc), 'sform_code': int(sc),
                           'qform_sform_conflict': bool(qc and sc and not np.allclose(q, s, atol=1e-3))}
                    if suffix in ('T1', 'T1FS', 'T2', 'T2FS'):
                        row['kind'] = 'image'
                        sequence_records.append(row)
                    else:
                        row['kind'] = 'mask'
                        data = np.asarray(image.dataobj)
                        values = np.unique(data)
                        foreground = data > 0
                        row.update(finite=bool(np.isfinite(data).all()),
                                   label_values=values.tolist(), nonzero=int(foreground.sum()),
                                   connected_components=int(label(foreground)[1]))
                        if foreground.any():
                            indices = np.where(foreground)
                            row['foreground_bbox'] = [[int(v.min()), int(v.max())] for v in indices]
                        mask_records.append(row)
                    inventory.append(row)
                except Exception as error:
                    errors.append({'path': str(path.relative_to(args.root)), 'error': repr(error)})
            uterus = [m for m in mask_records if m['suffix'].split('_')[0].strip() == 'ut']
            candidates = []
            for image in sequence_records:
                if image['suffix'] not in ('T2', 'T2FS'):
                    continue
                for mask in uterus:
                    same = (image['shape'] == mask['shape'] and
                            np.allclose(image['affine'], mask['affine'], atol=1e-3))
                    pair = {'participant': directory.name, 'cohort': cohort,
                            'image': image['path'], 'mask': mask['path'],
                            'sequence': image['suffix'], 'reference': mask['suffix'],
                            'same_physical_grid': bool(same),
                            'qform_sform_conflict': image['qform_sform_conflict'] or mask['qform_sform_conflict'],
                            'binary_nonempty_mask': mask['label_values'] == [0, 1] and mask['nonzero'] > 0,
                            'overlay_review': 'pending', 'admission': 'not_approved'}
                    pairs.append(pair)
                    if same:
                        candidates.append(pair)
            participants.append({'participant': directory.name, 'cohort': cohort,
                                 'sequences': [x['suffix'] for x in sequence_records],
                                 'uterus_references': [x['suffix'] for x in uterus],
                                 'same_grid_t2_pairs': len(candidates), 'files': len(files)})
            print(directory.name, 'files', len(files), 'same-grid uterus candidates', len(candidates), flush=True)
    write_new(args.output/'inventory.json', inventory)
    write_new(args.output/'participants.json', participants)
    write_new(args.output/'pair_candidates.json', pairs)
    write_new(args.output/'read_errors.json', errors)
    groups = collections.defaultdict(list)
    for row in inventory:
        if row['kind'] == 'image':
            groups[row['sha256']].append(row['path'])
    write_new(args.output/'duplicate_image_bytes.json', [v for v in groups.values() if len(v) > 1])
    summary = {'cohort_directories': dict(collections.Counter(p['cohort'] for p in participants)),
               'nifti_files': len(inventory), 'read_errors': len(errors),
               't2_uterus_pair_candidates': len(pairs),
               'same_grid_pairs': sum(p['same_physical_grid'] for p in pairs),
               'participants_with_same_grid_t2_uterus': dict(collections.Counter(
                   p['cohort'] for p in participants if p['same_grid_t2_pairs'])),
               'qform_sform_conflicts': sum(r['qform_sform_conflict'] for r in inventory),
               'clinical_performance_estimated': False,
               'limitation': 'Header and label audit only. Overlay review and original-image intensity QA pending.'}
    write_new(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
