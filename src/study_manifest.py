"""Dependency-free admission and partition rules. Never infers label approval.

Input JSON: {"cases": [{"participant": ..., "cohort": "D1"|"D2",
 "image": relative_path, "image_sha256": ..., "sequence": "T2"|"T2FS",
 "axial_verified": true, "identity_verified": true,
 "geometry_review": "approved", "overlay_review": "approved",
 "complete_uterus_label": true, "review_record": relative_path,
 "masks": [{"path": ..., "sha256": ..., "rater": ...}],
 "training_rater": ...}]}.

One primary sequence per globally resolved participant. Repeated examinations
are excluded from this primary manifest, with one partition per participant.
"""
import hashlib
import json
import math
import random
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def within(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Manifest paths must be relative to the data root')
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes data root')
    if not target.is_file():
        raise ValueError('Missing file: ' + relative)
    return target


def validate_manifest(document, root, verify_bytes=True):
    cases = document.get('cases', [])
    if not cases:
        raise ValueError('An empty manifest cannot authorize experiments')
    ids, hashes = set(), set()
    for row in cases:
        participant = row.get('participant')
        if not isinstance(participant, str) or not participant.strip() or participant in ids:
            raise ValueError('Unique globally resolved participant IDs required')
        ids.add(participant)
        if row.get('cohort') not in ('D1', 'D2') or row.get('sequence') not in ('T2', 'T2FS'):
            raise ValueError('Verified cohort and T2-family sequence required')
        for key in ('axial_verified', 'identity_verified', 'complete_uterus_label'):
            if row.get(key) is not True:
                raise ValueError(key + ' must be explicitly true')
        for key in ('geometry_review', 'overlay_review'):
            if row.get(key) != 'approved':
                raise ValueError(key + ' must be approved')
        within(root, row.get('review_record'))
        image = within(root, row.get('image'))
        digest = row.get('image_sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('A lowercase SHA-256 is required')
        if digest in hashes:
            raise ValueError('Duplicate primary images require adjudication before splitting')
        hashes.add(digest)
        if verify_bytes and sha256(image) != digest:
            raise ValueError('Image checksum changed')
        masks = row.get('masks', [])
        raters = [m.get('rater') for m in masks]
        if not masks or any(not isinstance(r, str) or not r for r in raters) or len(set(raters)) != len(raters):
            raise ValueError('Nonempty uniquely identified reference masks required')
        if row.get('training_rater') not in raters:
            raise ValueError('Prespecify the training/validation reference rater')
        for mask in masks:
            target = within(root, mask.get('path'))
            if verify_bytes and sha256(target) != mask.get('sha256'):
                raise ValueError('Mask checksum changed')
    return cases


def make_splits(cases, seed=61900):
    if len({r['participant'] for r in cases}) != len(cases):
        raise ValueError('Resolve repeated participant IDs before partitioning')
    source = sorted(r['participant'] for r in cases if r['cohort'] == 'D2')
    external = sorted(r['participant'] for r in cases if r['cohort'] == 'D1')
    if len(source) < 10 or not external:
        raise ValueError('Insufficient cases for the proposed four-way source split')
    random.Random(seed).shuffle(source)
    # Largest-remainder allocation avoids implementation-dependent rounding.
    fractions = (.5, .15, .15, .2)
    counts = [math.floor(len(source) * f) for f in fractions]
    order = sorted(range(4), key=lambda i: (-(len(source) * fractions[i] - counts[i]), i))
    for i in order[:len(source) - sum(counts)]:
        counts[i] += 1
    result, offset = {}, 0
    for name, count in zip(('train', 'validation', 'calibration', 'internal_test'), counts):
        result[name] = sorted(source[offset:offset + count])
        offset += count
    result['external_test'] = external
    return {'seed': seed, 'allocation': 'largest remainder; fixed order breaks ties', 'partitions': result}


def validate_splits(cases, split):
    parts = split['partitions']
    required = {'train', 'validation', 'calibration', 'internal_test', 'external_test'}
    if set(parts) != required or any(not values for values in parts.values()):
        raise ValueError('Five nonempty partitions required')
    flattened = [p for values in parts.values() for p in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != {r['participant'] for r in cases}:
        raise ValueError('Partitions must cover the manifest once, without overlap')
    cohorts = {r['participant']: r['cohort'] for r in cases}
    for name, ids in parts.items():
        if any(cohorts[p] != ('D1' if name == 'external_test' else 'D2') for p in ids):
            raise ValueError('Cohort leakage detected')
    return parts


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    cases = validate_manifest(json.loads(args.manifest.read_text()), args.data_root)
    split = make_splits(cases)
    validate_splits(cases, split)
    split['manifest_sha256'] = sha256(args.manifest)
    write_new(args.output, split)
