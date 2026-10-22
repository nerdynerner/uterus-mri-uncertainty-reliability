"""Native-grid secondary endpoints. Grid and binary-mask checks precede use.

Surface metrics crop only their joint evaluation domain, never inference data.
Only orthogonal voxel axes are supported; oblique orthogonal directions are OK.
"""
import itertools
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from monai.metrics import compute_surface_dice, compute_hausdorff_distance


def overlap(a, b):
    total = int(a.sum())+int(b.sum())
    return 1. if not total else 2*int((a & b).sum())/total


def surfaces(prediction, reference, spacing):
    a, b = np.asarray(prediction, dtype=bool), np.asarray(reference, dtype=bool)
    if a.shape != b.shape or a.ndim != 3 or len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError('Matched 3D grids and positive spacing required')
    if not a.any() or not b.any():
        return {'hd95_mm': None, 'surface_dice': {'2': None,'3': None,'5': None},
                'empty_case': 'both' if not a.any() and not b.any() else 'one',
                'scope': 'Undefined empty surfaces; retain case in overlap/failure summaries'}
    indices = np.where(a | b)
    box = tuple(slice(max(0,int(i.min())-1), min(a.shape[j],int(i.max())+2)) for j,i in enumerate(indices))
    # Explicit background padding retains closed surfaces at the crop limits.
    x = torch.from_numpy(np.pad(a[box],1).astype(np.float32))[None,None]
    y = torch.from_numpy(np.pad(b[box],1).astype(np.float32))[None,None]
    result = {'surface_dice': {}, 'empty_case': None,
              'hd95_definition': 'MONAI symmetric maximum of directed 95th-percentile surface distances'}
    result['hd95_mm'] = float(compute_hausdorff_distance(x,y,include_background=True,
        percentile=95., spacing=list(spacing), directed=False)[0,0])
    for tolerance in (2.,3.,5.):
        result['surface_dice'][str(int(tolerance))] = float(compute_surface_dice(x,y,
            class_thresholds=[tolerance], include_background=True, spacing=list(spacing),
            use_subvoxels=True)[0,0])
    return result


def expert_disagreement(mean_probability, references, spacing):
    """Exact existing domain, with redundant distant background removed.

    All reference foreground plus a >5 mm halo is retained. Consequently every
    voxel in the prespecified union-plus-5-mm domain and its reference boundary
    neighbourhood has the same distances, entropy and reference values.
    """
    if len(references) < 2:
        return _expert_disagreement_full_grid(mean_probability,references,spacing)
    masks=[np.asarray(x,dtype=bool) for x in references]
    if any(m.shape != mean_probability.shape for m in masks):
        raise ValueError('Reference/probability grids differ')
    union=np.logical_or.reduce(masks)
    if not union.any():
        return _expert_disagreement_full_grid(mean_probability,masks,spacing)
    positions=np.where(union)
    margin=np.ceil(5./np.asarray(spacing)).astype(int)+2
    box=tuple(slice(max(0,int(p.min())-int(margin[i])),
                    min(union.shape[i],int(p.max())+int(margin[i])+1))
              for i,p in enumerate(positions))
    return _expert_disagreement_full_grid(mean_probability[box],
        [m[box] for m in masks],spacing)


def _expert_disagreement_full_grid(mean_probability, references, spacing):
    if len(references) < 2:
        return {'status': 'undefined_fewer_than_two_compatible_references'}
    masks = [np.asarray(x,dtype=bool) for x in references]
    shape = mean_probability.shape
    if any(m.shape != shape for m in masks):
        raise ValueError('Reference/probability grids differ')
    union = np.logical_or.reduce(masks)
    if not union.any():
        return {'status': 'undefined_all_references_empty'}
    intersection = np.logical_and.reduce(masks)
    disagreement = union & ~intersection
    region = distance_transform_edt(~union,sampling=spacing) <= 5.
    p = np.clip(mean_probability,1e-7,1-1e-7)
    entropy = -(p*np.log2(p)+(1-p)*np.log2(1-p))
    boundary = np.logical_or.reduce([m & ~binary_erosion(m) for m in masks])
    boundary_band = distance_transform_edt(~boundary,sampling=spacing) <= 2.
    def enrichment(domain):
        disagree, agree = domain & disagreement, domain & ~disagreement
        return {'disagreement_voxels': int(disagree.sum()), 'agreement_voxels': int(agree.sum()),
                'mean_entropy_difference': float(entropy[disagree].mean()-entropy[agree].mean())
                    if disagree.any() and agree.any() else None}
    return {'status': 'computed',
            'pairwise_reference_dice': [overlap(a,b) for a,b in itertools.combinations(masks,2)],
            'union_plus_5mm': enrichment(region), 'boundary_2mm_sensitivity': enrichment(region & boundary_band),
            'soft_reference_brier': float(((mean_probability[region]-np.mean(masks,axis=0)[region])**2).mean()),
            'majority_vote_available': len(masks) >= 3,
            'definition': 'Entropy difference per participant; no voxel-level inferential sample size'}


if __name__ == '__main__':
    import argparse
    import json
    from study_manifest import write_new
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    a = np.zeros((24,24,24),dtype=bool)
    a[6:18,6:18,6:18] = True
    identical = surfaces(a,a,[2.,1.,3.])
    assert identical['hd95_mm'] == 0.
    assert all(v == 1. for v in identical['surface_dice'].values())
    b = np.roll(a,1,axis=0)
    shifted = surfaces(a,b,[2.,1.,3.])
    assert abs(shifted['hd95_mm']-2.) < 1e-6
    assert shifted['surface_dice']['3'] == 1.
    assert surfaces(a,np.zeros_like(a),[2.,1.,3.])['hd95_mm'] is None
    disagreement = expert_disagreement(a.astype(float)*.8+.1,[a,b],[2.,1.,3.])
    assert disagreement['status'] == 'computed'
    record = {'status':'passed','clinical_data_used':False,
              'identical_surface':identical,'known_2mm_shift':shifted,
              'empty_surface_handled':True,'disagreement_domain_computed':True,
              'scope':'Artificial cubes only; real endpoint integration still required'}
    write_new(args.output,record)
    print(json.dumps(record,indent=2))
