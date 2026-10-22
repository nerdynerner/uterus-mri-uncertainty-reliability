"""Prespecified participant analysis. Temperatures fit before held-out inference.

No thresholds, checkpoints, exclusions or endpoints are selected from test outcomes.
"""
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np
import nibabel as nib
from evaluation_runner import ROOT,CODE,EVALUATION,MANIFEST,SPLITS,METHODS,store_once
from study_manifest import sha256,write_new
from analyse_policy import analyse
from secondary_policy_metrics import reference_sensitivity,paired_method_contrast,spearman,fixed_review_budget
from reliability_metrics import auc_failure,average_precision,paired_bootstrap,frozen_policy,quantile
from probability_calibration import sample_patient,fit_temperature,evaluate


def read(path): return json.loads(Path(path).read_text())
def case_map(): return {r['participant']:r for r in read(MANIFEST)['cases']}
def jobs(partition,method):
    marker=read(EVALUATION/f'complete_{partition}_{method}.json')
    folder=EVALUATION/marker['folder']
    if sha256(folder/'participant_outputs.json')!=marker['participant_outputs_sha256']:
        raise ValueError('Output provenance mismatch')
    rows=read(folder/'participant_outputs.json')
    expected=set(read(SPLITS)['partitions'][partition])
    if {r['participant'] for r in rows}!=expected: raise ValueError('Wrong patient set')
    return [(r,folder/f'case_{i:04}') for i,r in enumerate(rows)]

def arrays(row,folder,cases):
    c=cases[row['participant']]
    image=nib.load(str(ROOT/c['image']))
    affine=np.load(folder/'native_affine.npy')
    probability=np.load(folder/'mean_native_probability.npy')
    if probability.shape!=image.shape or not np.allclose(affine,image.affine,atol=1e-5):
        raise ValueError('Native prediction geometry mismatch')
    refs=[]
    for ref in c['masks']:
        obj=nib.load(str(ROOT/ref['path']))
        if obj.shape!=image.shape or not np.allclose(obj.affine,image.affine,atol=1e-3):
            raise ValueError('Native reference geometry mismatch')
        values=np.asanyarray(obj.dataobj)
        if not np.isin(values,[0,1]).all(): raise ValueError('Nonbinary reference')
        refs.append((ref['rater'],values.astype(bool)))
    return c,image,probability,refs

def probability_samples(partition,method,cases):
    result=[]
    for row,folder in jobs(partition,method):
        c,image,p,refs=arrays(row,folder,cases)
        y=next(mask for rater,mask in refs if rater==c['training_rater'])
        # ID-based seed keeps voxel samples matched across methods and reruns.
        seed=61941+int(row['participant'].split('-')[1])+(1000 if c['cohort']=='D1' else 0)
        result.append(sample_patient(p,y,np.asanyarray(image.dataobj),seed))
    return result

def fit():
    cases=case_map()
    result={m:fit_temperature(probability_samples('calibration',m,cases)) for m in METHODS}
    store_once(EVALUATION/'frozen_temperatures.json',{'methods':result,
        'calibration_scores_sha256':sha256(EVALUATION/'calibration_scores.json'),
        'sampling':'4096 image-nonzero voxels per participant; ID-seeded; equal participant weight',
        'code_sha256':sha256(CODE/'probability_calibration.py')})
    print('Secondary temperatures frozen from calibration participants only',flush=True)

def transport(rows_internal,rows_external,threshold,draws=2000):
    rng=np.random.default_rng(61923)
    def effect(rows):
        return frozen_policy([statistics.fmean(r['errors']) for r in rows],
            [r['uncertainty'] for r in rows],threshold)['retained_minus_overall']
    a,b=effect(rows_internal),effect(rows_external)
    values=[]
    for _ in range(draws):
        x=effect([rows_internal[i] for i in rng.integers(len(rows_internal),size=len(rows_internal))])
        y=effect([rows_external[i] for i in rng.integers(len(rows_external),size=len(rows_external))])
        if x is not None and y is not None: values.append(y-x)
    return {'external_minus_internal_effect':b-a if a is not None and b is not None else None,
        'ci95':[quantile(values,.025),quantile(values,.975)] if values else None,
        'valid_draws':len(values),'undefined_draws':draws-len(values),
        'scope':'Independent cohort participant bootstrap; frozen calibration threshold and trained models'}

def run():
    import torch
    from spatial_endpoints import surfaces,expert_disagreement,overlap
    torch.set_num_threads(2)
    frozen=read(EVALUATION/'heldout_code_freeze.json')
    for name,digest in frozen['code_sha256'].items():
        if sha256(CODE/name)!=digest:
            corrections=read(EVALUATION/'analysis_implementation_corrections_v2.json')['files']
            correction=corrections.get(name,{})
            if not (digest==correction.get('original_sha256')
                    and sha256(CODE/name)==correction.get('corrected_sha256')):
                raise ValueError('Unrecorded frozen analysis code change: '+name)
    if not (EVALUATION/'all_heldout_inference_complete.json').exists(): raise ValueError('Incomplete inference')
    cases=case_map(); policy=read(EVALUATION/'frozen_review_policy.json')
    if sha256(EVALUATION/'frozen_review_policy.json')!=frozen['policy_sha256']: raise ValueError('Policy changed')
    temps=read(EVALUATION/'frozen_temperatures.json'); cal=read(EVALUATION/'calibration_scores.json')
    all_results={}; source_rows={}; native=[]
    for part in ('internal_test','external_test'):
        rows=read(EVALUATION/(part+'_scores.json')); source_rows[part]=rows
        result=analyse(rows,policy,part,cal)
        result['secondary']={}
        for method in METHODS:
            group=[r for r in rows if r['method']==method]
            errors=[statistics.fmean(r['errors']) for r in group]; scores=[r['uncertainty'] for r in group]
            samples=probability_samples(part,method,cases)
            secondary={'spearman_error_uncertainty':spearman(errors,scores),
                'reference_sensitivity':reference_sensitivity(group,policy['thresholds'][method]),
                'failure_sensitivity':{str(c):{'auroc':auc_failure([int(e>1-c) for e in errors],scores),
                    'average_precision':average_precision([int(e>1-c) for e in errors],scores),
                    'fixed_review_budget':fixed_review_budget(errors,scores,c)} for c in (.3,.5,.7)},
                'probability_calibration':{'raw':evaluate(samples),
                    'temperature_scaled':evaluate(samples,temps['methods'][method]['temperature'])},
                'empty_prediction_n':sum(r['empty_prediction'] for r in group),
                'both_empty_member_pairs':sum(r['both_empty_member_pairs'] for r in group)}
            disagreement=[]
            for row,folder in jobs(part,method):
                print('Native endpoints',part,method,row['participant'],flush=True)
                c,image,p,refs=arrays(row,folder,cases)
                prediction=p>=.5; spacing=np.linalg.norm(image.affine[:3,:3],axis=0).tolist()
                record={'participant':row['participant'],'partition':part,'method':method,
                    'references':[{'rater':rater,**surfaces(prediction,mask,spacing)} for rater,mask in refs],
                    'expert_disagreement':expert_disagreement(p,[mask for _,mask in refs],spacing)}
                if len(refs)>=3:
                    majority=np.mean([mask for _,mask in refs],axis=0)>.5
                    record['majority_dice_descriptive_only']=overlap(prediction,majority)
                native.append(record)
                if record['expert_disagreement']['status']=='computed': disagreement.append(record['expert_disagreement'])
            secondary['expert_disagreement']={}
            for domain in ('union_plus_5mm','boundary_2mm_sensitivity'):
                values=[d[domain]['mean_entropy_difference'] for d in disagreement if d[domain]['mean_entropy_difference'] is not None]
                secondary['expert_disagreement'][domain]={'n':len(values),
                    'mean_difference_ci':paired_bootstrap(values,statistics.fmean) if values else None}
            result['secondary'][method]=secondary
        result['paired_method_contrasts']=[paired_method_contrast(rows,policy['thresholds'],a,b)
            for a,b in [('ensemble','deterministic'),('dropout','deterministic'),('ensemble','dropout')]]
        all_results[part]=result
    all_results['transport']={m:transport([r for r in source_rows['internal_test'] if r['method']==m],
        [r for r in source_rows['external_test'] if r['method']==m],policy['thresholds'][m]) for m in METHODS}
    all_results['provenance']={'code_freeze_sha256':sha256(EVALUATION/'heldout_code_freeze.json'),
        'temperatures_sha256':sha256(EVALUATION/'frozen_temperatures.json'),
        'interpretation':'Technical retrospective segmentation study. No diagnosis, clinician trust or clinical benefit measured.'}
    write_new(EVALUATION/'native_secondary_endpoints.json',native)
    write_new(EVALUATION/'final_analysis.json',all_results)
    with (EVALUATION/'participant_results.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['partition','participant','method','uncertainty','mean_reference_dice','retained','empty_prediction'])
        writer.writeheader()
        for rows in source_rows.values():
            for r in rows: writer.writerow({'partition':r['partition'],'participant':r['participant'],'method':r['method'],
                'uncertainty':r['uncertainty'],'mean_reference_dice':1-statistics.fmean(r['errors']),
                'retained':r['uncertainty']<=policy['thresholds'][r['method']],'empty_prediction':r['empty_prediction']})
    print('Frozen participant-level analysis complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['fit','analyse']);a=parser.parse_args()
    fit() if a.phase=='fit' else run()
