"""Independent recomputation checks for saved native predictions and policy rows."""
import json,statistics
import numpy as np
from evaluation_runner import EVALUATION,ROOT,CODE,MANIFEST,SPLITS,METHODS
from study_manifest import sha256,write_new
from research_analysis import jobs,arrays,case_map
from spatial_endpoints import overlap

def main():
    cases=case_map();checked=0
    policy=json.loads((EVALUATION/'frozen_review_policy.json').read_text())
    frozen=json.loads((EVALUATION/'heldout_code_freeze.json').read_text())
    assert frozen['temperatures_sha256']==sha256(EVALUATION/'frozen_temperatures.json')
    final=json.loads((EVALUATION/'final_analysis.json').read_text())
    for part in ('internal_test','external_test'):
        rows=json.loads((EVALUATION/(part+'_scores.json')).read_text())
        for method in METHODS:
            group=[r for r in rows if r['method']==method]
            for row,folder in jobs(part,method):
                _,image,p,refs=arrays(row,folder,cases)
                pred=p>=.5
                for rater,mask in refs:
                    saved=next(r for r in row['references'] if r['rater']==rater)
                    score=overlap(pred,mask)
                    assert abs(score-saved['dice'])<1e-12
                    union=np.logical_or(pred,mask).sum()
                    iou=float(np.logical_and(pred,mask).sum()/union) if union else 1.
                    assert abs(iou-saved['iou'])<1e-12
                    import nibabel as nib
                    refpath=next(x['path'] for x in cases[row['participant']]['masks'] if x['rater']==rater)
                    reference=nib.load(str(ROOT/refpath))
                    voxvol=abs(np.linalg.det(reference.affine[:3,:3]))/1000
                    vol=abs(int(pred.sum())-int(mask.sum()))*voxvol
                    assert abs(vol-saved['absolute_volume_error_ml'])<1e-6
                checked+=1
            kept=[r for r in group if r['uncertainty']<=policy['thresholds'][method]]
            overall=statistics.fmean(statistics.fmean(r['errors']) for r in group)
            risk=statistics.fmean(statistics.fmean(r['errors']) for r in kept) if kept else None
            saved=final[part]['methods'][method]['policy']
            assert len(kept)==saved['retained'] and abs(overall-saved['overall_error'])<1e-12
            assert risk==saved['retained_error']
    assert checked==111
    write_new(EVALUATION/'independent_final_verification.json',{'status':'passed',
        'participant_method_native_outputs_checked':checked,'required_outputs':111,
        'native_dice_iou_volume_recomputed':True,'policy_retention_and_risk_recomputed':True,
        'frozen_temperatures_unchanged':True,'analysis_sha256':sha256(EVALUATION/'final_analysis.json'),
        'implementation_amendments_sha256':sha256(EVALUATION/'analysis_implementation_corrections_v2.json'),
        'executed_analysis_sha256':sha256(CODE/'research_analysis.py'),
        'executed_spatial_endpoints_sha256':sha256(CODE/'spatial_endpoints.py')})
    print('Independent native-output and frozen-policy verification passed',checked,flush=True)
if __name__=='__main__':main()
