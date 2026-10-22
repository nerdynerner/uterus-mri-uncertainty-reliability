"""Validate exact domain-crop acceleration before using the amended secondary code."""
import datetime,json,time
import numpy as np
import torch
from spatial_endpoints import expert_disagreement,_expert_disagreement_full_grid
from research_analysis import arrays,case_map,jobs,read
from evaluation_runner import EVALUATION,CODE
from study_manifest import sha256,write_new
torch.set_num_threads(2)
def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)
rng=np.random.default_rng(61951)
fixtures=[]
for edge,spacing in [(False,[.6,.9,4.]),(True,[2.,1.,3.]),(False,[1.,1.,1.])]:
    first=np.zeros((33,35,29),bool)
    first[(slice(0,9) if edge else slice(10,19)),12:24,9:18]=True
    second=np.roll(first,2,axis=1)
    p=rng.random(first.shape).astype(np.float32)
    compare(_expert_disagreement_full_grid(p,[first,second],spacing),expert_disagreement(p,[first,second],spacing))
    fixtures.append({'edge_foreground':edge,'spacing':spacing,'status':'passed'})
cases=case_map()
row,folder=next((r,f) for r,f in jobs('external_test','ensemble') if r['participant']=='D1-000')
_,image,p,refs=arrays(row,folder,cases)
spacing=np.linalg.norm(image.affine[:3,:3],axis=0).tolist()
t=time.perf_counter();original=_expert_disagreement_full_grid(p,[m for _,m in refs],spacing);full_seconds=time.perf_counter()-t
t=time.perf_counter();cropped=expert_disagreement(p,[m for _,m in refs],spacing);crop_seconds=time.perf_counter()-t
compare(original,cropped)
write_new(EVALUATION/'secondary_crop_equivalence.json',{'status':'passed','synthetic_fixtures':fixtures,
    'real_case':'D1-000','real_method':'ensemble','absolute_float_tolerance':1e-12,
    'all_domain_counts_and_summary_fields_match':True,'full_grid_seconds':full_seconds,
    'cropped_seconds':crop_seconds,'original_result':original,'cropped_result':cropped})
frozen=read(EVALUATION/'heldout_code_freeze.json')
assert sha256(EVALUATION/'spatial_endpoints_before_crop_acceleration.py')==frozen['code_sha256']['spatial_endpoints.py']
assert sha256(EVALUATION/'research_analysis_before_tolerance_correction.py')==frozen['code_sha256']['research_analysis.py']
write_new(EVALUATION/'analysis_implementation_corrections_v2.json',{
    'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Secondary loader tolerance alignment and exact background-domain crop acceleration; primary predictions, cohort, thresholds and statistical endpoints unchanged.',
    'after_primary_test_outcomes_available':True,'equivalence_record_sha256':sha256(EVALUATION/'secondary_crop_equivalence.json'),
    'files':{name:{'original_sha256':frozen['code_sha256'][name],'corrected_sha256':sha256(CODE/name)}
             for name in ['research_analysis.py','spatial_endpoints.py']},
    'prior_tolerance_record':'analysis_implementation_correction.json'})
print('Exact secondary crop equivalence passed; full/cropped seconds:',full_seconds,crop_seconds,flush=True)
