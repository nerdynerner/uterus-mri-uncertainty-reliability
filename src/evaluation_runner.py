"""Frozen calibration and held-out inference for the completed model set.

Use calibrate first, then heldout. Existing complete jobs are verified and reused;
partial jobs are retained and a new attempt directory is used. No retraining.
"""

import os
import argparse
import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from study_manifest import (sha256, validate_manifest, validate_splits, write_new)
from analyse_policy import freeze, validate_rows


ROOT = Path(os.environ['UTERUS_MRI_WORKSPACE']).resolve()
CODE = Path(__file__).resolve().parent
EVALUATION = ROOT/'outputs/evaluation_v1_2026-10-21'
MANIFEST = ROOT/'outputs/primary_manifest_v1_2026-10-21.json'
SPLITS = ROOT/'outputs/primary_splits_v1_2026-10-21.json'
METHODS = ('deterministic','ensemble','dropout')


def selected_models():
    candidates = sorted(p for p in (ROOT/'outputs').glob('models_*')
                        if (p/'all_training_complete.json').is_file())
    if len(candidates) != 1:
        raise ValueError('Resolve the exact completed training run before evaluation')
    run = candidates[0]
    completion = json.loads((run/'all_training_complete.json').read_text())
    models = {}
    for item in completion['models']:
        path = run/f"seed_{item['seed']}"/item['selection']['path']
        if sha256(path) != item['selection']['sha256']:
            raise ValueError('Selected checkpoint checksum mismatch')
        models[item['seed']] = path
    if set(models) != {61901,61902,61903,61904,61905,61911}:
        raise ValueError('Complete prespecified model set required')
    return {'deterministic':[models[61901]],
            'ensemble':[models[i] for i in range(61901,61906)],
            'dropout':[models[61911]]},run


def store_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing frozen record differs: '+str(path))
    else:
        write_new(path,value)


def run_job(pipeline, cases, partitions, models, partition, method):
    marker = EVALUATION/f'complete_{partition}_{method}.json'
    expected = {'checkpoint_sha256':[sha256(p) for p in models[method]],
                'manifest_sha256':sha256(MANIFEST),'split_sha256':sha256(SPLITS)}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['inputs'] != expected:
            raise ValueError('Completed job input provenance changed')
        folder = EVALUATION/saved['folder']
        if sha256(folder/'participant_outputs.json') != saved['participant_outputs_sha256']:
            raise ValueError('Completed participant outputs changed')
        rows = json.loads((folder/'participant_outputs.json').read_text())
        print('Reusing completed',partition,method,flush=True)
    else:
        stem = partition+'_'+method
        folder = EVALUATION/stem
        attempt = 1
        while folder.exists():
            attempt += 1
            folder = EVALUATION/f'{stem}_attempt_{attempt}'
        args = SimpleNamespace(root=ROOT,manifest=MANIFEST,splits=SPLITS,output=folder,
            partition=partition,method=method,checkpoints=models[method],
            frozen_policy=EVALUATION/'frozen_review_policy.json' if partition != 'calibration' else None)
        pipeline.infer(args,cases,partitions)
        rows = json.loads((folder/'participant_outputs.json').read_text())
        if {r['participant'] for r in rows} != set(partitions[partition]):
            raise ValueError('Inference partition is incomplete or has extra IDs')
        validate_rows(rows,partition,partition != 'calibration')
        write_new(marker,{'folder':folder.name,'inputs':expected,
                         'participant_outputs_sha256':sha256(folder/'participant_outputs.json')})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['calibrate','heldout'])
    args = parser.parse_args()
    EVALUATION.mkdir(parents=True,exist_ok=True)
    models,run = selected_models()
    config = {'training_run':str(run.relative_to(ROOT)),
              'manifest_sha256':sha256(MANIFEST),'splits_sha256':sha256(SPLITS),
              'checkpoints':{m:[{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for p in v]
                             for m,v in models.items()},
              'review_quantile':.8,'dropout_passes':30,'seed':61930,
              'segmentation_threshold':.5,'inference_device_policy':'MPS if available in this execution context else CPU',
              'primary_effect':'retained minus overall mean participant 1-Dice; mean across compatible references',
              'primary_bootstrap_draws':2000,'test_parameters_tuned':False}
    store_once(EVALUATION/'evaluation_configuration.json',config)
    print('Verifying source files before inference',flush=True)
    cases = validate_manifest(json.loads(MANIFEST.read_text()),ROOT)
    split = json.loads(SPLITS.read_text())
    if split['manifest_sha256'] != sha256(MANIFEST):
        raise ValueError('Manifest/split provenance mismatch')
    partitions = validate_splits(cases,split)
    import pipeline as pipeline
    pipeline.dependencies()
    if args.phase == 'calibrate':
        rows = []
        for method in METHODS:
            rows.extend(run_job(pipeline,cases,partitions,models,'calibration',method))
        store_once(EVALUATION/'calibration_scores.json',rows)
        provenance = {'score_sha256':sha256(EVALUATION/'calibration_scores.json'),
                      'checkpoint_sha256':{m:[sha256(p) for p in v] for m,v in models.items()},
                      'configuration_sha256':sha256(EVALUATION/'evaluation_configuration.json')}
        policy = freeze(rows,provenance)
        store_once(EVALUATION/'frozen_review_policy.json',policy)
        print('Calibration scores complete. Review thresholds frozen; no test performance evaluated.',flush=True)
    else:
        policy_path = EVALUATION/'frozen_review_policy.json'
        if not policy_path.is_file():
            raise ValueError('Freeze calibration review thresholds first')
        # Complete evaluation implementation is recorded before the first held-out job.
        filenames = ['pipeline.py','packed_disagreement.py','evaluation_runner.py',
            'analyse_policy.py','secondary_policy_metrics.py','spatial_endpoints.py',
            'probability_calibration.py','research_analysis.py','reliability_metrics.py']
        freeze_record = {'policy_sha256':sha256(policy_path),
                         'temperatures_sha256':sha256(EVALUATION/'frozen_temperatures.json'),
                         'code_sha256':{name:sha256(CODE/name) for name in filenames}}
        store_once(EVALUATION/'heldout_code_freeze.json',freeze_record)
        for partition in ('internal_test','external_test'):
            rows = []
            for method in METHODS:
                rows.extend(run_job(pipeline,cases,partitions,models,partition,method))
            store_once(EVALUATION/(partition+'_scores.json'),rows)
        store_once(EVALUATION/'all_heldout_inference_complete.json',{
            'partitions':{'internal_test':len(partitions['internal_test']),
                          'external_test':len(partitions['external_test'])},
            'methods':list(METHODS),'policy_sha256':sha256(policy_path)})
        print('Held-out inference complete. Saved outputs are ready for the frozen analysis.',flush=True)


if __name__ == '__main__':
    main()
