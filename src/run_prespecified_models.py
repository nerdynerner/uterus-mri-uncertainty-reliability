"""Sequential training of the prespecified five models and dropout model.

No calibration/test evaluation is performed here. Every model is independently
initialized. A failed model stops the sequence and leaves all outputs intact.
"""
import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from study_manifest import sha256, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='models_v1')
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('.','..'):
        parser.error('Output name must be a single new folder name')
    root = Path(os.environ['UTERUS_MRI_WORKSPACE']).resolve()
    code = Path(__file__).resolve().parent
    output = root/'outputs'/args.output_name
    output.mkdir(parents=True,exist_ok=False)
    configuration = {
        'model_seeds':[61901,61902,61903,61904,61905,61911],
        'maximum_epochs':100,'validation_patience':15,'batch_size':1,
        'manifest_sha256':sha256(root/'outputs/primary_manifest_v1_2026-10-21.json'),
        'split_sha256':sha256(root/'outputs/primary_splits_v1_2026-10-21.json'),
        'pipeline_sha256':sha256(code/'pipeline.py'),
        'scope':'Training and source validation only; no calibration/test performance opened',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(output/'run_configuration.json',configuration)
    completed = []
    for seed in configuration['model_seeds']:
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_new(output/f'started_{seed}.json',{'seed':seed,'started_utc':start})
        if sha256(code/'pipeline.py') != configuration['pipeline_sha256']:
            raise RuntimeError('Training implementation changed during the frozen run')
        command = [sys.executable,str(code/'pipeline.py'),'train',
            '--manifest',str(root/'outputs/primary_manifest_v1_2026-10-21.json'),
            '--splits',str(root/'outputs/primary_splits_v1_2026-10-21.json'),
            '--root',str(root),'--output',str(output/f'seed_{seed}'),
            '--benchmark',str(root/'outputs/benchmark_installed_256_2026-10-21.json'),
            '--integration',str(root/'outputs/integration_dtypefix_2026-10-21.json'),
            '--epochs','100','--batch-size','1','--seed',str(seed),
            '--dropout','.2' if seed == 61911 else '0.0']
        print('Starting independent model',seed,flush=True)
        with (output/f'training_{seed}.log').open('x') as stream:
            process = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            latest_epoch = 0
            while True:
                try:
                    exit_code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    history = sorted((output/f'seed_{seed}').glob('history_epoch_*.json'))
                    if len(history) > latest_epoch:
                        latest_epoch = len(history)
                        print('Model',seed,'completed epoch',latest_epoch,flush=True)
        if exit_code:
            write_new(output/'failed_model.json',{'seed':seed,'exit_code':exit_code,
                'log':f'training_{seed}.log','completed_models':completed})
            raise SystemExit(exit_code)
        selection = json.loads((output/f'seed_{seed}/selected_checkpoint.json').read_text())
        if sha256(output/f'seed_{seed}'/selection['path']) != selection['sha256']:
            raise RuntimeError('Selected checkpoint hash mismatch')
        completed.append({'seed':seed,'selection':selection})
        write_new(output/f'completed_{seed}.json',completed[-1])
        print('Completed independent model',seed,flush=True)
    write_new(output/'all_training_complete.json',{'models':completed,
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'calibration_test_evaluated':False})
    print('Prespecified training complete. Calibration and test evaluation remain separate.',flush=True)


if __name__ == '__main__':
    main()
