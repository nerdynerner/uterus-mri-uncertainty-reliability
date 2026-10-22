"""Analyse saved participant outputs; calibration and test are separate inputs.

Each JSON document contains rows with participant, method, uncertainty, errors
(one 1-Dice per reference), and partition. No slice rows are accepted. Calibration
requires scores only; held-out errors cannot affect thresholds.
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path
from reliability_metrics import (auc_failure, average_precision, aurc,
                                 frozen_policy, paired_bootstrap, quantile,
                                 risk_coverage)
from study_manifest import sha256, write_new


def validate_rows(rows, partition, require_errors):
    seen = set()
    if not rows:
        raise ValueError('No output rows')
    for r in rows:
        key = (r['participant'], r['method'])
        if key in seen or r.get('partition') != partition:
            raise ValueError('Duplicate participant/method or wrong partition')
        seen.add(key)
        if r['method'] not in ('deterministic', 'dropout', 'ensemble'):
            raise ValueError('Unknown method')
        if not math.isfinite(r['uncertainty']) or not 0 <= r['uncertainty'] <= 1:
            raise ValueError('Uncertainty must be finite in [0,1]')
        if require_errors and (not r.get('errors') or any(
                not math.isfinite(e) or not 0 <= e <= 1 for e in r['errors'])):
            raise ValueError('Per-reference errors must be finite in [0,1]')
    by_method = {}
    for participant, method in seen:
        by_method.setdefault(method, set()).add(participant)
    if any(s != next(iter(by_method.values())) for s in by_method.values()):
        raise ValueError('Methods must use the same participants')


def freeze(calibration, provenance):
    validate_rows(calibration, 'calibration', False)
    methods = sorted({r['method'] for r in calibration})
    return {'quantile': .8, 'tie_rule': 'retain uncertainty <= threshold',
            'provenance': provenance,
            'calibration_ids': sorted({r['participant'] for r in calibration}),
            'thresholds': {m: quantile([r['uncertainty'] for r in calibration if r['method'] == m], .8)
                           for m in methods}}


def two_stage(calibration, test, draws=2000, seed=61922):
    rng, values = random.Random(seed), []
    for _ in range(draws):
        source = [calibration[rng.randrange(len(calibration))] for _ in calibration]
        target = [test[rng.randrange(len(test))] for _ in test]
        threshold = quantile([r['uncertainty'] for r in source], .8)
        value = frozen_policy([statistics.fmean(r['errors']) for r in target],
                              [r['uncertainty'] for r in target], threshold)['retained_minus_overall']
        if value is not None:
            values.append(value)
    return {'draws': draws, 'valid': len(values), 'undefined': draws-len(values),
            'ci95': [quantile(values, .025), quantile(values, .975)] if values else None,
            'scope': 'resamples calibration and test participants; conditions on trained models'}


def analyse(test, policy, partition, calibration=None, draws=2000):
    validate_rows(test, partition, True)
    if set(policy['calibration_ids']) & {r['participant'] for r in test}:
        raise ValueError('Calibration/test participant overlap')
    if set(policy['thresholds']) != {r['method'] for r in test}:
        raise ValueError('Frozen and evaluated methods differ')
    results = {}
    for method, threshold in policy['thresholds'].items():
        rows = sorted((r for r in test if r['method'] == method), key=lambda r: r['participant'])
        errors = [statistics.fmean(r['errors']) for r in rows]
        scores = [r['uncertainty'] for r in rows]
        def effect(sample):
            return frozen_policy([statistics.fmean(r['errors']) for r in sample],
                                 [r['uncertainty'] for r in sample], threshold)['retained_minus_overall']
        # Failure against mean Dice is a summary; each reference is retained in input.
        failures = [int(e > .5) for e in errors]
        result = {'threshold': threshold, 'policy': frozen_policy(errors, scores, threshold),
                  'primary_effect_ci': paired_bootstrap(rows, effect, draws=draws),
                  'aurc': aurc(errors, scores), 'risk_coverage': risk_coverage(errors, scores),
                  'failure_prevalence': statistics.fmean(failures),
                  'failure_auroc': auc_failure(failures, scores),
                  'failure_average_precision': average_precision(failures, scores),
                  'scope': 'participant mean-reference error; per-rater sensitivity still required'}
        if calibration is not None:
            validate_rows(calibration, 'calibration', False)
            source = [r for r in calibration if r['method'] == method]
            if sorted(r['participant'] for r in source) != policy['calibration_ids']:
                raise ValueError('Calibration provenance does not match frozen policy')
            if quantile([r['uncertainty'] for r in source], .8) != threshold:
                raise ValueError('Calibration scores do not reproduce frozen threshold')
            result['two_stage_ci'] = two_stage(source, rows, draws=draws)
        results[method] = result
    return {'partition': partition, 'methods': results,
            'interpretation': 'Technical segmentation error; no diagnostic or clinical-benefit claim'}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('calibration', type=Path)
    f.add_argument('output', type=Path)
    f.add_argument('--checkpoint-provenance', type=Path, required=True,
                   help='JSON object mapping each method to its ordered checkpoint SHA-256 list')
    a = sub.add_parser('evaluate')
    a.add_argument('test', type=Path)
    a.add_argument('policy', type=Path)
    a.add_argument('output', type=Path)
    a.add_argument('--partition', choices=['internal_test', 'external_test'], required=True)
    a.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        checkpoint_hashes = json.loads(args.checkpoint_provenance.read_text())
        calibration = json.loads(args.calibration.read_text())
        if set(checkpoint_hashes) != {r['method'] for r in calibration}:
            raise ValueError('Checkpoint provenance must cover exactly the calibrated methods')
        for hashes in checkpoint_hashes.values():
            if not hashes or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
                raise ValueError('Valid checkpoint SHA-256 lists required')
        result = freeze(calibration, {'score_sha256': sha256(args.calibration),
                                     'checkpoint_sha256': checkpoint_hashes})
    else:
        policy = json.loads(args.policy.read_text())
        calibration = json.loads(args.calibration.read_text()) if args.calibration else None
        result = analyse(json.loads(args.test.read_text()), policy, args.partition, calibration)
        result['provenance'] = {'test_sha256': sha256(args.test), 'policy_sha256': sha256(args.policy)}
    write_new(args.output, result)


if __name__ == '__main__':
    main()
