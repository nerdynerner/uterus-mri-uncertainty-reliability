"""Additional participant-level policy summaries, with explicit tie handling."""
import math
import statistics
from reliability_metrics import (auc_failure, average_precision, frozen_policy,
                                 paired_bootstrap, risk_coverage)


def ranks(values):
    groups = {}
    for i, value in enumerate(values):
        groups.setdefault(value, []).append(i)
    result, offset = [None]*len(values), 0
    for value in sorted(groups):
        group = groups[value]
        rank = offset+(len(group)+1)/2
        for i in group:
            result[i] = rank
        offset += len(group)
    return result


def spearman(errors, scores):
    if len(errors) != len(scores) or len(errors) < 2:
        return None
    a, b = ranks(errors), ranks(scores)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    denominator = math.sqrt(sum((v-ma)**2 for v in a)*sum((v-mb)**2 for v in b))
    return None if denominator == 0 else sum((x-ma)*(y-mb) for x,y in zip(a,b))/denominator


def fixed_review_budget(errors, scores, failure_dice=.5, review_fraction=.2):
    if not errors or len(errors) != len(scores) or not 0 < review_fraction < 1:
        raise ValueError('Nonempty paired rows and interior review fraction required')
    n = len(errors)
    kept = max(1, math.floor((1-review_fraction)*n))
    reviewed = n-kept
    curve = risk_coverage(errors, scores)
    failures = [int(e > 1-failure_dice) for e in errors]
    # Expected retained failures under uniform tie-breaking at the cutoff.
    retained_failure_rate = risk_coverage(failures, scores)[kept-1][1]
    found = sum(failures)-kept*retained_failure_rate
    return {'requested_review_fraction': review_fraction,
            'reviewed_n': reviewed, 'achieved_review_fraction': reviewed/n,
            'retained_n': kept, 'retained_risk': curve[kept-1][1],
            'expected_failures_found': found,
            'failure_capture_fraction': found/sum(failures) if sum(failures) else None,
            'tie_rule': 'expected performance under uniform random ordering within exact score ties'}


def reference_sensitivity(rows, threshold, draws=2000):
    raters = sorted({reference['rater'] for row in rows for reference in row.get('references', [])})
    results = {}
    for rater in raters:
        subset = []
        for row in rows:
            matches = [r for r in row.get('references', []) if r['rater'] == rater]
            if len(matches) > 1:
                raise ValueError('Duplicate rater within participant')
            if matches:
                subset.append({'participant': row['participant'], 'error': 1-matches[0]['dice'],
                               'score': row['uncertainty']})
        errors = [r['error'] for r in subset]
        scores = [r['score'] for r in subset]
        def effect(sample):
            return frozen_policy([r['error'] for r in sample], [r['score'] for r in sample],
                                 threshold)['retained_minus_overall']
        results[rater] = {'n': len(subset), 'participant_ids': [r['participant'] for r in subset],
                          'policy': frozen_policy(errors, scores, threshold),
                          'effect_ci': paired_bootstrap(subset, effect, draws=draws),
                          'spearman_error_uncertainty': spearman(errors, scores),
                          'failure_sensitivity': {str(cutoff): {
                              'auroc': auc_failure([int(e > 1-cutoff) for e in errors], scores),
                              'average_precision': average_precision([int(e > 1-cutoff) for e in errors], scores),
                              'fixed_review_budget': fixed_review_budget(errors, scores, cutoff)
                          } for cutoff in (.3,.5,.7)}}
    return {'raters': results,
            'limitation': 'Rater subsets may differ; do not interpret unpaired rater differences as annotation effects'}


def paired_method_contrast(rows, thresholds, method_a, method_b, draws=2000):
    grouped = {}
    for row in rows:
        if row['method'] in (method_a, method_b):
            group = grouped.setdefault(row['participant'], {})
            if row['method'] in group:
                raise ValueError('Duplicate participant/method')
            group[row['method']] = row
    if not grouped or any(set(group) != {method_a, method_b} for group in grouped.values()):
        raise ValueError('A complete paired participant set is required')
    def effect(sample):
        effects = []
        for method in (method_a, method_b):
            values = [group[method] for group in sample]
            value = frozen_policy([statistics.fmean(r['errors']) for r in values],
                                  [r['uncertainty'] for r in values],
                                  thresholds[method])['retained_minus_overall']
            if value is None:
                return None
            effects.append(value)
        return effects[0]-effects[1]
    return {'contrast': method_a+' minus '+method_b,
            'paired_participants': len(grouped),
            'effect_difference_ci': paired_bootstrap(list(grouped.values()), effect, draws=draws),
            'limitation': 'Policies can retain different fractions and use different predictions; report both coverages'}
