"""Patient-level reliability arithmetic. No dependencies. No patient data bundled.

Scores increase with uncertainty. Curves use expected risk within score ties.
Only pass one aggregated row per participant/target/method/reference analysis.
"""
import math
import random
import statistics


def _check(errors, scores):
    if not errors or len(errors) != len(scores):
        raise ValueError('Nonempty equal-length inputs required')
    if any(not math.isfinite(x) for x in errors + scores):
        raise ValueError('Finite inputs required')
    if any(x < 0 or x > 1 for x in errors):
        raise ValueError('Errors must be in [0,1]')


def risk_coverage(errors, scores):
    """At every k/n, expected risk under random ordering within exact score ties."""
    errors, scores = list(errors), list(scores)
    _check(errors, scores)
    groups = {}
    for e, s in zip(errors, scores):
        groups.setdefault(s, []).append(e)
    curve, total, count = [], 0., 0
    for score in sorted(groups):
        group = groups[score]
        mean = statistics.fmean(group)
        for j in range(1, len(group) + 1):
            curve.append(((count+j)/len(errors), (total+j*mean)/(count+j)))
        total += sum(group)
        count += len(group)
    return curve


def aurc(errors, scores):
    """Discrete average over k=1..n; no extrapolated zero-coverage point."""
    return statistics.fmean(r for _, r in risk_coverage(errors, scores))


def auc_failure(failures, scores):
    """Mann-Whitney AUROC with half-credit for ties; None if a class absent."""
    f, s = list(failures), list(scores)
    _check(f, s)
    if any(x not in (0, 1) for x in f):
        raise ValueError('Failure labels must be binary')
    pos = [x for y,x in zip(f,s) if y]
    neg = [x for y,x in zip(f,s) if not y]
    if not pos or not neg:
        return None
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def average_precision(failures, scores):
    """Threshold-grouped average precision, not trapezoidal PR area."""
    f, s = list(failures), list(scores)
    _check(f, s)
    if any(x not in (0,1) for x in f):
        raise ValueError('Failure labels must be binary')
    positives = sum(f)
    if positives == 0:
        return None
    groups = {}
    for y,x in zip(f,s):
        groups.setdefault(x, []).append(y)
    tp, seen, ap = 0, 0, 0.
    for score in sorted(groups, reverse=True):
        ys = groups[score]
        new = sum(ys)
        tp += new
        seen += len(ys)
        ap += new/positives * tp/seen
    return ap


def quantile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError('Nonempty values and q in [0,1] required')
    xs = sorted(values)
    p = (len(xs)-1)*q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi]-xs[lo])*(p-lo)


def frozen_policy(errors, scores, threshold):
    e, s = list(errors), list(scores)
    _check(e,s)
    kept = [x for x,u in zip(e,s) if u <= threshold]
    base = statistics.fmean(e)
    risk = statistics.fmean(kept) if kept else None
    return dict(n=len(e), retained=len(kept), coverage=len(kept)/len(e),
                overall_error=base, retained_error=risk,
                retained_minus_overall=None if risk is None else risk-base)


def paired_bootstrap(rows, statistic, draws=2000, seed=61921):
    """Rows must each contain a whole participant, including all paired methods.

    Statistic is recomputed on each resample. Reports undefined replicates.
    This conditions on trained models and any fixed calibration threshold.
    """
    if not rows or draws < 1:
        raise ValueError('Rows and positive draws required')
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        v = statistic([rows[rng.randrange(len(rows))] for _ in rows])
        if v is not None and math.isfinite(v):
            vals.append(v)
    return dict(estimate=statistic(rows), draws=draws, valid=len(vals),
                undefined=draws-len(vals),
                ci95=[quantile(vals,.025),quantile(vals,.975)] if vals else None)
