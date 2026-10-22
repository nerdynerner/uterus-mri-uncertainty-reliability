"""Independent arithmetic checks on artificial fixtures, never clinical results."""
import itertools
import json
import math
import random
import statistics
from fractions import Fraction
from reliability_metrics import (aurc, auc_failure, average_precision,
                                  frozen_policy, paired_bootstrap)


def brute_aurc(errors,scores):
    # Enumerate all orderings compatible with weak score ordering.
    groups=[]
    for score in sorted(set(scores)):
        groups.append([i for i,x in enumerate(scores) if x==score])
    values=[]
    for pieces in itertools.product(*(list(itertools.permutations(g)) for g in groups)):
        order=list(itertools.chain.from_iterable(pieces))
        risk=[sum(errors[i] for i in order[:k])/k for k in range(1,len(order)+1)]
        values.append(sum(risk)/len(risk))
    return sum(values)/len(values)


def rank_auc(labels,scores):
    # Independent rank-sum formulation, with average ranks at ties.
    npos=sum(labels)
    nneg=len(labels)-npos
    if not npos or not nneg:
        return None
    ranks={}
    sorted_scores=sorted(scores)
    for x in set(scores):
        places=[i+1 for i,v in enumerate(sorted_scores) if v==x]
        ranks[x]=sum(places)/len(places)
    pos_rank=sum(ranks[x] for y,x in zip(labels,scores) if y)
    return (pos_rank-npos*(npos+1)/2)/(npos*nneg)


def main():
    checks=0
    rng=random.Random(61931)
    for _ in range(100):
        n=rng.randrange(2,7)
        errors=[rng.randrange(11)/10 for _ in range(n)]
        scores=[rng.randrange(3) for _ in range(n)]
        assert math.isclose(aurc(errors,scores),brute_aurc(errors,scores),abs_tol=1e-12)
        checks+=1
        labels=[int(e>.5) for e in errors]
        a,b=auc_failure(labels,scores),rank_auc(labels,scores)
        assert (a is None and b is None) or math.isclose(a,b,abs_tol=1e-12)
        checks+=1
    assert math.isclose(average_precision([1,0,1],[3,2,1]),float(Fraction(5,6)),abs_tol=1e-12)
    assert average_precision([1,0,1,0],[1,1,1,1])==.5
    assert average_precision([0,0],[1,2]) is None
    assert auc_failure([1,1],[1,2]) is None
    assert math.isclose(frozen_policy([.1,.8],[.1,.9],.5)['retained_minus_overall'],-.35,abs_tol=1e-12)
    assert frozen_policy([.1,.8],[.1,.9],-.1)['retained_error'] is None
    checks+=6
    # Compare bootstrap reproducibility and interval to direct order statistics.
    rows=[.1,.3,.6,.9]
    result=paired_bootstrap(rows,statistics.fmean,draws=2000,seed=61921)
    # Verify the exact sequence with separately calculated order statistics.
    rng=random.Random(61921)
    means=sorted(sum(rows[rng.randrange(4)] for _ in rows)/4 for _ in range(2000))
    for q,bound in zip([.025,.975],result['ci95']):
        p=1999*q
        expected=means[int(p)]+(means[math.ceil(p)]-means[int(p)])*(p-int(p))
        assert math.isclose(bound,expected,abs_tol=1e-12)
        checks+=1
    exhaustive=[sum(x)/4 for x in itertools.product(rows,repeat=4)]
    assert math.isclose(statistics.fmean(exhaustive),statistics.fmean(rows),abs_tol=1e-12)
    checks+=1
    print(json.dumps(dict(status='passed',checks=checks,
        clinical_data_used=False,models_trained=0,
        scope='AURC exhaustive ties; AUROC rank-sum; AP hand arithmetic; policy empty retention; bootstrap order statistics',
        limitation='Independent numerical formulas; no independent clinical reproduction'),indent=2))


if __name__=='__main__':
    main()
