"""Exact pairwise binary-mask Dice using packed bytes to bound memory traffic."""
import itertools
import numpy as np


def pairwise_disagreement(masks):
    if masks.ndim < 2 or masks.shape[0] < 2 or masks.dtype != np.bool_:
        raise ValueError('At least two full-volume boolean member masks required')
    flat = masks.reshape(masks.shape[0],-1)
    counts = np.count_nonzero(flat,axis=1)
    packed = np.packbits(flat,axis=1)
    lookup = np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    values, both_empty = [], 0
    for a,b in itertools.combinations(range(len(counts)),2):
        denominator = int(counts[a])+int(counts[b])
        if denominator == 0:
            both_empty += 1
            values.append(0.)
        else:
            intersection_bytes = np.bitwise_and(packed[a],packed[b])
            if hasattr(np,'bitwise_count'):
                intersection = int(np.bitwise_count(intersection_bytes).sum(dtype=np.uint64))
            else:
                intersection = int(lookup[intersection_bytes].sum(dtype=np.uint64))
            values.append(1.-2*intersection/denominator)
    return float(sum(values)/len(values)), both_empty


if __name__ == '__main__':
    rng = np.random.default_rng(61950)
    for shape in [(5,11,13,3),(3,5,6,7),(30,7,8,9)]:
        masks = rng.random(shape) > .8
        masks[:2] = False
        exact = []
        for a,b in itertools.combinations(masks,2):
            n = int(a.sum())+int(b.sum())
            exact.append(0. if not n else 1.-2*int((a&b).sum())/n)
        score, empty = pairwise_disagreement(masks)
        assert abs(score-float(np.mean(exact))) < 1e-14
        assert empty == 1
    print('Packed pairwise Dice matches direct voxel arithmetic on three artificial fixtures.')
