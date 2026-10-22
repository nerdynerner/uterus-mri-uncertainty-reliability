import math
import unittest
from secondary_policy_metrics import fixed_review_budget, spearman, paired_method_contrast


class Secondary(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(spearman([1,2,2,3], [3,2,2,1]), -1.)
        self.assertIsNone(spearman([1,1], [0,1]))

    def test_tied_budget(self):
        result = fixed_review_budget([0,0,0,0,1], [.5]*5)
        self.assertAlmostEqual(result['failure_capture_fraction'], .2)
        self.assertAlmostEqual(result['retained_risk'], .2)
        self.assertEqual(result['reviewed_n'], 1)

    def test_perfect_review(self):
        result = fixed_review_budget([0,0,0,0,1], [.1,.2,.3,.4,.9])
        self.assertEqual(result['failure_capture_fraction'], 1.)
        self.assertEqual(result['retained_risk'], 0.)

    def test_paired_contrast(self):
        rows = [{'participant': str(i), 'method': m, 'errors': [e], 'uncertainty': s}
                for i,(e,s) in enumerate([(.1,.1),(.9,.9)]) for m in ['a','b']]
        result = paired_method_contrast(rows, {'a': .5,'b': .5}, 'a','b',draws=100)
        self.assertEqual(result['effect_difference_ci']['estimate'], 0.)
        with self.assertRaises(ValueError):
            paired_method_contrast(rows[:-1], {'a': .5,'b': .5},'a','b')


if __name__ == '__main__':
    unittest.main()
