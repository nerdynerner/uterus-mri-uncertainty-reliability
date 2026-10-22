"""Artificial-fixture checks for leakage, frozen thresholds and undefined risks."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from study_manifest import make_splits, validate_splits, validate_manifest, sha256, within
from analyse_policy import analyse, freeze


class Controls(unittest.TestCase):
    def test_partition_conservation(self):
        for n in range(10, 135):
            rows = [{'participant': f'D2-{i:03}', 'cohort': 'D2'} for i in range(n)]
            rows += [{'participant': 'D1-000', 'cohort': 'D1'}]
            split = make_splits(rows)
            validate_splits(rows, split)
            self.assertEqual(split, make_splits(list(reversed(rows))))
            self.assertEqual(sum(len(v) for v in split['partitions'].values()), n+1)
        eighty_one = make_splits([{'participant': str(i), 'cohort': 'D2'} for i in range(81)] +
                               [{'participant': 'external', 'cohort': 'D1'}])
        self.assertEqual([len(eighty_one['partitions'][k]) for k in
                          ('train', 'validation', 'calibration', 'internal_test')], [41, 12, 12, 16])

    def test_cohort_leakage_rejected(self):
        cases = [{'participant': str(i), 'cohort': 'D2'} for i in range(20)] + [{'participant': 'x', 'cohort': 'D1'}]
        split = make_splits(cases)
        split['partitions']['train'][0], split['partitions']['external_test'][0] = 'x', split['partitions']['train'][0]
        with self.assertRaises(ValueError):
            validate_splits(cases, split)

    def test_data_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('image', 'mask', 'review'):
                (root/name).write_text(name)
            row = {'participant': 'p', 'cohort': 'D2', 'sequence': 'T2',
                   'axial_verified': True, 'identity_verified': True, 'complete_uterus_label': True,
                   'geometry_review': 'approved', 'overlay_review': 'approved', 'review_record': 'review',
                   'image': 'image', 'image_sha256': sha256(root/'image'), 'training_rater': 'r1',
                   'masks': [{'path': 'mask', 'sha256': sha256(root/'mask'), 'rater': 'r1'}]}
            validate_manifest({'cases': [row]}, root)
            changed = copy.deepcopy(row)
            changed['overlay_review'] = 'pending'
            with self.assertRaises(ValueError):
                validate_manifest({'cases': [changed]}, root)
            (root/'image').write_text('changed')
            with self.assertRaises(ValueError):
                validate_manifest({'cases': [row]}, root)
            with self.assertRaises(ValueError):
                within(root, '../escape')

    def test_frozen_effect_and_two_stage(self):
        calibration = [{'participant': f'c{i}', 'method': 'ensemble', 'partition': 'calibration',
                        'uncertainty': score} for i, score in enumerate([.1, .2, .3, .4, .5])]
        policy = freeze(calibration, {})
        self.assertTrue(math.isclose(policy['thresholds']['ensemble'], .42))
        test = [{'participant': 'a', 'method': 'ensemble', 'partition': 'external_test',
                 'uncertainty': .1, 'errors': [.1, .3]},
                {'participant': 'b', 'method': 'ensemble', 'partition': 'external_test',
                 'uncertainty': .9, 'errors': [.8]}]
        result = analyse(test, policy, 'external_test', calibration, draws=100)['methods']['ensemble']
        self.assertTrue(math.isclose(result['policy']['retained_minus_overall'], -.3))
        self.assertEqual(result['policy']['coverage'], .5)
        self.assertEqual(result['failure_auroc'], 1.)
        self.assertGreater(result['primary_effect_ci']['undefined'], 0)
        # Changing held-out errors cannot change an already frozen threshold.
        test[0]['errors'] = [.9]
        self.assertEqual(analyse(test, policy, 'external_test', draws=10)['methods']['ensemble']['threshold'], policy['thresholds']['ensemble'])
        test[0]['participant'] = 'c0'
        with self.assertRaises(ValueError):
            analyse(test, policy, 'external_test')

    def test_pairing_and_empty_retention(self):
        calibration = [{'participant': 'c', 'method': 'ensemble', 'partition': 'calibration', 'uncertainty': 0.}]
        policy = freeze(calibration, {})
        test = [{'participant': 't', 'method': 'ensemble', 'partition': 'internal_test',
                 'uncertainty': 1., 'errors': [1.]}]
        result = analyse(test, policy, 'internal_test', draws=10)['methods']['ensemble']
        self.assertIsNone(result['policy']['retained_error'])
        self.assertIsNone(result['primary_effect_ci']['ci95'])
        self.assertIsNone(result['failure_auroc'])
        with self.assertRaises(ValueError):
            analyse(test+test, policy, 'internal_test')


if __name__ == '__main__':
    unittest.main()
