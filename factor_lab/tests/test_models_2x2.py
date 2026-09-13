"""Aequivalenz zum bestehenden Ridge-Pfad, Lambda-Skalierung, GBM-Determinismus."""
import unittest

import numpy as np
import pandas as pd

from factor_lab.daily_comparison import build_dataset
from factor_lab.daily_models import fit_predict_models
from factor_lab.features_2x2 import build_2x2_dataset
from factor_lab.models_2x2 import (
    GBM_PARAMS,
    GBM_ROUNDS,
    RIDGE_LAMBDA_PER_FEATURE,
    SEQUENCE,
    fit_predict_gbm,
    fit_predict_ridge,
    fit_scaler,
    standardize,
    training_pairs,
)


def synthetic_prices(periods=700, assets=('SPY', 'AAA', 'BBB'), seed=11):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2011-01-03', periods=periods)
    data = {}
    for offset, name in enumerate(assets):
        steps = 0.0003 * (offset + 1) + 0.012 * rng.standard_normal(periods)
        data[name] = 50.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=index)


class RidgeEquivalenceTests(unittest.TestCase):
    def test_matches_existing_daily_models_ridge_exactly(self):
        """Bei 5 Kanaelen ist lambda = 1.0 * 100 der heutige Wert 100."""
        prices = synthetic_prices()
        reference = build_dataset(prices)
        first = len(reference['dates']) - 60
        expected = fit_predict_models(
            reference['X'], reference['y'], first_test=first, label_delay=2, seed=7
        )['ridge']
        got = fit_predict_ridge(
            reference['X'], reference['y'], first_test=first,
            last_test=len(reference['dates']), label_delay=2
        )
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-12)

    def test_lambda_scales_with_feature_count(self):
        self.assertEqual(RIDGE_LAMBDA_PER_FEATURE, 1.0)
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 40
        for block in ('X_base', 'X_ext'):
            out = fit_predict_ridge(data[block], data['y'], first, len(data['dates']))
            self.assertEqual(out.shape, (40, data['y'].shape[1]))
            self.assertTrue(np.isfinite(out).all())

    def test_base_and_extended_predictions_differ_and_stay_bounded(self):
        """Sanity: mehr Merkmale bei gleichem lambda pro Merkmal bleibt stabil."""
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 40
        base = fit_predict_ridge(data['X_base'], data['y'], first, len(data['dates']))
        ext = fit_predict_ridge(data['X_ext'], data['y'], first, len(data['dates']))
        self.assertFalse(np.allclose(base, ext))
        self.assertLess(np.abs(ext).max(), 50.0)

    def test_penalty_is_one_per_feature_at_both_channel_counts(self):
        """Die Anti-Confounding-Regel der Spec: 100 bei 5 Kanaelen, 200 bei 10."""
        data = build_2x2_dataset(synthetic_prices())
        for block, expected in (('X_base', 100.0), ('X_ext', 200.0)):
            features = SEQUENCE * data[block].shape[3]
            self.assertEqual(RIDGE_LAMBDA_PER_FEATURE * features, expected)


class ScalerTests(unittest.TestCase):
    def test_training_pairs_respect_the_label_delay(self):
        y = np.zeros((10, 2), dtype='float32')
        y[9] = np.nan
        pairs = training_pairs(y, first_test=8, label_delay=2)
        self.assertEqual(int(pairs[:, 0].max()), 6)
        self.assertEqual(len(pairs), 14)

    def test_scaler_pools_timesteps_and_guards_zero_scale(self):
        X = np.ones((6, 2, 20, 3), dtype='float32')
        X[..., 1] = np.arange(20, dtype='float32')
        y = np.zeros((6, 2), dtype='float32')
        pairs = training_pairs(y, first_test=6, label_delay=2)
        mean, scale = fit_scaler(X, pairs)
        self.assertEqual(mean.shape, (3,))
        np.testing.assert_allclose(mean[0], 1.0)
        np.testing.assert_allclose(scale[0], 1.0)  # konstante Spalte -> Guard
        np.testing.assert_allclose(mean[1], 9.5)
        standardized = standardize(X, pairs, mean, scale)
        self.assertEqual(standardized.shape, (len(pairs), 20, 3))
        self.assertLessEqual(np.abs(standardized).max(), 10.0)

    def test_standardize_clips_at_ten(self):
        X = np.zeros((4, 1, 20, 1), dtype='float32')
        X[2, 0, :, 0] = 1000.0
        y = np.zeros((4, 1), dtype='float32')
        pairs = training_pairs(y, first_test=4, label_delay=2)
        mean, scale = fit_scaler(X, pairs)
        self.assertLessEqual(np.abs(standardize(X, pairs, mean, scale)).max(), 10.0)


class GbmTests(unittest.TestCase):
    def test_recipe_is_frozen(self):
        self.assertEqual(GBM_ROUNDS, 300)
        self.assertEqual(GBM_PARAMS['learning_rate'], 0.05)
        self.assertEqual(GBM_PARAMS['num_leaves'], 15)
        self.assertEqual(GBM_PARAMS['min_data_in_leaf'], 200)
        self.assertEqual(GBM_PARAMS['feature_fraction'], 0.7)
        self.assertEqual(GBM_PARAMS['bagging_fraction'], 0.7)
        self.assertEqual(GBM_PARAMS['bagging_freq'], 1)
        self.assertEqual(GBM_PARAMS['seed'], 7)
        self.assertTrue(GBM_PARAMS['deterministic'])

    def test_predictions_have_the_expected_shape_and_are_finite(self):
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 30
        out = fit_predict_gbm(data['X_ext'], data['y'], first, len(data['dates']))
        self.assertEqual(out.shape, (30, data['y'].shape[1]))
        self.assertTrue(np.isfinite(out).all())

    def test_repeated_runs_are_bit_identical(self):
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 30
        a = fit_predict_gbm(data['X_base'], data['y'], first, len(data['dates']))
        b = fit_predict_gbm(data['X_base'], data['y'], first, len(data['dates']))
        np.testing.assert_array_equal(a, b)

    def test_uses_no_labels_from_the_test_block(self):
        """Label im Testblock veraendern darf die Prognosen nicht bewegen."""
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 30
        clean = fit_predict_gbm(data['X_base'], data['y'], first, len(data['dates']))
        tampered = data['y'].copy()
        tampered[first - 1:] = 9.0
        dirty = fit_predict_gbm(data['X_base'], tampered, first, len(data['dates']))
        np.testing.assert_array_equal(clean, dirty)


if __name__ == '__main__':
    unittest.main()
