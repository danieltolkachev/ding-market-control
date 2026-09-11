"""Holding-horizon labels and inclusive model maturity boundaries."""
import unittest

import numpy as np
import pandas as pd

from factor_lab.daily_models import fit_predict_models


class HorizonLabelsTests(unittest.TestCase):
    def labels(self, prices, vol, horizon):
        from factor_lab.horizon_data import horizon_labels
        return horizon_labels(prices, vol, horizon)

    def data(self):
        prices = pd.DataFrame({'a': [100., 100., 110., 120., 150., 180.]},
                              index=pd.bdate_range('2020-01-01', periods=6))
        return prices, pd.DataFrame(.1, index=prices.index, columns=prices.columns)

    def test_hand_computed_alignment_clipping_and_incomplete_tail(self):
        prices, vol = self.data()
        result = self.labels(prices, vol.iloc[1:], 2)
        expected = pd.DataFrame({'a': [40/11/np.sqrt(2), 5/np.sqrt(2),
                                       np.nan, np.nan, np.nan]}, index=vol.index[1:])
        pd.testing.assert_frame_equal(result, expected)
        vol.iloc[0] = .0001
        self.assertEqual(self.labels(prices, vol, 2).iloc[0, 0], 10)
        prices.iloc[3] = 1
        self.assertEqual(self.labels(prices, vol, 2).iloc[0, 0], -10)
        self.assertTrue(self.labels(prices, vol, 21).isna().all().all())

    def test_daily_matches_existing_dataset(self):
        from factor_lab.daily_comparison import build_dataset
        rng = np.random.default_rng(17)
        prices = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0, .01, (180, 2)), axis=0)),
                              index=pd.bdate_range('2020-01-01', periods=180))
        dataset = build_dataset(prices)
        result = self.labels(prices, dataset['vol'], 1)
        np.testing.assert_allclose(result, dataset['y'], rtol=1e-6, atol=1e-7)

    def test_future_price_change_cannot_change_mature_label(self):
        prices, vol = self.data()
        before = self.labels(prices, vol, 2)
        prices.iloc[4:] *= 2
        after = self.labels(prices, vol, 2)
        pd.testing.assert_frame_equal(before.iloc[:1], after.iloc[:1])
        self.assertNotEqual(before.iloc[1, 0], after.iloc[1, 0])

    def test_reject_invalid_horizon_and_calendars(self):
        prices, vol = self.data()
        for horizon in (0, -1, 1.5, True, np.bool_(True)):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                self.labels(prices, vol, horizon)
        for bad in (prices.iloc[::-1], prices.iloc[[0, 0, 1]], prices*0,
                    prices*np.inf, prices.rename(columns={'a': 'b'})):
            with self.assertRaises(ValueError):
                self.labels(bad, vol, 2)
        for bad in (vol.iloc[[0, 2, 3]], vol.iloc[::-1], vol*0, vol*np.nan):
            with self.assertRaises(ValueError):
                self.labels(prices, bad, 2)
        pd.testing.assert_frame_equal(self.labels(prices, vol, np.int64(2)),
                                      self.labels(prices, vol, 2))


class HorizonModelsTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(34, 1, 20, 5)).astype(np.float32)
        return X, X[:, :, -1, 0].copy()

    def test_default_exactly_matches_explicit_two(self):
        X, y = self.data()
        default = fit_predict_models(X, y, 25)
        explicit = fit_predict_models(X, y, 25, label_delay=2)
        for name in default:
            np.testing.assert_array_equal(default[name], explicit[name])

    def test_initial_and_online_maturity_boundaries(self):
        X, y = self.data()
        for delay in (6, 22):
            with self.subTest(delay=delay):
                def fit(labels):
                    return fit_predict_models(X, labels, 25, update_every=3, label_delay=delay)
                baseline = fit(y)
                # Initial unavailable boundary stays out of scaling and fitting.
                changed = y.copy()
                changed[25-delay+1] = 9
                initial = fit(changed)
                for name in ('ridge', 'lstm_frozen'):
                    np.testing.assert_array_equal(baseline[name], initial[name])
                np.testing.assert_array_equal(baseline['lstm_online'][:3], initial['lstm_online'][:3])
                self.assertFalse(np.array_equal(baseline['lstm_online'][3:], initial['lstm_online'][3:]))
                # At first update d=28, the next label is still unavailable.
                changed = y.copy()
                changed[28-delay+1] = 9
                unavailable = fit(changed)
                for name in baseline:
                    np.testing.assert_array_equal(baseline[name][:6], unavailable[name][:6])
                self.assertFalse(np.array_equal(baseline['lstm_online'][6:], unavailable['lstm_online'][6:]))
                # Exact maturity boundaries must be included, not discarded.
                changed = y.copy()
                changed[25-delay] = 9
                mature_initial = fit(changed)
                self.assertFalse(np.array_equal(baseline['ridge'], mature_initial['ridge']))
                changed = y.copy()
                changed[28-delay] = 9
                mature_online = fit(changed)
                np.testing.assert_array_equal(baseline['lstm_online'][:3], mature_online['lstm_online'][:3])
                self.assertFalse(np.array_equal(baseline['lstm_online'][3], mature_online['lstm_online'][3]))
                np.testing.assert_array_equal(baseline['lstm_frozen'], mature_online['lstm_frozen'])
                np.testing.assert_array_equal(baseline['lstm_frozen'][:3], baseline['lstm_online'][:3])

    def test_delay_validation_and_minimum_initial_sample(self):
        X, y = self.data()
        for delay in (1, -1, 2.5, True, np.bool_(True)):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                fit_predict_models(X, y, 25, label_delay=delay)
        with self.assertRaises(ValueError):
            fit_predict_models(X, y, 21, label_delay=22)
        result = fit_predict_models(X, y, 22, label_delay=np.int64(22))
        np.testing.assert_allclose(result['ridge'], y[0, 0], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
