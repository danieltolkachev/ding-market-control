"""Causal regression tests; run with python -m unittest factor_lab.tests.test_daily_models."""
import unittest

import numpy as np


class DailyModelsTests(unittest.TestCase):
    def fit(self, X, y, first=12, **kwargs):
        from factor_lab.daily_models import fit_predict_models
        return fit_predict_models(X, y, first, **kwargs)

    def data(self, days=58):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(days, 2, 20, 5)).astype(np.float32)
        y = (X[:, :, -1, 0] * .4).copy()
        y[-2:] = np.nan
        return X, y

    def test_identity_until_first_update_and_repeatability(self):
        X, y = self.data()
        before = X.copy(), y.copy()
        a, b = self.fit(X, y), self.fit(X, y)
        self.assertEqual(set(a), {'ridge', 'lstm_frozen', 'lstm_online'})
        for name in a:
            self.assertEqual(a[name].shape, (46, 2))
            self.assertTrue(np.isfinite(a[name]).all())
            np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(a['lstm_frozen'][:21], a['lstm_online'][:21])
        self.assertFalse(np.array_equal(a['lstm_frozen'][21:], a['lstm_online'][21:]))
        np.testing.assert_array_equal(X, before[0])
        np.testing.assert_array_equal(y, before[1])

    def test_unobservable_labels_and_future_features_cannot_change_past(self):
        X, y = self.data()
        a = self.fit(X, y)
        poisoned = y.copy()
        # First update is decision33: label32 is still unavailable.
        poisoned[32:-2] = 9
        b = self.fit(X, poisoned)
        for name in a:
            np.testing.assert_array_equal(a[name][:42], b[name][:42])
        self.assertFalse(np.array_equal(a['lstm_online'][42:], b['lstm_online'][42:]))
        future = X.copy()
        future[34:] = 10000
        c = self.fit(future, y)
        for name in a:
            np.testing.assert_array_equal(a[name][:22], c[name][:22])

    def test_initial_and_online_inclusive_maturity_boundary(self):
        X, y = self.data()
        a = self.fit(X, y)
        changed = y.copy()
        changed[10] = 9  # first_test - 2 is included in initial fit
        b = self.fit(X, changed)
        self.assertFalse(np.array_equal(a['ridge'], b['ridge']))
        changed = y.copy()
        changed[31] = 9  # decision33 - 2 must enter the first update
        c = self.fit(X, changed)
        np.testing.assert_array_equal(a['lstm_online'][:21], c['lstm_online'][:21])
        self.assertFalse(np.array_equal(a['lstm_online'][21], c['lstm_online'][21]))
        np.testing.assert_array_equal(a['lstm_frozen'], c['lstm_frozen'])

    def test_initial_fit_excludes_label_immediately_before_first_test(self):
        X, y = self.data()
        a = self.fit(X, y)
        poisoned = y.copy()
        poisoned[11] = 9  # first_test-1 matures one decision after initial fit
        b = self.fit(X, poisoned)
        for name in ('ridge', 'lstm_frozen'):
            np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(a['lstm_online'][:21], b['lstm_online'][:21])
        self.assertFalse(np.array_equal(a['lstm_online'][21:], b['lstm_online'][21:]))

    def test_ridge_sum_objective_nonfinite_exclusion_and_fixed_scaling(self):
        # Twenty identical varying coordinates, each +/-1, y=+/-1.
        # Summed squared loss + 100*||w||^2 gives20/(20+100/2).
        X = np.zeros((7, 1, 20, 5))
        X[:, 0, :, 0] = np.array([-1, 1, 999, 2, 1, 20, -1])[:, None]
        y = np.array([[-1], [1], [np.inf], [0], [0], [np.nan], [np.nan]])
        result = self.fit(X, y, first=4)
        np.testing.assert_allclose(result['ridge'][:, 0], [2/7, 20/7, -2/7], atol=1e-7)

    def test_ridge_unpenalized_intercept_recovers_constant_target(self):
        X, y = self.data()
        y[:-2] = 3.25
        result = self.fit(X, y)
        np.testing.assert_allclose(result['ridge'], 3.25, atol=1e-12)

    def test_no_training_labels_fails_clearly(self):
        X, y = self.data()
        y[:11] = np.nan
        with self.assertRaisesRegex(ValueError, 'training'):
            self.fit(X, y)

    def test_online_window_excludes_old_labels_but_includes_oldest_eligible(self):
        X, y = self.data(days=287)
        # First update d=285 uses 32..283 inclusive: exactly252 days.
        a = self.fit(X, y, update_every=273)
        old = y.copy()
        old[12:32] = 9
        b = self.fit(X, old, update_every=273)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        boundary = y.copy()
        boundary[32] = 9
        c = self.fit(X, boundary, update_every=273)
        np.testing.assert_array_equal(a['lstm_online'][:273], c['lstm_online'][:273])
        self.assertFalse(np.array_equal(a['lstm_online'][273:], c['lstm_online'][273:]))


if __name__ == '__main__':
    unittest.main()
