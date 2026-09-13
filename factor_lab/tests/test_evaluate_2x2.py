"""Fenstergrenzen, Purge und Zusammensetzen der Prognosen."""
import unittest

import numpy as np
import pandas as pd

from factor_lab.evaluate_2x2 import (
    CELLS,
    N_WINDOWS,
    expanding_windows,
    predict_over_windows,
)
from factor_lab.features_2x2 import build_2x2_dataset


def synthetic_prices(periods=700, assets=('SPY', 'AAA', 'BBB'), seed=23):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2011-01-03', periods=periods)
    data = {}
    for offset, name in enumerate(assets):
        steps = 0.0002 * (offset + 1) + 0.011 * rng.standard_normal(periods)
        data[name] = 80.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=index)


class WindowTests(unittest.TestCase):
    def test_windows_partition_the_evaluation_range_exactly(self):
        windows = expanding_windows(1501, 4433)
        self.assertEqual(len(windows), N_WINDOWS)
        self.assertEqual(windows[0][0], 1501)
        self.assertEqual(windows[-1][1], 4433)
        for (_, end), (start, _) in zip(windows, windows[1:]):
            self.assertEqual(end, start)
        sizes = [end - start for start, end in windows]
        self.assertEqual(sum(sizes), 2932)
        # 2932 // 5 = 586 Rest 2: der gesamte Rest geht an den letzten Block
        # (siehe test_block_sizes_for_the_real_evaluation_range), daher ist die
        # maximale Differenz hier der Rest (2), nicht die uebliche 1.
        self.assertLessEqual(max(sizes) - min(sizes), 2)

    def test_block_sizes_for_the_real_evaluation_range(self):
        sizes = [end - start for start, end in expanding_windows(1501, 4433)]
        self.assertEqual(sizes, [586, 586, 586, 586, 588])

    def test_rejects_degenerate_ranges(self):
        with self.assertRaises(ValueError):
            expanding_windows(100, 100)
        with self.assertRaises(ValueError):
            expanding_windows(100, 102, n_windows=5)

    def test_cells_are_the_four_preregistered_ones(self):
        self.assertEqual([name for name, _, _ in CELLS],
                         ['A_base_ridge', 'B_base_gbm', 'C_ext_ridge', 'D_ext_gbm'])
        self.assertEqual({block for _, block, _ in CELLS}, {'X_base', 'X_ext'})
        self.assertEqual({model for _, _, model in CELLS}, {'ridge', 'gbm'})


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.data = build_2x2_dataset(synthetic_prices())
        self.days = len(self.data['dates'])
        self.first = self.days - 100
        self.windows = expanding_windows(self.first, self.days)

    def test_composed_predictions_cover_every_evaluation_row(self):
        out = predict_over_windows(
            self.data['X_base'], self.data['y'], self.windows, 'ridge')
        self.assertEqual(out.shape, (100, self.data['y'].shape[1]))
        self.assertTrue(np.isfinite(out).all())

    def test_each_block_equals_its_own_single_window_fit(self):
        out = predict_over_windows(
            self.data['X_base'], self.data['y'], self.windows, 'ridge')
        from factor_lab.models_2x2 import fit_predict_ridge
        start, end = self.windows[2]
        direct = fit_predict_ridge(self.data['X_base'], self.data['y'], start, end)
        np.testing.assert_allclose(out[start - self.first:end - self.first], direct)

    def test_no_window_sees_labels_from_its_own_or_later_blocks(self):
        clean = predict_over_windows(
            self.data['X_base'], self.data['y'], self.windows, 'ridge')
        tampered = self.data['y'].copy()
        last_start = self.windows[-1][0]
        tampered[last_start - 1:] = 7.0
        dirty = predict_over_windows(
            self.data['X_base'], tampered, self.windows, 'ridge')
        np.testing.assert_array_equal(clean, dirty)

    def test_unknown_model_name_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_over_windows(
                self.data['X_base'], self.data['y'], self.windows, 'lstm')


if __name__ == '__main__':
    unittest.main()
