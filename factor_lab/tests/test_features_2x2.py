"""Kanaeldefinitionen, Kausalitaet und Zeilenausrichtung des 2x2-Datensatzes."""
import unittest

import numpy as np
import pandas as pd

from factor_lab.daily_comparison import build_dataset
from factor_lab.features_2x2 import (
    BASE_HORIZONS,
    EXTENDED_NAMES,
    build_2x2_dataset,
    extended_channels,
)


def synthetic_prices(periods=900, assets=('SPY', 'AAA', 'BBB'), seed=3):
    """Deterministische, positive Preisreihen auf einem Boersenkalender."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2010-01-04', periods=periods)
    data = {}
    for offset, name in enumerate(assets):
        steps = 0.0004 * (offset + 1) + 0.01 * rng.standard_normal(periods)
        data[name] = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=index)


class ExtendedChannelTests(unittest.TestCase):
    def test_channel_names_and_shapes(self):
        prices = synthetic_prices()
        channels = extended_channels(prices)
        self.assertEqual(tuple(channels), EXTENDED_NAMES)
        for frame in channels.values():
            self.assertTrue(frame.index.equals(prices.index))
            self.assertTrue(frame.columns.equals(prices.columns))

    def test_skip_momentum_matches_definition(self):
        prices = synthetic_prices()
        returns = prices.pct_change(fill_method=None)
        vol = returns.rolling(63, min_periods=63).std().clip(lower=1e-6)
        expected = (prices.shift(21) / prices.shift(147) - 1) / vol / np.sqrt(126)
        got = extended_channels(prices)['skip_momentum']
        pd.testing.assert_frame_equal(got, expected, check_names=False)

    def test_cross_rank_is_bounded_and_purely_cross_sectional(self):
        prices = synthetic_prices()
        channels = extended_channels(prices)
        rank = channels['cross_rank'].dropna(how='all')
        self.assertTrue(((rank >= -1 - 1e-12) & (rank <= 1 + 1e-12)).all().all())
        # Drei Assets: exakt ein Minimum bei -1, ein Maximum bei +1 je Tag.
        row = rank.dropna().iloc[-1]
        np.testing.assert_allclose(sorted(row.to_numpy()), [-1.0, 0.0, 1.0])

    def test_spy_correlation_channel_is_one_for_the_market_itself(self):
        prices = synthetic_prices()
        spy = extended_channels(prices)['spy_corr']['SPY'].dropna()
        self.assertGreater(len(spy), 0)
        np.testing.assert_allclose(spy.to_numpy(), 1.0, atol=1e-10)

    def test_channels_are_causal(self):
        """Aenderung eines zukuenftigen Preises darf Kanaele bei t nicht bewegen."""
        prices = synthetic_prices()
        cut = prices.index[-30]
        tampered = prices.copy()
        tampered.loc[prices.index[-10]:] *= 1.5
        before = extended_channels(prices).items()
        after = dict(extended_channels(tampered))
        for name, frame in before:
            pd.testing.assert_frame_equal(
                frame.loc[:cut], after[name].loc[:cut], check_names=False
            )


class DatasetAlignmentTests(unittest.TestCase):
    def test_base_block_equals_existing_builder_on_shared_rows(self):
        prices = synthetic_prices()
        data = build_2x2_dataset(prices)
        reference = build_dataset(prices)
        shared = data['dates']
        self.assertTrue(shared.isin(reference['dates']).all())
        take = reference['dates'].get_indexer(shared)
        np.testing.assert_allclose(data['X_base'], reference['X'][take], atol=0, rtol=0)
        np.testing.assert_array_equal(
            np.isfinite(data['y']), np.isfinite(reference['y'][take])
        )
        np.testing.assert_allclose(
            np.nan_to_num(data['y']), np.nan_to_num(reference['y'][take]), atol=0, rtol=0
        )

    def test_extended_block_carries_base_channels_first(self):
        data = build_2x2_dataset(synthetic_prices())
        np.testing.assert_allclose(data['X_ext'][..., :5], data['X_base'], atol=0, rtol=0)
        self.assertEqual(data['X_ext'].shape[-1], len(BASE_HORIZONS) + len(EXTENDED_NAMES))

    def test_all_extended_features_are_finite_on_kept_rows(self):
        data = build_2x2_dataset(synthetic_prices())
        self.assertTrue(np.isfinite(data['X_ext']).all())
        self.assertGreater(len(data['dates']), 0)

    def test_alignment_starts_later_than_the_base_only_dataset(self):
        prices = synthetic_prices()
        self.assertGreater(
            build_2x2_dataset(prices)['dates'][0], build_dataset(prices)['dates'][0]
        )


if __name__ == '__main__':
    unittest.main()
