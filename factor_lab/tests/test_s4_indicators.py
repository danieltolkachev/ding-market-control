"""Indikatoren und Featurevektor der S4-Mechanik.

ALLE Daten in dieser Datei sind SYNTHETISCH und dienen ausschliesslich der
Mechanikpruefung. Sie stammen aus keiner Marktquelle und stellen kein
Marktverhalten dar.
"""
import unittest

import numpy as np
import pandas as pd

from factor_lab.s4_features import build_features
from factor_lab.s4_indicators import SEQUENCE, WARMUP, compute_indicators


def synthetic_bars(n=400, seed=1, start='2010-01-04'):
    """Synthetische, streng positive OHLC-Bars auf einem Werktagskalender."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(0.0004 + 0.01 * rng.standard_normal(n)))
    spread = close * 0.01
    high = close + spread * rng.uniform(0.2, 1.0, n)
    low = close - spread * rng.uniform(0.2, 1.0, n)
    open_ = low + (high - low) * rng.uniform(0.0, 1.0, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close},
                        index=pd.bdate_range(start, periods=n))


class IndicatorTests(unittest.TestCase):
    def test_constants_are_frozen(self):
        self.assertEqual(WARMUP, 260)
        self.assertEqual(SEQUENCE, 60)

    def test_true_range_matches_definition(self):
        bars = synthetic_bars(50)
        ind = compute_indicators(bars)
        h, l, c = bars['high'].to_numpy(), bars['low'].to_numpy(), bars['close'].to_numpy()
        for i in range(1, 50):
            expected = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            self.assertAlmostEqual(ind['tr'][i], expected, places=12)
        self.assertTrue(np.isnan(ind['tr'][0]))

    def test_atr_is_a_simple_moving_average_not_wilder(self):
        bars = synthetic_bars(60)
        ind = compute_indicators(bars)
        self.assertAlmostEqual(ind['atr20'][25], float(np.mean(ind['tr'][6:26])), places=12)
        self.assertTrue(np.isnan(ind['atr20'][19]))
        self.assertFalse(np.isnan(ind['atr20'][20]))

    def test_ema_is_seeded_with_the_sma_of_the_first_n_closes(self):
        bars = synthetic_bars(80)
        ind = compute_indicators(bars)
        c = bars['close'].to_numpy()
        self.assertTrue(np.isnan(ind['ema20'][18]))
        self.assertAlmostEqual(ind['ema20'][19], float(np.mean(c[:20])), places=12)
        alpha = 2 / 21
        self.assertAlmostEqual(ind['ema20'][20], alpha * c[20] + (1 - alpha) * ind['ema20'][19],
                               places=12)

    def test_sigma_uses_sample_standard_deviation_of_log_returns(self):
        bars = synthetic_bars(60)
        ind = compute_indicators(bars)
        r = ind['logret']
        self.assertAlmostEqual(ind['sigma20'][30], float(np.std(r[11:31], ddof=1)), places=12)
        self.assertAlmostEqual(ind['sigma5'][30], float(np.std(r[26:31], ddof=1)), places=12)

    def test_efficiency_ratio_matches_definition(self):
        bars = synthetic_bars(60)
        ind = compute_indicators(bars)
        c = bars['close'].to_numpy()
        j = 40
        expected = abs(c[j] - c[j - 20]) / np.abs(np.diff(c[j - 20:j + 1])).sum()
        self.assertAlmostEqual(ind['er20'][j], expected, places=12)

    def test_uptrend_requires_all_three_conditions(self):
        bars = synthetic_bars(400)
        ind = compute_indicators(bars)
        j = 350
        manual = (ind['ema50'][j] > ind['ema200'][j]
                  and ind['ema50'][j] > ind['ema50'][j - 10]
                  and bars['close'].to_numpy()[j] > ind['ema50'][j])
        self.assertEqual(bool(ind['up'][j]), manual)
        # EMA200 ist erst ab lauf-lokalem Index 199 definiert; davor immer False.
        self.assertFalse(bool(ind['up'][198]))

    def test_indicators_are_causal(self):
        """Spaetere Bars aendern fruehere Indikatorwerte nicht."""
        bars = synthetic_bars(300)
        cut = 200
        tampered = bars.copy()
        tampered.iloc[cut + 1:] *= 1.5
        a = compute_indicators(bars)
        b = compute_indicators(tampered)
        for key in ('logret', 'tr', 'atr20', 'sigma5', 'sigma20', 'er20', 'ema20', 'ema50',
                    'ema200', 'up'):
            np.testing.assert_allclose(np.nan_to_num(a[key][:cut + 1].astype(float)),
                                       np.nan_to_num(b[key][:cut + 1].astype(float)))


class RunSegmentationTests(unittest.TestCase):
    def test_run_length_counts_contiguous_valid_bars(self):
        bars = synthetic_bars(50)
        bars.iloc[20] = np.nan
        ind = compute_indicators(bars)
        self.assertEqual(int(ind['run_length'][19]), 20)
        self.assertFalse(bool(ind['valid'][20]))
        self.assertEqual(int(ind['run_length'][20]), 0)
        self.assertEqual(int(ind['run_length'][21]), 1)

    def test_ema_state_restarts_after_a_data_break(self):
        bars = synthetic_bars(120)
        bars.iloc[60] = np.nan
        ind = compute_indicators(bars)
        c = bars['close'].to_numpy()
        self.assertTrue(np.isnan(ind['ema20'][79]))
        self.assertAlmostEqual(ind['ema20'][80], float(np.mean(c[61:81])), places=12)

    def test_nonpositive_and_inverted_bars_are_invalid(self):
        bars = synthetic_bars(40)
        bars.iloc[10, bars.columns.get_loc('close')] = 0.0
        bars.iloc[20, bars.columns.get_loc('high')] = bars['low'].iloc[20] - 1.0
        ind = compute_indicators(bars)
        self.assertFalse(bool(ind['valid'][10]))
        self.assertFalse(bool(ind['valid'][20]))

    def test_rejects_malformed_frames(self):
        with self.assertRaises(ValueError):
            compute_indicators(pd.DataFrame({'close': [1.0, 2.0]}))

    def test_rejects_non_datetime_index(self):
        bars = synthetic_bars(10)
        bars = bars.reset_index(drop=True)
        with self.assertRaises(ValueError):
            compute_indicators(bars)


class FeatureTests(unittest.TestCase):
    def test_shapes_and_exact_channel_order(self):
        bars = synthetic_bars(400)
        ind = compute_indicators(bars)
        t = 399
        p = t - 2
        seq, ctx = build_features(ind, bars, p, t)
        self.assertEqual(seq.shape, (60, 10))
        self.assertEqual(ctx.shape, (4,))
        c = bars['close'].to_numpy()
        j = t
        a, s = ind['atr20'][j - 1], ind['sigma20'][j - 1]
        row = seq[-1]
        self.assertAlmostEqual(row[0], ind['logret'][j] / s, places=12)
        self.assertAlmostEqual(row[1], np.log(c[j] / c[j - 5]) / (s * np.sqrt(5)), places=12)
        self.assertAlmostEqual(row[2], np.log(c[j] / c[j - 20]) / (s * np.sqrt(20)), places=12)
        self.assertAlmostEqual(row[3], (c[j] - ind['ema20'][j - 1]) / a, places=12)
        self.assertAlmostEqual(row[4], (c[j] - ind['ema50'][j - 1]) / a, places=12)
        self.assertAlmostEqual(row[5], (ind['ema50'][j - 1] - ind['ema50'][j - 11]) / a, places=12)
        self.assertAlmostEqual(row[6], ind['tr'][j] / a, places=12)
        self.assertAlmostEqual(row[7], ind['sigma5'][j - 1] / s, places=12)
        self.assertAlmostEqual(row[8], ind['er20'][j], places=12)

    def test_context_values_match_definition(self):
        bars = synthetic_bars(400)
        ind = compute_indicators(bars)
        t, p = 399, 396
        seq, ctx = build_features(ind, bars, p, t)
        h, l, c = bars['high'].to_numpy(), bars['low'].to_numpy(), bars['close'].to_numpy()
        self.assertAlmostEqual(ctx[0], (t - p) / 3, places=12)
        expected = (h[p - 20:p].max() - l[p:t + 1].min()) / ind['atr20'][p - 1]
        self.assertAlmostEqual(ctx[1], expected, places=12)
        self.assertAlmostEqual(ctx[2], (c[t] - h[t - 1]) / ind['atr20'][t - 1], places=12)
        self.assertAlmostEqual(ctx[3], ind['atr20'][t - 1] / c[t], places=12)

    def test_zero_range_bar_uses_the_defined_convention(self):
        bars = synthetic_bars(400)
        j = 380
        for column in ('open', 'high', 'low'):
            bars.iloc[j, bars.columns.get_loc(column)] = bars['close'].iloc[j]
        ind = compute_indicators(bars)
        seq, _ = build_features(ind, bars, 397, 399)
        self.assertAlmostEqual(seq[j - (399 - 59)][9], 0.5, places=12)

    def test_missing_history_raises_rather_than_imputing(self):
        bars = synthetic_bars(400)
        ind = compute_indicators(bars)
        with self.assertRaises(ValueError):
            build_features(ind, bars, 58, 60)


if __name__ == '__main__':
    unittest.main()
