"""Labelkernel der S4-Mechanik, Spec Abschnitt 8.

ALLE Daten in dieser Datei sind SYNTHETISCH und handgebaut, damit jeder
Ausgang exakt bestimmbar ist. Sie stammen aus keiner Marktquelle.
"""
import unittest

import numpy as np
import pandas as pd

from factor_lab.s4_labels import (
    ACTION_UNSUPPORTED,
    CENSORED,
    GAP_STOP,
    GAP_TARGET,
    STOP,
    TARGET,
    TARGET_FIRST,
    TIME,
    label_event,
)


def frame(rows):
    """rows: Liste von (open, high, low, close); Index ist ein Werktagskalender."""
    return pd.DataFrame(np.array(rows, dtype=float),
                        index=pd.bdate_range('2020-01-01', periods=len(rows)),
                        columns=['open', 'high', 'low', 'close'])


def plain_actions(bars):
    return pd.DataFrame({'split': 1.0, 'dividend': 0.0}, index=bars.index)


def event(t=0, R=1.0, q=1.5, N=10):
    return {'id': ('SYNTH', t, 'S4-v1'), 't': t, 'R': R, 'q': q, 'N': N}


class BarrierResolutionTests(unittest.TestCase):
    def test_target_hit_gives_plus_one_and_the_expected_net_r(self):
        # Low 99.2 liegt UEBER dem Stop 99 -- sonst waere die Bar mehrdeutig
        # und wuerde nach stop_first als Verlust aufgeloest.
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 99.2, 101.5)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['barrier_class'], 1)
        self.assertEqual(out['reason'], TARGET)
        self.assertAlmostEqual(out['entry'], 100.0)
        self.assertAlmostEqual(out['exit'], 101.5)
        self.assertAlmostEqual(out['net_R'], 1.5)
        self.assertEqual(out['meta_label'], 1)
        self.assertFalse(out['ambiguous'])

    def test_stop_hit_gives_minus_one(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 100.5, 98.9, 99)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['barrier_class'], -1)
        self.assertEqual(out['reason'], STOP)
        self.assertAlmostEqual(out['net_R'], -1.0)
        self.assertEqual(out['meta_label'], 0)

    def test_timeout_closes_at_the_last_held_bar(self):
        rows = [(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.1)] * 10
        bars = frame(rows)
        out = label_event(event(N=10), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['barrier_class'], 0)
        self.assertEqual(out['reason'], TIME)
        self.assertEqual(out['exit_bar'], 10)
        self.assertAlmostEqual(out['exit'], 100.1)
        self.assertEqual(out['label_available_after'], bars.index[10])


class GapTests(unittest.TestCase):
    def test_gap_below_the_stop_fills_at_the_open_not_at_the_stop(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (95, 96, 94, 95)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['reason'], GAP_STOP)
        self.assertAlmostEqual(out['exit'], 95.0)
        self.assertAlmostEqual(out['net_R'], -5.0)   # Gap verliert mehr als 1R
        self.assertFalse(out['ambiguous'])

    def test_gap_above_the_target_fills_at_the_open(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (104, 105, 103, 104)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['reason'], GAP_TARGET)
        self.assertAlmostEqual(out['exit'], 104.0)
        self.assertAlmostEqual(out['net_R'], 4.0)

    def test_the_open_is_resolved_before_high_and_low(self):
        """Eine Bar, die beide Barrieren enthaelt, aber unter dem Stop eroeffnet."""
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (98, 102, 97, 101)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['reason'], GAP_STOP)
        self.assertFalse(out['ambiguous'])


class AmbiguityTests(unittest.TestCase):
    def test_dual_touch_resolves_stop_first_and_is_flagged(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 98.5, 101)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['barrier_class'], -1)
        self.assertEqual(out['reason'], STOP)
        self.assertTrue(out['ambiguous'])

    def test_target_priority_variant_reverses_the_same_bar(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 98.5, 101)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0, tie=TARGET_FIRST)
        self.assertEqual(out['barrier_class'], 1)
        self.assertEqual(out['reason'], TARGET)
        self.assertTrue(out['ambiguous'])


class CostTests(unittest.TestCase):
    def test_haircut_moves_entry_up_and_exit_down(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 200, 99.9, 150)])
        out = label_event(event(), bars, plain_actions(bars), 0.001, 0.0)
        self.assertAlmostEqual(out['entry'], 100.1)
        self.assertAlmostEqual(out['exit'], (100.1 + 1.5) * 0.999, places=9)

    def test_commission_is_charged_on_both_sides(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 99.9, 101.5)])
        free = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        paid = label_event(event(), bars, plain_actions(bars), 0.0, 0.0001)
        self.assertLess(paid['net_R'], free['net_R'])
        expected = free['net_R'] - 0.0001 * (free['entry'] + free['exit'])
        self.assertAlmostEqual(paid['net_R'], expected, places=9)

    def test_costs_can_turn_a_barrier_win_into_a_negative_meta_label(self):
        """Kleines R gegen hohe Kosten: der Target-Treffer deckt sie nicht.

        R=0,1 bei O=100 und b=0,0005: der Eintritts-Haircut betraegt 0,05 und
        liegt damit unter R, der Trade wird also nicht sofort ausgestoppt.
        Die Provision von 10 bp je Seite ist hier bewusst hoch gewaehlt, um den
        Effekt zu zeigen -- sie ist kein Anbieterpreis.
        """
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 100.25, 99.99, 100.2)])
        out = label_event(event(R=0.1), bars, plain_actions(bars), 0.0005, 0.001)
        self.assertEqual(out['barrier_class'], 1)
        self.assertEqual(out['reason'], TARGET)
        self.assertEqual(out['meta_label'], 0)
        self.assertLess(out['net_R'], 0.0)


class CorporateActionTests(unittest.TestCase):
    def test_a_split_before_entry_converts_r_to_the_entry_share_basis(self):
        # R wird zu 0,5, Stop 49,5, Target 50,75. Low 49,6 liegt ueber dem Stop.
        bars = frame([(100, 100, 100, 100), (50, 50, 50, 50), (50, 51, 49.6, 50.8)])
        actions = plain_actions(bars)
        actions.iloc[1, actions.columns.get_loc('split')] = 2.0
        out = label_event(event(R=1.0), bars, actions, 0.0, 0.0)
        self.assertEqual(out['reason'], TARGET)
        self.assertAlmostEqual(out['net_R'], 1.5, places=9)

    def test_a_split_after_entry_leaves_net_r_unchanged(self):
        clean = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                       (100, 102, 99.5, 101.5)])
        out_clean = label_event(event(), clean, plain_actions(clean), 0.0, 0.0)
        split = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                       (50, 51, 49.75, 50.75)])
        actions = plain_actions(split)
        actions.iloc[2, actions.columns.get_loc('split')] = 2.0
        out_split = label_event(event(), split, actions, 0.0, 0.0)
        self.assertAlmostEqual(out_split['net_R'], out_clean['net_R'], places=9)

    def test_a_dividend_after_entry_is_booked_and_not_a_price_gain(self):
        rows = [(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.0)] * 10
        bars = frame(rows)
        actions = plain_actions(bars)
        actions.iloc[3, actions.columns.get_loc('dividend')] = 0.5
        out = label_event(event(N=10), bars, actions, 0.0, 0.0)
        self.assertEqual(out['reason'], TIME)
        self.assertAlmostEqual(out['net_R'], 0.5, places=9)

    def test_no_dividend_claim_when_buying_on_the_ex_day(self):
        rows = [(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.0)] * 10
        bars = frame(rows)
        actions = plain_actions(bars)
        actions.iloc[1, actions.columns.get_loc('dividend')] = 0.5
        out = label_event(event(N=10), bars, actions, 0.0, 0.0)
        self.assertAlmostEqual(out['net_R'], 0.0, places=9)

    def test_nonpositive_split_factor_is_rejected(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 99, 101.5)])
        actions = plain_actions(bars)
        actions.iloc[2, actions.columns.get_loc('split')] = 0.0
        out = label_event(event(), bars, actions, 0.0, 0.0)
        self.assertEqual(out['status'], ACTION_UNSUPPORTED)


class CensoringTests(unittest.TestCase):
    def test_missing_entry_bar_is_censored_not_zero(self):
        bars = frame([(100, 100, 100, 100)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['status'], CENSORED)

    def test_a_gap_inside_the_holding_window_is_censored(self):
        rows = [(100, 100, 100, 100), (100, 100, 100, 100),
                (np.nan, np.nan, np.nan, np.nan)]
        bars = frame(rows)
        out = label_event(event(N=10), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['status'], CENSORED)
        self.assertEqual(out['event_id'], ('SYNTH', 0, 'S4-v1'))

    def test_truncated_history_before_the_time_exit_is_censored(self):
        rows = [(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.0)] * 5
        bars = frame(rows)
        out = label_event(event(N=10), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['status'], CENSORED)


if __name__ == '__main__':
    unittest.main()
