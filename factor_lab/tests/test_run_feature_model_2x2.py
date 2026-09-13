"""Die vorab festgelegte Entscheidungsregel (compute_verdict), losgeloest vom Rest der Pipeline.

Keine Daten, kein Snapshot, kein Modellfit - nur synthetische primary-Dicts,
so wie main() sie aus stationary_block_bootstrap zusammensetzen wuerde.
"""
import unittest

from factor_lab.evaluate_2x2 import CELLS
from factor_lab.run_feature_model_2x2 import PRIMARY_COST, compute_verdict


def interval(low, high):
    return {PRIMARY_COST: {'ci_low_95': low, 'ci_high_95': high}}


def primary_for_all_cells(interval_by_name, default=(-0.01, 0.01)):
    """Baut ein primary-Dict fuer alle vier Zellen; Abweichler ueber interval_by_name."""
    names = [name for name, _, _ in CELLS]
    out = {}
    for name in names:
        low, high = interval_by_name.get(name, default)
        out[name] = interval(low, high)
    return out


class ComputeVerdictTests(unittest.TestCase):
    def test_all_intervals_span_zero_gives_null_result(self):
        primary = primary_for_all_cells({})
        verdict = compute_verdict(primary)
        self.assertEqual(verdict['cells_with_interval_excluding_zero'], [])
        self.assertEqual(verdict['reading'], 'null result')

    def test_one_cell_positive_interval_gives_hint_not_evidence(self):
        primary = primary_for_all_cells({'B_base_gbm': (0.01, 0.05)})
        verdict = compute_verdict(primary)
        self.assertEqual(verdict['cells_with_interval_excluding_zero'], ['B_base_gbm'])
        self.assertEqual(verdict['reading'], 'hint, not evidence')

    def test_one_cell_negative_interval_also_counts_as_excluding_zero(self):
        # Ein Intervall ganz unter Null schliesst die Null genauso aus wie eines
        # ganz darueber - dieser Zweig wird leicht falsch gemacht (nur > 0 geprueft)
        # und wuerde dann ein Ergebnis stillschweigend abschwaechen.
        primary = primary_for_all_cells({'C_ext_ridge': (-0.08, -0.02)})
        verdict = compute_verdict(primary)
        self.assertEqual(verdict['cells_with_interval_excluding_zero'], ['C_ext_ridge'])
        self.assertEqual(verdict['reading'], 'hint, not evidence')

    def test_two_cells_excluding_zero_gives_axis_consistency_check(self):
        primary = primary_for_all_cells({
            'A_base_ridge': (0.02, 0.06),
            'D_ext_gbm': (-0.07, -0.01),
        })
        verdict = compute_verdict(primary)
        self.assertEqual(verdict['cells_with_interval_excluding_zero'],
                         ['A_base_ridge', 'D_ext_gbm'])
        self.assertEqual(verdict['reading'], 'check axis consistency: C-A vs D-B')

    def test_order_follows_cells_not_insertion_order(self):
        # Reihenfolge der Ausschliesser muss der CELLS-Reihenfolge folgen, nicht der
        # Reihenfolge, in der die Zellen ins primary-Dict eingefuegt wurden.
        primary = primary_for_all_cells({
            'D_ext_gbm': (0.03, 0.09),
            'A_base_ridge': (0.01, 0.04),
        })
        verdict = compute_verdict(primary)
        self.assertEqual(verdict['cells_with_interval_excluding_zero'],
                         ['A_base_ridge', 'D_ext_gbm'])


if __name__ == '__main__':
    unittest.main()
