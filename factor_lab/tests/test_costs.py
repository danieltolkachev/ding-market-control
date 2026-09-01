"""
test_costs.py — prueft bp-Kostenmodell, Flip-Turnover und Borrow.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from factor_lab.costs import COST_BP, COST_LADDER, trade_cost_fraction, daily_borrow_cost_fraction


def check_trade_cost_and_flip() -> None:
    deltas = pd.Series({"SPY": 0.5, "EEM": -0.2})
    # 0.5*1.5bp + 0.2*3.0bp = 0.000075 + 0.00006
    assert abs(trade_cost_fraction(deltas, COST_BP) - 0.000135) < 1e-12
    assert abs(trade_cost_fraction(deltas, COST_BP, cost_multiplier=2.0) - 0.00027) < 1e-12

    # Pflichttest aus dem Review: Flip +1 -> -1 ist One-way-Turnover 2
    # und kostet entsprechend 2 * 1.5bp.
    flip = pd.Series({"SPY": -1.0 - 1.0})
    assert abs(flip.abs().sum() - 2.0) < 1e-12
    assert abs(trade_cost_fraction(flip, COST_BP) - 2.0 * 1.5 / 10_000.0) < 1e-12
    print("trade_cost_fraction (inkl. Flip=2): OK")


def check_borrow() -> None:
    weights = pd.Series({"TLT": -0.5, "SPY": 0.5})
    expected = 0.5 * 0.005 / 252
    assert abs(daily_borrow_cost_fraction(weights) - expected) < 1e-15
    assert daily_borrow_cost_fraction(pd.Series({"SPY": 1.0})) == 0.0
    # Borrow-Sensitivitaet ueber den Parameter, Stress ueber den Multiplier
    assert abs(daily_borrow_cost_fraction(weights, borrow_bp_pa=100.0) - 2 * expected) < 1e-15
    assert abs(daily_borrow_cost_fraction(weights, cost_multiplier=2.0) - 2 * expected) < 1e-15
    print("daily_borrow_cost_fraction: OK")


def check_tables() -> None:
    v1_symbols = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"]
    v2_new_symbols = ["UUP", "FXE", "FXY", "USO", "UNG", "DBA", "EMB"]
    assert sorted(COST_BP) == sorted(v1_symbols + v2_new_symbols), (
        f"Erwartete 19 Symbole (12 v1 + 7 neue v2), bekam {sorted(COST_BP)}"
    )
    # v1-Werte duerfen sich NICHT aendern (reine Erweiterung, Regressionsschutz).
    assert COST_BP["SPY"] == 1.5 and COST_BP["EEM"] == 3.0
    # Alle 7 neuen v2-Symbole bei 3.0bp (Spec trend-etf-v2 Abschnitt 4).
    for s in v2_new_symbols:
        assert COST_BP[s] == 3.0, f"{s}: erwartete 3.0bp, bekam {COST_BP[s]}"
    assert COST_LADDER == (1.0, 2.0, 5.0)
    print("COST_BP / COST_LADDER (19 Symbole inkl. trend-etf-v2): OK")


def run_consistency_check() -> None:
    check_tables()
    check_trade_cost_and_flip()
    check_borrow()
    print("\nAlle costs-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
