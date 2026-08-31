"""
test_cross_sectional_signal_metrics.py
=========================================

Prueft die reinen Statistikfunktionen in controller/cross_sectional_
signal_metrics.py mit synthetischen Werten (kein Modell/keine echten
Daten noetig).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))

import numpy as np

from cross_sectional_signal_metrics import (
    compute_rank_ic,
    compute_gross_spread,
    compute_breakeven_cost,
    compound_return,
    equity_curve,
    max_drawdown_from_returns,
    rolling_percentile_score,
    random_ranking_scores,
    momentum_scores,
    reversal_scores,
)


def check_rank_ic() -> None:
    scores = {"A": 1.0, "B": 2.0, "C": 3.0}
    perfect_positive = {"A": 0.01, "B": 0.02, "C": 0.03}
    ic = compute_rank_ic(scores, perfect_positive)
    assert abs(ic - 1.0) < 1e-9, f"Erwartete Rank-IC=1.0 bei perfekter positiver Korrelation, bekam {ic}"

    perfect_negative = {"A": 0.03, "B": 0.02, "C": 0.01}
    ic = compute_rank_ic(scores, perfect_negative)
    assert abs(ic - (-1.0)) < 1e-9, f"Erwartete Rank-IC=-1.0 bei perfekter negativer Korrelation, bekam {ic}"

    too_few = compute_rank_ic({"A": 1.0, "B": 2.0}, {"A": 0.1, "B": 0.2})
    assert np.isnan(too_few), "Bei < 3 gemeinsamen Symbolen sollte NaN zurueckkommen"

    print("compute_rank_ic: OK")


def check_gross_spread() -> None:
    scores = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
    forward_returns = {"A": 0.02, "B": 0.01, "C": -0.01, "D": -0.02}
    spread = compute_gross_spread(scores, forward_returns, n_long=1, n_short=1)
    # Long A (+0.02), Short D (-0.02) -> Spread = 0.02 - (-0.02) = 0.04
    assert abs(spread - 0.04) < 1e-9, f"Erwarteter Spread 0.04, bekam {spread}"
    print("compute_gross_spread: OK")


def check_breakeven_cost() -> None:
    gross = [0.01, 0.02, 0.0]
    turnover = [0.5, 0.5, 0.5]
    breakeven = compute_breakeven_cost(gross, turnover)
    expected = (sum(gross) / len(gross)) / (sum(turnover) / len(turnover))
    assert abs(breakeven - expected) < 1e-9
    print("compute_breakeven_cost: OK")


def check_compounding() -> None:
    returns = [0.1, 0.1]
    compounded = compound_return(returns)
    simple_sum = sum(returns)
    assert abs(compounded - 0.21) < 1e-9, f"Erwartetes Compound-Ergebnis 0.21, bekam {compounded}"
    assert abs(compounded - simple_sum) > 1e-6, (
        "Compounding und einfache Summe muessen bei diesen Werten unterschiedlich sein -- "
        "sonst wird faelschlich additiv gerechnet"
    )
    print("compound_return: OK (0.21, weicht bewusst von additiver Summe 0.20 ab)")


def check_drawdown() -> None:
    returns = [0.1, -0.2, 0.05]
    dd = max_drawdown_from_returns(returns)
    # equity: 1.1 -> 0.88 -> 0.924; running_max bleibt bei 1.1; Tiefpunkt 0.88/1.1-1=-0.2
    assert abs(dd - (-0.2)) < 1e-9, f"Erwarteter Max-Drawdown -0.2, bekam {dd}"
    print("max_drawdown_from_returns: OK")


def check_rolling_percentile() -> None:
    history = [1.0, 2.0, 3.0, 4.0, 100.0]  # letzter Wert ist der hoechste je gesehene
    pct = rolling_percentile_score(history, window=10)
    assert abs(pct - 1.0) < 1e-9, f"Hoechster je gesehener Wert sollte Perzentil 1.0 ergeben, bekam {pct}"

    too_short = rolling_percentile_score([1.0], window=10)
    assert too_short == 0.5, "Bei zu wenig Historie soll ein neutraler Default (0.5) zurueckkommen"
    print("rolling_percentile_score: OK")


def check_baselines() -> None:
    symbols = ["A", "B", "C"]
    scores_a = random_ranking_scores(symbols, seed=0)
    scores_b = random_ranking_scores(symbols, seed=0)
    scores_c = random_ranking_scores(symbols, seed=1)
    assert scores_a == scores_b, "Gleicher Seed muss identische Scores liefern"
    assert scores_a != scores_c, "Unterschiedlicher Seed sollte (fast sicher) unterschiedliche Scores liefern"
    assert set(scores_a) == set(symbols)

    prices = {
        "A": [100.0, 101.0, 102.0, 110.0],
        "B": [100.0, 99.0, 98.0, 90.0],
    }
    mom = momentum_scores(prices, lookback_bars=3)
    assert mom["A"] > 0, "A ist gestiegen -- Momentum-Score muss positiv sein"
    assert mom["B"] < 0, "B ist gefallen -- Momentum-Score muss negativ sein"

    rev = reversal_scores(prices, lookback_bars=3)
    assert rev["A"] < 0 and rev["B"] > 0, "Reversal muss exakt das Vorzeichen von Momentum umkehren"
    print("random_ranking_scores/momentum_scores/reversal_scores: OK")


def run_consistency_check() -> None:
    check_rank_ic()
    check_gross_spread()
    check_breakeven_cost()
    check_compounding()
    check_drawdown()
    check_rolling_percentile()
    check_baselines()
    print("\nAlle cross_sectional_signal_metrics-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
