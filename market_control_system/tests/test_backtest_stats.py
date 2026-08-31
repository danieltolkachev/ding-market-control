"""
test_backtest_stats.py
=========================

Prueft, dass backtest_stats einfache Returns MULTIPLIKATIV aggregiert
(prod(1+r)-1) statt additiv (sum/cumsum) -- der additive Pfad war Review-
Korrektur 3 (2026-08-31): die frueher berichteten -11%/-67%/-70% waren
dadurch keine korrekten Equity-Returns, und auch der Drawdown war auf
einer additiven statt einer echten Equity-Kurve berechnet.

Der Testfall [+100%, -20%, -20%] trennt die beiden Welten messbar:
additiv ergaebe +60% kumuliert und -40% Drawdown, multiplikativ korrekt
+28% und -36%.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import numpy as np
import pandas as pd

from backtest_stats import compute_period_statistics, summarize_period_returns


def check_period_aggregation_compounds() -> None:
    # Woche 1: zwei Schritte je +10% -> Wochenreturn 1.1*1.1-1 = +21%
    # (additiv waeren es +20%).
    timestamps = pd.DatetimeIndex([
        "2026-01-05 10:00", "2026-01-05 10:01",   # Montag Woche 1
        "2026-01-12 10:00", "2026-01-12 10:01",   # Montag Woche 2
    ])
    step_returns = np.array([0.1, 0.1, -0.2, -0.2])
    stats = compute_period_statistics(timestamps, step_returns, period="W")
    assert stats.n_periods == 2
    assert abs(stats.period_returns[0] - 0.21) < 1e-12, (
        f"Wochenreturn muss compounden (0.21), bekam {stats.period_returns[0]}"
    )
    assert abs(stats.period_returns[1] - (-0.36)) < 1e-12, (
        f"Wochenreturn muss compounden (-0.36), bekam {stats.period_returns[1]}"
    )
    print("compute_period_statistics compoundiert innerhalb der Periode: OK")


def check_summary_compounds() -> None:
    period_returns = pd.Series([1.0, -0.2, -0.2])  # +100%, -20%, -20%
    stats = summarize_period_returns(period_returns)

    # Kumulativ: 2.0 * 0.8 * 0.8 - 1 = +28% (additiv waere +60%).
    assert abs(stats.cumulative_return - 0.28) < 1e-12, (
        f"Kumulierter Return muss compounden (+0.28), bekam {stats.cumulative_return}"
    )

    # Drawdown auf der ECHTEN Equity-Kurve (1 -> 2.0 -> 1.6 -> 1.28):
    # 1.28/2.0 - 1 = -36% (additive cumsum-Kurve ergaebe -40%).
    assert abs(stats.max_drawdown - (-0.36)) < 1e-12, (
        f"Drawdown muss auf compoundender Equity-Kurve laufen (-0.36), bekam {stats.max_drawdown}"
    )

    # Perioden-Momente (mean/std/t/win_rate) bleiben Statistiken UEBER die
    # Perioden-Returns selbst -- unveraendert additiv gemittelt.
    assert abs(stats.mean_return - np.mean([1.0, -0.2, -0.2])) < 1e-12
    assert abs(stats.win_rate - (1.0 / 3.0)) < 1e-12
    print("summarize_period_returns compoundiert Kumulativ + Drawdown: OK")


def run_consistency_check() -> None:
    check_period_aggregation_compounds()
    check_summary_compounds()
    print("\nAlle backtest_stats-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
