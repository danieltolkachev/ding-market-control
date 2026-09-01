"""
test_portfolio.py — Vol-Schaetzung, Zielgewichte und (ab Task 4) der
taegliche Loop mit Lag-Konvention.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.portfolio import ewma_annualized_vol, rebalance_weights


def check_ewma_vol() -> None:
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    rets = pd.DataFrame({"A": [0.01, -0.01] * 40}, index=idx)
    vol = ewma_annualized_vol(rets, span=63)
    assert vol["A"].iloc[:62].isna().all(), "Vor min_periods muss NaN stehen"
    assert abs(vol["A"].iloc[-1] - 0.01 * np.sqrt(252)) < 0.02
    print("ewma_annualized_vol: OK")


def check_rebalance_weights() -> None:
    signal = pd.Series({"A": 1.0, "B": -1.0})
    vol = pd.Series({"A": 0.2, "B": 0.1})
    trailing = pd.DataFrame(0.0, index=range(63), columns=["A", "B"])

    # long_short: raw = [5, -10] -> base [1/3, -2/3], Gross exakt 1
    w = rebalance_weights(signal, vol, trailing, mode="long_short", vol_cap=0.10)
    assert abs(w["A"] - 1.0 / 3.0) < 1e-12 and abs(w["B"] + 2.0 / 3.0) < 1e-12
    assert abs(w.abs().sum() - 1.0) < 1e-12

    # long_flat: negatives Signal -> 0
    w = rebalance_weights(signal, vol, trailing, mode="long_flat", vol_cap=0.10)
    assert abs(w["A"] - 1.0) < 1e-12 and w["B"] == 0.0

    # Vol-Cap skaliert nur HERUNTER
    rng = np.random.default_rng(0)
    hot = pd.DataFrame({"A": rng.normal(0, 0.02, 63), "B": 0.0})
    w = rebalance_weights(pd.Series({"A": 1.0, "B": 0.0}), vol, hot, mode="long_flat", vol_cap=0.10)
    assert 0.0 < w["A"] < 1.0

    # NaN-Vol -> Gewicht 0; Alles-Null-Signal -> Cash
    w = rebalance_weights(pd.Series({"A": 1.0, "B": 1.0}), pd.Series({"A": 0.2, "B": np.nan}), trailing, mode="long_flat")
    assert w["B"] == 0.0 and abs(w["A"] - 1.0) < 1e-12
    w = rebalance_weights(pd.Series({"A": 0.0, "B": 0.0}), vol, trailing, mode="long_flat")
    assert (w == 0.0).all()
    print("rebalance_weights: OK")


def run_consistency_check() -> None:
    check_ewma_vol()
    check_rebalance_weights()
    print("\nAlle portfolio-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
