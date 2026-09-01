"""
test_signals.py — prueft die reinen Signalfunktionen mit handgerechneten Werten.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.signals import momentum_sign, combo_signal


def check_momentum_sign() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    prices = pd.Series([100.0, 101.0, 99.0, 99.0, 102.0, 98.0], index=idx)
    sig = momentum_sign(prices, lookback=2)
    assert sig.iloc[:2].isna().all(), f"Erste lookback Eintraege muessen NaN sein, bekam {sig.iloc[:2].tolist()}"
    # t=2: 99/100-1 < 0 -> -1 ; t=3: 99/101-1 < 0 -> -1 ; t=4: 102/99-1 > 0 -> +1
    assert sig.iloc[2] == -1.0 and sig.iloc[3] == -1.0 and sig.iloc[4] == 1.0
    flat = pd.Series([100.0, 100.0, 100.0], index=pd.date_range("2020-01-01", periods=3, freq="B"))
    assert momentum_sign(flat, lookback=1).iloc[2] == 0.0, "Exakter Null-Return muss Signal 0 geben"
    print("momentum_sign: OK")


def check_combo_signal() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    s1 = pd.Series([1.0, 1.0, -1.0], index=idx)
    s2 = pd.Series([1.0, -1.0, -1.0], index=idx)
    s3 = pd.Series([np.nan, 1.0, -1.0], index=idx)
    combo = combo_signal([s1, s2, s3])
    assert np.isnan(combo.iloc[0]), "NaN in irgendeinem Input muss NaN im Combo ergeben"
    assert abs(combo.iloc[1] - (1.0 / 3.0)) < 1e-12
    assert combo.iloc[2] == -1.0
    print("combo_signal: OK")


def run_consistency_check() -> None:
    check_momentum_sign()
    check_combo_signal()
    print("\nAlle signals-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
