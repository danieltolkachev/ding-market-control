"""
test_cross_sectional_signals.py — Querschnitts-Momentum-Signal
(Rang-Bucket relativ zu allen anderen Aktien am selben Tag, NICHT wie
signals.py's momentum_sign zeitreihenbasiert pro Instrument).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.cross_sectional_signals import cross_sectional_momentum_signal


def _prices(n_days: int = 300) -> pd.DataFrame:
    """10 Aktien mit fest verschiedenen Drifts -> eindeutige Rangfolge nach
    12-2-Monats-Return an jedem spaeten Testtag."""
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    drifts = np.linspace(-0.002, 0.002, 10)  # A (schlechtester) ... J (bester)
    cols = [chr(ord("A") + i) for i in range(10)]
    rng = np.random.default_rng(0)
    data = {}
    for i, col in enumerate(cols):
        noise = rng.normal(0.0, 0.001, n_days)
        data[col] = 100.0 * np.cumprod(1.0 + drifts[i] + noise)
    return pd.DataFrame(data, index=idx)


def check_top_bottom_buckets() -> None:
    prices = _prices()
    sig = cross_sectional_momentum_signal(prices, lookback=252, skip=21, top_frac=0.3, bottom_frac=0.3)
    assert sig.shape == prices.shape
    last = sig.iloc[-1]
    # 10 Aktien, top/bottom 30% -> je 3 Aktien; Drift A<B<...<J, also
    # top = H,I,J (hoechste Drift), bottom = A,B,C (niedrigste Drift).
    assert set(last[last == 1.0].index) == {"H", "I", "J"}, f"Top-Bucket falsch: {last.to_dict()}"
    assert set(last[last == -1.0].index) == {"A", "B", "C"}, f"Bottom-Bucket falsch: {last.to_dict()}"
    assert set(last[last == 0.0].index) == {"D", "E", "F", "G"}, f"Mittel-Bucket falsch: {last.to_dict()}"
    print("cross_sectional_momentum_signal Top/Bottom/Mitte-Buckets: OK")


def check_nan_before_warmup() -> None:
    prices = _prices(n_days=100)  # < lookback(252)+skip(21)
    sig = cross_sectional_momentum_signal(prices, lookback=252, skip=21, top_frac=0.3, bottom_frac=0.3)
    assert sig.iloc[-1].isna().all(), "Vor Ablauf von lookback+skip muss das Signal NaN sein (kein Fallback auf 0)"
    print("cross_sectional_momentum_signal NaN vor Warmup-Ende: OK")


def check_missing_symbol_excluded_from_ranking() -> None:
    """Eine Aktie mit NaN-Historie an einem Stichtag darf weder selbst
    einen Bucket bekommen noch die Rangfolge der anderen verschieben."""
    prices = _prices()
    prices_missing = prices.copy()
    prices_missing.loc[prices_missing.index[-60]:, "J"] = np.nan  # J faellt vor dem Stichtag aus
    sig = cross_sectional_momentum_signal(prices_missing, lookback=252, skip=21, top_frac=0.3, bottom_frac=0.3)
    last = sig.iloc[-1]
    assert pd.isna(last["J"]), "Aktie ohne Kurs am Stichtag darf keinen Bucket bekommen"
    # Naechstbeste 3 von den verbleibenden 9 (ohne J): G, H, I
    assert set(last[last == 1.0].index) == {"G", "H", "I"}, f"Top-Bucket nach Ausschluss falsch: {last.to_dict()}"
    print("cross_sectional_momentum_signal schliesst fehlende Aktien aus Ranking aus: OK")


def run_consistency_check() -> None:
    check_top_bottom_buckets()
    check_nan_before_warmup()
    check_missing_symbol_excluded_from_ranking()
    print("\nAlle cross_sectional_signals-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
