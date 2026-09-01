"""
signals.py — praeregistrierte Trend-Signale (Spec v2 Abschnitt 5).
Reine Funktionen; BEWUSST kein Mechanismus fuer weitere Varianten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_sign(prices: pd.Series, lookback: int) -> pd.Series:
    """Vorzeichen des Returns ueber `lookback` Handelstage; NaN fuer die
    ersten `lookback` Eintraege; exakter Null-Return -> 0."""
    return np.sign(prices / prices.shift(lookback) - 1.0)


def combo_signal(signals: list[pd.Series]) -> pd.Series:
    """Gleichgewichts-Mittel; NaN, wo irgendein Input NaN ist."""
    frame = pd.concat(signals, axis=1)
    return frame.mean(axis=1).where(frame.notna().all(axis=1))
