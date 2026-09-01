"""
cross_sectional_signals.py — Querschnitts-Momentum-Signal fuer das
Momentum-Leg-Mechanik-Testskript (KEIN Promotion-Kandidat, siehe
run_cross_sectional_momentum.py). Rang relativ zu allen anderen Aktien
AM SELBEN TAG (Jegadeesh & Titman 1993: 12-2-Monats-Formation) --
bewusst getrennt von signals.py's zeitreihenbasiertem momentum_sign
(Signal pro Instrument ueber die eigene Vergangenheit).
"""
from __future__ import annotations

import pandas as pd


def cross_sectional_momentum_signal(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21,
    top_frac: float = 0.3, bottom_frac: float = 0.3,
) -> pd.DataFrame:
    """+1 fuer die top_frac-Aktien mit dem hoechsten (skip..lookback+skip)-
    Trailing-Return am jeweiligen Tag, -1 fuer die bottom_frac niedrigsten,
    0 fuer den Rest. NaN, wo die Aktie selbst noch keinen gueltigen
    Trailing-Return hat (Warmup ODER fehlender Kurs an diesem Tag) --
    diese Aktien werden auch aus der Rangfolge der anderen ausgeschlossen
    (row-weise NaN-Skip beim Ranking)."""
    trailing_return = prices.shift(skip) / prices.shift(lookback + skip) - 1.0
    pct_rank = trailing_return.rank(axis=1, pct=True, na_option="keep")
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[pct_rank > (1.0 - top_frac)] = 1.0
    signal[pct_rank <= bottom_frac] = -1.0
    signal[trailing_return.isna()] = float("nan")
    return signal
