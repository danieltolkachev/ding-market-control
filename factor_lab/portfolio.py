"""
portfolio.py — Sizing, Lag-Konvention und taeglicher Loop (Spec v2
Abschnitte 6-7). Vol-Cap 0.10 ist eine OBERGRENZE: Gross-Cap 1.0 fuer
Zielgewichte, es wird nie gehebelt; zwischen Rebalances driften die
Gewichte (tageweises Max-Gross wird in Task 4 mitgemessen und reportet,
NICHT zwangsdeleveraged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PA = 252


def ewma_annualized_vol(returns: pd.DataFrame, span: int = 63) -> pd.DataFrame:
    """Kausale EWMA-Tagesvol (min_periods=span), annualisiert."""
    return returns.ewm(span=span, min_periods=span).std() * np.sqrt(TRADING_DAYS_PA)


def rebalance_weights(
    signal_row: pd.Series,
    vol_row: pd.Series,
    trailing_returns: pd.DataFrame,
    mode: str,
    vol_cap: float = 0.10,
) -> pd.Series:
    """Zielgewichte fuer EINEN Entscheidungszeitpunkt (Daten bis t)."""
    if mode not in ("long_short", "long_flat"):
        raise ValueError(f"Unbekannter Modus: {mode}")
    signal = signal_row.astype(float).copy()
    if mode == "long_flat":
        signal = signal.clip(lower=0.0)

    valid = vol_row.notna() & (vol_row > 0)
    raw = (signal / vol_row).where(valid, 0.0).fillna(0.0)
    gross = float(raw.abs().sum())
    if gross == 0.0:
        return pd.Series(0.0, index=signal_row.index)
    base = raw / gross  # Zielgewichte: Gross exakt 1.0

    # EINE praeregistrierte Formel (Spec v2 §6): einfache Std der mit den
    # Kandidatengewichten gewichteten letzten 63 Tagesreturns, annualisiert.
    portfolio_returns = (trailing_returns[base.index] * base).sum(axis=1)
    realized_vol = float(portfolio_returns.std()) * np.sqrt(TRADING_DAYS_PA)
    scale = min(1.0, vol_cap / realized_vol) if realized_vol > 0 else 1.0
    return base * scale
