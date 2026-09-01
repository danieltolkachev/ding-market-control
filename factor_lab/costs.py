"""
costs.py — Kostenmodell (Spec v2 Abschnitt 8). Basiskosten sind bewusst
als UNKALIBRIERTE konservative Schaetzung dokumentiert (Stand 2026-09-01,
keine Quoted-Spread-Studie) — deshalb laeuft die Kostenleiter 1x/2x/5x als
fester Bestandteil jedes Laufs. One-way-Turnover = Sum |Delta w|; ein Flip
+1 -> -1 ist Turnover 2.
"""
from __future__ import annotations

import pandas as pd

COST_BP: dict[str, float] = {
    "SPY": 1.5, "QQQ": 1.5, "IWM": 1.5, "TLT": 1.5, "IEF": 1.5, "GLD": 1.5,
    "EFA": 3.0, "EEM": 3.0, "LQD": 3.0, "SLV": 3.0, "DBC": 3.0, "VNQ": 3.0,
    # trend-etf-v2 (rein additiv, Spec 2026-09-01-daily-factor-lab-trend-v2-universe-design.md Abschnitt 4):
    "UUP": 3.0, "FXE": 3.0, "FXY": 3.0, "USO": 3.0, "UNG": 3.0, "DBA": 3.0, "EMB": 3.0,
}
BORROW_BP_PA: float = 50.0
TRADING_DAYS_PA: int = 252
COST_LADDER: tuple = (1.0, 2.0, 5.0)


def trade_cost_fraction(weight_deltas: pd.Series, cost_bp: dict[str, float], cost_multiplier: float = 1.0) -> float:
    """Rebalance-Kosten als Anteil am Portfoliowert."""
    total = 0.0
    for symbol, delta in weight_deltas.items():
        total += abs(float(delta)) * cost_bp[symbol] / 10_000.0
    return total * cost_multiplier


def daily_borrow_cost_fraction(weights: pd.Series, cost_multiplier: float = 1.0, borrow_bp_pa: float = BORROW_BP_PA) -> float:
    """Taegliche Borrow-Kosten auf das Short-Nominal."""
    short_nominal = float(weights[weights < 0].abs().sum())
    return short_nominal * (borrow_bp_pa / 10_000.0) / TRADING_DAYS_PA * cost_multiplier
