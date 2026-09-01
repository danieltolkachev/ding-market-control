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


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Letzter vorhandener Handelstag je (Jahr, Monat)."""
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.groupby([index.year, index.month]).max().sort_values().values)


def trend_weight_provider(returns, signals, vols, mode, vol_cap: float = 0.10, vol_window: int = 63):
    """Weight-Provider fuer Trend UND matched_long (Signale konstant +1):
    nutzt ausschliesslich Daten bis einschliesslich decision_date."""
    def provider(decision_date):
        trailing = returns.loc[:decision_date].tail(vol_window)
        return rebalance_weights(
            signals.loc[decision_date], vols.loc[decision_date], trailing, mode, vol_cap,
        )
    return provider


def fixed_mix_provider(target_weights: dict[str, float]):
    """Weight-Provider fuer feste Zielgewichte (SPY-B&H, 60/40)."""
    target = pd.Series(target_weights, dtype=float)
    def provider(decision_date):
        return target.copy()
    return provider


def run_lagged_backtest(
    returns: pd.DataFrame,
    cash_daily: pd.Series,
    decision_dates: pd.DatetimeIndex,
    weight_provider,
    cost_bp: dict[str, float],
    cost_multiplier: float = 1.0,
    borrow_bp_pa: float = 50.0,
) -> tuple[pd.Series, dict]:
    """Gemeinsamer taeglicher Loop fuer Strategie und ALLE Benchmarks
    (identisches Entry-Timing per Konstruktion, Spec v2 §6/§9):

    Tag s: (1) gestrige Gewichte verdienen r_s, Cash `max(0, 1-Gross)`
    verdient cash_daily_s, Borrow auf gestrige Shorts; (2) Drift;
    (3) wenn s Ausfuehrungstag (= Handelstag nach einer Entscheidung):
    Zielgewichte der GESTRIGEN Entscheidung werden gegen die gedrifteten
    Gewichte gehandelt (Kosten heute), wirken ab morgen. Kein Wert, der in
    die Gewichte eingeht, ist zum Fill unbekannt (Future-Poison-Test)."""
    from factor_lab.costs import trade_cost_fraction, daily_borrow_cost_fraction

    symbols = list(returns.columns)
    index = returns.index
    decisions = set(decision_dates)
    first_decision_pos = index.get_loc(decision_dates.min())
    weights = pd.Series(0.0, index=symbols)
    pending_target = None

    rows, out_index = [], []
    contributions = {s: 0.0 for s in symbols}
    total_turnover = 0.0
    n_rebalances = 0
    max_daily_gross = 0.0

    for pos in range(first_decision_pos, len(index)):
        t = index[pos]
        if pos > first_decision_pos:
            r_t = returns.iloc[pos]
            gross_pnl = float((weights * r_t).sum())
            for s in symbols:
                contributions[s] += float(weights[s] * r_t[s])
            cash_weight = max(0.0, 1.0 - float(weights.abs().sum()))
            cash_pnl = cash_weight * float(cash_daily.loc[t])
            borrow = daily_borrow_cost_fraction(weights, cost_multiplier, borrow_bp_pa)

            equity_growth = 1.0 + gross_pnl + cash_pnl - borrow
            if equity_growth > 0:
                weights = weights * (1.0 + r_t) / equity_growth
            max_daily_gross = max(max_daily_gross, float(weights.abs().sum()))

            trade_cost = 0.0
            if pending_target is not None:
                deltas = pending_target - weights
                trade_cost = trade_cost_fraction(deltas, cost_bp, cost_multiplier)
                total_turnover += float(deltas.abs().sum())
                n_rebalances += 1
                weights = pending_target
                pending_target = None

            rows.append((gross_pnl, cash_pnl, trade_cost, borrow))
            out_index.append(t)

        if t in decisions:
            pending_target = weight_provider(t).reindex(symbols, fill_value=0.0)

    per_day = pd.DataFrame(rows, index=pd.DatetimeIndex(out_index),
                           columns=["gross_pnl", "cash_pnl", "trade_cost", "borrow_cost"])
    net = per_day["gross_pnl"] + per_day["cash_pnl"] - per_day["trade_cost"] - per_day["borrow_cost"]
    info = {
        "per_day": per_day,
        "total_turnover": total_turnover,
        "n_rebalances": n_rebalances,
        "instrument_contributions": contributions,
        "max_daily_gross": max_daily_gross,
    }
    return net, info
