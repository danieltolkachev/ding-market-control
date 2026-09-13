"""Kausale Persistenz auf monatlichen Entscheidungswerten."""
import numpy as np
import pandas as pd
from factor_lab.portfolio import month_end_dates


def apply_signal_persistence(signal_frame: pd.DataFrame, min_confirm: int = 2) -> pd.DataFrame:
    """Bestaetigung zaehlt Monate, nicht taegliche Beobachtungen."""
    if isinstance(min_confirm, bool) or not isinstance(min_confirm, int) or min_confirm < 1:
        raise ValueError('min_confirm muss eine positive Ganzzahl sein')
    idx = signal_frame.index
    if not isinstance(idx, pd.DatetimeIndex) or not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError('Sortierter eindeutiger DatetimeIndex erforderlich')
    months = idx.to_period('M')
    if months.has_duplicates or signal_frame.columns.has_duplicates:
        raise ValueError('Nur ein Wert je Monat und eindeutige Instrumente')
    if np.isinf(signal_frame.to_numpy(dtype=float)).any():
        raise ValueError('Unendliche Signale')
    result = signal_frame.astype(float).copy()
    for col in signal_frame:
        effective = np.nan
        pending = None
        streak = 0
        previous_month = None
        for i, raw in enumerate(signal_frame[col].to_numpy(dtype=float)):
            ordinal = months[i].ordinal
            if previous_month is not None and ordinal != previous_month + 1:
                pending, streak = None, 0
            previous_month = ordinal
            if np.isnan(raw):
                effective, pending, streak = np.nan, None, 0
            elif np.isnan(effective) or np.sign(raw) == np.sign(effective):
                effective, pending, streak = raw, None, 0
            else:
                direction = np.sign(raw)
                streak = streak + 1 if pending == direction else 1
                pending = direction
                if streak >= min_confirm:
                    effective, pending, streak = raw, None, 0
            result.iat[i, result.columns.get_loc(col)] = effective
    return result


def daily_persistent_signals(raw: pd.DataFrame, min_confirm: int = 2) -> pd.DataFrame:
    """Monatsultimo sampeln; letzten Teilmonat konservativ auslassen.

    Historischer gemeinsamer Handelskalender wird vorausgesetzt. Bei einem
    Schlussmonat vor dem letzten Werktag wird keine Entscheidung erfunden.
    """
    if not isinstance(raw.index, pd.DatetimeIndex) or not raw.index.is_monotonic_increasing or raw.index.has_duplicates:
        raise ValueError('Sortierter eindeutiger Tagesindex erforderlich')
    if raw.empty:
        return raw.astype(float).copy()
    ends = month_end_dates(raw.index)
    last_business_day = raw.index[-1] + pd.offsets.BMonthEnd(0)
    if raw.index[-1].normalize() < last_business_day.normalize():
        ends = ends[ends.to_period('M') != raw.index[-1].to_period('M')]
    monthly = apply_signal_persistence(raw.loc[ends], min_confirm)
    return monthly.reindex(raw.index, method='ffill')
