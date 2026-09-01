"""
stats.py — Monatsaggregation, Stationary-Block-Bootstrap (Politis/Romano),
Sign-Flip-Permutation, Kennzahlen und Gates (Spec v2 Abschnitte 9-10).

Warum Stationary-Block statt unabhaengiger Monats-Bloecke: 63-252-Tage-
Signale erzeugen Persistenz UEBER Monatsgrenzen; unabhaengiges Resampling
einzelner Monate zerstoert genau diese Struktur. Erwartete Blocklaenge 6
Monate ist praeregistriert primaer (3/12 nur Sensitivitaet).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market_control_system", "controller"))

import numpy as np
import pandas as pd

from cross_sectional_signal_metrics import compound_return, max_drawdown_from_returns

TRADING_DAYS_PA = 252
MONTHS_PA = 12


def monthly_log_returns(net: pd.Series) -> pd.Series:
    """Summe log(1+r) je Kalendermonat; Index = Monatsanfang."""
    log_r = np.log1p(net.astype(float))
    grouped = log_r.groupby(net.index.to_period("M")).sum()
    grouped.index = grouped.index.to_timestamp()
    return grouped


def stationary_bootstrap_indices(n: int, expected_block_len: float, rng: np.random.Generator) -> np.ndarray:
    """Politis/Romano: nach jedem Schritt mit Wahrscheinlichkeit 1/L neu
    starten (uniformer Startindex), sonst zum Nachfolger (Wrap-around)."""
    p_restart = 1.0 / expected_block_len
    indices = np.empty(n, dtype=int)
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p_restart:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def _annualize_monthly_log(mean_monthly_log: float) -> float:
    return float(np.exp(MONTHS_PA * mean_monthly_log) - 1.0)


def stationary_block_bootstrap(
    monthly_values: np.ndarray, expected_block_len: float = 6.0, n_boot: int = 10_000, seed: int = 0,
) -> dict:
    """Bootstrap-Verteilung des annualisierten geometrischen Ertrags aus
    monatlichen Log-Ertraegen. Fester Seed -> identische Ziehungsfolge fuer
    alle Varianten (Spec v2 §9)."""
    values = np.asarray(monthly_values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    ann = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, expected_block_len, rng)
        ann[b] = _annualize_monthly_log(float(values[idx].mean()))
    return {
        "mean_monthly": float(values.mean()),
        "ann_geom": _annualize_monthly_log(float(values.mean())),
        "ann_geom_lower_1s95": float(np.percentile(ann, 5.0)),
        "ci_low_95": float(np.percentile(ann, 2.5)),
        "ci_high_95": float(np.percentile(ann, 97.5)),
        "p_leq_zero": float(((ann <= 0).sum() + 1) / (n_boot + 1)),
        "n_months": n,
        "n_boot": n_boot,
        "expected_block_len": expected_block_len,
    }


def monthly_sign_flip_pvalue(monthly_values: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> dict:
    """Sign-Flip-Permutation auf Monatsebene, p = (extreme+1)/(B+1)."""
    values = np.asarray(monthly_values, dtype=float)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(values)))
    perm_means = (signs * values).mean(axis=1)
    return {
        "p_two_sided": float(((np.abs(perm_means) >= abs(observed)).sum() + 1) / (n_perm + 1)),
        "p_greater_zero": float(((perm_means >= observed).sum() + 1) / (n_perm + 1)),
        "n_months": len(values),
        "n_perm": n_perm,
    }
