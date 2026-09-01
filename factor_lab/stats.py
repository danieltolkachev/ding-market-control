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


def annualized_stats(net: pd.Series, cash_daily: pd.Series | None = None) -> dict:
    """Multiplikative Kennzahlen; Drawdown inkl. Start-Equity 1.0 (der
    Helper in market_control_system ist entsprechend gefixt)."""
    r = net.astype(float)
    n = len(r)
    total = compound_return(r.tolist())
    years = n / TRADING_DAYS_PA
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    vol_pa = float(r.std()) * np.sqrt(TRADING_DAYS_PA) if n > 1 else 0.0
    sharpe = (float(r.mean()) * TRADING_DAYS_PA) / vol_pa if vol_pa > 0 else 0.0
    if cash_daily is not None:
        excess = r - cash_daily.loc[r.index]
        ex_vol = float(excess.std()) * np.sqrt(TRADING_DAYS_PA)
        excess_sharpe = (float(excess.mean()) * TRADING_DAYS_PA) / ex_vol if ex_vol > 0 else 0.0
    else:
        excess_sharpe = float("nan")
    return {
        "cagr": float(cagr),
        "vol_pa": vol_pa,
        "sharpe": float(sharpe),
        "excess_sharpe": float(excess_sharpe),
        "max_drawdown": max_drawdown_from_returns(r.tolist()),
        "n_days": n,
    }


def monthly_excess_sharpe(monthly_excess: pd.Series) -> float:
    """Annualisierte Sharpe Ratio des monatlichen Mehrertrags (Log-Differenz-
    Konvention, siehe monthly_log_returns): mean*12 / (std*sqrt(12)).
    Nullvol-Guard und ddof-Konvention (pandas .std()-Default, ddof=1)
    identisch zu sharpe/excess_sharpe in annualized_stats() -- siehe dort
    fuer die Begruendung des n>1-Schutzes vor dem std()-Aufruf."""
    m = monthly_excess.astype(float)
    n = len(m)
    vol = float(m.std()) * np.sqrt(MONTHS_PA) if n > 1 else 0.0
    return (float(m.mean()) * MONTHS_PA) / vol if vol > 0 else 0.0


def full_year_excess(excess_daily_log: pd.Series) -> dict[int, float]:
    """Compoundierte Jahres-Mehrertraege NUR vollstaendiger Kalenderjahre
    (Handelstage in Januar UND Dezember vorhanden) — unvollstaendige
    Randjahre verzerren das Bestes-Jahr-Gate sonst (Review-Punkt).

    Erwartet TAEGLICHE LOG-Mehrertraege (log1p(net) - log1p(benchmark)) als
    Input -- dieselbe Log-Differenz-Konvention wie ueberall sonst in der
    Screening-Pipeline (monatlicher Bootstrap, LOO, Kostenleiter; siehe
    monthly_log_returns). Pro Jahr wird die Summe der Log-Differenzen
    gebildet und via expm1() in einen compoundierten (einfachen) Jahres-
    Mehrertrag zurueckverwandelt -- NICHT compound_return()/prod(1+r)-1,
    denn die Werte hier sind bereits Log-Differenzen und duerfen nicht wie
    einfache Ertraege behandelt werden (Review-Fund: vorher wurde hier eine
    taegliche EINFACHE Renditedifferenz verwendet, inkonsistent mit dem Rest
    dieser Datei -- Abschnitt 9 der Design-Spec verlangt durchgehend die
    Log-Konvention)."""
    out = {}
    for year, group in excess_daily_log.groupby(excess_daily_log.index.year):
        months = set(group.index.month)
        if 1 in months and 12 in months:
            out[int(year)] = float(np.expm1(group.sum()))
    return out


def evaluate_screening_gates(
    excess_boot: dict,
    dd_base: float,
    dd_stress: float,
    stressed_cagr: float,
    yearly_excess: dict[int, float],
    loo_excess_compounds: dict[str, float],
    dd_cap: float = 0.15,
    gate_c_floor: float = 0.02,
) -> dict:
    gate_a = excess_boot["ann_geom_lower_1s95"] > 0
    gate_b = abs(dd_base) <= dd_cap and abs(dd_stress) <= dd_cap
    gate_c = stressed_cagr >= gate_c_floor
    if yearly_excess:
        best = max(yearly_excess, key=yearly_excess.get)
        rest = [v for k, v in yearly_excess.items() if k != best]
        years_ok = compound_return(rest) > 0 if rest else False
    else:
        years_ok = False
    loo_ok = bool(loo_excess_compounds) and all(v > 0 for v in loo_excess_compounds.values())
    gate_d = years_ok and loo_ok
    return {
        "gate_a_excess_ci": bool(gate_a),
        "gate_b_drawdown": bool(gate_b),
        "gate_c_stressed_floor": bool(gate_c),
        "gate_d_no_single_driver": bool(gate_d),
        "passed_all": bool(gate_a and gate_b and gate_c and gate_d),
    }


def evaluate_holdout_gates(
    excess_boot: dict, dd_base: float, dd_stress: float, stressed_cagr: float,
    dd_cap: float = 0.15, gate_c_floor: float = 0.02,
) -> dict:
    gate_a = excess_boot["ann_geom_lower_1s95"] > 0
    gate_b = abs(dd_base) <= dd_cap and abs(dd_stress) <= dd_cap
    gate_c = stressed_cagr >= gate_c_floor
    return {
        "gate_a_excess_ci": bool(gate_a),
        "gate_b_drawdown": bool(gate_b),
        "gate_c_stressed_floor": bool(gate_c),
        "passed_all": bool(gate_a and gate_b and gate_c),
    }


def screening_verdict(gates_by_variant: dict[str, dict], excess_sharpe_by_variant: dict[str, float]) -> tuple[str, str | None]:
    """Screening-Verdikt + versiegelbare Kandidatin. Tie-Break bei gleichem
    Excess-Sharpe: alphabetisch erster Variantenname (praeregistriert)."""
    passed = sorted(n for n, g in gates_by_variant.items() if g["passed_all"])
    if not passed:
        return (
            "VERDICT: keine Variante besteht alle Screening-Gates -- das Holdout wird "
            "NICHT angefasst; Ergebnis als Nullbefund der Familie dokumentieren. "
            "(Screening ist ausdruecklich KEINE Bestaetigung.)",
            None,
        )
    best_sharpe = max(excess_sharpe_by_variant[n] for n in passed)
    candidate = sorted(n for n in passed if excess_sharpe_by_variant[n] == best_sharpe)[0]
    return (
        f"VERDICT: {len(passed)} Variante(n) bestehen das Screening ({', '.join(passed)}). "
        f"Versiegelte Holdout-Kandidatin: {candidate}. Screening ist KEINE Bestaetigung -- "
        f"bestaetigen kann nur der eine Holdout-Lauf (run_trend_holdout.py, liest candidate.json).",
        candidate,
    )
