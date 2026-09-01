"""
test_stats.py — Monatsaggregation, Stationary-Block-Bootstrap, Permutation
(ab Task 6 auch Kennzahlen/Gates).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.stats import (
    monthly_log_returns, stationary_bootstrap_indices, stationary_block_bootstrap,
    monthly_sign_flip_pvalue,
)
from factor_lab.stats import (
    annualized_stats, full_year_excess, monthly_excess_sharpe, evaluate_screening_gates,
    evaluate_holdout_gates, screening_verdict,
)
from cross_sectional_signal_metrics import compound_return


def check_monthly_log_returns() -> None:
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-02-03"])
    net = pd.Series([0.01, 0.01, -0.02], index=idx)
    monthly = monthly_log_returns(net)
    assert len(monthly) == 2
    assert abs(monthly.iloc[0] - 2 * np.log(1.01)) < 1e-12
    assert abs(monthly.iloc[1] - np.log(0.98)) < 1e-12
    print("monthly_log_returns: OK")


def check_stationary_indices() -> None:
    rng = np.random.default_rng(0)
    # Riesige erwartete Blocklaenge -> Indizes praktisch durchgehend
    # zusammenhaengend (mod n). Genau das unterscheidet den Stationary-
    # Bootstrap vom unabhaengigen Monats-Resampling aus Plan v1.
    idx = stationary_bootstrap_indices(n=10, expected_block_len=1000.0, rng=rng)
    diffs = (np.diff(idx) - 1) % 10
    assert (diffs == 0).sum() >= 8, f"Erwartete fast nur zusammenhaengende Schritte, bekam {idx}"

    # Blocklaenge 1 -> im Mittel viele Neustarts (nicht fast alle Schritte +1)
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_indices(n=200, expected_block_len=1.0, rng=rng)
    contiguous = ((np.diff(idx) - 1) % 200 == 0).mean()
    assert contiguous < 0.5, f"Blocklaenge 1 darf nicht ueberwiegend zusammenhaengen ({contiguous})"
    assert len(idx) == 200 and idx.min() >= 0 and idx.max() < 200
    print("stationary_bootstrap_indices: OK")


def check_stationary_bootstrap() -> None:
    rng = np.random.default_rng(2)
    strong = rng.normal(0.01, 0.002, 120)  # 10 Jahre stark positiver Monats-Log-Ertrag
    result = stationary_block_bootstrap(strong, expected_block_len=6.0, n_boot=500, seed=0)
    assert result["ann_geom_lower_1s95"] > 0, "Einseitige Untergrenze muss bei starkem Signal > 0 sein"
    assert result["p_leq_zero"] <= (0 + 1) / (500 + 1) + 1e-12, "p muss (extreme+1)/(B+1) sein, minimal 1/(B+1)"
    assert abs(result["ann_geom"] - (np.exp(12 * strong.mean()) - 1)) < 1e-9
    again = stationary_block_bootstrap(strong, expected_block_len=6.0, n_boot=500, seed=0)
    assert result["ann_geom_lower_1s95"] == again["ann_geom_lower_1s95"], "Gleicher Seed -> identisch"

    noise = rng.normal(0.0, 0.02, 120)
    weak = stationary_block_bootstrap(noise, expected_block_len=6.0, n_boot=500, seed=0)
    assert weak["ci_low_95"] < 0 < weak["ci_high_95"], "Rauschen: CI muss 0 einschliessen"
    print("stationary_block_bootstrap: OK")


def check_sign_flip() -> None:
    strong = np.full(60, 0.01)
    p = monthly_sign_flip_pvalue(strong, n_perm=500, seed=0)
    assert p["p_two_sided"] <= 2 * (0 + 1) / (500 + 1), f"Konstant positives Signal: p minimal, bekam {p}"
    rng = np.random.default_rng(4)
    p = monthly_sign_flip_pvalue(rng.normal(0.0, 0.01, 60), n_perm=500, seed=0)
    assert p["p_two_sided"] > 0.05
    print("monthly_sign_flip_pvalue: OK")


def check_annualized_stats() -> None:
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    stats = annualized_stats(pd.Series([0.001] * 252, index=idx))
    assert abs(stats["cagr"] - (1.001 ** 252 - 1)) < 1e-9
    assert stats["max_drawdown"] == 0.0
    # Pflichttest Review: Start-Equity zaehlt als Peak.
    dd = annualized_stats(pd.Series([-0.10], index=idx[:1]))["max_drawdown"]
    assert abs(dd - (-0.10)) < 1e-12, f"[-0.10] muss -10% Drawdown ergeben, bekam {dd}"
    print("annualized_stats: OK")


def check_full_year_excess() -> None:
    idx = (list(pd.date_range("2019-11-01", "2019-12-31", freq="B"))
           + list(pd.date_range("2020-01-02", "2020-12-30", freq="B"))
           + list(pd.date_range("2021-01-04", "2021-03-31", freq="B")))
    excess_log = pd.Series(0.001, index=pd.DatetimeIndex(idx))
    years = full_year_excess(excess_log)
    assert list(years) == [2020], f"Nur 2020 ist vollstaendig (Jan+Dez), bekam {list(years)}"
    print("full_year_excess: OK")


def check_full_year_excess_log_convention() -> None:
    """Review-Fund: full_year_excess() muss die Summe der TAEGLICHEN LOG-
    Mehrertraege pro Jahr via expm1() zurueckverwandeln, NICHT
    compound_return()/prod(1+r)-1 auf denselben (Log-)Werten anwenden --
    das waere eine andere, mathematisch falsche Operation."""
    idx = pd.DatetimeIndex(["2020-01-02", "2020-06-15", "2020-12-30"])
    log_diffs = [0.01, -0.02, 0.03]
    excess_log = pd.Series(log_diffs, index=idx)
    years = full_year_excess(excess_log)
    assert list(years) == [2020]

    expected = float(np.expm1(sum(log_diffs)))
    assert abs(years[2020] - expected) < 1e-12, f"Erwartete {expected}, bekam {years[2020]}"

    # Klare Abgrenzung zur (falschen) compound_return-Anwendung auf
    # Log-Differenzen -- die beiden duerfen sich sichtbar unterscheiden.
    wrong = compound_return(log_diffs)
    assert abs(years[2020] - wrong) > 1e-6, "Log-Konvention (expm1(sum)) darf nicht mit compound_return uebereinstimmen"
    print("full_year_excess (Log-Konvention, expm1(sum) statt compound_return): OK")


def check_monthly_excess_sharpe() -> None:
    # Handgerechnet: [0.01, 0.02, 0.03, 0.04, 0.05] hat mean=0.03 und
    # (ddof=1-)Stichprobenvarianz 2.5*0.01^2 (bekannte Formel fuer 1..5),
    # also std = 0.01*sqrt(2.5). Sharpe = (0.03*12) / (0.01*sqrt(2.5)*sqrt(12))
    # = 36/sqrt(30).
    m = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    got = monthly_excess_sharpe(m)
    expected = 36.0 / np.sqrt(30.0)
    assert abs(got - expected) < 1e-9, f"Erwartete {expected}, bekam {got}"

    # Unabhaengige Gegenprobe ueber numpy mean/std (ddof=1, wie pandas-Default):
    arr = m.to_numpy()
    expected2 = (arr.mean() * 12) / (arr.std(ddof=1) * np.sqrt(12))
    assert abs(got - expected2) < 1e-9

    # Nullvol-Guard: konstante Werte -> std=0 -> Sharpe muss 0.0 sein (nicht inf/nan).
    assert monthly_excess_sharpe(pd.Series([0.02, 0.02, 0.02])) == 0.0
    # Einzelwert -> ddof=1-Std ist NaN -> muss ebenfalls sauber auf 0.0 fallen.
    assert monthly_excess_sharpe(pd.Series([0.02])) == 0.0
    print("monthly_excess_sharpe: OK")


def check_gates_and_verdict() -> None:
    good_boot = {"ann_geom_lower_1s95": 0.005, "ann_geom": 0.03}
    yearly = {2020: 0.02, 2021: 0.03, 2022: 0.01}
    loo = {"loo_SPY": 0.01, "loo_sleeve_bonds": 0.02}
    gates = evaluate_screening_gates(good_boot, dd_base=-0.10, dd_stress=-0.12,
                                     stressed_cagr=0.03, yearly_excess=yearly,
                                     loo_excess_compounds=loo)
    assert gates["passed_all"], f"Muesste bestehen: {gates}"

    # Gate A: Untergrenze <= 0 -> fail
    assert not evaluate_screening_gates({**good_boot, "ann_geom_lower_1s95": -0.001},
                                        -0.10, -0.12, 0.03, yearly, loo)["gate_a_excess_ci"]
    # Gate B: Stress-DD -20% -> fail
    assert not evaluate_screening_gates(good_boot, -0.10, -0.20, 0.03, yearly, loo)["gate_b_drawdown"]
    # Gate C: 1.5% < 2%-Floor -> fail (nicht bloss > 0!)
    assert not evaluate_screening_gates(good_boot, -0.10, -0.12, 0.015, yearly, loo)["gate_c_stressed_floor"]
    # Gate D: ein LOO-Rerun negativ -> fail
    assert not evaluate_screening_gates(good_boot, -0.10, -0.12, 0.03, yearly,
                                        {**loo, "loo_GLD": -0.001})["gate_d_no_single_driver"]
    # Gate D: bestes Jahr traegt alles -> fail
    assert not evaluate_screening_gates(good_boot, -0.10, -0.12, 0.03,
                                        {2020: 0.50, 2021: -0.01, 2022: -0.01}, loo)["gate_d_no_single_driver"]
    # Gate D: leeres LOO-Dict darf nicht vakuos bestehen -> fail
    assert not evaluate_screening_gates(good_boot, -0.10, -0.12, 0.03, yearly, {})["gate_d_no_single_driver"]
    # Gate D: einzelnes vollstaendiges Jahr -> leerer Rest nach Entfernen
    # des "besten" Jahres darf nicht vakuos bestehen -> fail
    assert evaluate_screening_gates(good_boot, -0.10, -0.12, 0.03,
                                    {2020: 0.05}, loo)["gate_d_no_single_driver"] is False

    hold = evaluate_holdout_gates(good_boot, -0.10, -0.12, 0.03)
    assert hold["passed_all"] and "gate_d_no_single_driver" not in hold

    verdict, candidate = screening_verdict(
        {"combo_long_flat": gates, "mom63_long_flat": gates},
        {"combo_long_flat": 0.5, "mom63_long_flat": 0.5},
    )
    assert candidate == "combo_long_flat", f"Tie-Break alphabetisch, bekam {candidate}"
    verdict, candidate = screening_verdict({"mom63_long_flat": {**gates, "passed_all": False}},
                                           {"mom63_long_flat": 0.5})
    assert candidate is None and "keine" in verdict.lower()
    print("Gates + Verdikt: OK")


def run_consistency_check() -> None:
    check_monthly_log_returns()
    check_stationary_indices()
    check_stationary_bootstrap()
    check_sign_flip()
    check_annualized_stats()
    check_full_year_excess()
    check_full_year_excess_log_convention()
    check_monthly_excess_sharpe()
    check_gates_and_verdict()
    print("\nAlle stats-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
