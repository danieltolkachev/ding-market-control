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


def run_consistency_check() -> None:
    check_monthly_log_returns()
    check_stationary_indices()
    check_stationary_bootstrap()
    check_sign_flip()
    print("\nAlle stats-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
