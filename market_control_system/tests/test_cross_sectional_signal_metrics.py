"""
test_cross_sectional_signal_metrics.py
=========================================

Prueft die reinen Statistikfunktionen in controller/cross_sectional_
signal_metrics.py mit synthetischen Werten (kein Modell/keine echten
Daten noetig).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))

import numpy as np
import pandas as pd

from cross_sectional_signal_metrics import (
    compute_rank_ic,
    compute_gross_spread,
    compute_breakeven_cost,
    compound_return,
    equity_curve,
    max_drawdown_from_returns,
    rolling_percentile_score,
    random_ranking_scores,
    momentum_scores,
    reversal_scores,
    one_minute_transition_mask,
    day_block_bootstrap,
    bootstrap_signal_verdict,
    day_sign_flip_pvalue,
)


def check_day_sign_flip_pvalue() -> None:
    # Stark positives Signal ueber 28 Tage -> beide p-Werte winzig.
    rng = np.random.default_rng(2)
    ts, vals = [], []
    for day in range(1, 29):
        for minute in range(30):
            ts.append(pd.Timestamp(f"2026-07-{day:02d} 09:30") + pd.Timedelta(minutes=minute))
            vals.append(float(rng.normal(0.5, 0.1)))
    strong = day_sign_flip_pvalue(vals, pd.DatetimeIndex(ts), n_perm=500, seed=0)
    assert strong["p_two_sided"] < 0.01, f"Erwartete winziges zweiseitiges p, bekam {strong['p_two_sided']}"
    assert strong["p_greater_zero"] < 0.01, f"Erwartete winziges einseitiges p, bekam {strong['p_greater_zero']}"
    assert strong["n_days"] == 28

    # Um 0 symmetrisches Rauschen -> p gross (kein Signal).
    noise_vals = [float(v) for v in rng.normal(0.0, 1.0, size=len(ts))]
    noise = day_sign_flip_pvalue(noise_vals, pd.DatetimeIndex(ts), n_perm=500, seed=0)
    assert noise["p_two_sided"] > 0.05, f"Rauschen darf nicht signifikant sein, bekam {noise['p_two_sided']}"

    # Determinismus pro Seed.
    again = day_sign_flip_pvalue(vals, pd.DatetimeIndex(ts), n_perm=500, seed=0)
    assert strong["p_two_sided"] == again["p_two_sided"], "Gleicher Seed muss identisches p liefern"
    print("day_sign_flip_pvalue: OK")


def check_bootstrap_signal_verdict() -> None:
    null_ci = {"ci_low_95": -0.002, "ci_high_95": 0.002, "p_leq_zero": 0.4}
    positive_ci = {"ci_low_95": 0.01, "ci_high_95": 0.03, "p_leq_zero": 0.001}

    # Alle Modell-Varianten straddeln 0 -> "kein Signal", auch wenn eine
    # BASELINE (reversal) ein positives CI hat -- Baselines zaehlen nicht.
    bootstrap_by_variant = {"mu": null_ci, "p_up": null_ci, "reversal": positive_ci}
    verdict = bootstrap_signal_verdict(bootstrap_by_variant, model_variant_names=["mu", "p_up"])
    assert "kein" in verdict.lower(), f"Erwartete Kein-Signal-Verdikt, bekam: {verdict}"

    # Eine Modell-Variante mit CI komplett > 0 -> Signal-Verdikt.
    bootstrap_by_variant = {"mu": null_ci, "p_up": positive_ci}
    verdict = bootstrap_signal_verdict(bootstrap_by_variant, model_variant_names=["mu", "p_up"])
    assert "kein" not in verdict.lower(), f"Erwartete Signal-Verdikt, bekam: {verdict}"
    assert "p_up" in verdict, f"Signal-Verdikt muss die Variante benennen, bekam: {verdict}"

    # Oekonomisches Band: liegen ALLE oberen CI-Grenzen unter dem Band,
    # muss das Kein-Signal-Verdikt die staerkere Equivalence-Aussage
    # enthalten (selbst optimistisch unterhalb oekonomischer Relevanz).
    bootstrap_by_variant = {"mu": null_ci, "p_up": null_ci}
    verdict = bootstrap_signal_verdict(
        bootstrap_by_variant, model_variant_names=["mu", "p_up"], economic_ic_band=0.01,
    )
    assert "kein" in verdict.lower(), f"Erwartete Kein-Signal-Verdikt, bekam: {verdict}"
    assert "0.01" in verdict, f"Band-Aussage muss den Schwellenwert nennen, bekam: {verdict}"

    # Ragt eine obere CI-Grenze UEBER das Band, darf die Equivalence-
    # Aussage NICHT erscheinen (nur das normale Kein-Signal-Verdikt).
    wide_ci = {"ci_low_95": -0.002, "ci_high_95": 0.05, "p_leq_zero": 0.2}
    verdict = bootstrap_signal_verdict(
        {"mu": null_ci, "p_up": wide_ci}, model_variant_names=["mu", "p_up"], economic_ic_band=0.01,
    )
    assert "kein" in verdict.lower() and "0.01" not in verdict, (
        f"Bei CI-Obergrenze ueber dem Band darf keine Band-Aussage kommen, bekam: {verdict}"
    )
    print("bootstrap_signal_verdict: OK")


def check_day_block_bootstrap() -> None:
    # 1) Blockstruktur: Tag A liefert konstant +1, Tag B konstant -1. Beim
    #    Resampling GANZER TAGE kann jedes Bootstrap-Mittel nur -1, 0 oder +1
    #    sein (AA/AB/BB) -- Bar-weises Resampling ergaebe ein Kontinuum.
    timestamps = pd.DatetimeIndex(
        [f"2026-08-03 09:{30+i}" for i in range(10)]
        + [f"2026-08-04 09:{30+i}" for i in range(10)]
    )
    values = [1.0] * 10 + [-1.0] * 10
    result = day_block_bootstrap(values, timestamps, n_boot=500, seed=0)
    unique_means = set(np.round(result["bootstrap_means"], 12))
    assert unique_means <= {-1.0, 0.0, 1.0}, (
        f"Tages-Block-Resampling darf nur Mittel aus {{-1,0,1}} erzeugen, bekam {sorted(unique_means)}"
    )
    assert result["n_days"] == 2, f"Erwartete 2 Tagesbloecke, bekam {result['n_days']}"

    # 2) Deutlich positives Signal ueber viele Tage -> CI-Untergrenze > 0,
    #    Anteil der Bootstrap-Mittel <= 0 praktisch null.
    rng = np.random.default_rng(1)
    ts_many, vals_many = [], []
    for day in range(1, 29):
        for minute in range(30):
            ts_many.append(pd.Timestamp(f"2026-07-{day:02d} 09:30") + pd.Timedelta(minutes=minute))
            vals_many.append(float(rng.normal(0.5, 0.1)))
    strong = day_block_bootstrap(vals_many, pd.DatetimeIndex(ts_many), n_boot=500, seed=0)
    assert strong["ci_low_95"] > 0, f"CI-Untergrenze muss bei starkem Signal > 0 sein, bekam {strong['ci_low_95']}"
    assert strong["p_leq_zero"] < 0.01, f"p_leq_zero muss winzig sein, bekam {strong['p_leq_zero']}"
    assert abs(strong["mean"] - 0.5) < 0.05, f"Punktschaetzer muss nahe 0.5 liegen, bekam {strong['mean']}"

    # 3) Determinismus: derselbe Seed muss exakt dieselben Ergebnisse liefern.
    again = day_block_bootstrap(vals_many, pd.DatetimeIndex(ts_many), n_boot=500, seed=0)
    assert strong["ci_low_95"] == again["ci_low_95"] and strong["p_leq_zero"] == again["p_leq_zero"], (
        "Gleicher Seed muss identische Bootstrap-Ergebnisse liefern"
    )
    print("day_block_bootstrap: OK")


def check_one_minute_transition_mask() -> None:
    # 09:30 -> 09:31 -> 09:32 sind echte 1-Minuten-Uebergaenge; 09:32 -> 09:35
    # ist eine Luecke (fehlende Bars); 09:35 -> naechster Handelstag ist eine
    # Session-Grenze. Nur Positionen, deren NAECHSTE Zeile exakt 60s spaeter
    # liegt, duerfen True sein; die letzte Position hat keinen Nachfolger.
    index = pd.DatetimeIndex([
        "2026-08-03 09:30", "2026-08-03 09:31", "2026-08-03 09:32",
        "2026-08-03 09:35", "2026-08-04 09:30", "2026-08-04 09:31",
    ])
    mask = one_minute_transition_mask(index)
    expected = np.array([True, True, False, False, True, False])
    assert mask.dtype == bool, f"Maske muss bool sein, bekam {mask.dtype}"
    assert len(mask) == len(index), "Maske muss dieselbe Laenge wie der Index haben"
    assert (mask == expected).all(), f"Erwartete {expected.tolist()}, bekam {mask.tolist()}"
    print("one_minute_transition_mask: OK")


def check_rank_ic() -> None:
    scores = {"A": 1.0, "B": 2.0, "C": 3.0}
    perfect_positive = {"A": 0.01, "B": 0.02, "C": 0.03}
    ic = compute_rank_ic(scores, perfect_positive)
    assert abs(ic - 1.0) < 1e-9, f"Erwartete Rank-IC=1.0 bei perfekter positiver Korrelation, bekam {ic}"

    perfect_negative = {"A": 0.03, "B": 0.02, "C": 0.01}
    ic = compute_rank_ic(scores, perfect_negative)
    assert abs(ic - (-1.0)) < 1e-9, f"Erwartete Rank-IC=-1.0 bei perfekter negativer Korrelation, bekam {ic}"

    too_few = compute_rank_ic({"A": 1.0, "B": 2.0}, {"A": 0.1, "B": 0.2})
    assert np.isnan(too_few), "Bei < 3 gemeinsamen Symbolen sollte NaN zurueckkommen"

    print("compute_rank_ic: OK")


def check_gross_spread() -> None:
    scores = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
    forward_returns = {"A": 0.02, "B": 0.01, "C": -0.01, "D": -0.02}
    spread = compute_gross_spread(scores, forward_returns, n_long=1, n_short=1)
    # Long A (+0.02), Short D (-0.02) -> Spread = 0.02 - (-0.02) = 0.04
    assert abs(spread - 0.04) < 1e-9, f"Erwarteter Spread 0.04, bekam {spread}"
    print("compute_gross_spread: OK")


def check_breakeven_cost() -> None:
    gross = [0.01, 0.02, 0.0]
    turnover = [0.5, 0.5, 0.5]
    breakeven = compute_breakeven_cost(gross, turnover)
    # Handberechnet (nicht ueber die Implementierungsformel re-abgeleitet):
    # mean(gross) = 0.01, mean(turnover) = 0.5 -> 0.01 / 0.5 = 0.02
    assert abs(breakeven - 0.02) < 1e-9, f"Erwarteter Breakeven-Kostensatz 0.02, bekam {breakeven}"
    print("compute_breakeven_cost: OK")


def check_compounding() -> None:
    returns = [0.1, 0.1]
    compounded = compound_return(returns)
    simple_sum = sum(returns)
    assert abs(compounded - 0.21) < 1e-9, f"Erwartetes Compound-Ergebnis 0.21, bekam {compounded}"
    assert abs(compounded - simple_sum) > 1e-6, (
        "Compounding und einfache Summe muessen bei diesen Werten unterschiedlich sein -- "
        "sonst wird faelschlich additiv gerechnet"
    )
    print("compound_return: OK (0.21, weicht bewusst von additiver Summe 0.20 ab)")


def check_drawdown() -> None:
    returns = [0.1, -0.2, 0.05]
    dd = max_drawdown_from_returns(returns)
    # equity: 1.1 -> 0.88 -> 0.924; running_max bleibt bei 1.1; Tiefpunkt 0.88/1.1-1=-0.2
    assert abs(dd - (-0.2)) < 1e-9, f"Erwarteter Max-Drawdown -0.2, bekam {dd}"
    print("max_drawdown_from_returns: OK")


def check_rolling_percentile() -> None:
    history = [1.0, 2.0, 3.0, 4.0, 100.0]  # letzter Wert ist der hoechste je gesehene
    pct = rolling_percentile_score(history, window=10)
    assert abs(pct - 1.0) < 1e-9, f"Hoechster je gesehener Wert sollte Perzentil 1.0 ergeben, bekam {pct}"

    too_short = rolling_percentile_score([1.0], window=10)
    assert too_short == 0.5, "Bei zu wenig Historie soll ein neutraler Default (0.5) zurueckkommen"
    print("rolling_percentile_score: OK")


def check_baselines() -> None:
    symbols = ["A", "B", "C"]
    scores_a = random_ranking_scores(symbols, seed=0)
    scores_b = random_ranking_scores(symbols, seed=0)
    scores_c = random_ranking_scores(symbols, seed=1)
    assert scores_a == scores_b, "Gleicher Seed muss identische Scores liefern"
    assert scores_a != scores_c, "Unterschiedlicher Seed sollte (fast sicher) unterschiedliche Scores liefern"
    assert set(scores_a) == set(symbols)

    prices = {
        "A": [100.0, 101.0, 102.0, 110.0],
        "B": [100.0, 99.0, 98.0, 90.0],
    }
    mom = momentum_scores(prices, lookback_bars=3)
    assert mom["A"] > 0, "A ist gestiegen -- Momentum-Score muss positiv sein"
    assert mom["B"] < 0, "B ist gefallen -- Momentum-Score muss negativ sein"

    rev = reversal_scores(prices, lookback_bars=3)
    assert rev["A"] < 0 and rev["B"] > 0, "Reversal muss exakt das Vorzeichen von Momentum umkehren"
    print("random_ranking_scores/momentum_scores/reversal_scores: OK")


def run_consistency_check() -> None:
    check_rank_ic()
    check_gross_spread()
    check_breakeven_cost()
    check_compounding()
    check_drawdown()
    check_rolling_percentile()
    check_baselines()
    check_one_minute_transition_mask()
    check_day_block_bootstrap()
    check_bootstrap_signal_verdict()
    check_day_sign_flip_pvalue()
    print("\nAlle cross_sectional_signal_metrics-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
