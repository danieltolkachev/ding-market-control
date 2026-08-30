"""
run_cross_sectional_multi_seed.py
====================================

Multi-Seed-Vergleich fuer CrossSectionalPortfolioConfig-Varianten
(n_long/n_short/hysteresis_zone), nach demselben Muster wie
orchestration/run_multi_seed_comparison.py fuer das Single-Symbol-System:
ein einzelner Lauf ist wegen Seed-Streuung keine verlaessliche Grundlage
fuer eine Entscheidung (siehe Projekt-Notizen zum Overnight-Backtest).

Zusaetzlich zu den 5 Kennzahlen, die run_multi_seed_comparison.py bereits
vergleicht, werden hier auch total_slippage_cost und total_turnover
verglichen: der erste reale Cross-Sectional-Lauf (Task 4) zeigte einen
grossen Verlust (-71% bis -75%), der bei genauerer Pruefung ueberwiegend
transaktionskosten-/turnover-getrieben war, nicht schlechtes Modellsignal.
Die hier verglichenen Varianten unterscheiden sich gerade in der Breite
der Hysterese-Zone (schmaler/mehr erwarteter Churn vs. breiter/weniger
erwarteter Churn) -- ob die breitere Variante Turnover/Slippage tatsaechlich
senkt, ist damit eine direkt relevante, ohnehin verfuegbare Messung.

Ausfuehren: py -3.12 orchestration/run_cross_sectional_multi_seed.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from cross_sectional_portfolio import CrossSectionalPortfolioConfig
from backtest_stats import (
    summarize_seed_distribution, format_seed_distribution_report,
    paired_comparison, format_paired_comparison_report,
)
import run_cross_sectional_backtest as backtest_module

SEEDS = [0, 1, 2, 3, 4]
COMPARISON_METRICS = [
    "cumulative_return", "t_statistic", "sharpe_like", "win_rate", "max_drawdown",
    "total_slippage_cost", "total_turnover",
]

VARIANTS = {
    "default_3x3_h5": CrossSectionalPortfolioConfig(n_long=3, n_short=3, hysteresis_zone=5),
    "wider_5x5_h7": CrossSectionalPortfolioConfig(n_long=5, n_short=5, hysteresis_zone=7),
}


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"cross_sectional_multiseed_{run_id}")
    os.makedirs(base_dir, exist_ok=True)
    print(f"Cross-Sectional-Multi-Seed-Run-ID: {run_id}")
    print(f"Varianten: {list(VARIANTS)}, Seeds: {SEEDS}")

    all_distributions = {}
    for variant_name, config in VARIANTS.items():
        summaries = []
        for seed in SEEDS:
            print(f"\n{'#'*70}\n# Variante '{variant_name}', Seed {seed}\n{'#'*70}")
            summary = backtest_module.run_backtest(seed=seed, portfolio_config=config)
            summary_path = os.path.join(base_dir, f"{variant_name}_seed{seed}_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            summaries.append(summary)

        print(f"\n=== Seed-Verteilung: {variant_name} (n={len(summaries)} Seeds) ===")
        distributions = {}
        for metric in COMPARISON_METRICS:
            values = [s[metric] for s in summaries]
            dist = summarize_seed_distribution(metric, values)
            print(format_seed_distribution_report(dist))
            distributions[metric] = dist.values.tolist()
        all_distributions[variant_name] = distributions

        dist_path = os.path.join(base_dir, f"{variant_name}_seed_distribution.json")
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(distributions, f, indent=2)

    variant_names = list(all_distributions)
    for i in range(len(variant_names)):
        for j in range(i + 1, len(variant_names)):
            a, b = variant_names[i], variant_names[j]
            print(f"\n=== Vergleich: {b} vs. {a} (gepaart) ===")
            comparisons = {}
            for metric in COMPARISON_METRICS:
                cmp = paired_comparison(all_distributions[a][metric], all_distributions[b][metric], metric_name=metric)
                print(format_paired_comparison_report(cmp))
                comparisons[metric] = {
                    "mean_diff": cmp.mean_diff, "t_statistic": cmp.t_statistic, "b_wins_rate": cmp.b_wins_rate,
                }
            cmp_path = os.path.join(base_dir, f"comparison_{a}_vs_{b}.json")
            with open(cmp_path, "w", encoding="utf-8") as f:
                json.dump(comparisons, f, indent=2)

    print(f"\nAlle Ergebnisse in: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    main()
