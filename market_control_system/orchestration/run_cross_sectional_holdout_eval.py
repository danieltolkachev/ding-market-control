"""
run_cross_sectional_holdout_eval.py
======================================

Wertet das RESERVIERTE Holdout-Fenster aus (siehe Design-Spec
docs/superpowers/specs/2026-08-31-cross-sectional-signal-diagnostics-
design.md) -- der letzte WALKFORWARD_FRACTION-Komplementanteil des
eingefrorenen Snapshots, der von run_cross_sectional_signal_diagnostics.py
NIE gelesen wird.

ABSICHTLICH ein separates Skript, kein Kommandozeilen-Flag auf dem
Walk-Forward-Skript: das macht es unmoeglich, das Holdout versehentlich
durch wiederholtes Ausfuehren desselben Befehls "anzupeeken". Nur einmalig
ausfuehren, NACHDEM die Walk-Forward-Folds einen robusten (im Mittel
positiven, in der Mehrheit der Folds positiven) Rank-IC gezeigt haben --
sonst ist das Ergebnis wertlos (siehe Design-Spec, Nicht-Ziel-Abschnitt
und Review-Punkt 5).

Ausfuehren: py -3.12 orchestration/run_cross_sectional_holdout_eval.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from run_cross_sectional_signal_diagnostics import run_diagnostics


def main():
    print("=== HOLDOUT-AUSWERTUNG ===")
    print("Dieses Skript liest das reservierte Holdout-Fenster. Nur ausfuehren, ")
    print("wenn die Walk-Forward-Folds (run_cross_sectional_signal_diagnostics.py) ")
    print("bereits einen robusten, positiven Rank-IC gezeigt haben.\n")

    result = run_diagnostics(use_holdout=True)

    print(f"\n{'='*70}\n=== Holdout-Ergebnis ===\n{'='*70}")
    for name, stats in result["summary"].items():
        boot = stats["bootstrap"]
        print(f"{name:>15}: mean_rank_ic={stats['mean_rank_ic']:+.4f}  "
              f"ci95=[{boot['ci_low_95']:+.4f}, {boot['ci_high_95']:+.4f}]  "
              f"p_leq_zero={boot['p_leq_zero']:.3f}  "
              f"n={stats['n_observations']}  "
              f"compounded_gross_return={stats['compounded_gross_return']:+.4%}  "
              f"max_drawdown={stats['max_drawdown']:+.4%}  "
              f"breakeven_cost={stats['breakeven_cost']:+.6f}")

    print(f"\n{result['verdict']}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"holdout_eval_{run_id}")
    os.makedirs(results_dir, exist_ok=True)

    per_bar_frame = result.pop("_per_bar_frame")
    per_bar_path = os.path.join(results_dir, "per_bar_rank_ic.csv")
    per_bar_frame.to_csv(per_bar_path)
    print(f"\nPer-Bar-Rank-IC gespeichert: {per_bar_path}")

    summary_path = os.path.join(results_dir, "holdout_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Summary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
