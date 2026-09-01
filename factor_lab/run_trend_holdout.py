"""
run_trend_holdout.py — EINMALIGE Holdout-Bestaetigung der versiegelten
Kandidatin (Spec v2 §10). Nimmt KEINE Argumente: liest candidate.json,
verifiziert Familie/Config-Hash/Snapshot-Hash, prueft den Tombstone und
schreibt ihn nach dem Lauf. Startet in CASH nach dev_end -- der volle
Positionsaufbau inkl. Kosten gehoert zum Holdout-Ergebnis.

Ausfuehren: py -3.12 factor_lab/run_trend_holdout.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from factor_lab.costs import COST_BP
from factor_lab.portfolio import month_end_dates, run_lagged_backtest, trend_weight_provider
from factor_lab.stats import (
    monthly_log_returns, stationary_block_bootstrap, monthly_sign_flip_pvalue,
    annualized_stats, evaluate_holdout_gates,
)
from factor_lab.registration import (
    REGISTRATION, read_and_verify_candidate, assert_no_tombstone, write_tombstone,
)
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.run_trend_baseline import prepare_inputs, run_variant, VARIANT_NAMES

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


def _holdout_run(inputs: dict, variant: str, dev_end: pd.Timestamp, cost_multiplier: float = 1.0):
    """Wie run_variant, aber Entscheidungen NUR nach dev_end -> der Lauf
    beginnt flach in Cash und baut die Position im Holdout auf."""
    holdout_decisions = inputs["eval_decisions"][inputs["eval_decisions"] > dev_end]
    saved = inputs["eval_decisions"]
    inputs["eval_decisions"] = holdout_decisions
    try:
        return run_variant(inputs, variant, cost_multiplier=cost_multiplier)
    finally:
        inputs["eval_decisions"] = saved


def run_holdout(dfs: dict, candidate: dict) -> dict:
    variant = candidate["variant"]
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unbekannte Variante im Siegel: {variant}")
    dev_end = pd.Timestamp(candidate["dev_end"])
    inputs = prepare_inputs(dfs)

    net, info = _holdout_run(inputs, variant, dev_end)
    net_stress, _ = _holdout_run(inputs, variant, dev_end, cost_multiplier=2.0)
    matched, _ = _holdout_run(inputs, "matched_long", dev_end)
    cash = inputs["cash_daily"].loc[net.index]

    monthly_excess = (monthly_log_returns(net) - monthly_log_returns(matched)).dropna()
    boot = stationary_block_bootstrap(monthly_excess.to_numpy(),
                                      REGISTRATION["bootstrap"]["expected_block_len_months"],
                                      REGISTRATION["bootstrap"]["n_boot"],
                                      REGISTRATION["bootstrap"]["seed"])
    stats = annualized_stats(net, cash)
    stats_stress = annualized_stats(net_stress, cash)
    gates = evaluate_holdout_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                   stats_stress["cagr"], REGISTRATION["dd_cap"],
                                   REGISTRATION["gate_c_floor"])
    return {
        "variant": variant,
        "summary": {
            "stats": stats,
            "stats_2x": stats_stress,
            "excess_bootstrap": boot,
            "permutation": monthly_sign_flip_pvalue(monthly_excess.to_numpy(),
                                                    REGISTRATION["permutation_n"],
                                                    REGISTRATION["bootstrap"]["seed"]),
            "gates": gates,
            "holdout_entry_cost": float(info["per_day"]["trade_cost"].iloc[:2].sum()),
            "instrument_contributions": info["instrument_contributions"],
            "attribution_days": ["holdout"],
            "matched_long_stats": annualized_stats(matched, cash),
        },
        "provenance": {
            "holdout_start": str(net.index.min()),
            "holdout_end": str(net.index.max()),
            "dev_end": str(dev_end),
            "snapshot_content_sha256": snapshot_content_sha256(dfs),
        },
    }


def main() -> None:
    print("*** VERSIEGELTES ONE-SHOT-HOLDOUT (keine Argumente, keine zweite Chance) ***")
    assert_no_tombstone(LOGS_DIR)
    dfs = load_trend_snapshot()
    candidates = sorted(glob.glob(os.path.join(LOGS_DIR, "trend_screening_*", "candidate.json")))
    if not candidates:
        raise ValueError("Kein candidate.json gefunden -- erst das Screening muss eine Kandidatin versiegeln")
    candidate = read_and_verify_candidate(candidates[-1], snapshot_content_sha256(dfs))
    result = run_holdout(dfs, candidate)

    g = result["summary"]["gates"]
    print(f"\n{result['variant']}: holdout_gates={'PASS' if g['passed_all'] else 'FAIL'}  "
          f"excess_lower95={result['summary']['excess_bootstrap']['ann_geom_lower_1s95']:+.4f}")
    if not g["passed_all"]:
        print("Familie trend-etf-v1 ist damit BEENDET (Spec v2 Abschnitt 10 Punkt 5) -- keine Runner-up-Variante.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LOGS_DIR, f"trend_holdout_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "holdout_summary.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_tombstone(LOGS_DIR, result_path)
    print(f"Ergebnis gespeichert: {result_path} (Tombstone geschrieben)")


if __name__ == "__main__":
    main()
