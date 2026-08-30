"""
run_cross_sectional_baseline.py
==================================

Trivialer Vergleichs-Massstab fuer run_cross_sectional_backtest.py: ein
gleichgewichtetes Long-Only-Portfolio ueber cross_sectional_universe.
UNIVERSE, OHNE Modell/Prognose/Ranking -- reine Durchschnittsbildung der
Log-Returns. Kein Transaktionskosten-Modell (wird nie umgeschichtet), das
ist bewusst eine GUENSTIGERE Referenz als das echte Portfolio bekommt --
nur wenn das Ranking-Portfolio DAS hier schlaegt, traegt das Ranking
selbst etwas bei (siehe Design-Spec, "Testen/Validierung", Punkt 4).

Nutzt denselben Zeitraum/Split wie run_cross_sectional_backtest.py
(gleiche UNIVERSE/BACKTEST_LOOKBACK_DAYS/PRETRAIN_FRACTION-Konstanten),
damit der Vergleich auf demselben Out-of-Sample-Fenster steht.

Ausfuehren: py -3.12 orchestration/run_cross_sectional_baseline.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import numpy as np
import pandas as pd

from alpaca_client import fetch_historical_bars_approximate
from backtest_stats import compute_period_statistics, format_statistics_report
from cross_sectional_universe import UNIVERSE, BACKTEST_LOOKBACK_DAYS, PRETRAIN_FRACTION

CAPITAL = 10_000.0


def main():
    print(f"=== Lade {len(UNIVERSE)} Symbole ({BACKTEST_LOOKBACK_DAYS} Tage) ===")
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=BACKTEST_LOOKBACK_DAYS)
    dfs = {}
    for symbol in UNIVERSE:
        dfs[symbol] = fetch_historical_bars_approximate(symbol, start, end)
        print(f"  {symbol}: {dfs[symbol].shape[0]} Bars")

    aligned_index = dfs[UNIVERSE[0]].index
    for symbol in UNIVERSE[1:]:
        aligned_index = aligned_index.intersection(dfs[symbol].index)
    aligned_index = aligned_index.sort_values()

    split_idx = int(len(aligned_index) * PRETRAIN_FRACTION)
    cutoff = aligned_index[split_idx]
    replay_index = aligned_index[aligned_index >= cutoff]
    print(f"\nGemeinsamer Zeitindex: {len(aligned_index)} Bars, Cutoff: {cutoff}, "
          f"Replay-Fenster: {len(replay_index)} Bars")

    log_returns = pd.DataFrame({
        symbol: np.log(dfs[symbol].loc[replay_index, "price"]).diff()
        for symbol in UNIVERSE
    })
    equal_weight_return = log_returns.mean(axis=1).dropna()

    period_stats = compute_period_statistics(
        pd.DatetimeIndex(equal_weight_return.index), equal_weight_return.to_numpy(), period="W"
    )
    print(f"\n=== Baseline: gleichgewichtetes Long-Only-Portfolio ({len(UNIVERSE)} Symbole) ===")
    print(format_statistics_report(period_stats, period_label="Woche", capital=CAPITAL))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"cross_sectional_baseline_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    summary = {
        "run_id": run_id,
        "universe": UNIVERSE,
        "n_periods": period_stats.n_periods,
        "mean_period_return": period_stats.mean_return,
        "std_period_return": period_stats.std_return,
        "t_statistic": period_stats.t_statistic,
        "sharpe_like": period_stats.sharpe_like,
        "win_rate": period_stats.win_rate,
        "cumulative_return": period_stats.cumulative_return,
        "max_drawdown": period_stats.max_drawdown,
    }
    summary_path = os.path.join(results_dir, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
