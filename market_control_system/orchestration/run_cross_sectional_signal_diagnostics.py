"""
run_cross_sectional_signal_diagnostics.py
============================================

Cross-Sectional-Walk-Forward-Signaldiagnostik OHNE Controller/Ausfuehrung/
Kosten (siehe Design-Spec docs/superpowers/specs/2026-08-31-cross-
sectional-signal-diagnostics-design.md). Misst Rank-IC und Brutto-Top-
minus-Bottom-Spread ueber mehrere Score-Varianten und vergleicht gegen
Random/Momentum/Reversal-Baselines -- bevor irgendeine weitere Arbeit an
Positionsgroessen, Deadbands oder Hysterese investiert wird.

Nutzt NIRGENDS ControlLoop/PaperExecutionEngine/CrossSectionalPortfolio.
Nutzt horizon=1 (naechster Bar), nicht den sonst ueblichen Default 5 --
das eliminiert jede Entscheidungs-/Fuellzeitpunkt-Ambiguitaet durch
Konstruktion (siehe Global Constraints im Plan).

Das Walk-Forward-Fenster (WALKFORWARD_FRACTION des eingefrorenen
Snapshots) wird hier automatisch in Folds durchlaufen. Der Rest
(Holdout) wird NIE von diesem Skript gelesen -- nur von
run_cross_sectional_holdout_eval.py (Task 5), manuell ausgeloest.

Ausfuehren: py -3.12 orchestration/run_cross_sectional_signal_diagnostics.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import numpy as np
import torch

from frozen_snapshot import load_or_build_snapshot
from cross_sectional_signal_metrics import (
    compute_rank_ic, compute_gross_spread, compute_breakeven_cost,
    compound_return, max_drawdown_from_returns, rolling_percentile_score,
    random_ranking_scores, momentum_scores, reversal_scores,
)
from cross_sectional_fold_training import build_symbol_sequences, train_fold_model
from walk_forward import WalkForwardConfig, generate_fold_slices
from exposure_controller import calibrate_k
from cross_sectional_universe import UNIVERSE, BACKTEST_LOOKBACK_DAYS

WALKFORWARD_FRACTION = 0.8
N_LONG = 3
N_SHORT = 3
ROLLING_PERCENTILE_WINDOW = 500
MOMENTUM_LOOKBACK_BARS = 20

SCORE_VARIANT_NAMES = ["mu", "mu_over_sigma", "kelly_edge", "p_up", "mu_percentile"]
BASELINE_NAMES = ["random", "momentum", "reversal"]


def split_walkforward_and_holdout(aligned_index):
    split_idx = int(len(aligned_index) * WALKFORWARD_FRACTION)
    return aligned_index[:split_idx], aligned_index[split_idx:]


def build_aligned_index(dfs: dict) -> "pd.DatetimeIndex":
    aligned_index = dfs[UNIVERSE[0]].index
    for symbol in UNIVERSE[1:]:
        aligned_index = aligned_index.intersection(dfs[symbol].index)
    return aligned_index.sort_values()


def run_diagnostics(use_holdout: bool = False) -> dict:
    print(f"=== Lade eingefrorenen Snapshot ({len(UNIVERSE)} Symbole, {BACKTEST_LOOKBACK_DAYS} Tage) ===")
    dfs = load_or_build_snapshot(UNIVERSE, BACKTEST_LOOKBACK_DAYS)
    aligned_index = build_aligned_index(dfs)
    walkforward_index, holdout_index = split_walkforward_and_holdout(aligned_index)
    print(f"  Gemeinsamer Zeitindex: {len(aligned_index)} Bars")
    print(f"  Walk-Forward-Fenster: {len(walkforward_index)} Bars, Holdout: {len(holdout_index)} Bars (nie automatisch gelesen)")

    eval_index = holdout_index if use_holdout else walkforward_index
    if use_holdout:
        print("  *** HOLDOUT-MODUS: dieser Lauf liest das reservierte Fenster. Nur manuell, nach positivem Walk-Forward-Ergebnis. ***")

    cfg = WalkForwardConfig(horizon=1, seed=0)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    sequences = {}
    for symbol in UNIVERSE:
        df_eval = dfs[symbol].loc[eval_index]
        X, y, end_idx = build_symbol_sequences(df_eval, cfg)
        prices = df_eval["price"].to_numpy()
        # build_symbol_sequences() liefert end_idx als pandas.Index von
        # Zeitstempel-Labels (siehe SequenceWindowBuilder.build docstring),
        # nicht als Ganzzahl-Positionen -- fuer die Positions-Slices in
        # price_history unten (prices ist ein rohes numpy-Array) auf
        # Ganzzahl-Positionen relativ zu df_eval.index umrechnen.
        end_idx = df_eval.index.get_indexer(end_idx)
        if (end_idx < 0).any():
            raise ValueError(f"{symbol}: get_indexer() konnte {int((end_idx < 0).sum())} Zeitstempel nicht in df_eval.index finden")
        sequences[symbol] = (X, y, end_idx, prices)

    n_samples = min(len(sequences[s][0]) for s in UNIVERSE)
    sample_counts = {s: len(sequences[s][0]) for s in UNIVERSE}
    if len(set(sample_counts.values())) != 1:
        raise ValueError(f"Symbole haben unterschiedliche Sequenz-Anzahlen, Cross-Sectional-Pairing waere falsch: {sample_counts}")
    folds = generate_fold_slices(n_samples, cfg)
    if not folds:
        raise ValueError(f"Nicht genug Sequenzen ({n_samples}) fuer train_size={cfg.train_size}+test_size={cfg.test_size}")
    print(f"  {n_samples} gemeinsame Sequenzen, {len(folds)} Folds")

    results = {name: {"rank_ic": [], "gross_spread": [], "turnover": []} for name in SCORE_VARIANT_NAMES + BASELINE_NAMES}

    for fold_i, (train_slice, test_slice) in enumerate(folds):
        print(f"\n=== Fold {fold_i+1}/{len(folds)} ===")
        models, k_calibrated, mu_histories = {}, {}, {}
        for symbol in UNIVERSE:
            X, y, _, _ = sequences[symbol]
            X_train, y_train = X[train_slice], y[train_slice]
            model = train_fold_model(X_train, y_train, cfg)
            models[symbol] = model
            forecast_train = model.predict(X_train)
            k_calibrated[symbol] = calibrate_k(
                forecast_train.expected_return, forecast_train.expected_volatility,
                max_position=1.0, target_utilization=0.5, percentile=95.0,
            )
            mu_histories[symbol] = list(forecast_train.expected_return[-ROLLING_PERCENTILE_WINDOW:])

        forecasts, test_len = {}, None
        for symbol in UNIVERSE:
            X, y, end_idx, prices = sequences[symbol]
            X_test, y_test = X[test_slice], y[test_slice]
            forecast = models[symbol].predict(X_test)
            forecasts[symbol] = (forecast, y_test, end_idx[test_slice], prices)
            test_len = len(y_test) if test_len is None else min(test_len, len(y_test))

        previous_weights = {name: {s: 0.0 for s in UNIVERSE} for name in SCORE_VARIANT_NAMES + BASELINE_NAMES}

        for t in range(test_len):
            mus = {s: float(forecasts[s][0].expected_return[t]) for s in UNIVERSE}
            sigmas = {s: float(forecasts[s][0].expected_volatility[t]) for s in UNIVERSE}
            p_ups = {s: float(forecasts[s][0].probability_up[t]) for s in UNIVERSE}
            forward_returns = {s: float(forecasts[s][1][t]) for s in UNIVERSE}

            for symbol in UNIVERSE:
                mu_histories[symbol].append(mus[symbol])

            price_history = {}
            for symbol in UNIVERSE:
                _, _, end_idx_test, prices = forecasts[symbol]
                row = end_idx_test[t]
                price_history[symbol] = list(prices[: row + 1])

            score_variants = {
                "mu": mus,
                "mu_over_sigma": {s: mus[s] / (sigmas[s] + 1e-6) for s in UNIVERSE},
                "kelly_edge": {s: k_calibrated[s] * mus[s] / (sigmas[s] ** 2 + 1e-6) for s in UNIVERSE},
                "p_up": p_ups,
                "mu_percentile": {s: rolling_percentile_score(mu_histories[s], ROLLING_PERCENTILE_WINDOW) for s in UNIVERSE},
                # Ein Zufalls-Seed PRO (Fold, Zeitschritt) statt mehrerer
                # gemittelter Wiederholungen PRO Zeitschritt -- ueber die
                # tausenden Zeitschritte aller Folds hinweg mittelt sich das
                # Ergebnis ohnehin zu einer stabilen Zufalls-Baseline, ohne
                # den N-fachen Rechenaufwand pro Schritt (YAGNI).
                "random": random_ranking_scores(UNIVERSE, seed=fold_i * 10_000 + t),
                "momentum": momentum_scores(price_history, MOMENTUM_LOOKBACK_BARS),
                "reversal": reversal_scores(price_history, MOMENTUM_LOOKBACK_BARS),
            }

            for name, scores in score_variants.items():
                if len(scores) < N_LONG + N_SHORT:
                    continue
                ic = compute_rank_ic(scores, forward_returns)
                spread = compute_gross_spread(scores, forward_returns, N_LONG, N_SHORT)
                ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
                new_weights = {s: 0.0 for s in UNIVERSE}
                for s in ranked[:N_LONG]:
                    new_weights[s] = 1.0 / N_LONG
                for s in ranked[-N_SHORT:]:
                    new_weights[s] = -1.0 / N_SHORT
                turnover = sum(abs(new_weights[s] - previous_weights[name][s]) for s in UNIVERSE)
                previous_weights[name] = new_weights

                if not np.isnan(ic):
                    results[name]["rank_ic"].append(ic)
                if not np.isnan(spread):
                    results[name]["gross_spread"].append(spread)
                results[name]["turnover"].append(turnover)

        print(f"  Fold {fold_i+1} abgeschlossen ({test_len} Test-Bars)")

    summary = {}
    for name in SCORE_VARIANT_NAMES + BASELINE_NAMES:
        rank_ics = results[name]["rank_ic"]
        spreads = results[name]["gross_spread"]
        turnovers = results[name]["turnover"]
        summary[name] = {
            "mean_rank_ic": float(np.mean(rank_ics)) if rank_ics else float("nan"),
            "n_observations": len(rank_ics),
            "mean_gross_spread_per_bar": float(np.mean(spreads)) if spreads else float("nan"),
            "compounded_gross_return": compound_return(spreads) if spreads else float("nan"),
            "max_drawdown": max_drawdown_from_returns(spreads) if spreads else float("nan"),
            "breakeven_cost": compute_breakeven_cost(spreads, turnovers) if spreads and turnovers else float("nan"),
        }

    return {"use_holdout": use_holdout, "n_folds": len(folds), "summary": summary}


def main():
    result = run_diagnostics(use_holdout=False)

    print(f"\n{'='*70}\n=== Zusammenfassung ueber {result['n_folds']} Walk-Forward-Folds ===\n{'='*70}")
    for name, stats in result["summary"].items():
        print(f"{name:>15}: mean_rank_ic={stats['mean_rank_ic']:+.4f}  "
              f"n={stats['n_observations']}  "
              f"compounded_gross_return={stats['compounded_gross_return']:+.4%}  "
              f"max_drawdown={stats['max_drawdown']:+.4%}  "
              f"breakeven_cost={stats['breakeven_cost']:+.6f}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"signal_diagnostics_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "diagnostics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSummary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
