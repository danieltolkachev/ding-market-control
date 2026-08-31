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
import pandas as pd
import torch

from frozen_snapshot import load_or_build_snapshot, snapshot_content_sha256, snapshot_path
from cross_sectional_signal_metrics import (
    compute_rank_ic, compute_gross_spread, compute_breakeven_cost,
    compound_return, max_drawdown_from_returns, rolling_percentile_score,
    random_ranking_scores, momentum_scores, reversal_scores,
    one_minute_transition_mask, day_block_bootstrap, bootstrap_signal_verdict,
)
from cross_sectional_fold_training import build_symbol_sequences, train_fold_model, purge_train_slice
from walk_forward import WalkForwardConfig, generate_fold_slices
from exposure_controller import calibrate_k
from cross_sectional_universe import UNIVERSE, BACKTEST_LOOKBACK_DAYS

WALKFORWARD_FRACTION = 0.8
N_LONG = 3
N_SHORT = 3
ROLLING_PERCENTILE_WINDOW = 500
MOMENTUM_LOOKBACK_BARS = 20
BOOTSTRAP_N_BOOT = 2000
BOOTSTRAP_SEED = 0

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

    # epochs_per_fold explizit auf 10 setzen (Design-Spec-Vorgabe, passend zu
    # run_backtest.py/run_cross_sectional_backtest.py/run_replay.py/run_live.py,
    # die alle `for epoch in range(10)` nutzen) -- WalkForwardConfig() alleine
    # wuerde still auf den Default 5 zurueckfallen.
    cfg = WalkForwardConfig(horizon=1, seed=0, epochs_per_fold=10)

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

    # Staerkere Ausrichtungspruefung als ein reiner Anzahl-Vergleich: zwei
    # Symbole koennten je genau eine Zeile an UNTERSCHIEDLICHEN Positionen
    # verlieren (unterschiedliche interne NaN-Stellen) und damit gleiche
    # Sequenz-ANZAHLEN, aber tatsaechlich divergierende end_idx-Arrays haben
    # -- das wuerde Bar t von Symbol A still mit einem anderen echten
    # Zeitstempel von Symbol B paaren. Deshalb die end_idx-Arrays selbst
    # (Ganzzahl-Positionen in eval_index nach der get_indexer-Umrechnung
    # oben) auf exakte Gleichheit pruefen -- das impliziert automatisch
    # gleiche Anzahlen, macht eine separate Anzahl-Pruefung ueberfluessig.
    reference_end_idx = sequences[UNIVERSE[0]][2]
    for symbol in UNIVERSE[1:]:
        if not np.array_equal(sequences[symbol][2], reference_end_idx):
            raise ValueError(f"{symbol}: end_idx weicht von {UNIVERSE[0]} ab -- Cross-Sectional-Pairing waere falsch")

    n_samples = min(len(sequences[s][0]) for s in UNIVERSE)
    folds = generate_fold_slices(n_samples, cfg)
    if not folds:
        raise ValueError(f"Nicht genug Sequenzen ({n_samples}) fuer train_size={cfg.train_size}+test_size={cfg.test_size}")
    print(f"  {n_samples} gemeinsame Sequenzen, {len(folds)} Folds")

    # Nur Bars auswerten, deren horizon=1-Forward-Return ein ECHTER
    # 1-Minuten-Uebergang ist: eval_index ist die Schnittmenge aller 12
    # Symbole, "naechste Zeile" kann also Luecken/Session-Grenzen
    # ueberspannen -- solche Uebergaenge messen keinen 1-Minuten-Effekt
    # und werden uebersprungen (gezaehlt in der Provenance).
    bar_is_true_1min = one_minute_transition_mask(eval_index)
    end_positions = sequences[UNIVERSE[0]][2]

    results = {
        name: {"rank_ic": [], "gross_spread": [], "turnover": [], "rank_ic_by_fold": []}
        for name in SCORE_VARIANT_NAMES + BASELINE_NAMES
    }
    per_bar_records: dict = {}
    fold_meta = []
    n_bars_skipped_non_1min = 0

    for fold_i, (train_slice, test_slice) in enumerate(folds):
        print(f"\n=== Fold {fold_i+1}/{len(folds)} ===")
        # Purge an der Fold-Grenze: das Target des letzten Trainings-
        # Samples reicht `horizon` Zeilen in den Testbereich hinein.
        purged_train = purge_train_slice(train_slice, cfg.horizon)
        models, k_calibrated, mu_histories = {}, {}, {}
        for symbol in UNIVERSE:
            X, y, _, _ = sequences[symbol]
            X_train, y_train = X[purged_train], y[purged_train]
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
        fold_rank_ics = {name: [] for name in SCORE_VARIANT_NAMES + BASELINE_NAMES}
        end_idx_test_ref = forecasts[UNIVERSE[0]][2]
        fold_n_evaluated = 0
        fold_n_skipped = 0

        for t in range(test_len):
            mus = {s: float(forecasts[s][0].expected_return[t]) for s in UNIVERSE}
            sigmas = {s: float(forecasts[s][0].expected_volatility[t]) for s in UNIVERSE}
            p_ups = {s: float(forecasts[s][0].probability_up[t]) for s in UNIVERSE}
            # mu-Historien IMMER fortschreiben (kausal verfuegbare Information,
            # unabhaengig davon, ob der Bar unten ausgewertet wird) -- sonst
            # bekaeme mu_percentile an uebersprungenen Bars Loecher.
            for symbol in UNIVERSE:
                mu_histories[symbol].append(mus[symbol])

            bar_position = int(end_idx_test_ref[t])
            if not bar_is_true_1min[bar_position]:
                fold_n_skipped += 1
                n_bars_skipped_non_1min += 1
                continue
            fold_n_evaluated += 1
            bar_ts = eval_index[bar_position]

            # build_scaled_features_and_target() liefert die Zielvariable als
            # LOG-Return (log(price).shift(-h) - log(price)), nicht als
            # einfachen Return -- compound_return()/equity_curve() weiter
            # unten (cross_sectional_signal_metrics.py) nehmen aber einfache
            # Returns an (prod(1+r)-1). Deshalb hier per expm1 in einfache
            # Returns umrechnen, BEVOR sie in compute_rank_ic/
            # compute_gross_spread verwendet werden. Rank-IC selbst ist von
            # dieser monotonen Transformation unveraendert (nur die
            # Compounding-Metriken waeren sonst um ca. 8% relativ verzerrt).
            forward_returns = {s: float(np.expm1(forecasts[s][1][t])) for s in UNIVERSE}

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
                    fold_rank_ics[name].append(ic)
                    per_bar_records.setdefault(bar_ts, {})[name] = ic
                if not np.isnan(spread):
                    results[name]["gross_spread"].append(spread)
                results[name]["turnover"].append(turnover)

        for name in SCORE_VARIANT_NAMES + BASELINE_NAMES:
            if fold_rank_ics[name]:
                results[name]["rank_ic_by_fold"].append(float(np.mean(fold_rank_ics[name])))

        train_positions = end_positions[purged_train]
        test_positions = end_positions[test_slice]
        fold_meta.append({
            "fold": fold_i,
            "train_start": str(eval_index[int(train_positions[0])]),
            "train_end": str(eval_index[int(train_positions[-1])]),
            "test_start": str(eval_index[int(test_positions[0])]),
            "test_end": str(eval_index[int(test_positions[-1])]),
            "n_test_bars_evaluated": fold_n_evaluated,
            "n_test_bars_skipped_non_1min": fold_n_skipped,
        })

        print(f"  Fold {fold_i+1} abgeschlossen ({fold_n_evaluated} ausgewertete, {fold_n_skipped} uebersprungene Test-Bars)")

    summary = {}
    for name in SCORE_VARIANT_NAMES + BASELINE_NAMES:
        rank_ics = results[name]["rank_ic"]
        spreads = results[name]["gross_spread"]
        turnovers = results[name]["turnover"]
        rank_ic_by_fold = results[name]["rank_ic_by_fold"]
        summary[name] = {
            "mean_rank_ic": float(np.mean(rank_ics)) if rank_ics else float("nan"),
            "n_observations": len(rank_ics),
            "mean_gross_spread_per_bar": float(np.mean(spreads)) if spreads else float("nan"),
            "compounded_gross_return": compound_return(spreads) if spreads else float("nan"),
            "max_drawdown": max_drawdown_from_returns(spreads) if spreads else float("nan"),
            "breakeven_cost": compute_breakeven_cost(spreads, turnovers) if spreads and turnovers else float("nan"),
            "mean_turnover": float(np.mean(turnovers)) if turnovers else float("nan"),
            # Streuung ueber Folds statt nur der flachen Mittelwert-Kennzahl
            # oben -- die Design-Spec's eigenes Gate fuer den Holdout-Zugriff
            # ("mean Rank-IC positiv UND positiv in der MEHRHEIT der Folds")
            # laesst sich aus mean_rank_ic allein nicht pruefen.
            "rank_ic_std_across_folds": float(np.std(rank_ic_by_fold)) if rank_ic_by_fold else float("nan"),
            "frac_folds_positive": (
                sum(1 for v in rank_ic_by_fold if v > 0) / len(rank_ic_by_fold) if rank_ic_by_fold else float("nan")
            ),
            "n_folds_with_data": len(rank_ic_by_fold),
            # Die einzelnen Fold-Mittel selbst persistieren, nicht nur ihre
            # Aggregate -- sonst ist jede spaetere Inferenz (Bootstrap,
            # Sensitivitaetsanalysen) auf einen Re-Run der ~90min Folds
            # angewiesen.
            "rank_ic_by_fold": [float(v) for v in rank_ic_by_fold],
        }

    # Tages-Block-Bootstrap ueber die per-Bar-Rank-IC-Serien: Inferenz, die
    # Intraday-Abhaengigkeit respektiert, statt des frueheren willkuerlichen
    # 5x-|Random-IC|-Schwellenwerts gegen einen einzelnen Zufallspfad.
    per_bar_frame = pd.DataFrame.from_dict(per_bar_records, orient="index").sort_index()
    per_bar_frame.index.name = "timestamp"
    bootstrap_by_variant = {}
    for name in SCORE_VARIANT_NAMES + BASELINE_NAMES:
        series = per_bar_frame[name].dropna() if name in per_bar_frame.columns else pd.Series(dtype=float)
        if series.empty:
            bootstrap_by_variant[name] = {
                "mean": float("nan"), "ci_low_95": float("nan"), "ci_high_95": float("nan"),
                "p_leq_zero": float("nan"), "n_days": 0, "n_boot": BOOTSTRAP_N_BOOT,
            }
        else:
            boot = day_block_bootstrap(
                series.tolist(), series.index, n_boot=BOOTSTRAP_N_BOOT, seed=BOOTSTRAP_SEED,
            )
            boot.pop("bootstrap_means")  # Rohverteilung nicht ins JSON
            bootstrap_by_variant[name] = boot
        summary[name]["bootstrap"] = bootstrap_by_variant[name]

    verdict = bootstrap_signal_verdict(bootstrap_by_variant, SCORE_VARIANT_NAMES)

    provenance = {
        "universe": UNIVERSE,
        "backtest_lookback_days": BACKTEST_LOOKBACK_DAYS,
        "walkforward_fraction": WALKFORWARD_FRACTION,
        "n_long": N_LONG,
        "n_short": N_SHORT,
        "rolling_percentile_window": ROLLING_PERCENTILE_WINDOW,
        "momentum_lookback_bars": MOMENTUM_LOOKBACK_BARS,
        "cfg_horizon": cfg.horizon,
        "cfg_epochs_per_fold": cfg.epochs_per_fold,
        "cfg_seed": cfg.seed,
        "eval_index_start": str(eval_index.min()),
        "eval_index_end": str(eval_index.max()),
        # Content-Hash pinnt fest, auf welchen BARS gerechnet wurde -- der
        # Parameter-Hash im Dateinamen wuerde nach Loeschen+Neubau des
        # Snapshots auf andere Daten zeigen (build_snapshot() fetcht
        # relativ zu datetime.now()).
        "snapshot_path": snapshot_path(UNIVERSE, BACKTEST_LOOKBACK_DAYS),
        "snapshot_content_sha256": snapshot_content_sha256(dfs),
        "bootstrap_n_boot": BOOTSTRAP_N_BOOT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bars_evaluated": int(len(per_bar_frame)),
        "n_bars_skipped_non_1min": int(n_bars_skipped_non_1min),
    }

    return {
        "use_holdout": use_holdout,
        "n_folds": len(folds),
        "summary": summary,
        "fold_meta": fold_meta,
        "verdict": verdict,
        "provenance": provenance,
        # DataFrame, absichtlich mit Unterstrich: wird vom Aufrufer als CSV
        # gespeichert und VOR json.dump() entfernt (nicht JSON-serialisierbar).
        "_per_bar_frame": per_bar_frame,
    }


def main():
    result = run_diagnostics(use_holdout=False)

    print(f"\n{'='*70}\n=== Zusammenfassung ueber {result['n_folds']} Walk-Forward-Folds ===\n{'='*70}")
    for name, stats in result["summary"].items():
        boot = stats["bootstrap"]
        print(f"{name:>15}: mean_rank_ic={stats['mean_rank_ic']:+.4f}  "
              f"ci95=[{boot['ci_low_95']:+.4f}, {boot['ci_high_95']:+.4f}]  "
              f"p_leq_zero={boot['p_leq_zero']:.3f}  "
              f"n={stats['n_observations']}  "
              f"compounded_gross_return={stats['compounded_gross_return']:+.4%}  "
              f"max_drawdown={stats['max_drawdown']:+.4%}  "
              f"breakeven_cost={stats['breakeven_cost']:+.6f}  "
              f"mean_turnover={stats['mean_turnover']:.4f}  "
              f"rank_ic_std_across_folds={stats['rank_ic_std_across_folds']:.4f}  "
              f"frac_folds_positive={stats['frac_folds_positive']:.2%}  "
              f"n_folds_with_data={stats['n_folds_with_data']}")

    print(f"\n{result['verdict']}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"signal_diagnostics_{run_id}")
    os.makedirs(results_dir, exist_ok=True)

    per_bar_frame = result.pop("_per_bar_frame")
    per_bar_path = os.path.join(results_dir, "per_bar_rank_ic.csv")
    per_bar_frame.to_csv(per_bar_path)
    print(f"\nPer-Bar-Rank-IC gespeichert: {per_bar_path}")

    summary_path = os.path.join(results_dir, "diagnostics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Summary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
