"""
run_backtest.py
================

Langzeit-Multi-Symbol-Backtest ueber ~1 Jahr echte historische Daten, mit
statistischer Auswertung ueber viele unabhaengige Wochen statt einer
einzelnen kumulierten Zahl (siehe training/backtest_stats.py).

Nutzt data_layer.alpaca_client.fetch_historical_bars_approximate() statt
der quote-genauen fetch_historical_market_data() -- bei diesem Umfang
(1 Jahr x mehrere Symbole) ist echte Quote-Historie nicht mehr praktikabel
(siehe Docstring dort: >200 Mio. Zeilen/Symbol/Jahr, mehrere Stunden allein
zum Laden). Fuer Live-Betrieb bleibt der quote-genaue Pfad die richtige
Wahl -- dieser Backtest beantwortet eine andere Frage ("hat der Ansatz
ueber viele Wochen/Monate/Symbole hinweg irgendein Signal, das ueber
Rauschen hinausgeht?"), nicht "wie praezise ist ein einzelner Live-Tick".

Checkpointing: Ergebnisse (CSV + JSON-Summary) werden PRO SYMBOL sofort
nach Abschluss auf Platte geschrieben. Bricht der Prozess z.B. bei Symbol 3
von 3 ab, bleiben die Ergebnisse der vorherigen Symbole erhalten und das
kombinierte Portfolio-Summary wird trotzdem (aus den verfuegbaren Symbolen)
berechnet.

Ausfuehren: py -3.12 orchestration/run_backtest.py
"""

from __future__ import annotations

import sys
import os
import csv
import json
import traceback
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import numpy as np
import pandas as pd
import torch

from feature_pipeline import FeaturePipeline, FEATURE_NAMES, build_scaled_features_and_target
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import LSTMForecaster, train_epoch
from exposure_controller import ControllerConfig, calibrate_k
from risk_overlay import RiskOverlayConfig
from online_trainer import OnlineTrainerConfig
from alpaca_client import fetch_historical_bars_approximate
from backtest_stats import compute_period_statistics, summarize_period_returns, format_statistics_report

from control_loop import ControlLoop, ExecutionConfig
from run_live import TIMESTEPS, HORIZON, CSV_HEADER


SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
BACKTEST_LOOKBACK_DAYS = 365
PRETRAIN_FRACTION = 0.25    # ~3 Monate Vortraining, ~9 Monate Out-of-Sample-Replay
CAPITAL = 10_000.0          # nur fuer die $-Darstellung im Report, geht nicht in die Logik ein


def default_risk_config() -> RiskOverlayConfig:
    return RiskOverlayConfig(max_step_change=0.1, max_sigma=0.08, drawdown_limit=-0.03, cooldown_steps=15, min_rebalance_threshold=0.30)


def default_online_trainer_config() -> OnlineTrainerConfig:
    return OnlineTrainerConfig(retrain_every=100, batch_size=32, max_training_window=256)


def run_symbol_backtest(
    symbol: str,
    results_dir: str,
    seed: int | None = None,
    risk_config: RiskOverlayConfig | None = None,
    online_trainer_config: OnlineTrainerConfig | None = None,
    scale_features: bool = True,
) -> dict:
    """Fuehrt den kompletten Vortrainings- + Out-of-Sample-Replay-Zyklus fuer
    EIN Symbol aus. Schreibt CSV (alle Einzelschritte) + JSON-Summary sofort
    nach Abschluss. Gibt ein Summary-dict zurueck (auch fuer die Portfolio-
    Aggregation in main()/build_portfolio_summary() genutzt).

    seed/risk_config/online_trainer_config/scale_features sind optional und
    fallen auf das bisherige (hardcodierte) Verhalten zurueck, wenn nicht
    gesetzt -- damit bleibt der Standardlauf (main()) unveraendert. Sie
    existieren, damit run_multi_seed_comparison.py denselben Lauf mit
    unterschiedlichen Seeds und/oder Configs starten kann, ohne diese
    Funktion zu duplizieren (siehe Docstring dort: einzelne Full-Year-Laeufe
    streuen zu stark, um Config-Aenderungen daran zu beurteilen).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    risk_config = risk_config if risk_config is not None else default_risk_config()
    online_trainer_config = online_trainer_config if online_trainer_config is not None else default_online_trainer_config()

    print(f"\n{'='*70}\n=== {symbol}: Historische Daten laden ({BACKTEST_LOOKBACK_DAYS} Tage) ===\n{'='*70}")
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=BACKTEST_LOOKBACK_DAYS)
    full_df = fetch_historical_bars_approximate(symbol, start, end)
    print(f"  {full_df.shape[0]} Bars geladen ({full_df.index.min()} bis {full_df.index.max()})")

    split_idx = int(len(full_df) * PRETRAIN_FRACTION)
    pretrain_df = full_df.iloc[:split_idx]
    replay_df = full_df.iloc[split_idx:]
    print(f"  Vortraining: {pretrain_df.shape[0]} Bars, "
          f"Replay (out-of-sample): {replay_df.shape[0]} Bars")

    if scale_features:
        features, target = build_scaled_features_and_target(pretrain_df, horizon=HORIZON)
    else:
        features, target = FeaturePipeline().transform_with_target(pretrain_df, horizon=HORIZON)
    builder = SequenceWindowBuilder(timesteps=TIMESTEPS, feature_names=list(FEATURE_NAMES))
    X, y, _ = builder.build(features, target)
    print(f"  {X.shape[0]} Trainings-Sequenzen")

    print(f"\n=== {symbol}: Offline-Vortraining ===")
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        stats = train_epoch(model, optimizer, X, y, batch_size=64)
        print(f"  Epoch {epoch+1}: mean_loss={stats['mean_loss']:.4f}")

    model.eval()
    forecast = model.predict(X)
    k_calibrated = calibrate_k(
        forecast.expected_return, forecast.expected_volatility,
        max_position=1.0, target_utilization=0.5, percentile=95.0,
    )
    print(f"  Kalibriertes k: {k_calibrated:.5f}")

    loop = ControlLoop(
        timesteps=TIMESTEPS,
        model=model,
        horizon=HORIZON,
        controller_config=ControllerConfig(k=k_calibrated, max_position=1.0),
        risk_config=risk_config,
        execution_config=ExecutionConfig(slippage_bps=0.5),
        online_trainer_config=online_trainer_config,
        scale_features=scale_features,
    )

    csv_path = os.path.join(results_dir, f"{symbol}.csv")
    print(f"\n=== {symbol}: Replay startet ({replay_df.shape[0]} Bars, out-of-sample) ===")
    print(f"  Logging nach: {csv_path}")

    timestamps_list = []
    returns_list = []
    n_active_steps = 0
    n_train_updates = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(CSV_HEADER)

        for i, (bar_timestamp, row) in enumerate(replay_df.iterrows()):
            raw_event = row.to_dict()
            result = loop.step(raw_event)
            if result is None:
                continue

            n_active_steps += 1
            timestamps_list.append(bar_timestamp)
            returns_list.append(result.fill.realized_return)

            writer.writerow([
                bar_timestamp.isoformat(),
                raw_event["price"], result.mu, result.sigma, result.p_up,
                result.raw_target_position, result.target_position,
                result.fill.fill_price, result.fill.position_delta,
                result.fill.slippage_cost, result.fill.realized_return,
                result.train_stats is not None,
                result.train_stats["n_samples"] if result.train_stats else "",
                result.train_stats["mean_loss"] if result.train_stats else "",
            ])

            if result.train_stats is not None:
                n_train_updates += 1

            if n_active_steps % 5000 == 0:
                print(f"  [{symbol}] {n_active_steps}/{len(replay_df)} Bars verarbeitet "
                      f"({bar_timestamp.date()})")

    period_stats = compute_period_statistics(
        pd.DatetimeIndex(timestamps_list), np.array(returns_list), period="W"
    )

    summary = {
        "symbol": symbol,
        "n_bars_total": int(full_df.shape[0]),
        "n_bars_pretrain": int(pretrain_df.shape[0]),
        "n_bars_replay": int(replay_df.shape[0]),
        "n_active_steps": n_active_steps,
        "n_train_updates": n_train_updates,
        "k_calibrated": k_calibrated,
        "n_periods": period_stats.n_periods,
        "mean_period_return": period_stats.mean_return,
        "std_period_return": period_stats.std_return,
        "t_statistic": period_stats.t_statistic,
        "sharpe_like": period_stats.sharpe_like,
        "win_rate": period_stats.win_rate,
        "cumulative_return": period_stats.cumulative_return,
        "max_drawdown": period_stats.max_drawdown,
        "period_returns": period_stats.period_returns.tolist(),
        "period_timestamps": [str(t) for t in
                               pd.Series(returns_list, index=pd.DatetimeIndex(timestamps_list))
                               .resample("W").sum().index],
    }

    summary_path = os.path.join(results_dir, f"{symbol}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {symbol}: Ergebnis ===")
    print(format_statistics_report(period_stats, period_label="Woche", capital=CAPITAL))
    print(f"Summary gespeichert: {summary_path}")

    return summary


def build_portfolio_summary(summaries: dict, run_id: str, results_dir: str, capital: float = CAPITAL) -> dict | None:
    """Kombiniert Pro-Symbol-Summaries (wie von run_symbol_backtest()
    zurueckgegeben) zu einem gleichgewichteten Portfolio-Summary, schreibt
    es nach results_dir/portfolio_summary.json und gibt es zurueck.

    Ausgelagert aus main(), damit run_multi_seed_comparison.py dieselbe
    Aggregation pro (Variante, Seed) wiederverwenden kann, statt sie zu
    duplizieren. Gibt None zurueck, wenn summaries leer ist (kein Symbol
    erfolgreich durchgelaufen)."""
    if not summaries:
        print("\nKein Symbol erfolgreich durchgelaufen -- kein Portfolio-Summary moeglich.")
        return None

    print(f"\n{'='*70}\n=== Kombiniertes Portfolio ({', '.join(summaries.keys())}, gleichgewichtet) ===\n{'='*70}")

    per_symbol_weekly = {}
    for symbol, summary in summaries.items():
        idx = pd.to_datetime(summary["period_timestamps"])
        per_symbol_weekly[symbol] = pd.Series(summary["period_returns"], index=idx)

    portfolio_df = pd.DataFrame(per_symbol_weekly)
    # Gleichgewichtet: Durchschnitt ueber die Symbole, die in dieser Woche
    # Daten hatten (NaN-robust via .mean(), skipna=True per Default)
    portfolio_weekly = portfolio_df.mean(axis=1).dropna()

    portfolio_stats = summarize_period_returns(portfolio_weekly)
    print(format_statistics_report(portfolio_stats, period_label="Woche", capital=capital))

    portfolio_summary = {
        "run_id": run_id,
        "symbols": list(summaries.keys()),
        "capital_per_symbol": capital / len(summaries),
        "n_periods": portfolio_stats.n_periods,
        "mean_period_return": portfolio_stats.mean_return,
        "std_period_return": portfolio_stats.std_return,
        "t_statistic": portfolio_stats.t_statistic,
        "sharpe_like": portfolio_stats.sharpe_like,
        "win_rate": portfolio_stats.win_rate,
        "cumulative_return": portfolio_stats.cumulative_return,
        "max_drawdown": portfolio_stats.max_drawdown,
        "per_symbol": {s: {k: v for k, v in summ.items() if k not in ("period_returns", "period_timestamps")}
                       for s, summ in summaries.items()},
    }
    portfolio_path = os.path.join(results_dir, "portfolio_summary.json")
    with open(portfolio_path, "w", encoding="utf-8") as f:
        json.dump(portfolio_summary, f, indent=2)
    print(f"\nPortfolio-Summary gespeichert: {portfolio_path}")
    print(f"Alle Ergebnisse in: {os.path.abspath(results_dir)}")
    return portfolio_summary


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"backtest_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Backtest-Run-ID: {run_id}")
    print(f"Ergebnisse werden nach {os.path.abspath(results_dir)} geschrieben (pro Symbol sofort, "
          f"nicht erst am Ende -- ueberlebt einen Absturz mitten im Lauf).")

    summaries = {}
    for symbol in SYMBOLS:
        try:
            summaries[symbol] = run_symbol_backtest(symbol, results_dir)
        except Exception:
            print(f"\n!!! FEHLER bei Symbol {symbol}, wird uebersprungen: !!!")
            traceback.print_exc()
            continue

    build_portfolio_summary(summaries, run_id, results_dir)


if __name__ == "__main__":
    main()
