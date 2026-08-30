"""
run_replay.py
==============

Offline-Test des kompletten Regelkreises auf ECHTEN historischen Daten,
OHNE auf die Boersenoeffnung zu warten. Nutzt exakt denselben ControlLoop-
Code wie run_live.py -- der einzige Unterschied ist die Datenquelle:
statt AlpacaLiveMarketDataStream (Websocket, Echtzeit) wird ein bereits
geladenes historisches DataFrame Zeile fuer Zeile durchgereicht. Das laeuft
in Sekunden statt Stunden, weil nicht auf echte Zeit gewartet wird.

Wichtige Design-Entscheidung: Out-of-Sample-Split.
---------------------------------------------------
Der geladene Zeitraum wird in zwei Teile gesplittet:
- PRETRAIN_FRACTION (aeltere Bars): fuer Offline-Vortraining + k-Kalibrierung
- Rest (neuere Bars): fuer den eigentlichen Replay

Waere der Replay-Zeitraum derselbe wie der Trainings-Zeitraum, wuerde man
nur sehen, wie gut sich das Modell an bereits gesehene Daten erinnert --
nicht, wie es sich auf neuen Daten verhaelt. Das ist derselbe Grundsatz
wie beim Walk-Forward-Test (training/walk_forward.py), hier aber End-to-
End durch den kompletten Regelkreis (inkl. Controller, Risk Overlay,
Execution, Online-Training) statt nur durchs Modell.

Ausfuehren: py -3.12 orchestration/run_replay.py
"""

from __future__ import annotations

import sys
import os
import csv
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import numpy as np
import torch

from feature_pipeline import FeaturePipeline, FEATURE_NAMES, build_scaled_features_and_target
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import LSTMForecaster, train_epoch
from exposure_controller import ControllerConfig, calibrate_k
from risk_overlay import RiskOverlayConfig
from online_trainer import OnlineTrainerConfig
from alpaca_client import fetch_historical_market_data

from control_loop import ControlLoop, ExecutionConfig
from run_live import SYMBOL, TIMESTEPS, HORIZON, LOG_DIR, CSV_HEADER


REPLAY_LOOKBACK_DAYS = 20   # Gesamtzeitraum, der geladen wird
PRETRAIN_FRACTION = 0.6     # aelterer Anteil davon fuers Vortraining; Rest = Out-of-Sample-Replay


def build_replay_loop_and_data():
    print(f"=== Historische Daten laden ({SYMBOL}, letzte {REPLAY_LOOKBACK_DAYS} Tage) ===")
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=REPLAY_LOOKBACK_DAYS)
    full_df = fetch_historical_market_data(SYMBOL, start, end)
    print(f"  {full_df.shape[0]} Bars geladen ({full_df.index.min()} bis {full_df.index.max()})")

    split_idx = int(len(full_df) * PRETRAIN_FRACTION)
    pretrain_df = full_df.iloc[:split_idx]
    replay_df = full_df.iloc[split_idx:]
    print(f"  Vortraining: {pretrain_df.shape[0]} Bars ({pretrain_df.index.min()} bis {pretrain_df.index.max()})")
    print(f"  Replay (out-of-sample): {replay_df.shape[0]} Bars "
          f"({replay_df.index.min()} bis {replay_df.index.max()})")

    features, target = build_scaled_features_and_target(pretrain_df, horizon=HORIZON)
    builder = SequenceWindowBuilder(timesteps=TIMESTEPS, feature_names=list(FEATURE_NAMES))
    X, y, _ = builder.build(features, target)
    print(f"  {X.shape[0]} Trainings-Sequenzen (X: {X.shape})")

    print("\n=== Offline-Vortraining (nur auf dem aelteren Teil) ===")
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        stats = train_epoch(model, optimizer, X, y, batch_size=64)
        print(f"  Epoch {epoch+1}: mean_loss={stats['mean_loss']:.4f}")

    print("\n=== k-Kalibrierung ===")
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
        risk_config=RiskOverlayConfig(max_step_change=0.1, max_sigma=0.08, drawdown_limit=-0.03, cooldown_steps=15, min_rebalance_threshold=0.30),
        execution_config=ExecutionConfig(slippage_bps=0.5),
        online_trainer_config=OnlineTrainerConfig(retrain_every=100, batch_size=32, max_training_window=256),
    )
    return loop, replay_df


def run_replay(loop: ControlLoop, replay_df) -> None:
    log_path = os.path.join(
        LOG_DIR, f"run_replay_{SYMBOL}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"\n=== Replay startet ({replay_df.shape[0]} Bars, out-of-sample) ===")
    print(f"Logging nach: {os.path.abspath(log_path)}\n")

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(CSV_HEADER)

        n_active_steps = 0
        n_train_updates = 0
        cumulative_return = 0.0
        equity_curve = []

        for bar_timestamp, row in replay_df.iterrows():
            raw_event = row.to_dict()
            result = loop.step(raw_event)
            if result is None:
                continue

            n_active_steps += 1
            cumulative_return += result.fill.realized_return
            equity_curve.append(cumulative_return)

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
                print(f"  [{bar_timestamp}] Online-Fine-Tuning: "
                      f"n_samples={result.train_stats['n_samples']} "
                      f"mean_loss={result.train_stats['mean_loss']:.4f}")

            if n_active_steps % 200 == 0:
                print(
                    f"[{bar_timestamp}] mu={result.mu:+.5f} sigma={result.sigma:.5f} "
                    f"target_pos={result.target_position:+.3f} "
                    f"cumulative_return={cumulative_return:+.5f}"
                )

    print("\n=== Zusammenfassung ===")
    print(f"Aktive Regelkreis-Schritte (nach Warm-up): {n_active_steps}")
    print(f"Online-Fine-Tuning-Updates:                {n_train_updates}")
    print(f"Kumulierter Return (inkl. Slippage):        {cumulative_return:+.5f}")
    if equity_curve:
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        max_drawdown = float((equity_arr - running_max).min())
        print(f"Maximaler Drawdown (auf kum. Return-Kurve): {max_drawdown:+.5f}")
    print(f"\nCSV-Log: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    loop, replay_df = build_replay_loop_and_data()
    run_replay(loop, replay_df)
