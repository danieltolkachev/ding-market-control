"""
run_live.py
============

Echter Einstiegspunkt fuer den Live-/Paper-Betrieb gegen Alpaca (im
Gegensatz zu control_loop.py's __main__, der auf synthetischen Daten
demonstriert).

Ablauf:
1. Historische Bars+Quotes fuer SYMBOL laden (echte Alpaca-Daten).
2. Offline-Vortraining des LSTM-Forecasters auf diesen echten Daten.
3. k aus den EIGENEN mu/sigma-Vorhersagen des vortrainierten Modells auf
   den echten Daten kalibrieren (calibrate_k) -- nicht die synthetische
   Kalibrierung aus control_loop.py wiederverwenden, die war nur ein
   Anhaltspunkt fuer den Mechanismus, keine Kalibrierungsquelle fuer
   echtes Symbol/Kapital.
4. ControlLoop mit vortrainiertem Modell + kalibriertem k aufbauen.
5. AlpacaLiveMarketDataStream starten -- BLOCKIERT ab hier unbegrenzt,
   bis der Prozess beendet wird (Ctrl+C). Das ist gewolltes Verhalten
   fuer einen Live-Loop, nicht zum automatischen Ausfuehren in einer
   Sandbox/CI gedacht.

Ausfuehren: py -3.12 orchestration/run_live.py
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

import torch

from feature_pipeline import FeaturePipeline, FEATURE_NAMES, build_scaled_features_and_target
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import LSTMForecaster, train_epoch
from exposure_controller import ControllerConfig, calibrate_k
from risk_overlay import RiskOverlayConfig
from online_trainer import OnlineTrainerConfig
from alpaca_client import fetch_historical_market_data, AlpacaLiveMarketDataStream

from control_loop import ControlLoop, ExecutionConfig


SYMBOL = "AAPL"          # MVP-Scope: ein Symbol, siehe README
TIMESTEPS = 20
HORIZON = 5
OFFLINE_LOOKBACK_DAYS = 10  # Kalendertage historischer Daten fuer Vortraining + k-Kalibrierung
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

CSV_HEADER = [
    "timestamp", "price", "mu", "sigma", "p_up",
    "raw_target_position", "target_position",
    "fill_price", "position_delta", "slippage_cost", "realized_return",
    "train_update", "train_n_samples", "train_mean_loss",
]


def build_pretrained_loop() -> ControlLoop:
    print(f"=== Historische Daten laden ({SYMBOL}, letzte {OFFLINE_LOOKBACK_DAYS} Tage) ===")
    end = datetime.now(timezone.utc) - timedelta(minutes=20)  # Free-Tier: 15min verzoegerte Daten
    start = end - timedelta(days=OFFLINE_LOOKBACK_DAYS)
    df = fetch_historical_market_data(SYMBOL, start, end)
    print(f"  {df.shape[0]} Bars geladen ({df.index.min()} bis {df.index.max()})")

    features, target = build_scaled_features_and_target(df, horizon=HORIZON)
    builder = SequenceWindowBuilder(timesteps=TIMESTEPS, feature_names=list(FEATURE_NAMES))
    X, y, _ = builder.build(features, target)
    print(f"  {X.shape[0]} Trainings-Sequenzen (X: {X.shape})")

    print("\n=== Offline-Vortraining ===")
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        stats = train_epoch(model, optimizer, X, y, batch_size=64)
        print(f"  Epoch {epoch+1}: mean_loss={stats['mean_loss']:.4f}")

    print("\n=== k-Kalibrierung auf ECHTEN mu/sigma-Vorhersagen ===")
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
    return loop


def make_bar_handler(loop: ControlLoop, log_path: str):
    """
    Baut den on_bar-Handler UND haelt eine offene CSV-Datei fuer die Dauer
    des Laufs (append-Modus, nach jeder Zeile geflusht -- bei einem Absturz/
    Ausschalten mittendrin gehen so nur die letzten Millisekunden verloren,
    nicht der ganze Lauf). Konsolen-Prints sind bewusst flush=True, damit
    sie auch bei Umleitung in eine Log-Datei sofort sichtbar sind, nicht
    erst gepuffert am Prozessende.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    is_new_file = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if is_new_file:
        writer.writerow(CSV_HEADER)
        log_file.flush()

    def on_bar(raw_event: dict) -> None:
        result = loop.step(raw_event)
        if result is None:
            return  # noch Warm-up-Phase des Live-Feature-Buffers

        print(
            f"[{SYMBOL}] mu={result.mu:+.5f} sigma={result.sigma:.5f} "
            f"target_pos={result.target_position:+.3f} "
            f"fill_price={result.fill.fill_price:.2f} "
            f"realized_return={result.fill.realized_return:+.5f}",
            flush=True,
        )
        if result.train_stats is not None:
            print(f"  [Online-Fine-Tuning] n_samples={result.train_stats['n_samples']} "
                  f"mean_loss={result.train_stats['mean_loss']:.4f}", flush=True)

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            raw_event["price"], result.mu, result.sigma, result.p_up,
            result.raw_target_position, result.target_position,
            result.fill.fill_price, result.fill.position_delta,
            result.fill.slippage_cost, result.fill.realized_return,
            result.train_stats is not None,
            result.train_stats["n_samples"] if result.train_stats else "",
            result.train_stats["mean_loss"] if result.train_stats else "",
        ])
        log_file.flush()

    return on_bar


if __name__ == "__main__":
    loop = build_pretrained_loop()

    log_path = os.path.join(LOG_DIR, f"run_live_{SYMBOL}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv")
    print(f"\n=== Starte Live-/Paper-Stream fuer {SYMBOL} ===")
    print(f"Logging nach: {os.path.abspath(log_path)}")
    print("(Blockiert ab hier -- Ctrl+C zum Beenden)\n")
    stream = AlpacaLiveMarketDataStream(SYMBOL, on_bar=make_bar_handler(loop, log_path))
    stream.run()
