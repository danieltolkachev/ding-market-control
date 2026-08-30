"""
run_market_relative_comparison.py
====================================

Vergleicht zwei identische Backtests auf demselben Symbol/Zeitraum/Seed:
7 Basis-Features vs. 8 Features inkl. market_relative_return (siehe
data_layer/market_relative.py). Gleiche Methodik wie
run_news_comparison.py -- erst validieren, bevor es in den grossen
Mehr-Symbol-1-Jahres-Backtest uebernommen wird.

Ausfuehren: py -3.12 orchestration/run_market_relative_comparison.py
"""

from __future__ import annotations

import sys
import os
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

from feature_pipeline import FeaturePipeline, FEATURE_NAMES, MARKET_RELATIVE_FEATURE_NAMES_SET, RollingZScoreScaler
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import LSTMForecaster, train_epoch
from exposure_controller import ControllerConfig, calibrate_k
from risk_overlay import RiskOverlayConfig
from online_trainer import OnlineTrainerConfig
from alpaca_client import fetch_historical_bars_approximate
from market_relative import compute_market_relative_features
from backtest_stats import compute_period_statistics, format_statistics_report

from control_loop import ControlLoop, ExecutionConfig


SYMBOL = "AAPL"
REFERENCE_SYMBOL = "SPY"
COMPARISON_LOOKBACK_DAYS = 90
PRETRAIN_FRACTION = 0.4
TIMESTEPS = 20
HORIZON = 5
SEED = 42


def run_variant(label: str, feature_names: tuple[str, ...], full_df: pd.DataFrame) -> None:
    print(f"\n{'='*70}\n=== Variante: {label} ({len(feature_names)} Features) ===\n{'='*70}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    split_idx = int(len(full_df) * PRETRAIN_FRACTION)
    pretrain_df = full_df.iloc[:split_idx]
    replay_df = full_df.iloc[split_idx:]

    features, target = FeaturePipeline().transform_with_target(pretrain_df, horizon=HORIZON)
    for name in feature_names:
        if name not in features.columns:
            features[name] = pretrain_df[name]
    # Skalierung ERST NACH dem Zusammenfuehren aller Features (Basis + Zusatz)
    # anwenden, damit auch das Zusatz-Feature dieselbe Rolling-Z-Score-
    # Behandlung bekommt wie die Basis-Features (siehe IncrementalZScoreScaler
    # -Docstring in feature_pipeline.py fuer den Bugfix-Kontext).
    features = RollingZScoreScaler(window=100).transform(features)

    builder = SequenceWindowBuilder(timesteps=TIMESTEPS, feature_names=list(feature_names))
    X, y, _ = builder.build(features, target)
    print(f"  {X.shape[0]} Trainings-Sequenzen ({X.shape})")

    model = LSTMForecaster(n_features=len(feature_names), hidden_size=32, num_layers=2)
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
        risk_config=RiskOverlayConfig(max_step_change=0.1, max_sigma=0.08, drawdown_limit=-0.03, cooldown_steps=15, min_rebalance_threshold=0.15),
        execution_config=ExecutionConfig(slippage_bps=0.5),
        online_trainer_config=OnlineTrainerConfig(retrain_every=100, batch_size=32, max_training_window=256),
        feature_names=list(feature_names),
    )

    timestamps_list, returns_list = [], []
    for bar_timestamp, row in replay_df.iterrows():
        result = loop.step(row.to_dict())
        if result is None:
            continue
        timestamps_list.append(bar_timestamp)
        returns_list.append(result.fill.realized_return)

    period_stats = compute_period_statistics(pd.DatetimeIndex(timestamps_list), np.array(returns_list), period="W")
    print(f"\n--- {label}: Ergebnis ---")
    print(format_statistics_report(period_stats, period_label="Woche", capital=10_000.0))


if __name__ == "__main__":
    print(f"=== Historische Bars laden ({SYMBOL} + Referenz {REFERENCE_SYMBOL}, {COMPARISON_LOOKBACK_DAYS} Tage) ===")
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=COMPARISON_LOOKBACK_DAYS)
    bars_df = fetch_historical_bars_approximate(SYMBOL, start, end)
    reference_df = fetch_historical_bars_approximate(REFERENCE_SYMBOL, start, end)
    print(f"  {SYMBOL}: {bars_df.shape[0]} Bars, {REFERENCE_SYMBOL}: {reference_df.shape[0]} Bars")

    rel_features = compute_market_relative_features(bars_df, reference_df)
    full_df = pd.concat([bars_df, rel_features], axis=1)

    run_variant("Basis (7 Features)", FEATURE_NAMES, full_df)
    run_variant("Erweitert (8 Features inkl. market_relative_return)", MARKET_RELATIVE_FEATURE_NAMES_SET, full_df)
