"""
symbol_forecaster.py
======================

Layer 2-4 + 7-8 des Regelkreises (Feature-Engine -> Sequence-Buffer ->
LSTM-Prognose -> Feedback -> Online-Korrektur) fuer EIN Symbol, OHNE
Positions-/Risk-/Ausfuehrungslogik (Layer 5-6) -- die uebernimmt beim
Cross-Sectional-Portfolio die Ranking-Ebene (controller/cross_sectional_
portfolio.py) stattdessen, nicht ein Einzelsymbol-Controller.

Bewusst eine eigene, schlanke Klasse statt ControlLoop mit einem
"Positions-Override"-Parameter zu verbiegen: ControlLoop ist bereits
production-validiert (mehrere Backtest-Runs, siehe Projekt-Notizen) --
diese Klasse dupliziert nur den Forecast+Online-Training-Teil, statt das
bestehende, funktionierende Single-Symbol-System anzufassen. Die
Konsistenz beider Pfade (identische mu/sigma/train_stats-Sequenz bei
identischem Modell/Config/Daten) wird in
tests/test_symbol_forecaster_consistency.py explizit geprueft.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feedback"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from feature_pipeline import FeatureConfig, LiveFeatureEngine, FEATURE_NAMES, IncrementalZScoreScaler
from sequence_buffer import SequenceBuffer
from lstm_forecaster_torch import LSTMForecaster
from feedback_buffer import FeedbackBuffer
from online_trainer import OnlineTrainer, OnlineTrainerConfig


@dataclass
class ForecastResult:
    mu: float
    sigma: float
    p_up: float
    train_stats: dict | None


class SymbolForecaster:
    """Wie ControlLoop, aber ohne ExposureController/RiskOverlay/
    PaperExecutionEngine -- reine Prognose + Selbstlern-Schleife."""

    def __init__(
        self,
        timesteps: int,
        model: LSTMForecaster,
        feature_config: FeatureConfig = FeatureConfig(),
        horizon: int = 5,
        feedback_min_batch_size: int = 64,
        online_trainer_config: OnlineTrainerConfig = OnlineTrainerConfig(),
        feature_names: list[str] | None = None,
        scale_features: bool = True,
        zscore_window: int = 100,
    ):
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        self.extra_feature_names = [n for n in self.feature_names if n not in FEATURE_NAMES]
        self.scaler = IncrementalZScoreScaler(self.feature_names, window=zscore_window) if scale_features else None

        self.feature_engine = LiveFeatureEngine(feature_config)
        self.sequence_buffer = SequenceBuffer(timesteps=timesteps, feature_names=self.feature_names)
        self.model = model
        self.feedback_buffer = FeedbackBuffer(horizon=horizon, min_batch_size=feedback_min_batch_size)
        self.online_trainer = OnlineTrainer(model, online_trainer_config)

    def step(self, raw_event: dict) -> ForecastResult | None:
        price = raw_event["price"]

        # Muss vor record_window() fuer denselben Bar laufen -- siehe
        # feedback_buffer.py Docstring.
        self.feedback_buffer.resolve(price)

        result = None
        features = self.feature_engine.update(raw_event)
        if features is not None:
            for name in self.extra_feature_names:
                features[name] = raw_event[name]
            if self.scaler is not None:
                features = self.scaler.transform(features)
        if features is not None:
            self.sequence_buffer.push(features)
            if self.sequence_buffer.is_ready():
                X = self.sequence_buffer.get_batch_window()
                forecast = self.model.predict(X)
                mu = float(forecast.expected_return[0])
                sigma = float(forecast.expected_volatility[0])
                p_up = float(forecast.probability_up[0])

                self.feedback_buffer.record_window(X[0], entry_price=price)
                train_stats = self.online_trainer.maybe_update(self.feedback_buffer)

                result = ForecastResult(mu=mu, sigma=sigma, p_up=p_up, train_stats=train_stats)

        self.feedback_buffer.tick()
        return result
