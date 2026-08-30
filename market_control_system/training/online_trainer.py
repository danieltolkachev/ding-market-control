"""
online_trainer.py
==================

Layer 8 des Market-Control-Systems: periodisches Fine-Tuning des LSTM-
Forecasters aus den juengsten (X, y)-Paaren im FeedbackBuffer.

MVP-Entscheidung (siehe README): periodisches Retrain alle N Bars statt
fehler-getriggertem/adaptivem Retraining -- einfacher, vorhersehbar,
gut nachvollziehbar in Logs. Adaptive Trigger (z.B. bei ueberschrittenem
gleitenden Prognosefehler) sind eine spaetere Erweiterung.

Wichtig: eigener (kleinerer) Optimizer/Learning-Rate als das initiale
Offline-Training. Ziel ist sanftes Nachjustieren auf neue Daten, nicht
das Ueberschreiben der Offline-gelernten Gewichte durch wenige, ggf.
verrauschte Live-Datenpunkte.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import torch

from lstm_forecaster_torch import LSTMForecaster, train_epoch


@dataclass
class OnlineTrainerConfig:
    retrain_every: int = 60       # Bars zwischen zwei Fine-Tuning-Schritten
    epochs_per_update: int = 1    # wenige Gradientenschritte, kein Full-Retrain
    batch_size: int = 64
    max_training_window: int = 512  # nur die juengsten N Samples aus dem FeedbackBuffer nutzen
    lr: float = 1e-4              # bewusst kleiner als beim Offline-Training (1e-3)


class OnlineTrainer:
    """Haelt einen eigenen Optimizer-Zustand fuer das uebergebene Modell und
    entscheidet selbst, wann (Bar-Zaehler) ein Fine-Tuning-Schritt faellig ist.
    Die Datenverfuegbarkeit (genug Samples?) prueft der FeedbackBuffer."""

    def __init__(self, model: LSTMForecaster, config: OnlineTrainerConfig = OnlineTrainerConfig()):
        self.model = model
        self.cfg = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self._steps_since_last_update = 0

    def maybe_update(self, feedback_buffer) -> dict | None:
        """Vom control_loop einmal pro Bar aufgerufen. Gibt Trainings-Stats
        zurueck, wenn ein Update ausgefuehrt wurde, sonst None."""
        self._steps_since_last_update += 1
        if self._steps_since_last_update < self.cfg.retrain_every:
            return None
        if not feedback_buffer.is_ready_for_training():
            return None

        self._steps_since_last_update = 0
        X, y = feedback_buffer.get_training_batch(batch_size=self.cfg.max_training_window)

        stats = None
        for _ in range(self.cfg.epochs_per_update):
            stats = train_epoch(self.model, self.optimizer, X, y, batch_size=self.cfg.batch_size)
        stats["n_samples"] = len(X)
        return stats


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feedback"))
    from feedback_buffer import FeedbackBuffer

    torch.manual_seed(0)
    np.random.seed(0)

    n_features = 7
    timesteps = 20
    model = LSTMForecaster(n_features=n_features, hidden_size=16, num_layers=1)
    trainer = OnlineTrainer(model, OnlineTrainerConfig(retrain_every=5, batch_size=8, max_training_window=100))

    buf = FeedbackBuffer(horizon=1, min_batch_size=8)
    price = 100.0
    print("=== OnlineTrainer Sanity-Check (retrain_every=5) ===")
    for bar in range(30):
        price *= 1 + np.random.normal(0, 0.001)
        buf.resolve(price)
        window = np.random.normal(0, 1, size=(timesteps, n_features)).astype(np.float32)
        buf.record_window(window, entry_price=price)
        buf.tick()

        stats = trainer.maybe_update(buf)
        if stats is not None:
            print(f"  Bar {bar}: Fine-Tuning-Update ausgefuehrt, "
                  f"n_samples={stats['n_samples']}, mean_loss={stats['mean_loss']:.4f}")
