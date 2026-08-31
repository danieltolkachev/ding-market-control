"""
cross_sectional_fold_training.py
===================================

Wiederverwendbare Trainings-Bausteine fuer die Cross-Sectional-Walk-
Forward-Diagnostik (siehe Design-Spec docs/superpowers/specs/2026-08-31-
cross-sectional-signal-diagnostics-design.md). Nutzt dieselbe Fold-
Erzeugung wie training/walk_forward.py (generate_fold_slices, dort
unveraendert) -- neu ist hier nur, dass Training/Vorhersage als
wiederverwendbare Funktionen statt inline in einem Skript vorliegen,
weil run_cross_sectional_signal_diagnostics.py sie fuer 12 Symbole
gleichzeitig pro Fold aufrufen muss.

KEIN Online-Training hier -- jedes Fold trainiert ein frisches Modell
und wertet es eingefroren aus, exakt wie training/walk_forward.py es
fuer ein Symbol tut, hier verallgemeinert auf viele Symbole.
"""
from __future__ import annotations

import numpy as np
import torch

from feature_pipeline import FEATURE_NAMES, build_scaled_features_and_target
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import LSTMForecaster, train_epoch
from walk_forward import WalkForwardConfig


def build_symbol_sequences(df, cfg: WalkForwardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baut skalierte Features + Zielvariable (horizon=cfg.horizon, siehe
    Global Constraints: hier IMMER 1, nicht der sonst im Projekt uebliche
    Default 5) und daraus Sequenz-Fenster fuer EIN Symbol."""
    features, target = build_scaled_features_and_target(df, horizon=cfg.horizon)
    builder = SequenceWindowBuilder(timesteps=cfg.timesteps, feature_names=list(FEATURE_NAMES))
    X, y, end_idx = builder.build(features, target)
    return X, y, end_idx


def train_fold_model(X_train: np.ndarray, y_train: np.ndarray, cfg: WalkForwardConfig) -> LSTMForecaster:
    """Trainiert ein FRISCHES Modell auf dem Trainingsfenster eines Folds.
    Gibt das Modell im eval()-Modus zurueck (kein weiteres Training danach
    -- Vorhersagen im Testfenster sind eingefroren)."""
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs_per_fold):
        train_epoch(model, optimizer, X_train, y_train, batch_size=cfg.batch_size)
    model.eval()
    return model
