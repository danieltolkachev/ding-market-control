"""
walk_forward.py
================

Walk-Forward-Validation fuer den LSTM-Forecaster (Layer 4).

Ersetzt den statischen 80/20-Split aus der ersten Demo-Version durch
mehrere rollierende Trainings-/Test-Fenster ueber die Zeitachse. Grund:
`edge = expected_return / (expected_volatility^2 + eps)` im spaeteren
Exposure Controller verstaerkt Fehler in sigma quadratisch -- bevor
mu/sigma in eine Positionsgroesse uebersetzt werden, muss klar sein,
ob die Modellguete ueber mehrere Zeitfenster hinweg stabil ist oder in
bestimmten Regimes (z.B. hohe Vol-Cluster) kollabiert.

Methodik: rollierendes Fenster (kein expanding window). Jeder Fold
bekommt ein FRISCH initialisiertes Modell, trainiert NUR auf seinem
Trainingsfenster und wird NUR auf dem direkt folgenden, nicht
ueberlappenden Testfenster evaluiert. Das testet die Pipeline/Architektur
selbst (verallgemeinert sie ueber verschiedene Zeitabschnitte?) -- eine
andere Frage als die des spaeteren Online-Trainers (Layer 8), der ein
einzelnes Modell fortlaufend per Fine-Tuning aktualisiert.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import numpy as np
import torch

from feature_pipeline import FeaturePipeline, _generate_synthetic_market_data
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import (
    LSTMForecaster,
    combined_loss,
    train_epoch,
)


@dataclass
class WalkForwardConfig:
    n_synthetic_bars: int = 6000
    timesteps: int = 20
    horizon: int = 5
    train_size: int = 2000     # Samples pro Trainingsfenster
    test_size: int = 400       # Samples pro (nicht ueberlappendem) Testfenster
    step_size: int = 400       # Vorschub pro Fold; = test_size -> keine Test-Ueberlappung
    epochs_per_fold: int = 5
    batch_size: int = 64
    hidden_size: int = 32
    num_layers: int = 2
    lr: float = 1e-3
    seed: int = 0


def generate_fold_slices(n_samples: int, cfg: WalkForwardConfig) -> list[tuple[slice, slice]]:
    """Erzeugt (train_slice, test_slice)-Paare, die als rollierendes Fenster ueber
    den Sample-Index laufen. Traininsfenster hat feste Groesse (kein expanding
    window) -- das haelt jeden Fold vergleichbar und simuliert, dass ein
    Produktionsmodell auch nur begrenzt weit in die Vergangenheit zurueckschaut."""
    folds = []
    start = 0
    while True:
        train_end = start + cfg.train_size
        test_end = train_end + cfg.test_size
        if test_end > n_samples:
            break
        folds.append((slice(start, train_end), slice(train_end, test_end)))
        start += cfg.step_size
    return folds


def evaluate_fold(model: LSTMForecaster, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Berechnet Loss, Richtungsgenauigkeit und einen einfachen Kalibrierungs-
    Indikator (mittleres praediziertes sigma vs. tatsaechliche Residual-Streuung)
    fuer ein Testfenster."""
    model.eval()
    forecast = model.predict(X_test)

    with torch.no_grad():
        mu_t = torch.from_numpy(forecast.expected_return)
        sigma_t = torch.from_numpy(forecast.expected_volatility)
        p_up_t = torch.from_numpy(forecast.probability_up)
        y_t = torch.from_numpy(y_test.astype(np.float32))
        _, loss_parts = combined_loss(mu_t, sigma_t, p_up_t, y_t)

    residuals = y_test - forecast.expected_return
    pred_direction = forecast.probability_up > 0.5
    true_direction = y_test > 0
    direction_accuracy = float((pred_direction == true_direction).mean())

    return {
        "n_test": len(y_test),
        "mean_loss": loss_parts["total"],
        "nll": loss_parts["nll"],
        "bce": loss_parts["bce"],
        "direction_accuracy": direction_accuracy,
        "mean_predicted_sigma": float(forecast.expected_volatility.mean()),
        "realized_residual_std": float(residuals.std()),
    }


def run_walk_forward(cfg: WalkForwardConfig = WalkForwardConfig()) -> list[dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = _generate_synthetic_market_data(n=cfg.n_synthetic_bars, seed=cfg.seed + 11)
    features, target = FeaturePipeline().transform_with_target(df, horizon=cfg.horizon)
    feature_names = list(features.columns)

    builder = SequenceWindowBuilder(timesteps=cfg.timesteps, feature_names=feature_names)
    X, y, end_idx = builder.build(features, target)

    folds = generate_fold_slices(len(X), cfg)
    if not folds:
        raise ValueError(
            f"Nicht genug Samples ({len(X)}) fuer train_size={cfg.train_size} + "
            f"test_size={cfg.test_size}. n_synthetic_bars erhoehen."
        )

    print(f"Gesamt-Samples: {len(X)}, Folds: {len(folds)} "
          f"(train_size={cfg.train_size}, test_size={cfg.test_size}, step={cfg.step_size})\n")

    results = []
    for i, (train_slice, test_slice) in enumerate(folds):
        X_train, y_train = X[train_slice], y[train_slice]
        X_test, y_test = X[test_slice], y[test_slice]

        model = LSTMForecaster(
            n_features=len(feature_names),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        for _ in range(cfg.epochs_per_fold):
            train_epoch(model, optimizer, X_train, y_train, batch_size=cfg.batch_size)

        metrics = evaluate_fold(model, X_test, y_test)
        metrics["fold"] = i
        metrics["train_range"] = (int(train_slice.start), int(train_slice.stop))
        metrics["test_range"] = (int(test_slice.start), int(test_slice.stop))
        results.append(metrics)

        print(
            f"Fold {i}: train={metrics['train_range']} test={metrics['test_range']}  "
            f"loss={metrics['mean_loss']:.4f}  dir_acc={metrics['direction_accuracy']:.3f}  "
            f"pred_sigma={metrics['mean_predicted_sigma']:.5f}  "
            f"realized_std={metrics['realized_residual_std']:.5f}"
        )

    losses = [r["mean_loss"] for r in results]
    accs = [r["direction_accuracy"] for r in results]
    print("\n=== Zusammenfassung ueber alle Folds ===")
    print(f"mean_loss:           mean={np.mean(losses):.4f}  std={np.std(losses):.4f}  "
          f"min={np.min(losses):.4f}  max={np.max(losses):.4f}")
    print(f"direction_accuracy:  mean={np.mean(accs):.3f}  std={np.std(accs):.3f}  "
          f"min={np.min(accs):.3f}  max={np.max(accs):.3f}")
    print(
        "\nInterpretation: Auf synthetischen Random-Walk-Daten (kein echtes "
        "praediktives Signal) ist eine direction_accuracy nahe 0.5 in JEDEM Fold "
        "das erwartete, gesunde Ergebnis. Eine hohe Streuung (std) zwischen Folds "
        "waere trotzdem ein Warnsignal auf echten Marktdaten: sie zeigt "
        "Regime-Instabilitaet der Pipeline/Architektur, die der spaetere Online-"
        "Trainer (Layer 8) allein nicht ausgleichen kann, wenn sie strukturell ist."
    )
    return results


if __name__ == "__main__":
    run_walk_forward()
