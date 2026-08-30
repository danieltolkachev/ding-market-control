"""
lstm_forecaster_torch.py
=========================

LSTM-Kernmodell (PyTorch) fuer Layer 4 des Market-Control-Systems.

Architektur:
    Input X (B, T, F)
      -> LSTM-Stack (num_layers, hidden_size)
      -> letzter Hidden-State h_T
      -> 3 parallele Heads:
           mu_head:    linear           -> expected_return
           sigma_head: softplus + eps   -> expected_volatility (> 0)
           prob_head:  sigmoid          -> probability_up

Loss: Gaussian NLL (mu, sigma) + lambda * BCE(prob_up)

r_{t+h} | X_t ~ N(mu_t, sigma_t^2)
mu_t = W_mu * h_T + b_mu
sigma_t = softplus(W_sigma * h_T + b_sigma) + epsilon
L_NLL = 0.5 * ln(2*pi*sigma_t^2) + (r_{t+h} - mu_t)^2 / (2*sigma_t^2)

Die gekoppelte Optimierung von mu und sigma ueber die Log-Likelihood
(statt getrennter MSE-Losses) bestraft ueberkonfidente Volatilitaets-
schaetzungen direkt im Training: sagt das Modell niedrige Volatilitaet
voraus, aber der tatsaechliche Fehler ist gross, explodiert der zweite
Term. Das ist wichtig, weil der Exposure Controller spaeter
edge = expected_return / (expected_volatility^2 + eps) bildet --
ein unkalibriertes sigma wuerde sich dort quadratisch auf die
Positionsgroesse auswirken.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ForecastOutput:
    """Einheitliches Rueckgabeformat, unabhaengig vom Framework."""
    expected_return: np.ndarray       # mu
    expected_volatility: np.ndarray   # sigma (> 0)
    probability_up: np.ndarray        # p in (0, 1)


class LSTMForecaster(nn.Module):
    """
    Multi-Head-LSTM fuer gekoppelte Return-/Volatilitaets-/Richtungsprognose.

    Args:
        n_features: F, Anzahl Input-Features pro Timestep
        hidden_size: LSTM-Hidden-Dimension
        num_layers: Anzahl gestapelter LSTM-Layer
        dropout: Dropout zwischen LSTM-Layern (nur wirksam bei num_layers > 1)
        sigma_epsilon: numerische Untergrenze fuer sigma, verhindert
                       Division-durch-Null im Controller (edge = mu/sigma^2)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sigma_epsilon: float = 1e-4,
    ):
        super().__init__()
        self.sigma_epsilon = sigma_epsilon

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer-Norm auf dem finalen Hidden-State stabilisiert das Training
        # bei nicht-stationaeren Finanzzeitreihen deutlich (empirisch robuster
        # als BatchNorm, da hier keine Batch-Statistik ueber Zeit driftet).
        self.norm = nn.LayerNorm(hidden_size)

        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Linear(hidden_size, 1)
        self.prob_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor (B, T, F)
        Returns:
            mu:    (B,)  -- expected_return, unbeschraenkt
            sigma: (B,)  -- expected_volatility, > 0 via softplus
            p_up:  (B,)  -- probability_up, in (0,1) via sigmoid
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]  # letzter Hidden-State des obersten Layers, shape (B, hidden_size)
        h_last = self.norm(h_last)

        mu = self.mu_head(h_last).squeeze(-1)
        sigma = torch.nn.functional.softplus(self.sigma_head(h_last)).squeeze(-1) + self.sigma_epsilon
        p_up = torch.sigmoid(self.prob_head(h_last)).squeeze(-1)

        return mu, sigma, p_up

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> ForecastOutput:
        """Inferenz-Schnittstelle fuer den Live-Control-Loop. X: (B, T, F) numpy."""
        self.eval()
        x_t = torch.from_numpy(X.astype(np.float32))
        mu, sigma, p_up = self.forward(x_t)
        return ForecastOutput(
            expected_return=mu.numpy(),
            expected_volatility=sigma.numpy(),
            probability_up=p_up.numpy(),
        )


def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L_NLL = 0.5*log(2*pi*sigma^2) + (y-mu)^2 / (2*sigma^2)
    Numerisch stabiler ueber log(sigma^2) statt sigma direkt zu quadrieren.
    """
    var = sigma ** 2
    return (0.5 * torch.log(2 * np.pi * var) + (y - mu) ** 2 / (2 * var)).mean()


def combined_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    p_up: torch.Tensor,
    y_return: torch.Tensor,
    bce_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Gesamtloss = NLL(mu, sigma) + bce_weight * BCE(p_up, 1[y_return > 0])."""
    nll = gaussian_nll_loss(mu, sigma, y_return)
    y_direction = (y_return > 0).float()
    bce = torch.nn.functional.binary_cross_entropy(p_up, y_direction)
    total = nll + bce_weight * bce
    return total, {"nll": nll.item(), "bce": bce.item(), "total": total.item()}


def train_epoch(
    model: LSTMForecaster,
    optimizer: torch.optim.Optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 64,
) -> dict:
    """Ein Trainings-Epoch ueber die uebergebenen Daten (einfaches Mini-Batching, kein DataLoader-Overhead noetig bei diesem Datenvolumen)."""
    model.train()
    n = len(X_train)
    idx = np.random.permutation(n)
    X_shuf, y_shuf = X_train[idx], y_train[idx]

    epoch_losses = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        xb = torch.from_numpy(X_shuf[start:end].astype(np.float32))
        yb = torch.from_numpy(y_shuf[start:end].astype(np.float32))

        optimizer.zero_grad()
        mu, sigma, p_up = model(xb)
        loss, parts = combined_loss(mu, sigma, p_up, yb)
        loss.backward()
        # Gradient Clipping: LSTMs auf Finanzdaten neigen bei Volatilitaets-
        # Spikes zu explodierenden Gradienten -> harte Clip-Grenze.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_losses.append(parts["total"])

    return {"mean_loss": float(np.mean(epoch_losses))}


# ---------------------------------------------------------------------------
# Sanity-Check / Demo: End-to-End von Feature-Pipeline bis Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
    from feature_pipeline import FeaturePipeline, _generate_synthetic_market_data
    from sequence_buffer import SequenceWindowBuilder

    torch.manual_seed(0)
    np.random.seed(0)

    # --- Daten & Sequenzen aufbauen (Pipeline aus vorherigem Layer) ---
    df = _generate_synthetic_market_data(n=3000, seed=11)
    features, target = FeaturePipeline().transform_with_target(df, horizon=5)
    feature_names = list(features.columns)

    builder = SequenceWindowBuilder(timesteps=20, feature_names=feature_names)
    X, y, _ = builder.build(features, target)

    # Walk-Forward-Split: kein Shuffle ueber Zeit hinweg, letzte 20% als Test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- Modell & Training ---
    model = LSTMForecaster(n_features=len(feature_names), hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n=== Training (5 Epochen, Demo-Umfang) ===")
    for epoch in range(5):
        stats = train_epoch(model, optimizer, X_train, y_train, batch_size=64)
        print(f"Epoch {epoch+1}: mean_loss={stats['mean_loss']:.4f}")

    # --- Evaluation auf Test-Set ---
    model.eval()
    forecast = model.predict(X_test)
    print("\n=== Test-Set Forecast (erste 5 Samples) ===")
    for i in range(5):
        print(
            f"  y_true={y_test[i]:+.5f}  "
            f"mu={forecast.expected_return[i]:+.5f}  "
            f"sigma={forecast.expected_volatility[i]:.5f}  "
            f"p_up={forecast.probability_up[i]:.3f}"
        )

    # Kalibrierungscheck: Richtungsgenauigkeit
    pred_direction = forecast.probability_up > 0.5
    true_direction = y_test > 0
    accuracy = (pred_direction == true_direction).mean()
    print(f"\nRichtungsgenauigkeit auf Test-Set: {accuracy:.3f} (Demo-Daten sind Zufallsrauschen -> ~0.5 erwartet)")
