"""
lstm_forecaster_tf.py
======================

TensorFlow/Keras-Pendant zu lstm_forecaster_torch.py -- bewusst mit
identischer Architektur (gleiche Heads, gleiche Loss-Formel), um einen
fairen Framework-Vergleich zu ermoeglichen.

Unterschied zur PyTorch-Version: Keras nutzt ein funktionales Modell mit
einem custom-loss-faehigen Training-Loop via GradientTape, weil die
Kopplung von drei Heads ueber eine gemeinsame Loss-Funktion (NLL + BCE)
in der High-Level model.fit()-API umstaendlicher zu formulieren ist als
mit einem expliziten Loop -- hier wird bewusst dieselbe Kontrolle wie im
PyTorch-Pfad beibehalten.

Einschaetzung fuer dieses System: PyTorch ist der primaere Pfad, weil
Layer 8 (Online Trainer / Model Correction) im Live-Betrieb feingranulare
Kontrolle ueber einzelne Gradient-Updates braucht -- genau das, was der
explizite PyTorch-Loop von Haus aus bietet. Diese TF-Version dient als
Referenz/Vergleich, ist in dieser Umgebung nicht ausgefuehrt (kein
TensorFlow installiert) und strukturell 1:1 zur getesteten PyTorch-Version.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class ForecastOutput:
    expected_return: np.ndarray
    expected_volatility: np.ndarray
    probability_up: np.ndarray


def build_lstm_forecaster(
    timesteps: int,
    n_features: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    sigma_epsilon: float = 1e-4,
) -> keras.Model:
    """
    Baut das funktionale Keras-Modell mit denselben drei Heads wie die
    PyTorch-Version: mu (linear), sigma (softplus + eps), p_up (sigmoid).
    """
    inputs = keras.Input(shape=(timesteps, n_features), name="market_sequence")

    x = inputs
    for i in range(num_layers):
        return_sequences = i < num_layers - 1  # nur letzter Layer gibt (B, hidden) zurueck
        x = layers.LSTM(
            hidden_size,
            return_sequences=return_sequences,
            dropout=dropout if num_layers > 1 else 0.0,
            name=f"lstm_{i}",
        )(x)

    h_last = layers.LayerNormalization(name="norm")(x)

    mu = layers.Dense(1, name="mu_raw")(h_last)
    mu = layers.Reshape((), name="expected_return")(mu)

    sigma_raw = layers.Dense(1, name="sigma_raw")(h_last)
    sigma = layers.Lambda(
        lambda t: tf.nn.softplus(t) + sigma_epsilon, name="softplus_sigma"
    )(sigma_raw)
    sigma = layers.Reshape((), name="expected_volatility")(sigma)

    p_up_raw = layers.Dense(1, activation="sigmoid", name="p_up_raw")(h_last)
    p_up = layers.Reshape((), name="probability_up")(p_up_raw)

    model = keras.Model(inputs=inputs, outputs=[mu, sigma, p_up], name="lstm_forecaster")
    return model


def gaussian_nll_loss_tf(mu: tf.Tensor, sigma: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Identische Formel wie im PyTorch-Pfad: 0.5*log(2*pi*sigma^2) + (y-mu)^2/(2*sigma^2)."""
    var = tf.square(sigma)
    return tf.reduce_mean(
        0.5 * tf.math.log(2 * np.pi * var) + tf.square(y - mu) / (2 * var)
    )


def combined_loss_tf(mu, sigma, p_up, y_return, bce_weight: float = 0.5):
    nll = gaussian_nll_loss_tf(mu, sigma, y_return)
    y_direction = tf.cast(y_return > 0, tf.float32)
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_direction, p_up))
    total = nll + bce_weight * bce
    return total, {"nll": float(nll), "bce": float(bce)}


@tf.function
def train_step(model, optimizer, xb, yb, bce_weight=0.5):
    with tf.GradientTape() as tape:
        mu, sigma, p_up = model(xb, training=True)
        var = tf.square(sigma)
        nll = tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * var) + tf.square(yb - mu) / (2 * var))
        y_direction = tf.cast(yb > 0, tf.float32)
        bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_direction, p_up))
        loss = nll + bce_weight * bce

    grads = tape.gradient(loss, model.trainable_variables)
    # Gradient Clipping -- identisch zur PyTorch-Version, gleicher Grund:
    # Volatilitaets-Spikes in Finanzdaten fuehren sonst zu explodierenden Gradienten.
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_epoch_tf(model, optimizer, X_train, y_train, batch_size=64):
    n = len(X_train)
    idx = np.random.permutation(n)
    X_shuf, y_shuf = X_train[idx], y_train[idx]

    losses = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        xb = tf.constant(X_shuf[start:end], dtype=tf.float32)
        yb = tf.constant(y_shuf[start:end], dtype=tf.float32)
        loss = train_step(model, optimizer, xb, yb)
        losses.append(float(loss))

    return {"mean_loss": float(np.mean(losses))}


def predict_tf(model: keras.Model, X: np.ndarray) -> ForecastOutput:
    mu, sigma, p_up = model(tf.constant(X, dtype=tf.float32), training=False)
    return ForecastOutput(
        expected_return=mu.numpy(),
        expected_volatility=sigma.numpy(),
        probability_up=p_up.numpy(),
    )


# ---------------------------------------------------------------------------
# Demo -- identischer Ablauf wie lstm_forecaster_torch.py, zum Vergleich
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
    from feature_pipeline import FeaturePipeline, _generate_synthetic_market_data
    from sequence_buffer import SequenceWindowBuilder

    tf.random.set_seed(0)
    np.random.seed(0)

    df = _generate_synthetic_market_data(n=3000, seed=11)
    features, target = FeaturePipeline().transform_with_target(df, horizon=5)
    feature_names = list(features.columns)

    builder = SequenceWindowBuilder(timesteps=20, feature_names=feature_names)
    X, y, _ = builder.build(features, target)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_forecaster(timesteps=20, n_features=len(feature_names), hidden_size=32, num_layers=2)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    print("=== Training (5 Epochen, Demo-Umfang) ===")
    for epoch in range(5):
        stats = train_epoch_tf(model, optimizer, X_train, y_train, batch_size=64)
        print(f"Epoch {epoch+1}: mean_loss={stats['mean_loss']:.4f}")

    forecast = predict_tf(model, X_test)
    print("\n=== Test-Set Forecast (erste 5 Samples) ===")
    for i in range(5):
        print(
            f"  y_true={y_test[i]:+.5f}  mu={forecast.expected_return[i]:+.5f}  "
            f"sigma={forecast.expected_volatility[i]:.5f}  p_up={forecast.probability_up[i]:.3f}"
        )
