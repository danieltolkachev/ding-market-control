"""
control_loop.py
================

Verdrahtet den kompletten Regelkreis:

    Messen -> Sequenzdaten aufbauen -> LSTM-Prognose -> Controller-Entscheidung
       -> Paper-Ausfuehrung -> Feedback -> Online-Korrektur
       ^_____________________________________________________________________|

Diese Version laeuft auf synthetischen Streaming-Daten (dieselbe
_generate_synthetic_market_data-Funktion wie ueberall sonst im Projekt),
Bar fuer Bar, so als kaeme sie von einer echten Datenquelle. Das ist
bewusst so gebaut, dass NUR data_stream.py (Alpaca-Anbindung, noch nicht
gebaut) ausgetauscht werden muss, um von synthetisch auf echte Live-/
Paper-Daten umzuschalten -- ControlLoop.step() ist bereits jetzt
unabhaengig von der Datenquelle (nimmt ein generisches raw_event-dict).
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "execution"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feedback"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import numpy as np

from feature_pipeline import FeatureConfig, LiveFeatureEngine, FEATURE_NAMES, IncrementalZScoreScaler, _generate_synthetic_market_data
from sequence_buffer import SequenceBuffer
from lstm_forecaster_torch import LSTMForecaster
from exposure_controller import ExposureController, ControllerConfig, calibrate_k
from risk_overlay import RiskOverlay, RiskOverlayConfig
from paper_execution import PaperExecutionEngine, ExecutionConfig, Fill
from feedback_buffer import FeedbackBuffer
from online_trainer import OnlineTrainer, OnlineTrainerConfig


@dataclass
class StepResult:
    mu: float
    sigma: float
    p_up: float
    raw_target_position: float
    target_position: float
    fill: Fill
    train_stats: dict | None


class ControlLoop:
    """Orchestriert Layer 2-8 des Systems fuer EIN Symbol pro Instanz."""

    def __init__(
        self,
        timesteps: int,
        model: LSTMForecaster,
        feature_config: FeatureConfig = FeatureConfig(),
        controller_config: ControllerConfig = ControllerConfig(),
        risk_config: RiskOverlayConfig = RiskOverlayConfig(),
        execution_config: ExecutionConfig = ExecutionConfig(),
        horizon: int = 5,
        feedback_min_batch_size: int = 64,
        online_trainer_config: OnlineTrainerConfig = OnlineTrainerConfig(),
        recalibrate_k_on_retrain: bool = True,
        recalibrate_k_target_utilization: float = 0.5,
        recalibrate_k_percentile: float = 95.0,
        feature_names: list[str] | None = None,
        scale_features: bool = True,
        zscore_window: int = 100,
    ):
        # feature_names: komplette Feature-Liste inkl. optionaler Zusatz-
        # Features (z.B. news_intensity/news_sentiment, siehe
        # feature_pipeline.EXTENDED_FEATURE_NAMES). Default = die 7
        # Basis-Features (rueckwaertskompatibel). Alles ueber die 7 Basis-
        # Features hinaus MUSS bereits als Schluessel im raw_event-dict
        # vorliegen (siehe step()) -- LiveFeatureEngine berechnet nur die
        # Basis-Features, Zusatz-Features kommen von aussen (z.B. vorab in
        # der Backtest-DataFrame berechnet, oder von einer separaten Live-
        # Engine wie data_layer.news_client.LiveNewsFeatureEngine).
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        self.extra_feature_names = [n for n in self.feature_names if n not in FEATURE_NAMES]
        # scale_features: kausales Rolling-Z-Score-Scaling ueber ALLE
        # Features (Basis + Zusatz) hinweg, siehe feature_pipeline.
        # IncrementalZScoreScaler-Docstring fuer den Bugfix-Kontext (fehlende
        # Skalierung verzerrte fruehere Feature-Vergleiche systematisch
        # zuungunsten neuer Features). Default True.
        self.scaler = IncrementalZScoreScaler(self.feature_names, window=zscore_window) if scale_features else None

        self.feature_engine = LiveFeatureEngine(feature_config)
        self.sequence_buffer = SequenceBuffer(timesteps=timesteps, feature_names=self.feature_names)
        self.model = model
        self.controller = ExposureController(controller_config)
        self.risk_overlay = RiskOverlay(risk_config)
        self.execution = PaperExecutionEngine(execution_config)
        self.feedback_buffer = FeedbackBuffer(horizon=horizon, min_batch_size=feedback_min_batch_size)
        self.online_trainer = OnlineTrainer(model, online_trainer_config)
        self.recalibrate_k_on_retrain = recalibrate_k_on_retrain
        self.recalibrate_k_target_utilization = recalibrate_k_target_utilization
        self.recalibrate_k_percentile = recalibrate_k_percentile

    def step(self, raw_event: dict) -> StepResult | None:
        """
        Args:
            raw_event: dict mit price, volume, bid, ask, bid_volume, ask_volume, trade_count
                (Format wie von _generate_synthetic_market_data-Zeilen bzw.
                spaeter vom Alpaca-Stream-Adapter).

        Returns:
            StepResult, sobald der SequenceBuffer voll ist (nach der Warm-up-
            Phase), sonst None.
        """
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

                raw_target = self.controller.compute_target_position(mu, sigma)
                target_position = self.risk_overlay.apply(raw_target, expected_volatility=sigma)

                fill = self.execution.execute(target_position, raw_event)
                self.risk_overlay.record_realized_return(fill.realized_return)

                self.feedback_buffer.record_window(X[0], entry_price=price)
                train_stats = self.online_trainer.maybe_update(self.feedback_buffer)
                if train_stats is not None and self.recalibrate_k_on_retrain:
                    self._recalibrate_k()

                result = StepResult(
                    mu=mu, sigma=sigma, p_up=p_up,
                    raw_target_position=raw_target, target_position=target_position,
                    fill=fill, train_stats=train_stats,
                )

        self.feedback_buffer.tick()
        return result

    def _recalibrate_k(self) -> None:
        """
        Kalibriert k neu, im selben Takt wie das Online-Fine-Tuning (also
        immer, wenn OnlineTrainer.maybe_update() gerade einen Schritt
        ausgefuehrt hat).

        Grund (Bugfix nach Backtest-Laeufen vom 24./25.08.2026, siehe
        logs/backtest_20260824_225333 vs. 230406 fuer den Befund): k wurde
        bisher NUR EINMAL zu Beginn kalibriert, auf Basis der mu/sigma-
        Verteilung des frisch offline-vortrainierten Modells. Waehrend eines
        mehrmonatigen Out-of-Sample-Replays veraendert sich sigma aber
        laufend durch das Online-Fine-Tuning (typischerweise: es schrumpft
        ueber viele Updates hinweg) -- edge = mu/sigma^2 wird dadurch mit
        der Zeit systematisch groesser, als k urspruenglich dafuer kalibriert
        wurde. Ergebnis im Backtest: MSFT/GOOGL-Positionen lagen im Median
        bei 0,65-0,70 und saettigten im 90. Perzentil bereits bei 1,0 --
        obwohl calibrate_k() eigentlich "50% Auslastung beim 95. Perzentil"
        als Ziel hatte. Ohne periodisches Neu-Kalibrieren driftet der
        Controller also mit der Zeit in Richtung Dauer-Saettigung, was Cost-
        Turnover (haeufiges Traden grosser Positionsaenderungen) begünstigt.
        """
        if not self.feedback_buffer.is_ready_for_training():
            return
        X, _ = self.feedback_buffer.get_training_batch(batch_size=256)
        self.model.eval()
        forecast = self.model.predict(X)
        new_k = calibrate_k(
            forecast.expected_return,
            forecast.expected_volatility,
            max_position=self.controller.cfg.max_position,
            target_utilization=self.recalibrate_k_target_utilization,
            percentile=self.recalibrate_k_percentile,
            epsilon=self.controller.cfg.epsilon,
            max_edge=self.controller.cfg.max_edge,
        )
        self.controller.cfg.k = new_k


# ---------------------------------------------------------------------------
# End-to-End-Demo auf synthetischen Streaming-Daten
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from feature_pipeline import FeaturePipeline
    from sequence_buffer import SequenceWindowBuilder
    from lstm_forecaster_torch import train_epoch

    torch.manual_seed(0)
    np.random.seed(0)

    TIMESTEPS = 20
    HORIZON = 5
    N_OFFLINE_BARS = 2500   # fuer Offline-Vortraining, BEVOR der Live-Loop startet
    N_LIVE_BARS = 4000      # Bars, die anschliessend durch den Live-Regelkreis laufen

    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=32, num_layers=2)

    # --- Offline-Vortraining (siehe README: Pflicht vor dem Live-Schalten) ---
    # Ohne diesen Schritt startet der Live-Loop mit einem zufaellig initialisierten
    # Modell, dessen sigma-Head noch voellig unkalibriert ist (softplus(0)+eps ~ 0.7,
    # weit ausserhalb realistischer Vol-Groessenordnungen von ~0.01-0.05). Der
    # Online-Trainer wuerde dann versuchen, diese komplette Kalibrierung "on the
    # fly" aus wenigen Live-Bars zu lernen -- das produziert genau die Art von
    # Cold-Start-Instabilitaet (sigma faellt ueber viele Updates hinweg drastisch,
    # edge = mu/sigma^2 kann dabei zwischenzeitlich ueberschiessen), die eigentlich
    # durch Offline-Pretraining vermieden werden soll.
    print("=== Offline-Vortraining ===")
    offline_df = _generate_synthetic_market_data(n=N_OFFLINE_BARS, seed=42)
    offline_features, offline_target = FeaturePipeline().transform_with_target(offline_df, horizon=HORIZON)
    builder = SequenceWindowBuilder(timesteps=TIMESTEPS, feature_names=list(FEATURE_NAMES))
    X_off, y_off, _ = builder.build(offline_features, offline_target)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        stats = train_epoch(model, optimizer, X_off, y_off, batch_size=64)
        print(f"  Offline-Epoch {epoch+1}: mean_loss={stats['mean_loss']:.4f}")
    print()

    # --- k datengetrieben kalibrieren, statt eine Magic Number zu raten ---
    # Nutzt die eigenen mu/sigma-Vorhersagen DIESES Modells auf den Offline-
    # Daten: welches k sorgt dafuer, dass ein starkes (95. Perzentil), aber
    # nicht extremes Signal 50% des Risikobudgets ausnutzt -- statt dass
    # praktisch jedes Rauschsignal am max_position-Clip saettigt (siehe
    # Befund/Doku in controller/exposure_controller.py).
    model.eval()
    offline_forecast = model.predict(X_off)
    k_calibrated = calibrate_k(
        offline_forecast.expected_return,
        offline_forecast.expected_volatility,
        max_position=1.0,
        target_utilization=0.5,
        percentile=95.0,
    )
    print(f"=== k-Kalibrierung ===\nKalibriertes k aus Offline-Vorhersagen: {k_calibrated:.5f}\n")

    loop = ControlLoop(
        timesteps=TIMESTEPS,
        model=model,
        horizon=HORIZON,
        controller_config=ControllerConfig(k=k_calibrated, max_position=1.0),
        risk_config=RiskOverlayConfig(max_step_change=0.1, max_sigma=0.08, drawdown_limit=-0.03, cooldown_steps=15, min_rebalance_threshold=0.30),
        execution_config=ExecutionConfig(slippage_bps=0.5),
        online_trainer_config=OnlineTrainerConfig(retrain_every=100, batch_size=32, max_training_window=256),
    )

    # Live-Loop laeuft auf NEUEN, im Offline-Training ungesehenen Bars (anderer Seed).
    df = _generate_synthetic_market_data(n=N_LIVE_BARS, seed=99)

    print(f"=== ControlLoop End-to-End-Demo ({N_LIVE_BARS} synthetische Live-Bars, Modell offline vortrainiert) ===\n")
    n_active_steps = 0
    n_train_updates = 0
    cumulative_return = 0.0
    equity_curve = []

    for i, row in df.iterrows():
        raw_event = row.to_dict()
        result = loop.step(raw_event)
        if result is None:
            continue

        n_active_steps += 1
        cumulative_return += result.fill.realized_return
        equity_curve.append(cumulative_return)
        if result.train_stats is not None:
            n_train_updates += 1
            print(
                f"  [Bar {i}] Online-Fine-Tuning: n_samples={result.train_stats['n_samples']}, "
                f"mean_loss={result.train_stats['mean_loss']:.4f}"
            )

        if n_active_steps % 500 == 0:
            print(
                f"Bar {i}: mu={result.mu:+.5f} sigma={result.sigma:.5f} "
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
    print(
        "\nHinweis: Auf synthetischen Random-Walk-Daten ohne echtes Signal ist "
        "ein kumulierter Return nahe 0 (dominiert von Slippage-Kosten) das "
        "erwartete Ergebnis -- dieser Lauf verifiziert, dass der komplette "
        "Regelkreis fehlerfrei durchlaeuft, NICHT dass die Strategie profitabel "
        "ist. Profitabilitaet kann erst auf echten Marktdaten mit echtem Signal "
        "bewertet werden."
    )
