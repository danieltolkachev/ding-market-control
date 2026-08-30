"""
exposure_controller.py
=======================

Layer 5 des Market-Control-Systems: uebersetzt eine LSTM-Prognose
(expected_return, expected_volatility) in eine Zielposition.

edge = expected_return / (expected_volatility^2 + epsilon)
target_position = clip(k * edge, -max_position, +max_position)

Design-Entscheidung: reine floats statt ForecastOutput als Argument.
------------------------------------------------------------------
models.lstm_forecaster_torch.ForecastOutput haelt Batch-Arrays (auch bei
Batch-Groesse 1 im Live-Loop noch ein 1-elementiges np.ndarray). Der
Controller soll aber weder von PyTorch noch vom Forecast-Objektformat
abhaengen -- reine, testbare Kontrolllogik. Der Aufrufer (control_loop.py)
extrahiert `float(forecast.expected_return[0])` etc. und uebergibt
einfache Zahlen.

Das k-Problem (rohes Kelly-Sizing saettigt fast immer) und die Loesung
-----------------------------------------------------------------------
k=1.0 entspricht dem vollen ("rohen") Kelly-Betrag mu/sigma^2. Bei
typischen Finanzrenditen ist sigma^2 winzig (z.B. sigma~0.03 ->
sigma^2~0.0009), wodurch bereits kleines Rauschen in mu einen
zweistelligen edge-Wert erzeugt und die Position bei k=1 praktisch
IMMER am max_position-Clip saettigt (im End-to-End-Test von
orchestration/control_loop.py: bei k=0.5 lag der max. Drawdown auf
reinem Random-Walk-Rauschen bei -84%). Das ist KEINE Modell-
Fehlkalibrierung -- sigma trifft die tatsaechliche Residualstreuung
nachweislich gut (siehe training/walk_forward.py) -- sondern eine
bekannte Eigenschaft roher Kelly-Sizing: sie ignoriert Schaetz-
unsicherheit in mu selbst.

Statt k per Hand zu erraten (z.B. "0.02, weil das im Demo-Lauf gut
aussah"), bietet dieses Modul `calibrate_k()`: k wird aus der
tatsaechlichen historischen edge-Verteilung des Modells abgeleitet, so
dass ein bestimmtes Perzentil dieser Verteilung einer gewuenschten
Ziel-Exposure entspricht. Das macht die Kalibrierung reproduzierbar und
datengetrieben statt einer im Chat gefundenen Magic Number, und sie
laesst sich jederzeit neu ausfuehren, sobald neue mu/sigma-Historie
(z.B. aus echten Alpaca-Daten) vorliegt.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ControllerConfig:
    k: float = 1.0              # Skalierungsfaktor edge -> Position; siehe calibrate_k() statt Hand-Schaetzung
    max_position: float = 1.0   # Hartes Limit, symmetrisch (long/short)
    epsilon: float = 1e-6       # numerische Untergrenze im Nenner
    max_edge: float | None = None
    # Optionale Sicherheitsbegrenzung des ROHEN edge-Werts, VOR Multiplikation
    # mit k. Grund: sigma nahe 0 (z.B. durch einen schlechten Online-Trainings-
    # Schritt oder ein kurzes Datenartefakt) kann edge auf absurde Groessen-
    # ordnungen treiben. Der max_position-Clip am Ende faengt das zwar fuer die
    # Positionsgroesse selbst ab, aber ein unclipptes edge macht die Bedeutung
    # von k instabil (k=0.02 verhaelt sich bei edge=50 voellig anders als bei
    # edge=5000) und waere gefaehrlich, sobald edge kuenftig noch fuer anderes
    # verwendet wird (z.B. risikoproportionale Skalierung im RiskOverlay).
    # None = deaktiviert (Standard).


class ExposureController:
    """Reine Kontrolllogik: Forecast -> Zielposition, ohne Risk-Overlay.

    Bewusst zustandslos (kein internes Gedaechtnis an vorherige Positionen)
    -- Rate-Limiting, Regime-Filter und Cooldown gehoeren in RiskOverlay
    (risk_overlay.py), das den rohen Output dieses Controllers wrapped.
    Diese Trennung haelt die Kernformel isoliert testbar.
    """

    def __init__(self, config: ControllerConfig = ControllerConfig()):
        self.cfg = config

    def compute_edge(self, expected_return: float, expected_volatility: float) -> float:
        """Rohes, unskaliertes edge = mu / (sigma^2 + eps) -- vor k und Clip.
        Oeffentlich, damit calibrate_k() und Diagnose-Code dieselbe Formel
        wie compute_target_position() verwenden, statt sie zu duplizieren."""
        if expected_volatility < 0:
            raise ValueError(
                f"expected_volatility muss >= 0 sein, erhalten: {expected_volatility}"
            )
        edge = expected_return / (expected_volatility ** 2 + self.cfg.epsilon)
        if self.cfg.max_edge is not None:
            edge = float(np.clip(edge, -self.cfg.max_edge, self.cfg.max_edge))
        return edge

    def compute_target_position(self, expected_return: float, expected_volatility: float) -> float:
        """
        Args:
            expected_return: mu, vorzeichenbehaftet
            expected_volatility: sigma, muss > 0 sein (kommt so aus dem
                softplus-Head des LSTM-Forecasters)

        Returns:
            Zielposition in [-max_position, +max_position].
        """
        edge = self.compute_edge(expected_return, expected_volatility)
        return float(np.clip(self.cfg.k * edge, -self.cfg.max_position, self.cfg.max_position))


def calibrate_k(
    expected_returns: np.ndarray,
    expected_volatilities: np.ndarray,
    max_position: float = 1.0,
    target_utilization: float = 0.5,
    percentile: float = 95.0,
    epsilon: float = 1e-6,
    max_edge: float | None = None,
) -> float:
    """
    Leitet k aus einer historischen mu/sigma-Stichprobe ab (z.B. Vorhersagen
    des offline-vortrainierten Modells auf seinen eigenen Trainings-/
    Validierungsdaten, oder aus einem Walk-Forward-Lauf).

    Idee: statt k so zu waehlen, dass der SCHLIMMSTE denkbare edge gerade
    noch nicht saettigt (viel zu konservativ) oder per Bauchgefuehl (nicht
    reproduzierbar), wird k so gewaehlt, dass ein hohes, aber realistisches
    Perzentil (Standard: 95.) der tatsaechlich beobachteten |edge|-Werte
    genau `target_utilization * max_position` an Positionsgroesse ergibt.
    Seltenere, extremere edge-Werte duerfen den Clip weiterhin erreichen --
    aber der Normalfall saettigt nicht mehr bei praktisch jedem Rauschsignal.

    Args:
        expected_returns, expected_volatilities: gleich lange Arrays
            historischer mu/sigma-Vorhersagen desselben Modells, das auch
            live eingesetzt wird (Kalibrierung ist modellspezifisch!).
        target_utilization: welcher Anteil von max_position beim gewaehlten
            Perzentil erreicht werden soll (0.5 = "typische starke Signale
            nutzen die Haelfte des Risikobudgets aus").
        percentile: welches Perzentil der |edge|-Verteilung als Referenz
            dient (95. Perzentil = "die staerksten 5% der Signale duerfen
            durchaus stark reagieren").

    Returns:
        Kalibriertes k. Gibt 0.0 zurueck, falls die edge-Verteilung
        degeneriert ist (z.B. alle Werte nahe 0).
    """
    if len(expected_returns) != len(expected_volatilities):
        raise ValueError("expected_returns und expected_volatilities muessen gleich lang sein")
    if len(expected_returns) == 0:
        raise ValueError("Leere Stichprobe -- calibrate_k braucht historische mu/sigma-Paare")

    sigma = np.asarray(expected_volatilities, dtype=np.float64)
    mu = np.asarray(expected_returns, dtype=np.float64)
    edge = mu / (sigma ** 2 + epsilon)
    if max_edge is not None:
        edge = np.clip(edge, -max_edge, max_edge)

    reference_abs_edge = float(np.percentile(np.abs(edge), percentile))
    if reference_abs_edge < epsilon:
        return 0.0

    return (target_utilization * max_position) / reference_abs_edge


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    controller = ExposureController(ControllerConfig(k=1.0, max_position=1.0))

    cases = [
        ("hohe Confidence, niedrige Vol", 0.02, 0.01),
        ("hohe Confidence, hohe Vol", 0.02, 0.10),
        ("negativer Edge (Short-Signal)", -0.015, 0.02),
        ("sehr kleines sigma (Clip-Test)", 0.01, 0.001),
        ("kein Signal", 0.0, 0.02),
    ]
    print("=== ExposureController Sanity-Check (k=1.0, unkalibriert) ===")
    for label, mu, sigma in cases:
        pos = controller.compute_target_position(mu, sigma)
        print(f"  {label:35s}  mu={mu:+.4f}  sigma={sigma:.4f}  -> target_position={pos:+.4f}")

    print("\n=== calibrate_k() Demo: realistische mu/sigma-Historie ===")
    rng = np.random.default_rng(0)
    n = 2000
    sigma_hist = np.abs(rng.normal(0.03, 0.008, size=n)) + 1e-4       # typische Vol-Groessenordnung
    mu_hist = rng.normal(0.0, 0.006, size=n)                          # rein rauschbasiertes mu (kein echtes Signal)

    k_calibrated = calibrate_k(
        mu_hist, sigma_hist, max_position=1.0, target_utilization=0.5, percentile=95.0
    )
    print(f"Kalibriertes k (Ziel: 95. Perzentil von |edge| -> 50% von max_position): {k_calibrated:.5f}")

    controller_calibrated = ExposureController(ControllerConfig(k=k_calibrated, max_position=1.0))
    sample_positions = [
        controller_calibrated.compute_target_position(float(mu_hist[i]), float(sigma_hist[i]))
        for i in range(n)
    ]
    print(f"Verteilung der resultierenden Positionen: "
          f"median={np.median(np.abs(sample_positions)):.4f}  "
          f"p95={np.percentile(np.abs(sample_positions), 95):.4f}  "
          f"max={np.max(np.abs(sample_positions)):.4f}  "
          f"Anteil am Clip (|pos|>=0.99): {np.mean(np.abs(sample_positions) >= 0.99):.2%}")
