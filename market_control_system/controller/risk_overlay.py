"""
risk_overlay.py
================

Wrapped den rohen Output von ExposureController mit vier Schutz-/
Kosten-Mechanismen, bevor eine Zielposition an die Paper Execution Engine
geht:

1. Rebalancing-Totband: liegt |target - aktuelle Position| unter
   `min_rebalance_threshold`, wird GAR NICHT gehandelt. Verhindert, dass
   minimales Bar-zu-Bar-Rauschen im Forecast staendig winzige, aber in
   Summe teure Positionsanpassungen ausloest (siehe Docstring bei
   RiskOverlayConfig.min_rebalance_threshold fuer den konkreten Vorfall,
   der diesen Fix noetig machte: 123% kumulierte Slippage-Kosten ueber
   ein Jahr durch Positionsaenderungen auf 96,5% aller Bars).
2. Rate Limiter: begrenzt |Delta Position| pro Schritt (fuer Aenderungen,
   die das Totband ueberschreiten). Verhindert, dass ein einzelner
   verrauschter Forecast die Position abrupt umschlaegt (whipsaw), und
   simuliert nebenbei realistische Ausfuehrungsgeschwindigkeit.
3. Regime-Filter (sigma-Schwelle): liegt expected_volatility ueber
   `max_sigma`, wird die Position auf 0 gezwungen. Das LSTM wurde auf
   einem bestimmten Vol-Regime trainiert; ein sigma weit ausserhalb davon
   ist ein Signal, dass die Prognose extrapoliert statt interpoliert --
   dem Controller wird in diesem Zustand bewusst nicht vertraut.
4. Drawdown-Cooldown: der Feedback-Layer meldet realisierte Returns via
   `record_realized_return()`. Faellt die Summe der letzten
   `drawdown_lookback` realisierten Returns unter `drawdown_limit`, geht
   der Overlay fuer `cooldown_steps` Schritte zwangsweise auf Position 0,
   unabhaengig vom Controller-Signal.

Ohne aufgerufenes `record_realized_return()` bleibt Mechanismus 4 inaktiv
(Cooldown greift nie) -- der Overlay funktioniert also bereits jetzt
eigenstaendig mit den Mechanismen 1-3, auch bevor Feedback Buffer/
Execution Engine existieren.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RiskOverlayConfig:
    max_step_change: float = 0.3       # max. |Positionsaenderung| pro Schritt
    max_sigma: float | None = None     # Regime-Filter-Schwelle; None = deaktiviert
    drawdown_limit: float = -0.05      # Summe realisierter Returns, ab der Cooldown greift
    drawdown_lookback: int = 20        # Fenstergroesse fuer die Drawdown-Summe
    cooldown_steps: int = 10           # Dauer der Zwangs-Flat-Phase nach Trigger
    min_rebalance_threshold: float = 0.0
    # Totband: liegt |target - aktuelle Position| UNTER dieser Schwelle, wird
    # NICHT gehandelt (Position bleibt exakt gleich, keine Slippage-Kosten).
    # Grund (Bugfix nach Run 1 vom 24./25.08.2026): der LSTM-Forecast schwankt
    # von Bar zu Bar minimal, auch ohne echten Richtungswechsel -- ohne Totband
    # fuehrte das im 1-Jahres-Backtest zu Positionsaenderungen auf 96,5% aller
    # Bars und damit zu kumulierten Slippage-Kosten von 123% des Kapitals
    # ueber ein Jahr. Default 0.0 = deaktiviert (altes Verhalten, rueckwaerts-
    # kompatibel) -- die Live-/Backtest-Configs setzen bewusst einen Wert > 0.


class RiskOverlay:
    """Zustandsbehafteter Wrapper um die zustandslose ExposureController-Formel."""

    def __init__(self, config: RiskOverlayConfig = RiskOverlayConfig()):
        self.cfg = config
        self._previous_position: float = 0.0
        self._realized_returns: deque[float] = deque(maxlen=config.drawdown_lookback)
        self._cooldown_remaining: int = 0

    def record_realized_return(self, realized_return: float) -> None:
        """Vom Feedback Buffer nach jedem abgeschlossenen Schritt aufzurufen,
        BEVOR apply() fuer den naechsten Schritt aufgerufen wird."""
        self._realized_returns.append(realized_return)
        if self._cooldown_remaining == 0 and sum(self._realized_returns) <= self.cfg.drawdown_limit:
            self._cooldown_remaining = self.cfg.cooldown_steps
            self._realized_returns.clear()  # verhindert sofortiges Re-Triggern direkt nach Cooldown-Ende

    def apply(self, raw_target_position: float, expected_volatility: float | None = None) -> float:
        """
        Args:
            raw_target_position: Output von ExposureController.compute_target_position()
            expected_volatility: sigma des aktuellen Forecasts, fuer den Regime-Filter
                (optional -- ohne Wert bleibt der Regime-Filter fuer diesen Schritt inaktiv)

        Returns:
            Tatsaechliche Zielposition nach Risk-Overlay, in [-1, 1]-Schrittdistanz
            zur vorherigen Position begrenzt.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._previous_position = 0.0
            return 0.0

        target = raw_target_position
        if (
            self.cfg.max_sigma is not None
            and expected_volatility is not None
            and expected_volatility > self.cfg.max_sigma
        ):
            target = 0.0

        if abs(target - self._previous_position) < self.cfg.min_rebalance_threshold:
            return self._previous_position

        delta = np.clip(
            target - self._previous_position,
            -self.cfg.max_step_change,
            self.cfg.max_step_change,
        )
        new_position = self._previous_position + delta
        self._previous_position = new_position
        return float(new_position)

    def reset(self) -> None:
        self._previous_position = 0.0
        self._realized_returns.clear()
        self._cooldown_remaining = 0


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from exposure_controller import ExposureController, ControllerConfig

    controller = ExposureController(ControllerConfig(k=1.0, max_position=1.0))
    overlay = RiskOverlay(RiskOverlayConfig(
        max_step_change=0.2, max_sigma=0.05, drawdown_limit=-0.05,
        drawdown_lookback=5, cooldown_steps=3,
    ))

    print("=== Rate Limiter: grosser Sprung wird ueber mehrere Schritte geglaettet ===")
    raw = controller.compute_target_position(expected_return=0.03, expected_volatility=0.02)
    print(f"Roher Controller-Output: {raw:+.4f}")
    for step in range(4):
        pos = overlay.apply(raw, expected_volatility=0.02)
        print(f"  Schritt {step}: Zielposition nach Overlay = {pos:+.4f}")

    print("\n=== Regime-Filter: sigma ueber max_sigma zwingt Position auf 0 ===")
    overlay2 = RiskOverlay(RiskOverlayConfig(max_step_change=1.0, max_sigma=0.05))
    raw_high_vol = controller.compute_target_position(expected_return=0.03, expected_volatility=0.15)
    pos = overlay2.apply(raw_high_vol, expected_volatility=0.15)
    print(f"  raw_target={raw_high_vol:+.4f} (sigma=0.15 > max_sigma=0.05) -> {pos:+.4f}")

    print("\n=== Drawdown-Cooldown: mehrere Verlust-Ticks zwingen N Schritte auf 0 ===")
    overlay3 = RiskOverlay(RiskOverlayConfig(
        max_step_change=1.0, drawdown_limit=-0.05, drawdown_lookback=5, cooldown_steps=3,
    ))
    overlay3.apply(0.8)  # baut erst eine Position auf
    losses = [-0.02, -0.02, -0.02]
    for i, r in enumerate(losses):
        overlay3.record_realized_return(r)
        pos = overlay3.apply(0.8)
        print(f"  nach Verlust {i} (r={r:+.3f}, kumuliert={sum(overlay3._realized_returns) if overlay3._cooldown_remaining == 0 else 'reset (Cooldown aktiv)'}): position={pos:+.4f}")
