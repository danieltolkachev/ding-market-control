"""
feedback_buffer.py
===================

Layer 7 des Market-Control-Systems: sammelt (Sequenzfenster, tatsaechlicher
Future-Return)-Paare, sobald deren Forecast-Horizon erreicht ist, und
stellt sie dem Online Trainer (Layer 8) als Trainingsbatches zur Verfuegung.

WICHTIG -- zwei verschiedene "realisierte" Groessen im System:
-----------------------------------------------------------------
1. PaperExecutionEngine.Fill.realized_return: slippage-bereinigte
   Trading-PnL, mark-to-market der GEHALTENEN Position. Wird von
   RiskOverlay.record_realized_return() fuer den Drawdown-Cooldown
   konsumiert.
2. FeedbackBuffer actual_future_return (dieses Modul): roher Preis-Return
   ueber den Forecast-Horizon, exakt dieselbe Zieldefinition wie in
   feature_pipeline.FeaturePipeline.transform_with_target(). Wird vom
   Online Trainer konsumiert.

Diese beiden Groessen absichtlich NICHT vermischen: das Modell wurde auf
rohen Returns trainiert (log(price[t+h]) - log(price[t])), nicht auf
Trading-PnL inkl. Slippage/Positionsgroesse. Wuerde man stattdessen mit
Trading-PnL nachtrainieren, wuerde das Ziel systematisch verschoben
(kleinere Positionen erzeugen kleinere Ziel-Returns) und mu/sigma
verzerrt -- ein subtiler, aber schwerwiegender Fehler in geschlossenen
Regelkreisen dieser Art.
"""

from __future__ import annotations

from collections import deque
import numpy as np


class FeedbackBuffer:
    """
    Fuehrt einen eigenen Bar-Zaehler (`_current_step`), unabhaengig vom
    restlichen System -- der Aufrufer (control_loop.py) muss `resolve()`
    und `tick()` bei JEDEM Bar aufrufen, auch waehrend der Warm-up-Phase
    der Feature-Pipeline, damit die Horizon-Verzoegerung korrekt bleibt.
    """

    def __init__(self, horizon: int, min_batch_size: int = 64, max_size: int = 5000):
        self.horizon = horizon
        self.min_batch_size = min_batch_size
        self._pending: deque[tuple[int, np.ndarray, float]] = deque()  # (target_step, X_window, entry_price)
        self._ready_X: deque[np.ndarray] = deque(maxlen=max_size)
        self._ready_y: deque[float] = deque(maxlen=max_size)
        self._current_step: int = 0

    def record_window(self, X_window: np.ndarray, entry_price: float) -> None:
        """Vom control_loop aufgerufen, sobald fuer den aktuellen Bar ein
        vollstaendiges Sequenzfenster + Forecast vorliegt. X_window: Shape
        (timesteps, n_features) -- OHNE Batch-Dimension."""
        target_step = self._current_step + self.horizon
        self._pending.append((target_step, X_window, entry_price))

    def resolve(self, current_price: float) -> int:
        """Loest alle pending Eintraege auf, deren Horizon mit dem aktuellen
        Bar erreicht ist. Muss VOR record_window() fuer denselben Bar
        aufgerufen werden. Gibt die Anzahl aufgeloester Eintraege zurueck."""
        resolved = 0
        while self._pending and self._pending[0][0] <= self._current_step:
            _, X_window, entry_price = self._pending.popleft()
            actual_return = float(np.log(current_price) - np.log(entry_price))
            self._ready_X.append(X_window)
            self._ready_y.append(actual_return)
            resolved += 1
        return resolved

    def tick(self) -> None:
        """Erhoeht den internen Bar-Zaehler. Einmal pro Bar aufzurufen,
        nachdem resolve() und ggf. record_window() gelaufen sind."""
        self._current_step += 1

    def is_ready_for_training(self) -> bool:
        return len(self._ready_X) >= self.min_batch_size

    def get_training_batch(self, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Liefert die JUENGSTEN `batch_size` (X, y)-Paare (oder alle, falls
        batch_size None ist bzw. groesser als der Buffer)."""
        n_available = len(self._ready_X)
        n = n_available if batch_size is None else min(batch_size, n_available)
        X = np.stack(list(self._ready_X)[-n:], axis=0)
        y = np.array(list(self._ready_y)[-n:], dtype=np.float32)
        return X, y

    def __len__(self) -> int:
        return len(self._ready_X)


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    buf = FeedbackBuffer(horizon=3, min_batch_size=2)

    prices = [100.0, 100.5, 101.0, 102.0, 101.5, 103.0, 104.0]
    dummy_window = np.zeros((5, 3), dtype=np.float32)

    print("=== FeedbackBuffer Sanity-Check (horizon=3) ===")
    for step, price in enumerate(prices):
        n_resolved = buf.resolve(price)
        if n_resolved:
            print(f"  Bar {step}: {n_resolved} Eintrag/Eintraege aufgeloest "
                  f"(letzter y={buf._ready_y[-1]:+.5f})")
        # An jedem Bar ein neues Fenster aufzeichnen (vereinfachtes Beispiel)
        buf.record_window(dummy_window, entry_price=price)
        buf.tick()

    print(f"\nBereit fuer Training: {buf.is_ready_for_training()}  (n={len(buf)})")
    X, y = buf.get_training_batch()
    print(f"X shape: {X.shape}, y: {y}")

    # Erwartung: erstes aufgeloestes y bei Bar 3 = log(102.0) - log(100.0)
    expected_first_y = np.log(102.0) - np.log(100.0)
    print(f"Erwarteter erster y-Wert (Bar 0 -> Bar 3): {expected_first_y:+.5f}, "
          f"tatsaechlich im Buffer: {list(buf._ready_y)[0]:+.5f}")
