"""
paper_execution.py
===================

Layer 6 des Market-Control-Systems: simuliert die Ausfuehrung einer
Zielposition, die aus RiskOverlay.apply() kommt.

MVP-Fill-Modell (siehe README, Abschnitt "MVP vs. Produktion"):
Positionsaenderungen werden sofort zum Mid-Preis ausgefuehrt, zzgl.
halbem Spread als Slippage-Kosten in Richtung der Aenderung. Kein
Orderbook-Tiefe-Modell, keine Latenz-Simulation -- das ist bewusst die
einfachste Variante, die noch realistisch genug ist, um zu zeigen, dass
haeufiges Positions-Flippen (viel |position_delta|) Kosten produziert.

Wichtige Design-Entscheidung: realized_return wird aus der Position
berechnet, die VOR diesem Schritt bereits gehalten wurde (mark-to-market
der alten Position ueber die Preisbewegung), nicht aus der neuen
Zielposition -- alles andere waere Look-Ahead (man kann nicht von einer
Preisbewegung profitieren, die schon vorbei ist, wenn man die Position
erst danach eroeffnet).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fill:
    """Ergebnis eines Ausfuehrungsschritts."""
    position: float          # neue Position nach diesem Schritt
    fill_price: float        # angenommener Ausfuehrungspreis (Mid) fuer die Aenderung
    position_delta: float    # tatsaechlich gehandelte Aenderung (neu - alt)
    slippage_cost: float     # Kosten durch Spread-Crossing, als Return-Anteil (>= 0)
    realized_return: float   # Return dieses Schritts: alte Position * Preisbewegung - slippage_cost


@dataclass
class ExecutionConfig:
    slippage_bps: float = 0.0  # zusaetzlicher Slippage-Aufschlag in bps, ueber den halben Spread hinaus


class PaperExecutionEngine:
    """
    Haelt den aktuellen Positions- und Preis-Zustand und simuliert Fills.

    market_state (Argument von execute()) erwartet mindestens:
        bid, ask  -- fuer Mid-Preis und Spread
    """

    def __init__(self, config: ExecutionConfig = ExecutionConfig()):
        self.cfg = config
        self._position: float = 0.0
        self._previous_mid: float | None = None

    def execute(self, target_position: float, market_state: dict) -> Fill:
        bid = market_state["bid"]
        ask = market_state["ask"]
        mid = (bid + ask) / 2.0
        spread = ask - bid

        position_before = self._position
        position_delta = target_position - position_before

        half_spread_cost = abs(position_delta) * (spread / 2.0) / mid
        extra_slippage_cost = abs(position_delta) * (self.cfg.slippage_bps / 1e4)
        slippage_cost = half_spread_cost + extra_slippage_cost

        if self._previous_mid is not None:
            asset_return = (mid - self._previous_mid) / self._previous_mid
            realized_return = position_before * asset_return - slippage_cost
        else:
            # Erster Schritt: keine Vorperiode zum Mark-to-Market, nur Eintrittskosten.
            realized_return = -slippage_cost

        self._position = target_position
        self._previous_mid = mid

        return Fill(
            position=self._position,
            fill_price=mid,
            position_delta=position_delta,
            slippage_cost=slippage_cost,
            realized_return=realized_return,
        )

    @property
    def current_position(self) -> float:
        return self._position

    def reset(self) -> None:
        self._position = 0.0
        self._previous_mid = None


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = PaperExecutionEngine(ExecutionConfig(slippage_bps=1.0))

    ticks = [
        {"bid": 99.99, "ask": 100.01},   # mid=100.00, erster Tick
        {"bid": 100.49, "ask": 100.51},  # mid=100.50, Preis steigt
        {"bid": 100.29, "ask": 100.31},  # mid=100.30, Preis faellt wieder
    ]
    targets = [0.5, 0.5, -0.3]  # halten, dann auf -0.3 umswitchen

    print("=== PaperExecutionEngine Sanity-Check ===")
    cumulative = 0.0
    for i, (target, tick) in enumerate(zip(targets, ticks)):
        fill = engine.execute(target, tick)
        cumulative += fill.realized_return
        print(
            f"Step {i}: target={target:+.2f}  fill_price={fill.fill_price:.4f}  "
            f"delta={fill.position_delta:+.4f}  slippage={fill.slippage_cost:.5f}  "
            f"realized_return={fill.realized_return:+.5f}  cumulative={cumulative:+.5f}"
        )
