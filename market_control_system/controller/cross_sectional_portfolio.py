"""
cross_sectional_portfolio.py
==============================

Layer "Controller" fuer das Cross-Sectional-Portfolio (siehe Design-Spec
docs/superpowers/specs/2026-08-30-cross-sectional-portfolio-design.md):
rankt die edge-Werte (mu/sigma^2) mehrerer Symbole zueinander und leitet
daraus ein marktneutrales Long/Short-Buch ab, statt wie ExposureController
eine einzelne Position fuer EIN Symbol zu berechnen.

Hysterese-Prinzip (Analogon zu RiskOverlayConfig.min_rebalance_threshold
im Single-Symbol-System): ein Symbol MUSS in die strengere Top-/Bottom-N-
Zone fallen, um NEU aufgenommen zu werden, darf aber in der breiteren
Hysterese-Zone bleiben, ohne sofort wieder ausgetauscht zu werden. Ohne
das wuerde jedes kleine Rausch-Wackeln am Rang N/N+1 dieselbe Art von
Kosten-Turnover erzeugen, die im Single-Symbol-System vor dem Deadband-Fix
beobachtet wurde (siehe backtest_20260824_223957-Befund).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossSectionalPortfolioConfig:
    n_long: int = 3
    n_short: int = 3
    hysteresis_zone: int = 5    # muss >= n_long und >= n_short sein
    gross_exposure: float = 1.0  # Summe |Gewicht| ueber alle Positionen (Long+Short zusammen)
    epsilon: float = 1e-6        # numerische Untergrenze im edge-Nenner, wie ExposureController

    def __post_init__(self):
        if self.hysteresis_zone < self.n_long or self.hysteresis_zone < self.n_short:
            raise ValueError(
                f"hysteresis_zone ({self.hysteresis_zone}) muss >= n_long ({self.n_long}) "
                f"und >= n_short ({self.n_short}) sein -- sonst kann eine gehaltene "
                f"Position nie in der Hysterese-Zone liegen."
            )


def compute_edges(mus: dict[str, float], sigmas: dict[str, float], epsilon: float = 1e-6) -> dict[str, float]:
    """edge = mu / (sigma^2 + epsilon), pro Symbol -- dieselbe Formel wie
    ExposureController.compute_target_position(), hier aber fuer das
    Ranking ueber mehrere Symbole statt fuer eine Einzelposition."""
    return {symbol: mus[symbol] / (sigmas[symbol] ** 2 + epsilon) for symbol in mus}


def rank_and_select(
    edges: dict[str, float],
    current_longs: set[str],
    current_shorts: set[str],
    n_long: int,
    n_short: int,
    hysteresis_zone: int,
) -> tuple[set[str], set[str]]:
    """
    Bestimmt die neuen Long-/Short-Mengen fuer diesen Zeitschritt.

    Regel: ein Symbol, das NICHT bereits gehalten wird, muss in die
    strikte Top-n_long- (bzw. Bottom-n_short-)Zone fallen, um neu
    aufgenommen zu werden. Ein Symbol, das BEREITS long/short gehalten
    wird, bleibt darin, solange es innerhalb der breiteren
    hysteresis_zone bleibt. Erst beim Verlassen dieser Zone wird die
    Position geschlossen.
    """
    ranked = sorted(edges, key=lambda s: edges[s], reverse=True)

    top_entry = set(ranked[:n_long])
    bottom_entry = set(ranked[-n_short:]) if n_short > 0 else set()
    top_zone = set(ranked[:hysteresis_zone])
    bottom_zone = set(ranked[-hysteresis_zone:]) if hysteresis_zone > 0 else set()

    new_longs = (current_longs & top_zone) | top_entry
    new_shorts = (current_shorts & bottom_zone) | bottom_entry

    # Randfall bei sehr kleinem Universum/grossen Zonen: ein Symbol darf
    # nicht gleichzeitig long und short sein. Bestehende Long-Position hat
    # Vorrang vor einem neu eintretenden Short (und umgekehrt).
    overlap = new_longs & new_shorts
    for symbol in overlap:
        if symbol in current_longs:
            new_shorts.discard(symbol)
        else:
            new_longs.discard(symbol)

    return new_longs, new_shorts


def compute_target_weights(
    longs: set[str], shorts: set[str], universe: list[str], gross_exposure: float = 1.0,
) -> dict[str, float]:
    """Gleichgewichtet innerhalb jedes Legs, dollar-neutral: Summe der
    Long-Gewichte = +gross_exposure/2, Summe der Short-Gewichte =
    -gross_exposure/2 (zusammen also gross_exposure an eingesetztem
    Kapital, netto markt-neutral)."""
    weights = {symbol: 0.0 for symbol in universe}
    if longs:
        w_long = (gross_exposure / 2.0) / len(longs)
        for symbol in longs:
            weights[symbol] = w_long
    if shorts:
        w_short = -(gross_exposure / 2.0) / len(shorts)
        for symbol in shorts:
            weights[symbol] = w_short
    return weights


class CrossSectionalPortfolio:
    """Haelt den Long-/Short-Zustand ueber die Zeit (Hysterese braucht
    Gedaechtnis an die vorherige Zusammensetzung)."""

    def __init__(self, universe: list[str], config: CrossSectionalPortfolioConfig = CrossSectionalPortfolioConfig()):
        if config.hysteresis_zone * 2 > len(universe):
            raise ValueError(
                f"2 * hysteresis_zone ({config.hysteresis_zone * 2}) darf das Universum "
                f"({len(universe)} Symbole) nicht ueberschreiten -- sonst ueberlappen "
                f"sich Long- und Short-Zone."
            )
        self.universe = list(universe)
        self.cfg = config
        self.current_longs: set[str] = set()
        self.current_shorts: set[str] = set()

    def step(self, mus: dict[str, float], sigmas: dict[str, float]) -> dict[str, float]:
        edges = compute_edges(mus, sigmas, epsilon=self.cfg.epsilon)
        self.current_longs, self.current_shorts = rank_and_select(
            edges, self.current_longs, self.current_shorts,
            self.cfg.n_long, self.cfg.n_short, self.cfg.hysteresis_zone,
        )
        return compute_target_weights(self.current_longs, self.current_shorts, self.universe, self.cfg.gross_exposure)
