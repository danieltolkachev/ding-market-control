"""
test_cross_sectional_portfolio.py
====================================

Prueft die Ranking-/Hysterese-Logik in controller/cross_sectional_
portfolio.py mit synthetischen edge-Werten (kein Modell/keine echten
Daten noetig -- reine Zustandsmaschinen-Logik).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))

from cross_sectional_portfolio import (
    CrossSectionalPortfolioConfig,
    CrossSectionalPortfolio,
    compute_target_weights,
)


def check_config_validation() -> None:
    # hysteresis_zone < n_long muss ablehnen
    try:
        CrossSectionalPortfolioConfig(n_long=3, n_short=2, hysteresis_zone=2)
        raise AssertionError("Erwartete ValueError fuer hysteresis_zone < n_long, keine geworfen")
    except ValueError:
        pass

    # 2*hysteresis_zone > Universumsgroesse muss ablehnen
    try:
        CrossSectionalPortfolio(
            universe=["A", "B", "C"],
            config=CrossSectionalPortfolioConfig(n_long=1, n_short=1, hysteresis_zone=2),
        )
        raise AssertionError("Erwartete ValueError fuer 2*hysteresis_zone > Universum, keine geworfen")
    except ValueError:
        pass

    print("Config-Validierung: OK")


def check_target_weights() -> None:
    weights = compute_target_weights(
        longs={"X", "Y"}, shorts={"Z"}, universe=["X", "Y", "Z", "W"], gross_exposure=1.0,
    )
    assert abs(weights["X"] - 0.25) < 1e-9, weights
    assert abs(weights["Y"] - 0.25) < 1e-9, weights
    assert abs(weights["Z"] - (-0.5)) < 1e-9, weights
    assert weights["W"] == 0.0, weights
    assert abs(sum(abs(w) for w in weights.values()) - 1.0) < 1e-9, weights
    print("compute_target_weights: OK")


def check_ranking_entry_hysteresis_and_exit() -> None:
    """Drei-Schritt-Szenario ueber 6 Symbole (A..F), n_long=2, n_short=2,
    hysteresis_zone=3:
      Schritt 1: A,B gehen long, E,F gehen short (klarer Fall).
      Schritt 2: C ueberholt B im Rang, B faellt auf Rang 3 -- B bleibt
                 dank Hysterese-Zone (Top 3) trotzdem long, C kommt neu
                 dazu -> Long-Leg waechst voruebergehend auf 3.
      Schritt 3: B faellt auf Rang 4 (ausserhalb der Top-3-Zone) -- B
                 wird jetzt korrekt ausgeschlossen.
    """
    portfolio = CrossSectionalPortfolio(
        universe=["A", "B", "C", "D", "E", "F"],
        config=CrossSectionalPortfolioConfig(n_long=2, n_short=2, hysteresis_zone=3, gross_exposure=1.0),
    )

    # Schritt 1
    mus = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
    sigmas = {s: 1.0 for s in mus}
    weights = portfolio.step(mus, sigmas)
    assert portfolio.current_longs == {"A", "B"}, portfolio.current_longs
    assert portfolio.current_shorts == {"E", "F"}, portfolio.current_shorts
    assert weights["A"] == weights["B"] == 0.25, weights
    assert weights["E"] == weights["F"] == -0.25, weights

    # Schritt 2: C ueberholt B (Rang: A, C, B, D, E, F)
    mus = {"A": 5, "C": 4.5, "B": 4, "D": 2, "E": 1, "F": 0}
    weights = portfolio.step(mus, sigmas)
    assert portfolio.current_longs == {"A", "B", "C"}, (
        f"B sollte dank Hysterese noch gehalten werden, C neu dazu: {portfolio.current_longs}"
    )
    assert abs(weights["A"] - 1 / 6) < 1e-9, weights

    # Schritt 3: B faellt auf Rang 4 (ausserhalb Top-3: A, C, D, B, E, F)
    mus = {"A": 5, "C": 4.5, "D": 4, "B": 1, "E": 0.5, "F": 0}
    weights = portfolio.step(mus, sigmas)
    assert portfolio.current_longs == {"A", "C"}, (
        f"B sollte jetzt ausgeschlossen sein (ausserhalb Hysterese-Zone): {portfolio.current_longs}"
    )
    assert weights["B"] == 0.0, weights

    print("Ranking/Hysterese-Szenario (Entry/Retention/Exit): OK")


def run_consistency_check() -> None:
    check_config_validation()
    check_target_weights()
    check_ranking_entry_hysteresis_and_exit()
    print("\nAlle cross_sectional_portfolio-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
