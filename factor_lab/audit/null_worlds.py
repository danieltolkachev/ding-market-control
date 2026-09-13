"""
null_worlds.py — Generatoren der drei Nullwelten (Spec Abschnitt 1).

Gemeinsame, definierende Eigenschaft aller drei: der bedingte Erwartungswert
der Renditen ist konstruktionsbedingt exakt null. Nur die Struktur ZWEITER
Ordnung (Volatilitaetsregime, Clustering, Querschnittskorrelation) wird
nachgebildet -- genau die Strukturen, die scheinbare Prognostizierbarkeit
erzeugen koennen, ohne dass ein echter Edge existiert.

Der Cash-Satz (IRX) wird konstant gehalten: er ist kein Gegenstand der
Falsifikation, und ein zufaelliger Cash-Pfad wuerde nur Rauschen hinzufuegen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WORLDS = ("n1", "n2", "n3")
_CASH_RATE_PA_PCT = 2.0


def _garch_path(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    omega, alpha, beta = spec["omega"], spec["alpha"], spec["beta"]
    var = omega / max(1.0 - alpha - beta, 1e-8)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * out[i] ** 2 + beta * var
    return out


def _regime_states(transition: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    states = np.empty(n, dtype=int)
    state = 0
    for i in range(n):
        states[i] = state
        state = int(rng.random() >= transition[state, 0])
    return states


def _to_panel(returns: np.ndarray, symbols: list[str], calendar: pd.DatetimeIndex) -> dict:
    """Renditen -> Preise (Start 100). Die erste Zeile ist der Startpreis, daher
    werden n-1 Renditen kumuliert."""
    dfs = {}
    for j, symbol in enumerate(symbols):
        prices = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns[:, j]]))
        dfs[symbol] = pd.DataFrame({"price": prices}, index=calendar)
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": np.full(len(calendar), _CASH_RATE_PA_PCT)},
                              index=calendar)
    return dfs


def generate_null_panel(world: str, params: dict, seed: int) -> dict:
    if world not in WORLDS:
        raise ValueError(f"Unbekannte Nullwelt: {world}")
    symbols = list(params["symbols"])
    calendar = pd.DatetimeIndex(pd.to_datetime(params["calendar"]))
    n = len(calendar) - 1  # eine Rendite weniger als Preise
    rng = np.random.default_rng(seed)

    if world == "n1":
        transition = np.asarray(params["n1"]["transition"], dtype=float)
        states = _regime_states(transition, n, rng)
        returns = np.empty((n, len(symbols)))
        for j, symbol in enumerate(symbols):
            low, high = params["n1"]["state_vol"][symbol]
            sigma = np.where(states == 0, low, high)
            returns[:, j] = sigma * rng.standard_normal(n)
    elif world == "n2":
        returns = np.column_stack([_garch_path(params["n2"][s], n, rng) for s in symbols])
    else:
        loadings = np.asarray(params["n3"]["loadings"], dtype=float)
        idio = np.asarray(params["n3"]["idio_vol"], dtype=float)
        factors = np.column_stack([_garch_path(spec, n, rng)
                                   for spec in params["n3"]["factor_garch"]])
        # Faktoren auf Einheitsvarianz normieren, damit die Ladungen die
        # Skala bestimmen und nicht die GARCH-Parametrisierung.
        factors = factors / factors.std(axis=0, keepdims=True)
        returns = factors @ loadings.T + idio * rng.standard_normal((n, len(symbols)))

    return _to_panel(returns, symbols, calendar)
