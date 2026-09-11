"""
null_params.py — Schaetzung der Parameter fuer die drei Nullwelten des
Falsifikations-Audits (Spec Abschnitt 1), ausschliesslich LESEND auf dem
archivierten Preissnapshot.

Bewusst ohne scipy/sklearn: GARCH(1,1) wird per Varianz-Targeting plus
deterministischem Gitter-MLE geschaetzt. Das ist gruber als ein echter
Optimizer, aber reproduzierbar, testbar und ohne zusaetzliche Abhaengigkeit
-- fuer eine Nullwelt genuegt eine plausible, nicht eine optimale
Parametrisierung.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd

ARCHIVE_SNAPSHOT_PATH = (
    r"C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907"
    r"/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl"
)
ARCHIVE_SNAPSHOT_SHA256 = "36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89"

_ALPHA_GRID = np.round(np.arange(0.02, 0.205, 0.01), 3)
_BETA_GRID = np.round(np.arange(0.70, 0.975, 0.01), 3)


def load_archive_snapshot() -> dict:
    """Laedt den archivierten Snapshot read-only. Der Datei-Hash wird NICHT
    geprueft (das Pickle-Format ist nicht bytegleich reproduzierbar); geprueft
    wird der Inhalts-Hash ueber snapshot_content_sha256, der auch im
    Uebergabe-Prompt steht."""
    from factor_lab.data_snapshot import snapshot_content_sha256
    with open(ARCHIVE_SNAPSHOT_PATH, "rb") as f:
        dfs = pickle.load(f)
    digest = snapshot_content_sha256(dfs)
    if digest != ARCHIVE_SNAPSHOT_SHA256:
        raise ValueError(
            f"Archiv-Snapshot hat Inhalts-Hash {digest}, erwartet {ARCHIVE_SNAPSHOT_SHA256}"
        )
    return dfs


def _returns_frame(dfs: dict) -> pd.DataFrame:
    symbols = sorted(s for s in dfs if s != "IRX")
    common = dfs[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(dfs[s].index)
    common = common.sort_values()
    prices = pd.DataFrame({s: dfs[s].loc[common, "price"] for s in symbols})
    return prices.pct_change().dropna()


def fit_garch_grid(returns: np.ndarray) -> dict:
    """GARCH(1,1) mit Varianz-Targeting: omega folgt aus der Stichprobenvarianz
    und (alpha, beta); (alpha, beta) per deterministischem Gitter-MLE unter
    Gauss-Annahme. Mittelwert wird als null angenommen (Nullwelt-Zweck).

    VECTORIZED: The alpha-beta grid is carried through the time loop as arrays.
    The time recursion for var stays serial (inherent sequential dependency),
    but all grid points are updated in parallel."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    target_var = float(np.mean(r ** 2))

    # Create all valid (alpha, beta) pairs
    alpha_grid = []
    beta_grid = []
    for alpha in _ALPHA_GRID:
        for beta in _BETA_GRID:
            persistence = alpha + beta
            if persistence < 0.999:
                alpha_grid.append(alpha)
                beta_grid.append(beta)

    alpha_grid = np.array(alpha_grid, dtype=float)
    beta_grid = np.array(beta_grid, dtype=float)
    persistence = alpha_grid + beta_grid

    # Initialize omega, var, and loglik as arrays (one entry per grid point)
    omega = target_var * (1.0 - persistence)
    var = np.full(len(alpha_grid), target_var, dtype=float)
    loglik = np.zeros(len(alpha_grid), dtype=float)

    # Loop over observations, updating all grid points in parallel
    for value in r:
        loglik -= 0.5 * (np.log(var) + value * value / var)
        var = omega + alpha_grid * value * value + beta_grid * var

    # Find the best fit
    best_idx = np.argmax(loglik)
    return {
        "omega": float(omega[best_idx]),
        "alpha": float(alpha_grid[best_idx]),
        "beta": float(beta_grid[best_idx]),
        "loglik": float(loglik[best_idx])
    }


def estimate_null_params(dfs: dict) -> dict:
    returns = _returns_frame(dfs)
    symbols = list(returns.columns)

    # --- N1: zwei Vola-Zustaende ueber die Panel-Durchschnittsvola ---
    panel_vol = returns.mean(axis=1).rolling(21, min_periods=21).std().dropna()
    high = (panel_vol > panel_vol.median()).to_numpy()
    transition = np.zeros((2, 2))
    for a, b in zip(high[:-1], high[1:]):
        transition[int(a), int(b)] += 1.0
    transition = transition / transition.sum(axis=1, keepdims=True)
    aligned = returns.loc[panel_vol.index]
    state_vol = {
        s: [float(aligned.loc[~high, s].std()), float(aligned.loc[high, s].std())]
        for s in symbols
    }

    # --- N2: GARCH(1,1) je Instrument ---
    n2 = {s: fit_garch_grid(returns[s].to_numpy()) for s in symbols}

    # --- N3: zwei Hauptkomponenten aus der Korrelationsmatrix ---
    standardized = (returns - returns.mean()) / returns.std()
    eigvals, eigvecs = np.linalg.eigh(np.corrcoef(standardized.to_numpy(), rowvar=False))
    order = np.argsort(eigvals)[::-1][:2]
    loadings = eigvecs[:, order] * np.sqrt(eigvals[order])
    factors = standardized.to_numpy() @ eigvecs[:, order]
    residual = standardized.to_numpy() - factors @ loadings.T
    n3 = {
        "loadings": (loadings * returns.std().to_numpy()[:, None]).tolist(),
        "idio_vol": (residual.std(axis=0) * returns.std().to_numpy()).tolist(),
        "factor_garch": [fit_garch_grid(factors[:, k]) for k in range(2)],
    }

    return {
        "symbols": symbols,
        "calendar": [d.strftime("%Y-%m-%d") for d in returns.index],
        "n1": {"transition": transition.tolist(), "state_vol": state_vol},
        "n2": n2,
        "n3": n3,
    }


def freeze_params(params: dict, path: str) -> str:
    payload = json.dumps(params, indent=2, sort_keys=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
