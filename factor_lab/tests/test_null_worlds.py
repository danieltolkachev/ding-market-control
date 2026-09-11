"""
test_null_worlds.py — Generatoren fuer die drei Nullwelten. Zentrale
Eigenschaft: konstruktionsbedingt kein prognostizierbarer Mittelwert.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.audit.null_worlds import WORLDS, generate_null_panel


def _params(n_days: int = 600, n_symbols: int = 4) -> dict:
    symbols = [f"S{i}" for i in range(n_symbols)]
    calendar = [d.strftime("%Y-%m-%d")
                for d in pd.date_range("2010-01-01", periods=n_days, freq="B")]
    garch = {"omega": 1e-6, "alpha": 0.08, "beta": 0.90, "loglik": 0.0}
    return {
        "symbols": symbols,
        "calendar": calendar,
        "n1": {"transition": [[0.95, 0.05], [0.10, 0.90]],
               "state_vol": {s: [0.006, 0.018] for s in symbols}},
        "n2": {s: dict(garch) for s in symbols},
        "n3": {"loadings": [[0.01, 0.004]] * n_symbols,
               "idio_vol": [0.008] * n_symbols,
               "factor_garch": [dict(garch), dict(garch)]},
    }


def check_panel_shape_matches_pipeline_contract() -> None:
    params = _params()
    for world in WORLDS:
        dfs = generate_null_panel(world, params, seed=0)
        assert set(dfs) == set(params["symbols"]) | {"IRX"}, f"{world}: falsche Schluessel"
        for s in params["symbols"]:
            assert list(dfs[s].columns) == ["price"], f"{world}/{s}: Spalte muss 'price' heissen"
            assert len(dfs[s]) == len(params["calendar"])
            values = dfs[s]["price"].to_numpy()
            assert np.isfinite(values).all() and (values > 0).all(), (
                f"{world}/{s}: Preise muessen endlich und positiv sein"
            )
        assert list(dfs["IRX"].columns) == ["rate_pa_pct"]
    print("generate_null_panel: Struktur/Kalender/Positivitaet fuer alle Welten: OK")


def check_deterministic_per_seed() -> None:
    params = _params()
    for world in WORLDS:
        a = generate_null_panel(world, params, seed=7)["S0"]["price"].to_numpy()
        b = generate_null_panel(world, params, seed=7)["S0"]["price"].to_numpy()
        c = generate_null_panel(world, params, seed=8)["S0"]["price"].to_numpy()
        assert np.array_equal(a, b), f"{world}: gleicher Seed muss gleiches Panel geben"
        assert not np.array_equal(a, c), f"{world}: anderer Seed muss anderes Panel geben"
    print("generate_null_panel: deterministisch je Seed, verschieden ueber Seeds: OK")


def check_mean_is_statistically_zero() -> None:
    """Ueber viele Seeds gemittelt darf die Tagesrendite nicht signifikant
    von null abweichen -- das ist die definierende Eigenschaft der Nullwelt."""
    params = _params(n_days=1200)
    for world in WORLDS:
        means = []
        for seed in range(20):
            dfs = generate_null_panel(world, params, seed=seed)
            r = dfs["S0"]["price"].pct_change().dropna().to_numpy()
            means.append(r.mean())
        means = np.asarray(means)
        t_stat = means.mean() / (means.std(ddof=1) / np.sqrt(len(means)))
        assert abs(t_stat) < 3.0, f"{world}: Mittelwert weicht von null ab (t={t_stat:.2f})"
    print("generate_null_panel: Mittelwert statistisch null in allen Welten: OK")


def check_n3_induces_cross_sectional_correlation() -> None:
    """N3 muss echte Querschnittskorrelation erzeugen, sonst testet sie nicht,
    wozu sie da ist."""
    params = _params(n_days=1500)
    dfs = generate_null_panel("n3", params, seed=3)
    returns = pd.DataFrame({s: dfs[s]["price"].pct_change() for s in params["symbols"]}).dropna()
    corr = returns.corr().to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    assert off_diagonal.mean() > 0.2, (
        f"N3 erzeugt zu wenig Querschnittskorrelation (Mittel {off_diagonal.mean():.2f})"
    )
    print("generate_null_panel: N3 erzeugt Querschnittskorrelation: OK")


def run_consistency_check() -> None:
    check_panel_shape_matches_pipeline_contract()
    check_deterministic_per_seed()
    check_mean_is_statistically_zero()
    check_n3_induces_cross_sectional_correlation()
    print("\nAlle null_worlds-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
