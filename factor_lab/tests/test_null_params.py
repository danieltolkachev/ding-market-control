"""
test_null_params.py — Schaetzung der Nullwelt-Parameter auf dem
Archiv-Snapshot (ausschliesslich lesend) und deren Einfrieren.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile

import numpy as np
import pandas as pd

from factor_lab.audit.null_params import (
    fit_garch_grid, estimate_null_params, freeze_params, load_archive_snapshot, _returns_frame,
)
from factor_lab.audit.null_worlds import generate_null_panel


def check_garch_grid_recovers_persistence() -> None:
    """Auf simulierten GARCH-Daten mit bekannter Persistenz muss die
    geschaetzte Persistenz alpha+beta in der richtigen Groessenordnung
    liegen (kein Punktschaetzer-Anspruch, nur Plausibilitaet)."""
    rng = np.random.default_rng(0)
    n = 4000
    omega_true, alpha_true, beta_true = 1e-6, 0.08, 0.90
    r = np.empty(n)
    var = omega_true / (1 - alpha_true - beta_true)
    for i in range(n):
        r[i] = np.sqrt(var) * rng.standard_normal()
        var = omega_true + alpha_true * r[i] ** 2 + beta_true * var
    fit = fit_garch_grid(r)
    persistence = fit["alpha"] + fit["beta"]
    assert 0.80 <= persistence < 0.999, f"Persistenz {persistence:.3f} unplausibel"
    assert fit["omega"] > 0 and fit["alpha"] > 0 and fit["beta"] > 0
    print("fit_garch_grid: Persistenz plausibel rekonstruiert: OK")


def check_garch_grid_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    r = rng.standard_normal(2000) * 0.01
    assert fit_garch_grid(r) == fit_garch_grid(r), "Gitter-Fit muss deterministisch sein"
    print("fit_garch_grid: deterministisch: OK")


def check_estimate_null_params_structure() -> None:
    dfs = load_archive_snapshot()
    params = estimate_null_params(dfs)
    assert set(params) >= {"symbols", "calendar", "n1", "n2", "n3"}
    assert len(params["symbols"]) == 19, f"erwartet 19 Instrumente, bekam {len(params['symbols'])}"
    # N1: zwei Vola-Zustaende plus Uebergangsmatrix
    assert set(params["n1"]) >= {"transition", "state_vol"}
    t = np.asarray(params["n1"]["transition"], dtype=float)
    assert t.shape == (2, 2) and np.allclose(t.sum(axis=1), 1.0), "Uebergangszeilen muessen auf 1 summieren"
    # N2: je Instrument omega/alpha/beta mit Stationaritaet
    assert set(params["n2"]) == set(params["symbols"])
    for sym, p in params["n2"].items():
        assert p["alpha"] + p["beta"] < 1.0, f"{sym}: nichtstationaer"
    # N3: zwei Faktoren, Ladungen je Instrument, idiosynkratische Vola
    assert np.asarray(params["n3"]["loadings"]).shape == (19, 2)
    assert len(params["n3"]["idio_vol"]) == 19
    assert len(params["n3"]["factor_garch"]) == 2
    print("estimate_null_params: Struktur aller drei Welten: OK")


def check_freeze_params_roundtrip_and_hash() -> None:
    params = {"symbols": ["A"], "n1": {"transition": [[0.9, 0.1], [0.2, 0.8]]}}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.json")
        digest = freeze_params(params, path)
        assert len(digest) == 64
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == params
        assert freeze_params(params, os.path.join(tmp, "p2.json")) == digest, (
            "Gleicher Inhalt muss denselben Hash ergeben"
        )
    print("freeze_params: Roundtrip + stabiler Hash: OK")


def check_n3_reproduces_snapshot_correlation_and_vol() -> None:
    """Regressionstest fuer die N3-Skalierung: Ein aus den GESCHAETZTEN
    Parametern generiertes Panel muss die mittlere Querschnittskorrelation
    und die annualisierte Vola des echten Snapshots ungefaehr reproduzieren.

    Toleranz 0.03 auf beiden Groessen: die Seed-zu-Seed-Streuung liegt bei
    ca. 0.01 (Korrelation) bzw. 0.001 (Vola) -- weit innerhalb der Toleranz --
    waehrend der urspruengliche Skalierungsfehler (Faktor-Scores mit Varianz
    lambda statt 1) eine Abweichung von 0.076 (Korrelation: 0.086 vs. 0.162)
    und 0.080 (Vola: 0.285 vs. 0.205) erzeugte. 0.03 ist also eng genug, um
    diesen Bug sicher zu faengen, aber locher als die reine Seed-Varianz."""
    dfs = load_archive_snapshot()
    real = _returns_frame(dfs)
    off_diag = lambda c: c[~np.eye(len(c), dtype=bool)].mean()
    real_corr = off_diag(np.corrcoef(real.to_numpy(), rowvar=False))
    real_vol = real.std().mean() * np.sqrt(252)

    params = estimate_null_params(dfs)
    panel = generate_null_panel("n3", params, seed=0)
    synth = pd.DataFrame({s: panel[s]["price"].pct_change() for s in params["symbols"]}).dropna()
    synth_corr = off_diag(np.corrcoef(synth.to_numpy(), rowvar=False))
    synth_vol = synth.std().mean() * np.sqrt(252)

    assert abs(synth_corr - real_corr) < 0.03, (
        f"N3-Korrelation weicht zu stark vom Snapshot ab: synth={synth_corr:.3f} "
        f"echt={real_corr:.3f}"
    )
    assert abs(synth_vol - real_vol) < 0.03, (
        f"N3-Vola weicht zu stark vom Snapshot ab: synth={synth_vol:.3f} echt={real_vol:.3f}"
    )
    print(f"estimate_null_params: N3 reproduziert Snapshot-Korrelation/Vola "
          f"(synth={synth_corr:.3f}/{synth_vol:.3f}, echt={real_corr:.3f}/{real_vol:.3f}): OK")


def run_consistency_check() -> None:
    check_garch_grid_recovers_persistence()
    check_garch_grid_is_deterministic()
    check_estimate_null_params_structure()
    check_freeze_params_roundtrip_and_hash()
    check_n3_reproduces_snapshot_correlation_and_vol()
    print("\nAlle null_params-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
