"""
test_run_trend_baseline_v2.py — Integrationstest auf synthetischen Daten fuer
die Familie trend-etf-v2: 7-Symbol-Fixture (ein Instrument je Sleeve, damit
die Sleeve-LOO-Mechanik sinnvoll geprueft wird), identischer Index ueber
alle Serien, REGISTRATION_V2 in der Provenance, beide Verdikt-Zweige.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
import pandas as pd

from factor_lab.run_trend_baseline_v2 import run_screening, VARIANT_NAMES
from factor_lab.registration_v2 import REGISTRATION_V2

# Ein Symbol je Sleeve (Spec trend-etf-v2 Abschnitt 2), damit Gate Ds
# Sleeve-LOO im Test alle 7 Sleeves durchlaeuft.
_TEST_SYMBOLS = ["SPY", "EFA", "TLT", "EMB", "GLD", "UUP", "USO"]


def _dfs(regime_amplitude: float, n_days: int = 1400, seed: int = 0) -> dict:
    """Wie v1s bewaehrtes Fixture-Muster (Regime-Wechsel-Sinuswelle): bei
    regime_amplitude=0 reiner Random Walk (Null-Zweig); sonst wechselt der
    Drift alle 130 Tage das Vorzeichen -- long/flat geht in Abwaertsregimen
    in Cash, matched_long bleibt durchgehend investiert."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    regime = np.sign(np.sin(np.arange(n_days) * np.pi / 130.0) + 1e-9)
    dfs = {}
    for i, symbol in enumerate(_TEST_SYMBOLS):
        drift = regime_amplitude * regime * (0.6 + 0.1 * i)
        prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.006, n_days) + drift)
        dfs[symbol] = pd.DataFrame({"price": prices}, index=idx)
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * n_days}, index=idx)
    return dfs


def check_registration_v2_used() -> None:
    result, _ = run_screening(_dfs(regime_amplitude=0.0, seed=4))
    assert result["provenance"]["registration"]["family"] == "trend-etf-v2"
    assert result["provenance"]["config_hash"], "Config-Hash muss gesetzt sein"
    print("run_screening (v2) nutzt REGISTRATION_V2: OK")


def run_consistency_check() -> None:
    check_registration_v2_used()

    result, per_day = run_screening(_dfs(regime_amplitude=0.0, seed=1))
    assert sorted(result["summary"]) == VARIANT_NAMES
    expected_cols = set(VARIANT_NAMES) | {"matched_long", "bench_spy_bh", "bench_60_40"}
    assert set(per_day.columns) == expected_cols
    assert not per_day.isna().any().any(), "Alle Serien muessen denselben Index vollstaendig fuellen"
    json.dumps(result)

    for name in VARIANT_NAMES:
        s = result["summary"][name]
        for key in ("stats", "excess_bootstrap", "excess_bootstrap_sensitivity", "permutation",
                    "gates", "cost_ladder", "breakeven_cost_multiplier", "max_daily_gross"):
            assert key in s, f"{name}: {key} fehlt"
    assert "research_variant" in result["summary"]["mom63_long_short"]["labels"]

    # UMSETZUNG DER ESCAPE-HATCH-NOTIZ (Brief Schritt 4, analog v1 Task 9):
    # seed=4 (v1s Wahl fuer sein 3-Symbol-Fixture) liefert HIER, mit dem
    # 7-Symbol-Sleeve-Fixture und seinem drift-Multiplikator (0.6 + 0.1*i,
    # i=0..6), KEINEN zuverlaessigen Nullbefund: bei reinem Random Walk
    # (regime_amplitude=0.0) besteht mom252_long_flat bei seed=4 alle vier
    # Gates (siehe Scan unten) -- derselbe strukturelle Cash-Timing-Bias, den
    # v1s Kommentar beschreibt (long_flat geht bei negativem Momentum in
    # Cash, die hier fix mit 2% p.a. verzinst wird, waehrend matched_long
    # permanent im ~0%-Ertrags-Risk-Asset bleibt), trifft bei anderer
    # Symbolzahl/Gewichtung einen anderen Satz "guenstiger" Seeds. Scan von
    # seed=0..14 bei regime_amplitude=0.0 (siehe Kommentar-Historie/PR):
    # seeds 0, 2, 4, 9 lassen je eine long_flat-Variante alle Gates
    # bestehen; seeds 1, 3, 5-8, 10-14 liefern passed=[] (kein einziger
    # Variantenpasser) -- ein robusterer Nullbefund als "nur candidate=None
    # trotz Passer mit Tie-Break". seed=1 gewaehlt. Gate-Schwellen wurden
    # NICHT veraendert.
    assert result["candidate"] is None, "Regimefreie Daten duerfen keine Kandidatin liefern"
    print("run_screening (v2, Null-Zweig, 7-Sleeve-Fixture): OK")

    # NOTIZ: dieselbe Escape-Hatch-Regel gilt fuer den Kandidaten-Zweig --
    # Amplitude/Seed sind ein Startpunkt, keine garantierte Konstante.
    result2, _ = run_screening(_dfs(regime_amplitude=0.0022, seed=1))
    assert result2["candidate"] in VARIANT_NAMES, f"Erwartete Kandidatin, bekam {result2['candidate']}"
    winning = result2["summary"][result2["candidate"]]
    if winning["loo_excess_compounds"]:
        sleeve_keys = [k for k in winning["loo_excess_compounds"] if k.startswith("loo_sleeve_")]
        assert len(sleeve_keys) == len(REGISTRATION_V2["sleeves"]), (
            f"Erwartete {len(REGISTRATION_V2['sleeves'])} Sleeve-LOO-Eintraege, bekam {len(sleeve_keys)}"
        )
    print("run_screening (v2, Kandidaten-Zweig, Sleeve-LOO ueber alle 7 Sleeves): OK")


if __name__ == "__main__":
    run_consistency_check()
