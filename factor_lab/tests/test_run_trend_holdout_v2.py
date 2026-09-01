"""
test_run_trend_holdout_v2.py — Holdout der Familie trend-etf-v2: startet in
Cash NACH dev_end, wertet nur Holdout-Tage, volle Kostenleiter + Breakeven +
Borrow-Sensitivitaet (von Anfang an Teil der Familie, siehe die v1-
Nachbesserung dazu), verweigert ohne gueltiges Siegel und beim zweiten Mal --
unabhaengig von v1s eigenem Tombstone.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile
import numpy as np
import pandas as pd

from factor_lab import registration_v2
from factor_lab.run_trend_holdout_v2 import run_holdout
from factor_lab.run_trend_baseline_v2 import prepare_inputs

_TEST_SYMBOLS = ["SPY", "EFA", "TLT", "EMB", "GLD", "UUP", "USO"]


def _dfs(n_days: int = 1400, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    dfs = {s: pd.DataFrame({"price": 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.008, n_days))}, index=idx)
           for s in _TEST_SYMBOLS}
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * n_days}, index=idx)
    return dfs


def check_cost_ladder_and_breakeven(dfs: dict, dev_end, variant: str) -> dict:
    candidate = {"variant": variant, "dev_end": str(dev_end)}
    result = run_holdout(dfs, candidate)
    s = result["summary"]
    assert set(s["cost_ladder"]) == {"1.0", "2.0", "5.0"}, f"Erwartete volle Kostenleiter, bekam {sorted(s['cost_ladder'])}"
    assert "breakeven_cost_multiplier" in s
    return result


def check_borrow_sensitivity_long_short(dfs: dict, dev_end) -> None:
    result = check_cost_ladder_and_breakeven(dfs, dev_end, "mom63_long_short")
    sens = result["summary"]["borrow_sensitivity"]
    assert set(sens) == {"25.0", "50.0", "100.0"}, f"Erwartete 3 Borrow-Saetze, bekam {sorted(sens)}"
    cagrs = [sens[k]["cagr"] for k in ("25.0", "50.0", "100.0")]
    assert cagrs[0] >= cagrs[1] >= cagrs[2], f"Hoeherer Borrow-Satz darf CAGR nicht erhoehen, bekam {cagrs}"
    print("run_holdout_v2 Kostenleiter+Breakeven+Borrow-Sensitivitaet (long_short): OK")


def check_borrow_sensitivity_skipped_long_flat(dfs: dict, dev_end) -> None:
    result = check_cost_ladder_and_breakeven(dfs, dev_end, "mom63_long_flat")
    assert result["summary"]["borrow_sensitivity"] is None
    print("run_holdout_v2 Kostenleiter+Breakeven (long_flat, Borrow-Sensitivitaet uebersprungen): OK")


def run_consistency_check() -> None:
    dfs = _dfs()
    dev_end = prepare_inputs(dfs)["dev_end"]
    candidate = {"variant": "mom63_long_flat", "dev_end": str(dev_end)}
    result = run_holdout(dfs, candidate)

    first_day = pd.Timestamp(result["provenance"]["holdout_start"])
    assert first_day > dev_end
    assert result["summary"]["holdout_entry_cost"] > 0, "Positionsaufbau muss im Holdout bezahlt werden"
    assert "gate_d_no_single_driver" not in result["summary"]["gates"], "Holdout hat nur Gates A-C"
    assert set(result["summary"]["attribution_days"]) == {"holdout"}, "Attribution nur aus Holdout-Tagen"

    check_borrow_sensitivity_long_short(dfs, dev_end)
    check_borrow_sensitivity_skipped_long_flat(dfs, dev_end)

    with tempfile.TemporaryDirectory() as tmp:
        registration_v2.assert_no_tombstone_v2(tmp)
        registration_v2.write_tombstone_v2(tmp, "x.json")
        try:
            registration_v2.assert_no_tombstone_v2(tmp)
            raise AssertionError("Zweiter Holdout-Zugriff muss verweigert werden")
        except ValueError:
            pass
    print("run_holdout_v2: OK")


if __name__ == "__main__":
    run_consistency_check()
