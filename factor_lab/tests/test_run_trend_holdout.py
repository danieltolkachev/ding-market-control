"""
test_run_trend_holdout.py — Holdout: startet in Cash NACH dev_end, wertet
nur Holdout-Tage, verweigert ohne gueltiges Siegel und beim zweiten Mal.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile
import numpy as np
import pandas as pd

from factor_lab import registration
from factor_lab.run_trend_holdout import run_holdout
from factor_lab.run_trend_baseline import prepare_inputs


def _dfs(n_days: int = 1400, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    dfs = {s: pd.DataFrame({"price": 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.008, n_days))}, index=idx)
           for s in ["SPY", "TLT", "GLD"]}
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * n_days}, index=idx)
    return dfs


def run_consistency_check() -> None:
    dfs = _dfs()
    dev_end = prepare_inputs(dfs)["dev_end"]
    candidate = {"variant": "mom63_long_flat", "dev_end": str(dev_end)}
    result = run_holdout(dfs, candidate)

    # Alle PnL-Tage liegen NACH dev_end; erste Kosten (Positionsaufbau)
    # fallen im Holdout an -- keine geerbte Dev-Position.
    first_day = pd.Timestamp(result["provenance"]["holdout_start"])
    assert first_day > dev_end
    assert result["summary"]["holdout_entry_cost"] > 0, "Positionsaufbau muss im Holdout bezahlt werden"
    assert "gate_d_no_single_driver" not in result["summary"]["gates"], "Holdout hat nur Gates A-C"
    assert set(result["summary"]["attribution_days"]) == {"holdout"}, "Attribution nur aus Holdout-Tagen"

    # Tombstone-Einmaligkeit (auf Task-7-Funktionen, hier als Integrationspruefung)
    with tempfile.TemporaryDirectory() as tmp:
        registration.assert_no_tombstone(tmp)
        registration.write_tombstone(tmp, "x.json")
        try:
            registration.assert_no_tombstone(tmp)
            raise AssertionError("Zweiter Holdout-Zugriff muss verweigert werden")
        except ValueError:
            pass
    print("run_holdout: OK")


if __name__ == "__main__":
    run_consistency_check()
