"""
test_build_trend_snapshot_v2.py — netzwerkfreie Pruefungen: v2-Snapshot-Pfad
unterscheidet sich von v1s (unterschiedliches Universum -> unterschiedlicher
Hash, keine Kollision), resolve_dev_end() auf einem synthetischen Fixture.
fetch_all()/main() brauchen echten Netzwerkzugriff und sind (wie das
restliche Netzwerk-Fetching in diesem Package) bewusst NICHT unit-getestet.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.build_trend_snapshot_v2 import resolve_dev_end
from factor_lab.registration_v2 import trend_snapshot_path_v2
from factor_lab.data_snapshot import trend_snapshot_path


def check_v2_path_differs_from_v1() -> None:
    assert trend_snapshot_path_v2() != trend_snapshot_path(), (
        "v2-Snapshot-Pfad muss sich vom v1-Pfad unterscheiden (anderes Universum -> anderer Hash)"
    )
    print("trend_snapshot_path_v2 != trend_snapshot_path (v1): OK")


def check_resolve_dev_end() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    dfs = {s: pd.DataFrame({"price": 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, 500))}, index=idx)
           for s in ["SPY", "UUP", "USO", "EMB"]}
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * 500}, index=idx)
    dev_end = resolve_dev_end(dfs)
    assert dev_end <= idx[int(500 * 0.8)], "DEV_END muss innerhalb des 80%-Quantil-Fensters liegen"
    assert dev_end in idx, "DEV_END muss ein echter Handelstag aus dem gemeinsamen Kalender sein"
    print("resolve_dev_end (v2, synthetisches Fixture): OK")


def run_consistency_check() -> None:
    check_v2_path_differs_from_v1()
    check_resolve_dev_end()
    print("\nAlle build_trend_snapshot_v2-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
