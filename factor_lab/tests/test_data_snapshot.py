"""
test_data_snapshot.py — Load-only-Verhalten: Hash-Verifikation gegen das
Manifest, Manipulation -> Fehler, Sanity-Checks fail-closed. Kein Netzwerk.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pickle
import tempfile
import pandas as pd

from factor_lab import data_snapshot


def _fake_dfs():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    dfs = {s: pd.DataFrame({"price": [100.0 + i for i in range(5)]}, index=idx) for s in ["AAA", "BBB"]}
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [4.0] * 5}, index=idx)
    return dfs


def run_consistency_check() -> None:
    dfs = _fake_dfs()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trend_snapshot_test.pkl")
        with open(path, "wb") as f:
            pickle.dump(dfs, f)
        data_snapshot.write_snapshot_manifest(dfs, path + ".manifest.json")

        loaded = data_snapshot.load_trend_snapshot(path=path)
        pd.testing.assert_frame_equal(loaded["AAA"], dfs["AAA"])

        # Pflichttest Review: Manipulation des Pickles -> Hashfehler beim Load
        dfs_tampered = _fake_dfs()
        dfs_tampered["AAA"].iloc[0, 0] += 0.5
        with open(path, "wb") as f:
            pickle.dump(dfs_tampered, f)
        try:
            data_snapshot.load_trend_snapshot(path=path)
            raise AssertionError("Manipulierter Snapshot muss ValueError ausloesen")
        except ValueError:
            pass
    print("load_trend_snapshot (Hash-Verifikation): OK")

    # Sanity-Checks fail-closed
    bad = _fake_dfs()
    bad["AAA"].iloc[2, 0] = -1.0
    try:
        data_snapshot.sanity_check_snapshot(bad, min_common_days=3)
        raise AssertionError("Nichtpositiver Preis muss ValueError ausloesen")
    except ValueError:
        pass
    bad = _fake_dfs()
    bad["BBB"] = bad["BBB"].iloc[:2]  # gemeinsamer Kalender schrumpft unter min_common_days
    try:
        data_snapshot.sanity_check_snapshot(bad, min_common_days=3)
        raise AssertionError("Zu kurzer gemeinsamer Kalender muss ValueError ausloesen")
    except ValueError:
        pass
    data_snapshot.sanity_check_snapshot(_fake_dfs(), min_common_days=3)
    print("sanity_check_snapshot (fail-closed): OK")


if __name__ == "__main__":
    run_consistency_check()
