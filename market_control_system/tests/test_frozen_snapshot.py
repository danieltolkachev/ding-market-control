"""
test_frozen_snapshot.py
==========================

Prueft, dass load_or_build_snapshot() beim ersten Aufruf eine Datei
anlegt und bei jedem weiteren Aufruf mit denselben Parametern GENAU
dieselben Daten zurueckgibt, ohne erneut zu fetchen (kein Netzwerkzugriff
noetig fuer diesen Test -- build_snapshot() wird durch eine Stub-Funktion
ersetzt, die deterministische synthetische Daten liefert).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))

import pandas as pd

import frozen_snapshot


def _fake_build_snapshot(universe, lookback_days):
    """Deterministischer Ersatz fuer build_snapshot() -- keine echten
    Netzwerkaufrufe im Test."""
    return {
        symbol: pd.DataFrame({"price": [100.0 + i for i in range(5)]})
        for symbol in universe
    }


def run_consistency_check() -> None:
    universe = ["AAA", "BBB"]
    lookback_days = 7

    path = frozen_snapshot.snapshot_path(universe, lookback_days)
    if os.path.exists(path):
        os.remove(path)

    original_build = frozen_snapshot.build_snapshot
    frozen_snapshot.build_snapshot = _fake_build_snapshot
    try:
        first = frozen_snapshot.load_or_build_snapshot(universe, lookback_days)
        assert os.path.exists(path), f"Snapshot-Datei wurde nicht angelegt: {path}"

        second = frozen_snapshot.load_or_build_snapshot(universe, lookback_days)
        for symbol in universe:
            pd.testing.assert_frame_equal(first[symbol], second[symbol])

        # Andere lookback_days -> anderer Pfad, kein Ueberschreiben
        other_path = frozen_snapshot.snapshot_path(universe, lookback_days + 1)
        assert other_path != path, "Unterschiedliche Parameter muessen unterschiedliche Snapshot-Pfade ergeben"

        print("frozen_snapshot: Datei angelegt, wiederholter Load liefert identische Daten, "
              "unterschiedliche Parameter -> unterschiedlicher Pfad -- OK")
    finally:
        frozen_snapshot.build_snapshot = original_build
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    run_consistency_check()
