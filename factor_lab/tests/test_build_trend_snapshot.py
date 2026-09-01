"""
test_build_trend_snapshot.py — nur der netzwerkfreie Teil von
build_trend_snapshot.py: add_yfinance_version_to_manifest() als reiner
Datei-Roundtrip auf einem synthetischen Manifest. fetch_all()/main() selbst
brauchen echten yfinance-Netzwerkzugriff und sind (wie das restliche
Netzwerk-Fetching in diesem Package) bewusst NICHT unit-getestet.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile

from factor_lab.build_trend_snapshot import add_yfinance_version_to_manifest


def check_add_yfinance_version_to_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = os.path.join(tmp, "snap.pkl.manifest.json")
        original = {"content_sha256": "abc123", "symbols": {"SPY": {"rows": 5}}}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(original, f, indent=2)

        add_yfinance_version_to_manifest(manifest_path, "0.2.40")

        with open(manifest_path, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
        assert reloaded["yfinance_version"] == "0.2.40"
        # Bestehende Felder muessen unangetastet bleiben (reiner Zusatz).
        assert reloaded["content_sha256"] == "abc123"
        assert reloaded["symbols"] == {"SPY": {"rows": 5}}
    print("add_yfinance_version_to_manifest (Roundtrip, additiv): OK")


def run_consistency_check() -> None:
    check_add_yfinance_version_to_manifest()
    print("\nAlle build_trend_snapshot-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
