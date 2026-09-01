"""
test_registration.py — Config-Hash-Stabilitaet, Kandidaten-Siegel und
Tombstone-Einmaligkeit.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile

from factor_lab import registration


def run_consistency_check() -> None:
    h1 = registration.config_hash()
    h2 = registration.config_hash()
    assert h1 == h2 and len(h1) == 64, "Config-Hash muss deterministisch sein"
    changed = dict(registration.REGISTRATION)
    changed["gate_c_floor"] = 0.03
    assert registration.config_hash(changed) != h1, "Parameteraenderung muss den Hash aendern"

    # Insertions-Reihenfolge (top-level UND verschachtelt) darf den Hash nicht aendern,
    # sonst waere sort_keys=True wirkungslos bzw. unnoetig.
    reordered = {}
    for key in reversed(list(registration.REGISTRATION.keys())):
        reordered[key] = registration.REGISTRATION[key]
    reordered["cost_bp"] = dict(reversed(list(reordered["cost_bp"].items())))
    reordered["bootstrap"] = dict(reversed(list(reordered["bootstrap"].items())))
    assert registration.config_hash(reordered) == h1, \
        "Config-Hash darf nicht von der Dict-Insertions-Reihenfolge abhaengen"

    with tempfile.TemporaryDirectory() as tmp:
        cand = os.path.join(tmp, "candidate.json")
        registration.write_candidate(cand, variant="combo_long_flat", snapshot_sha256="abc",
                                     git_sha="deadbeef", dev_end="2022-10-31", results_sha256="123")
        loaded = registration.read_and_verify_candidate(cand, snapshot_sha256="abc")
        assert loaded["variant"] == "combo_long_flat" and loaded["family"] == registration.FAMILY

        # Ueberschreiben eines bereits versiegelten candidate.json -> Verweigerung
        try:
            registration.write_candidate(cand, variant="anderer_variant", snapshot_sha256="abc",
                                         git_sha="deadbeef", dev_end="2022-10-31", results_sha256="123")
            raise AssertionError("Ueberschreiben eines versiegelten candidate.json muss ValueError ausloesen")
        except ValueError:
            pass

        # Falscher Snapshot-Hash -> Verweigerung
        try:
            registration.read_and_verify_candidate(cand, snapshot_sha256="anders")
            raise AssertionError("Falscher Snapshot-Hash muss ValueError ausloesen")
        except ValueError:
            pass

        # Manipulierte Datei (anderer config_hash) -> Verweigerung
        with open(cand, encoding="utf-8") as f:
            payload = json.load(f)
        payload["config_hash"] = "0" * 64
        with open(cand, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        try:
            registration.read_and_verify_candidate(cand, snapshot_sha256="abc")
            raise AssertionError("Manipulierter config_hash muss ValueError ausloesen")
        except ValueError:
            pass

        # Tombstone: vorher ok, nachher Verweigerung
        registration.assert_no_tombstone(tmp)
        registration.write_tombstone(tmp, holdout_result_path="ergebnis.json")
        try:
            registration.assert_no_tombstone(tmp)
            raise AssertionError("Existierender Tombstone muss ValueError ausloesen")
        except ValueError:
            pass
    print("registration: Config-Hash, Siegel, Tombstone -- OK")


if __name__ == "__main__":
    run_consistency_check()
