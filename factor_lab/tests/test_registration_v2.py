"""
test_registration_v2.py — Config-Hash, Kandidaten-Siegel, Tombstone fuer die
Familie trend-etf-v2 (Wiederholung von test_registration.py's Pruefungen fuer
die neue Familie, PLUS Familien-Isolation: v2 darf v1 nicht beeinflussen und
umgekehrt -- das ist der ganze Zweck der Duplizierung statt Parametrisierung).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile

from factor_lab.registration import config_hash
from factor_lab import registration_v2
from factor_lab import registration as registration_v1


def check_config_hash() -> None:
    h1 = config_hash(registration_v2.REGISTRATION_V2)
    h2 = config_hash(registration_v2.REGISTRATION_V2)
    assert h1 == h2 and len(h1) == 64, "Config-Hash muss deterministisch sein"
    changed = dict(registration_v2.REGISTRATION_V2)
    changed["gate_c_floor"] = 0.03
    assert config_hash(changed) != h1, "Parameteraenderung muss den Hash aendern"
    assert config_hash(registration_v2.REGISTRATION_V2) != config_hash(registration_v1.REGISTRATION), (
        "v1- und v2-Config-Hash duerfen niemals zufaellig gleich sein"
    )
    assert registration_v2.REGISTRATION_V2["family"] == "trend-etf-v2"
    assert len(registration_v2.REGISTRATION_V2["universe"]) == 19
    assert len(registration_v2.REGISTRATION_V2["sleeves"]) == 7
    assert sum(len(v) for v in registration_v2.REGISTRATION_V2["sleeves"].values()) == 19, (
        "Jedes der 19 Instrumente muss in genau einem Sleeve vorkommen"
    )
    print("registration_v2 config_hash + Struktur: OK")


def check_candidate_seal_and_tamper_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cand = os.path.join(tmp, "candidate.json")
        registration_v2.write_candidate_v2(cand, variant="combo_long_flat", snapshot_sha256="abc",
                                           git_sha="deadbeef", dev_end="2022-10-31", results_sha256="123")
        loaded = registration_v2.read_and_verify_candidate_v2(cand, snapshot_sha256="abc")
        assert loaded["variant"] == "combo_long_flat" and loaded["family"] == registration_v2.FAMILY_V2

        try:
            registration_v2.read_and_verify_candidate_v2(cand, snapshot_sha256="anders")
            raise AssertionError("Falscher Snapshot-Hash muss ValueError ausloesen")
        except ValueError:
            pass

        with open(cand, encoding="utf-8") as f:
            payload = json.load(f)
        payload["config_hash"] = "0" * 64
        with open(cand, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        try:
            registration_v2.read_and_verify_candidate_v2(cand, snapshot_sha256="abc")
            raise AssertionError("Manipulierter config_hash muss ValueError ausloesen")
        except ValueError:
            pass

        fresh = os.path.join(tmp, "fresh_candidate.json")
        registration_v2.write_candidate_v2(fresh, variant="mom63_long_flat", snapshot_sha256="xyz",
                                           git_sha="cafebabe", dev_end="2022-10-31", results_sha256="456")
        try:
            registration_v2.write_candidate_v2(fresh, variant="mom63_long_short", snapshot_sha256="xyz",
                                               git_sha="cafebabe", dev_end="2022-10-31", results_sha256="456")
            raise AssertionError("Zweiter Write auf denselben Pfad muss ValueError ausloesen (unveraenderliches Siegel)")
        except ValueError:
            pass
    print("registration_v2 candidate seal + tamper detection + write-once: OK")


def check_family_isolation_from_v1_candidate() -> None:
    """Ein v1-Kandidat darf von der v2-Verifikation NIEMALS akzeptiert werden
    -- selbst bei identischem Snapshot-Hash."""
    with tempfile.TemporaryDirectory() as tmp:
        v1_cand = os.path.join(tmp, "v1_candidate.json")
        registration_v1.write_candidate(v1_cand, variant="combo_long_flat", snapshot_sha256="same_hash",
                                        git_sha="deadbeef", dev_end="2022-08-31", results_sha256="123")
        try:
            registration_v2.read_and_verify_candidate_v2(v1_cand, snapshot_sha256="same_hash")
            raise AssertionError("v1-Kandidat darf von v2-Verifikation nicht akzeptiert werden (Familie stimmt nicht)")
        except ValueError:
            pass
    print("registration_v2 Familien-Isolation gegen trend-etf-v1-Kandidaten: OK")


def check_tombstone_isolation() -> None:
    """v1- und v2-Tombstones im selben logs_dir duerfen sich nicht
    gegenseitig blockieren oder freigeben."""
    with tempfile.TemporaryDirectory() as tmp:
        registration_v1.assert_no_tombstone(tmp)
        registration_v2.assert_no_tombstone_v2(tmp)

        registration_v1.write_tombstone(tmp, "v1_result.json")
        registration_v2.assert_no_tombstone_v2(tmp)  # v1s Tombstone darf v2 NICHT blockieren

        registration_v2.write_tombstone_v2(tmp, "v2_result.json")
        try:
            registration_v2.assert_no_tombstone_v2(tmp)
            raise AssertionError("Zweiter v2-Holdout-Zugriff muss verweigert werden")
        except ValueError:
            pass
        try:
            registration_v1.assert_no_tombstone(tmp)
            raise AssertionError("v1-Tombstone muss unabhaengig weiterhin aktiv sein")
        except ValueError:
            pass
    print("registration_v2/v1 Tombstone-Isolation: OK")


def run_consistency_check() -> None:
    check_config_hash()
    check_candidate_seal_and_tamper_detection()
    check_family_isolation_from_v1_candidate()
    check_tombstone_isolation()
    print("\nAlle registration_v2-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
