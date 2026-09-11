"""
test_audit_evaluation.py — Fehlalarmraten, vorab festgelegte Kriterien
und Versiegelung der Kontroll-Seeds.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile

from factor_lab.audit.evaluation import (
    CONTROL_SEEDS, WORKING_SEEDS, false_alarm_rates, seal_control_seeds, verdict,
)


def _record(gate_a_hits: int, candidate: bool) -> dict:
    variants = [f"v{i}" for i in range(8)]
    return {
        "world": "n2", "seed": 0,
        "candidate": "v0" if candidate else None,
        "gate_a": {v: (i < gate_a_hits) for i, v in enumerate(variants)},
        "passed_all": {v: (candidate and v == "v0") for v in variants},
        "excess_lower95": {v: 0.0 for v in variants},
    }


def check_rates_counted_over_the_right_denominators() -> None:
    records = [_record(1, True)] + [_record(0, False)] * 9
    rates = false_alarm_rates(records)
    assert rates["n_replications"] == 10
    assert rates["n_variant_tests"] == 80, "8 Varianten x 10 Laeufe"
    assert abs(rates["fa_variant"] - 1 / 80) < 1e-12
    assert abs(rates["fa_pipeline"] - 1 / 10) < 1e-12
    low, high = rates["fa_variant_ci95"]
    assert low <= rates["fa_variant"] <= high
    print("false_alarm_rates: korrekte Nenner und CIs: OK")


def check_verdict_thresholds() -> None:
    calibrated = verdict({"fa_variant": 0.05, "fa_pipeline": 0.08})
    assert calibrated["calibrated"] and not calibrated["selection_inflated"]

    inflated = verdict({"fa_variant": 0.05, "fa_pipeline": 0.22})
    assert inflated["calibrated"] and inflated["selection_inflated"]

    broken = verdict({"fa_variant": 0.18, "fa_pipeline": 0.30})
    assert not broken["calibrated"], "FA_variant > 10% muss als fehlkalibriert gelten"

    conservative = verdict({"fa_variant": 0.01, "fa_pipeline": 0.02})
    assert not conservative["calibrated"], "FA_variant < 2.5% liegt ebenfalls ausserhalb"
    print("verdict: vorab festgelegte Schwellen: OK")


def check_seeds_are_disjoint_and_sealed() -> None:
    assert set(WORKING_SEEDS).isdisjoint(CONTROL_SEEDS), "Seed-Mengen muessen disjunkt sein"
    assert len(WORKING_SEEDS) == 200 and len(CONTROL_SEEDS) == 50
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "seeds.json")
        digest = seal_control_seeds(path)
        assert len(digest) == 64
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["control_seeds"] == list(CONTROL_SEEDS)
        assert seal_control_seeds(os.path.join(tmp, "s2.json")) == digest
    print("Kontroll-Seeds: disjunkt, versiegelbar, stabiler Hash: OK")


def run_consistency_check() -> None:
    check_rates_counted_over_the_right_denominators()
    check_verdict_thresholds()
    check_seeds_are_disjoint_and_sealed()
    print("\nAlle audit_evaluation-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
