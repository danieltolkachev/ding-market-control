"""
run_n1_extension.py — N1-Erweiterung zur Aufloesung des Deviation-Criterion-Ausloesers.

Das Audit loeste das Abweichungs-Kriterium fuer n1 aus: 7.4% auf dem Hauptlauf
(Samen 0-99) gegenueber 1.5% auf den versiegelten Kontrolsamen (10000-10049).
Da WORKING_SEEDS 0..199 umfasst, aber der Hauptlauf nur 0..99 verbrauchte, sind
die Samen 100..199 noch unbenuetzt. Dieser Lauf repliziert die n1-Beobachtung
auf diesen PRE-REGISTRIERTEN Kontrolsamen, um zu pruefeng, ob die Abweichung
bei der Verdopplung der Stichprobe reproduziert wird.

Nur n1 wird erweitert, da nur n1 das Kriterium ausloeste.

Ausfuehren: py -3.12 factor_lab/audit/run_n1_extension.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from factor_lab.audit.audit_runner import run_one_replication
from factor_lab.audit.evaluation import WORKING_SEEDS, false_alarm_rates, verdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "audit_data")


def main() -> None:
    with open(os.path.join(DATA_DIR, "null_params.json"), encoding="utf-8") as f:
        params = json.load(f)

    seeds = WORKING_SEEDS[100:200]
    world = "n1"
    records = []

    for i, seed in enumerate(seeds):
        records.append(run_one_replication(world, params, seed))
        if (i + 1) % 10 == 0:
            print(f"  {world}: {i + 1}/{len(seeds)}", flush=True)

    path = os.path.join(DATA_DIR, "audit_records_n1_extension.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "label": "n1_extension", "records": records}, f, indent=2)

    rates = false_alarm_rates(records)
    print(f"{world}: FA_variant={rates['fa_variant']:.1%} "
          f"FA_pipeline={rates['fa_pipeline']:.1%}", flush=True)

    print(f"\nN1-Erweiterung (n={len(seeds)}): "
          f"FA_variant={rates['fa_variant']:.2%} CI{rates['fa_variant_ci95']}, "
          f"FA_pipeline={rates['fa_pipeline']:.2%} CI{rates['fa_pipeline_ci95']}")
    print(verdict(rates)["text"])


if __name__ == "__main__":
    main()
