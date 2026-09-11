"""
run_audit.py — Haupt- und Kontrolllauf des Falsifikations-Audits.

Liest die im Piloten EINGEFRORENEN Parameter; schaetzt nicht neu. Der
Kontrolllauf darf erst laufen, wenn Code und Auswertung eingefroren sind
(--control), und wird getrennt gespeichert.

Ausfuehren: py -3.12 factor_lab/audit/run_audit.py [--n N] [--control]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from factor_lab.audit.audit_runner import run_one_replication
from factor_lab.audit.evaluation import CONTROL_SEEDS, WORKING_SEEDS, false_alarm_rates, verdict
from factor_lab.audit.null_worlds import WORLDS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "audit_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(DATA_DIR, "null_params.json"), encoding="utf-8") as f:
        params = json.load(f)

    seeds = CONTROL_SEEDS if args.control else WORKING_SEEDS[:args.n]
    label = "control" if args.control else "main"
    records = []
    for world in WORLDS:
        for i, seed in enumerate(seeds):
            records.append(run_one_replication(world, params, seed))
            if (i + 1) % 10 == 0:
                print(f"  {world}: {i + 1}/{len(seeds)}", flush=True)
        rates = false_alarm_rates([r for r in records if r["world"] == world])
        print(f"{world}: FA_variant={rates['fa_variant']:.1%} "
              f"FA_pipeline={rates['fa_pipeline']:.1%}", flush=True)

    path = os.path.join(DATA_DIR, f"audit_records_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "label": label, "records": records}, f, indent=2)

    overall = false_alarm_rates(records)
    print(f"\nGesamt ({label}, n={len(seeds)} je Welt): "
          f"FA_variant={overall['fa_variant']:.2%} CI{overall['fa_variant_ci95']}, "
          f"FA_pipeline={overall['fa_pipeline']:.2%} CI{overall['fa_pipeline_ci95']}")
    print(verdict(overall)["text"])


if __name__ == "__main__":
    main()
