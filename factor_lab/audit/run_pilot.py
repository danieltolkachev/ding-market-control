"""
run_pilot.py — Schritt 1 des Audits: Parameter auf dem Archiv-Snapshot
schaetzen und einfrieren, Kontroll-Seeds versiegeln, je Nullwelt 5
Replikationen fahren und die Laufzeit messen.

Der Pilot entscheidet ueber das Hauptbudget. Er behauptet KEINE
Fehlalarmrate -- 5 Replikationen je Welt haben dafuer keine Power.

Ausfuehren: py -3.12 factor_lab/audit/run_pilot.py [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from factor_lab.audit.audit_runner import run_one_replication
from factor_lab.audit.evaluation import false_alarm_rates, seal_control_seeds
from factor_lab.audit.null_params import estimate_null_params, freeze_params, load_archive_snapshot
from factor_lab.audit.null_worlds import WORLDS, generate_null_panel

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "audit_data")
PILOT_SEEDS = (0, 1, 2, 3, 4)


def _check_no_clobber(path: str, force: bool) -> None:
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"Verweigert: {path} existiert bereits und wuerde ueberschrieben "
            "(versiegeltes Artefakt). Mit --force erzwingen, falls beabsichtigt."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Erlaubt das Ueberschreiben bereits versiegelter Artefakte")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    params_path = os.path.join(DATA_DIR, "null_params.json")
    seeds_path = os.path.join(DATA_DIR, "audit_control_seeds.json")
    pilot_records_path = os.path.join(DATA_DIR, "pilot_records.json")
    _check_no_clobber(params_path, args.force)
    _check_no_clobber(seeds_path, args.force)
    _check_no_clobber(pilot_records_path, args.force)

    print("Schaetze Nullwelt-Parameter auf dem Archiv-Snapshot (read-only)...", flush=True)
    params = estimate_null_params(load_archive_snapshot())
    params_hash = freeze_params(params, params_path)
    print(f"  eingefroren: {params_path}\n  SHA256: {params_hash}", flush=True)

    seeds_hash = seal_control_seeds(seeds_path)
    print(f"  Kontroll-Seeds versiegelt: {seeds_path}\n  SHA256: {seeds_hash}", flush=True)

    for world in WORLDS:
        example = generate_null_panel(world, params, seed=0)
        frame = pd.DataFrame({s: df["price"] for s, df in example.items() if s != "IRX"})
        frame.to_csv(os.path.join(DATA_DIR, f"SYNTHETIC_NULL_{world}_seed0_example.csv"))

    records = []
    for world in WORLDS:
        for seed in PILOT_SEEDS:
            record = run_one_replication(world, params, seed)
            records.append(record)
            print(f"  {world} seed={seed}: candidate={record['candidate']} "
                  f"gate_a_hits={sum(record['gate_a'].values())}/8 "
                  f"{record['elapsed_s']:.1f}s", flush=True)

    with open(pilot_records_path, "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "params_sha256": params_hash,
                   "control_seeds_sha256": seeds_hash, "records": records}, f, indent=2)

    total = sum(r["elapsed_s"] for r in records)
    per_run = total / len(records)
    rates = false_alarm_rates(records)
    print(f"\nPilot: {len(records)} Replikationen, {total / 60:.1f} min gesamt, "
          f"{per_run:.1f} s je Lauf")
    print(f"Hochrechnung Hauptlauf (3 x 200): {per_run * 600 / 3600:.1f} h")
    print(f"Hochrechnung Kontrolllauf (3 x 50): {per_run * 150 / 3600:.1f} h")
    print(f"Orientierung (KEINE Aussage, zu wenig Power): FA_variant={rates['fa_variant']:.1%}, "
          f"FA_pipeline={rates['fa_pipeline']:.1%}")


if __name__ == "__main__":
    main()
