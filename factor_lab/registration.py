"""
registration.py — operationale Praeregistrierung der Familie trend-etf-v1
(Spec v2 Abschnitte 10+12): Config-Hash, unveraenderliches candidate.json,
Tombstone-Einmaligkeit fuer das Holdout.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

FAMILY = "trend-etf-v1"

REGISTRATION: dict = {
    "family": FAMILY,
    "universe": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"],
    "cash_series": "^IRX",
    "snapshot_start": "2007-01-01",
    "snapshot_end_exclusive": "2026-09-01",
    "dev_end_rule": "letzter Monatsultimo <= 80%-Quantil-Datum des gemeinsamen Kalenders",
    "warmup_days": 252 + 63,
    "lookbacks": [63, 126, 252],
    "signals": ["mom63", "mom126", "mom252", "combo"],
    "modes": ["long_short", "long_flat"],
    "ewma_span": 63,
    "vol_cap": 0.10,
    "gross_cap_targets_only": True,
    "rebalance": "monatsultimo_entscheid_fill_naechster_close",
    "cost_bp": {"SPY": 1.5, "QQQ": 1.5, "IWM": 1.5, "TLT": 1.5, "IEF": 1.5, "GLD": 1.5,
                "EFA": 3.0, "EEM": 3.0, "LQD": 3.0, "SLV": 3.0, "DBC": 3.0, "VNQ": 3.0},
    "cost_ladder": [1.0, 2.0, 5.0],
    "borrow_bp_pa": 50.0,
    "bootstrap": {"kind": "stationary", "expected_block_len_months": 6.0,
                  "sensitivity_block_lens": [3.0, 12.0], "n_boot": 10000, "seed": 0},
    "permutation_n": 10000,
    "dd_cap": 0.15,
    "gate_c_floor": 0.02,
    "candidate_rule": "hoechster Excess-Sharpe unter Bestehenden, Tie-Break alphabetisch",
    "holdout_gates": ["A", "B", "C"],
}


def config_hash(registration: dict = REGISTRATION) -> str:
    canonical = json.dumps(registration, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def write_candidate(path: str, variant: str, snapshot_sha256: str, git_sha: str,
                    dev_end: str, results_sha256: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"candidate.json bereits versiegelt: {path} -- kein Ueberschreiben erlaubt")
    payload = {
        "family": FAMILY,
        "config_hash": config_hash(),
        "variant": variant,
        "snapshot_sha256": snapshot_sha256,
        "git_sha": git_sha,
        "dev_end": dev_end,
        "results_sha256": results_sha256,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_and_verify_candidate(path: str, snapshot_sha256: str) -> dict:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("family") != FAMILY:
        raise ValueError(f"candidate.json gehoert zu Familie {payload.get('family')}, erwartet {FAMILY}")
    if payload.get("config_hash") != config_hash():
        raise ValueError("config_hash weicht ab -- Registrierung wurde nach dem Siegel geaendert (Amendment-Regel: neue Familie noetig)")
    if payload.get("snapshot_sha256") != snapshot_sha256:
        raise ValueError("Snapshot-Hash weicht ab -- das ist nicht der versiegelte Datensatz")
    return payload


def tombstone_path(logs_dir: str) -> str:
    return os.path.join(logs_dir, f"holdout_tombstone_{FAMILY}.json")


def assert_no_tombstone(logs_dir: str) -> None:
    path = tombstone_path(logs_dir)
    if os.path.exists(path):
        raise ValueError(f"Holdout der Familie {FAMILY} wurde bereits ausgefuehrt ({path}) -- kein zweiter Zugriff")


def write_tombstone(logs_dir: str, holdout_result_path: str) -> None:
    with open(tombstone_path(logs_dir), "w", encoding="utf-8") as f:
        json.dump({"family": FAMILY, "holdout_result": holdout_result_path,
                   "executed_utc": datetime.now(timezone.utc).isoformat()}, f, indent=2)
