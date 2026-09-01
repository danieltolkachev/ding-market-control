"""
registration_v2.py — operationale Praeregistrierung der Familie trend-etf-v2
(Spec docs/superpowers/specs/2026-09-01-daily-factor-lab-trend-v2-universe-design.md):
erweitertes 19-Instrumenten-Universum (die 12 aus trend-etf-v1 UNVERAENDERT
plus 7 neue: Waehrungen, disaggregierte Rohstoffe, EM-Anleihen).

Bewusst ein VOLLSTAENDIG eigenstaendiges Modul, das aus registration.py
(trend-etf-v1) NICHTS ausser den beiden bereits generischen, parametrisierten
Funktionen config_hash() und file_sha256() importiert. write_candidate/
read_and_verify_candidate/tombstone_path/assert_no_tombstone/write_tombstone
sind in registration.py hart an die Modul-Level-Konstante FAMILY="trend-etf-v1"
gekoppelt (keine Parameter) und werden deshalb hier bewusst DUPLIZIERT statt
wiederverwendet -- Ziel ist echte Isolation: ein Bug hier darf den bereits
gemergten, hart geprueften v1-Code (inkl. des irreversiblen Holdout-Pfads)
niemals beruehren koennen.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from factor_lab.registration import config_hash, file_sha256
from factor_lab.data_snapshot import SNAPSHOT_DIR
from factor_lab.costs import COST_BP

FAMILY_V2 = "trend-etf-v2"

REGISTRATION_V2: dict = {
    "family": FAMILY_V2,
    "universe": [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ",
        "UUP", "FXE", "FXY", "USO", "UNG", "DBA", "EMB",
    ],
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
    "cost_bp": {
        "SPY": 1.5, "QQQ": 1.5, "IWM": 1.5, "TLT": 1.5, "IEF": 1.5, "GLD": 1.5,
        "EFA": 3.0, "EEM": 3.0, "LQD": 3.0, "SLV": 3.0, "DBC": 3.0, "VNQ": 3.0,
        "UUP": 3.0, "FXE": 3.0, "FXY": 3.0, "USO": 3.0, "UNG": 3.0, "DBA": 3.0, "EMB": 3.0,
    },
    "cost_ladder": [1.0, 2.0, 5.0],
    "borrow_bp_pa": 50.0,
    "bootstrap": {"kind": "stationary", "expected_block_len_months": 6.0,
                  "sensitivity_block_lens": [3.0, 12.0], "n_boot": 10000, "seed": 0},
    "permutation_n": 10000,
    "dd_cap": 0.15,
    "gate_c_floor": 0.02,
    "candidate_rule": "hoechster Excess-Sharpe unter Bestehenden, Tie-Break alphabetisch",
    "holdout_gates": ["A", "B", "C"],
    "min_common_days": 4200,
    "sleeves": {
        "us_equity": ["SPY", "QQQ", "IWM"],
        "intl_equity": ["EFA", "EEM"],
        "bonds": ["TLT", "IEF", "LQD"],
        "em_bonds": ["EMB"],
        "real_assets": ["GLD", "SLV", "DBC", "VNQ"],
        "currencies": ["UUP", "FXE", "FXY"],
        "granular_commodities": ["USO", "UNG", "DBA"],
    },
}


def assert_cost_bp_consistency_v2() -> None:
    """Review-Fund: run_trend_baseline_v2.py fuehrt tatsaechlich mit
    factor_lab.costs.COST_BP aus (der geteilten, wachsenden Multi-Familien-
    Kostentabelle), aber der Wert, der ins Siegel gehasht wird, ist die
    separate, eingefrorene Kopie REGISTRATION_V2["cost_bp"]. Beide stimmen
    heute fuer alle 19 Symbole ueberein -- costs.py wird aber fuer
    kuenftige Familien weiter editiert, und eine NICHT-additive Aenderung
    dort (z.B. ein bestehendes Symbol-bp veraendert) wuerde v2s reale
    Wirtschaftlichkeit lautlos veraendern, OHNE dass sich config_hash
    aendert -- eine manipulierte/abgedriftete Ausfuehrung wuerde weiterhin
    als dieselbe versiegelte Kandidatin verifizieren. Diese Pruefung faengt
    das ab, bevor Screening oder Holdout irgendetwas Geldrelevantes
    berechnen (sie laeuft beim Import dieses Moduls, das beide Runner vor
    jeder Rechnung importieren). REGISTRATION_V2["cost_bp"] selbst bleibt
    dabei unveraendert -- nur eine Konsistenzpruefung, kein Fix."""
    mismatches = sorted(
        s for s, bp in REGISTRATION_V2["cost_bp"].items() if COST_BP.get(s) != bp
    )
    if mismatches:
        raise ValueError(
            "costs.COST_BP weicht von REGISTRATION_V2['cost_bp'] ab fuer: "
            f"{', '.join(mismatches)} -- Siegel-Oekonomie und tatsaechliche "
            "Ausfuehrung sind nicht mehr identisch (Amendment-Regel: costs.py "
            "wurde nicht-additiv geaendert, seit v2 versiegelt hat)."
        )


assert_cost_bp_consistency_v2()


def trend_snapshot_path_v2() -> str:
    key = (f"{sorted(REGISTRATION_V2['universe'])}|{REGISTRATION_V2['cash_series']}"
           f"|{REGISTRATION_V2['snapshot_start']}|{REGISTRATION_V2['snapshot_end_exclusive']}")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(SNAPSHOT_DIR, f"trend_snapshot_{digest}.pkl")


def write_candidate_v2(path: str, variant: str, snapshot_sha256: str, git_sha: str,
                       dev_end: str, results_sha256: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"candidate.json bereits versiegelt: {path} -- kein Ueberschreiben erlaubt")
    payload = {
        "family": FAMILY_V2,
        "config_hash": config_hash(REGISTRATION_V2),
        "variant": variant,
        "snapshot_sha256": snapshot_sha256,
        "git_sha": git_sha,
        "dev_end": dev_end,
        "results_sha256": results_sha256,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_and_verify_candidate_v2(path: str, snapshot_sha256: str) -> dict:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("family") != FAMILY_V2:
        raise ValueError(f"candidate.json gehoert zu Familie {payload.get('family')}, erwartet {FAMILY_V2}")
    if payload.get("config_hash") != config_hash(REGISTRATION_V2):
        raise ValueError("config_hash weicht ab -- Registrierung wurde nach dem Siegel geaendert (Amendment-Regel: neue Familie noetig)")
    if payload.get("snapshot_sha256") != snapshot_sha256:
        raise ValueError("Snapshot-Hash weicht ab -- das ist nicht der versiegelte Datensatz")
    return payload


def tombstone_path_v2(logs_dir: str) -> str:
    return os.path.join(logs_dir, f"holdout_tombstone_{FAMILY_V2}.json")


def assert_no_tombstone_v2(logs_dir: str) -> None:
    path = tombstone_path_v2(logs_dir)
    if os.path.exists(path):
        raise ValueError(f"Holdout der Familie {FAMILY_V2} wurde bereits ausgefuehrt ({path}) -- kein zweiter Zugriff")


def write_tombstone_v2(logs_dir: str, holdout_result_path: str) -> None:
    with open(tombstone_path_v2(logs_dir), "w", encoding="utf-8") as f:
        json.dump({"family": FAMILY_V2, "holdout_result": holdout_result_path,
                   "executed_utc": datetime.now(timezone.utc).isoformat()}, f, indent=2)
