# Daily-Factor-Lab trend-etf-v2 (Expanded Universe) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `trend-etf-v2` pre-registered family (19-instrument universe, 7 new instruments purely additive over `trend-etf-v1`'s 12) as a fully isolated module set, then run it for real.

**Architecture:** A parallel module set (`registration_v2.py`, `build_trend_snapshot_v2.py`, `run_trend_baseline_v2.py`, `run_trend_holdout_v2.py`) that reuses `signals.py`/`portfolio.py`/`stats.py` unchanged and reuses `data_snapshot.py`'s already-generic load/sanity/hash functions by passing an explicit v2-specific snapshot path — zero modifications to those files. The only shared file touched is `costs.py` (purely additive: 7 new `COST_BP` entries). `registration.py` (v1) is touched NOT AT ALL — `registration_v2.py` duplicates its family-specific sealing/tombstone functions (`write_candidate`/`read_and_verify_candidate`/`tombstone_path`/`assert_no_tombstone`/`write_tombstone` are hardcoded to v1's module-level `FAMILY`/`REGISTRATION` constants, not parametrized) rather than risk modifying that already-hardened, already-shipped tamper-evidence code. Only `config_hash()` and `file_sha256()` — already generic/parametrized — are imported and reused directly.

**Tech Stack:** Python 3.12 (`py -3.12`), pandas, numpy, yfinance (build script only).

**Spec:** `docs/superpowers/specs/2026-09-01-daily-factor-lab-trend-v2-universe-design.md` (references the still-authoritative `docs/superpowers/specs/2026-09-01-daily-factor-lab-trend-design.md` for everything unchanged from v1).

## Global Constraints

- Run EVERYTHING with `py -3.12` (never bare `python`).
- Test convention: plain scripts, `check_*` functions, `run_consistency_check()`, `__main__`. No pytest.
- German docstrings/comments (ASCII: ue/oe/ae).
- **Universe (19, exact list, exact order not significant but exact membership is):** SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, GLD, SLV, DBC, VNQ (unchanged from v1) + UUP, FXE, FXY, USO, UNG, DBA, EMB (new).
- **New instruments' cost_bp: all 3.0** (the "less liquid than SPY-tier" bucket, matching EFA/EEM/LQD/etc.).
- **`min_common_days` for v2: 4200** (lowered from v1's 4800 — mechanical consequence of later real-world inception dates for the new instruments, fixed before seeing any data).
- **Sleeves (6, NEW — part of `REGISTRATION_V2`, unlike v1 where `SLEEVES` was a bare module constant NOT covered by `config_hash()`):** us_equity=[SPY,QQQ,IWM], intl_equity=[EFA,EEM], bonds=[TLT,IEF,LQD], em_bonds=[EMB], real_assets=[GLD,SLV,DBC,VNQ], currencies=[UUP,FXE,FXY], granular_commodities=[USO,UNG,DBA].
- **All other REGISTRATION_V2 values identical to v1's REGISTRATION**: `family="trend-etf-v2"`, `cash_series="^IRX"`, `snapshot_start="2007-01-01"`, `snapshot_end_exclusive="2026-09-01"`, `warmup_days=252+63`, `lookbacks=[63,126,252]`, `signals=["mom63","mom126","mom252","combo"]`, `modes=["long_short","long_flat"]`, `ewma_span=63`, `vol_cap=0.10`, `gross_cap_targets_only=True`, `rebalance="monatsultimo_entscheid_fill_naechster_close"`, `cost_ladder=[1.0,2.0,5.0]`, `borrow_bp_pa=50.0`, `bootstrap={"kind":"stationary","expected_block_len_months":6.0,"sensitivity_block_lens":[3.0,12.0],"n_boot":10000,"seed":0}`, `permutation_n=10000`, `dd_cap=0.15`, `gate_c_floor=0.02`, `candidate_rule="hoechster Excess-Sharpe unter Bestehenden, Tie-Break alphabetisch"`, `holdout_gates=["A","B","C"]`.
- **Output directory naming is family-scoped**: `trend_screening_v2_<runid>` / `trend_holdout_v2_<runid>` (NOT the bare `trend_screening_*`/`trend_holdout_*` v1 uses) — `run_trend_holdout_v2.py`'s candidate glob MUST match only `trend_screening_v2_*/candidate.json`, never v1's directories.
- **Never modify** `registration.py`, `signals.py`, `portfolio.py`, `stats.py`, `data_snapshot.py`, `run_trend_baseline.py`, `run_trend_holdout.py`, `build_trend_snapshot.py`, or anything under `market_control_system/`. Only `costs.py` gets touched among existing files (Task 1, purely additive).
- Do NOT re-run `build_trend_snapshot.py` (v1) or touch `factor_lab/data_snapshots/trend_snapshot_41568b99...` (v1's already-sealed snapshot) or `factor_lab/logs/trend_screening_20260901_114430/` (v1's already-produced real result).
- Commit after each task; messages end with a blank line then `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: costs.py additive COST_BP extension

**Files:**
- Modify: `factor_lab/costs.py`
- Modify: `factor_lab/tests/test_costs.py`

**Interfaces:**
- Produces: `COST_BP` dict grows from 12 to 19 entries. All other exports (`BORROW_BP_PA`, `TRADING_DAYS_PA`, `COST_LADDER`, `trade_cost_fraction`, `daily_borrow_cost_fraction`) unchanged.

- [ ] **Step 1: Write the failing test (extend the existing `check_tables`)**

In `factor_lab/tests/test_costs.py`, replace the body of `check_tables()`:

```python
def check_tables() -> None:
    v1_symbols = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"]
    v2_new_symbols = ["UUP", "FXE", "FXY", "USO", "UNG", "DBA", "EMB"]
    assert sorted(COST_BP) == sorted(v1_symbols + v2_new_symbols), (
        f"Erwartete 19 Symbole (12 v1 + 7 neue v2), bekam {sorted(COST_BP)}"
    )
    # v1-Werte duerfen sich NICHT aendern (reine Erweiterung, Regressionsschutz).
    assert COST_BP["SPY"] == 1.5 and COST_BP["EEM"] == 3.0
    # Alle 7 neuen v2-Symbole bei 3.0bp (Spec trend-etf-v2 Abschnitt 4).
    for s in v2_new_symbols:
        assert COST_BP[s] == 3.0, f"{s}: erwartete 3.0bp, bekam {COST_BP[s]}"
    assert COST_LADDER == (1.0, 2.0, 5.0)
    print("COST_BP / COST_LADDER (19 Symbole inkl. trend-etf-v2): OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_costs.py`
Expected: FAIL — `AssertionError` on the `sorted(COST_BP)` comparison (only 12 symbols present).

- [ ] **Step 3: Implement — extend `COST_BP` in `factor_lab/costs.py`**

Change the `COST_BP` dict literal from:
```python
COST_BP: dict[str, float] = {
    "SPY": 1.5, "QQQ": 1.5, "IWM": 1.5, "TLT": 1.5, "IEF": 1.5, "GLD": 1.5,
    "EFA": 3.0, "EEM": 3.0, "LQD": 3.0, "SLV": 3.0, "DBC": 3.0, "VNQ": 3.0,
}
```
to:
```python
COST_BP: dict[str, float] = {
    "SPY": 1.5, "QQQ": 1.5, "IWM": 1.5, "TLT": 1.5, "IEF": 1.5, "GLD": 1.5,
    "EFA": 3.0, "EEM": 3.0, "LQD": 3.0, "SLV": 3.0, "DBC": 3.0, "VNQ": 3.0,
    # trend-etf-v2 (rein additiv, Spec 2026-09-01-daily-factor-lab-trend-v2-universe-design.md Abschnitt 4):
    "UUP": 3.0, "FXE": 3.0, "FXY": 3.0, "USO": 3.0, "UNG": 3.0, "DBA": 3.0, "EMB": 3.0,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_costs.py`
Expected: PASS. Also run `py -3.12 factor_lab/tests/test_portfolio.py` and `py -3.12 factor_lab/tests/test_run_trend_baseline.py` (v1's tests) to confirm the additive change doesn't break v1 (it shouldn't — v1's code only ever looks up its own 12 symbols).

- [ ] **Step 5: Commit**

```bash
git add factor_lab/costs.py factor_lab/tests/test_costs.py
git commit -m "feat(factor_lab): extend COST_BP with 7 trend-etf-v2 instruments (additive)"
```

---

### Task 2: registration_v2.py — REGISTRATION_V2, sleeves, duplicated sealing/tombstone

**Files:**
- Create: `factor_lab/registration_v2.py`
- Create: `factor_lab/tests/test_registration_v2.py`

**Interfaces:**
- Consumes: `config_hash(registration: dict) -> str`, `file_sha256(path: str) -> str` from `factor_lab.registration` (both already generic/parametrized — reused as-is). `SNAPSHOT_DIR` from `factor_lab.data_snapshot`.
- Produces: `FAMILY_V2 = "trend-etf-v2"`; `REGISTRATION_V2: dict` (see Global Constraints for exact contents); `trend_snapshot_path_v2() -> str`; `write_candidate_v2(path, variant, snapshot_sha256, git_sha, dev_end, results_sha256) -> None` (refuses to overwrite, mirrors v1's write-once guard); `read_and_verify_candidate_v2(path, snapshot_sha256) -> dict`; `tombstone_path_v2(logs_dir) -> str`; `assert_no_tombstone_v2(logs_dir) -> None`; `write_tombstone_v2(logs_dir, holdout_result_path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_registration_v2.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'factor_lab.registration_v2'`

- [ ] **Step 3: Implement `factor_lab/registration_v2.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_registration_v2.py`
Expected: PASS. Also run `py -3.12 factor_lab/tests/test_registration.py` to confirm v1's own tests are unaffected.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/registration_v2.py factor_lab/tests/test_registration_v2.py
git commit -m "feat(factor_lab): add trend-etf-v2 registration (19-instrument universe, sleeves in hash, isolated sealing)"
```

---

### Task 3: build_trend_snapshot_v2.py

**Files:**
- Create: `factor_lab/build_trend_snapshot_v2.py`
- Create: `factor_lab/tests/test_build_trend_snapshot_v2.py`

**Interfaces:**
- Consumes: `REGISTRATION_V2`, `trend_snapshot_path_v2()` from `factor_lab.registration_v2` (Task 2). `SNAPSHOT_DIR`, `sanity_check_snapshot(dfs, min_common_days)`, `snapshot_content_sha256(dfs)`, `write_snapshot_manifest(dfs, path)` from `factor_lab.data_snapshot` (unchanged, generic). `add_yfinance_version_to_manifest(manifest_path, version)` from `factor_lab.build_trend_snapshot` (v1's, already generic — takes a path and a version string, no v1-registration coupling). `month_end_dates` from `factor_lab.portfolio`.
- Produces: `fetch_all() -> dict[str, pd.DataFrame]`; `resolve_dev_end(dfs: dict) -> pd.Timestamp`; `main() -> None`.

- [ ] **Step 1: Write the failing test**

```python
"""
test_build_trend_snapshot_v2.py — netzwerkfreie Pruefungen: v2-Snapshot-Pfad
unterscheidet sich von v1s (unterschiedliches Universum -> unterschiedlicher
Hash, keine Kollision), resolve_dev_end() auf einem synthetischen Fixture.
fetch_all()/main() brauchen echten Netzwerkzugriff und sind (wie das
restliche Netzwerk-Fetching in diesem Package) bewusst NICHT unit-getestet.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.build_trend_snapshot_v2 import resolve_dev_end
from factor_lab.registration_v2 import trend_snapshot_path_v2
from factor_lab.data_snapshot import trend_snapshot_path


def check_v2_path_differs_from_v1() -> None:
    assert trend_snapshot_path_v2() != trend_snapshot_path(), (
        "v2-Snapshot-Pfad muss sich vom v1-Pfad unterscheiden (anderes Universum -> anderer Hash)"
    )
    print("trend_snapshot_path_v2 != trend_snapshot_path (v1): OK")


def check_resolve_dev_end() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    dfs = {s: pd.DataFrame({"price": 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, 500))}, index=idx)
           for s in ["SPY", "UUP", "USO", "EMB"]}
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * 500}, index=idx)
    dev_end = resolve_dev_end(dfs)
    assert dev_end <= idx[int(500 * 0.8)], "DEV_END muss innerhalb des 80%-Quantil-Fensters liegen"
    assert dev_end in idx, "DEV_END muss ein echter Handelstag aus dem gemeinsamen Kalender sein"
    print("resolve_dev_end (v2, synthetisches Fixture): OK")


def run_consistency_check() -> None:
    check_v2_path_differs_from_v1()
    check_resolve_dev_end()
    print("\nAlle build_trend_snapshot_v2-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_build_trend_snapshot_v2.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'factor_lab.build_trend_snapshot_v2'`

- [ ] **Step 3: Implement `factor_lab/build_trend_snapshot_v2.py`**

```python
"""
build_trend_snapshot_v2.py — EIGENER, einmaliger Build-Schritt fuer die
Familie trend-etf-v2 (19-Instrumenten-Universum, Spec
docs/superpowers/specs/2026-09-01-daily-factor-lab-trend-v2-universe-design.md).
Fetcht via yfinance im fixen Fenster, prueft fail-closed (min_common_days=4200,
siehe Spec Abschnitt 5), friert ein, schreibt das Manifest. Wiederverwendet
sanity_check_snapshot/snapshot_content_sha256/write_snapshot_manifest aus
data_snapshot.py sowie add_yfinance_version_to_manifest aus
build_trend_snapshot.py UNVERAENDERT -- nur der Snapshot-Pfad ist
v2-spezifisch (andere Universum/Datums-Kombination -> anderer Hash -> andere
Datei, keine Kollision mit dem v1-Snapshot).

Ausfuehren: py -3.12 factor_lab/build_trend_snapshot_v2.py
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from factor_lab.registration_v2 import REGISTRATION_V2, trend_snapshot_path_v2
from factor_lab.data_snapshot import (
    SNAPSHOT_DIR, sanity_check_snapshot, snapshot_content_sha256, write_snapshot_manifest,
)
from factor_lab.build_trend_snapshot import add_yfinance_version_to_manifest
from factor_lab.portfolio import month_end_dates


def _download(symbol: str, column: str) -> pd.Series:
    import yfinance as yf
    raw = yf.download(symbol, start=REGISTRATION_V2["snapshot_start"],
                      end=REGISTRATION_V2["snapshot_end_exclusive"], auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"{symbol}: yfinance lieferte keine Daten")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    series = close.astype(float)
    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    return series.rename(column)


def fetch_all() -> dict[str, pd.DataFrame]:
    dfs = {}
    for symbol in REGISTRATION_V2["universe"]:
        print(f"  lade {symbol}...")
        dfs[symbol] = _download(symbol, "price").to_frame()
    print("  lade ^IRX (T-Bill-Cash-Naeherung)...")
    irx = _download(REGISTRATION_V2["cash_series"], "rate_pa_pct").to_frame()
    dfs["IRX"] = irx.dropna()
    return dfs


def resolve_dev_end(dfs: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """DEV_END-Regel (praeregistriert): letzter Monatsultimo des gemeinsamen
    Kalenders <= dem 80%-Quantil-Datum."""
    common = None
    for name, df in dfs.items():
        if name == "IRX":
            continue
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    quantile_date = common[int(len(common) * 0.8) - 1]
    ends = month_end_dates(common)
    return ends[ends <= quantile_date].max()


def main() -> None:
    print(f"=== Versiegelter Snapshot-Build trend-etf-v2 ({REGISTRATION_V2['snapshot_start']} bis "
          f"exkl. {REGISTRATION_V2['snapshot_end_exclusive']}, {len(REGISTRATION_V2['universe'])} Instrumente) ===")
    dfs = fetch_all()
    sanity_check_snapshot(dfs, min_common_days=REGISTRATION_V2["min_common_days"])
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = trend_snapshot_path_v2()
    with open(path, "wb") as f:
        pickle.dump(dfs, f)
    manifest_path = path + ".manifest.json"
    write_snapshot_manifest(dfs, manifest_path)
    import yfinance as yf  # lazy wie in _download() -- kein Hard-Import auf Modulebene
    add_yfinance_version_to_manifest(manifest_path, yf.__version__)
    print(f"  Snapshot: {path}")
    print(f"  Content-SHA256: {snapshot_content_sha256(dfs)}")
    print(f"  DEV_END (aufgeloest): {resolve_dev_end(dfs).date()}")
    print("\nJETZT das Manifest committen (git add -f ...), DANN erst run_trend_baseline_v2.py.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_build_trend_snapshot_v2.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/build_trend_snapshot_v2.py factor_lab/tests/test_build_trend_snapshot_v2.py
git commit -m "feat(factor_lab): add sealed snapshot build for trend-etf-v2 (19-instrument universe)"
```

---

### Task 4: run_trend_baseline_v2.py — screening runner

**Files:**
- Create: `factor_lab/run_trend_baseline_v2.py`
- Create: `factor_lab/tests/test_run_trend_baseline_v2.py`

**Interfaces:**
- Consumes: `COST_BP`, `COST_LADDER` from `factor_lab.costs` (Task 1). `REGISTRATION_V2`, `write_candidate_v2`, `trend_snapshot_path_v2` from `factor_lab.registration_v2` (Task 2). `config_hash`, `file_sha256` from `factor_lab.registration` (v1, generic). `resolve_dev_end` from `factor_lab.build_trend_snapshot_v2` (Task 3). `load_trend_snapshot`, `snapshot_content_sha256` from `factor_lab.data_snapshot` (unchanged). `momentum_sign`, `combo_signal` from `factor_lab.signals`. `ewma_annualized_vol`, `month_end_dates`, `run_lagged_backtest`, `trend_weight_provider`, `fixed_mix_provider` from `factor_lab.portfolio`. `monthly_log_returns`, `stationary_block_bootstrap`, `monthly_sign_flip_pvalue`, `annualized_stats`, `full_year_excess`, `monthly_excess_sharpe`, `evaluate_screening_gates`, `screening_verdict` from `factor_lab.stats`.
- Produces: `LOOKBACKS`, `VARIANT_NAMES` (identical 8-variant grid to v1). `prepare_inputs(dfs) -> dict`. `run_variant(inputs, variant, cost_multiplier=1.0, exclude=frozenset(), borrow_bp_pa=None) -> tuple[pd.Series, dict]`. `_breakeven_multiplier(excess_by_mult) -> float`. `run_screening(dfs, dev_end=None) -> tuple[dict, pd.DataFrame]`. `main() -> None` (writes to `factor_lab/logs/trend_screening_v2_<runid>/`).

- [ ] **Step 1: Write the failing test**

```python
"""
test_run_trend_baseline_v2.py — Integrationstest auf synthetischen Daten fuer
die Familie trend-etf-v2: 7-Symbol-Fixture (ein Instrument je Sleeve, damit
die Sleeve-LOO-Mechanik sinnvoll geprueft wird), identischer Index ueber
alle Serien, REGISTRATION_V2 in der Provenance, beide Verdikt-Zweige.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
import pandas as pd

from factor_lab.run_trend_baseline_v2 import run_screening, VARIANT_NAMES
from factor_lab.registration_v2 import REGISTRATION_V2

# Ein Symbol je Sleeve (Spec trend-etf-v2 Abschnitt 2), damit Gate Ds
# Sleeve-LOO im Test alle 7 Sleeves durchlaeuft.
_TEST_SYMBOLS = ["SPY", "EFA", "TLT", "EMB", "GLD", "UUP", "USO"]


def _dfs(regime_amplitude: float, n_days: int = 1400, seed: int = 0) -> dict:
    """Wie v1s bewaehrtes Fixture-Muster (Regime-Wechsel-Sinuswelle): bei
    regime_amplitude=0 reiner Random Walk (Null-Zweig); sonst wechselt der
    Drift alle 130 Tage das Vorzeichen -- long/flat geht in Abwaertsregimen
    in Cash, matched_long bleibt durchgehend investiert."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    regime = np.sign(np.sin(np.arange(n_days) * np.pi / 130.0) + 1e-9)
    dfs = {}
    for i, symbol in enumerate(_TEST_SYMBOLS):
        drift = regime_amplitude * regime * (0.6 + 0.1 * i)
        prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.006, n_days) + drift)
        dfs[symbol] = pd.DataFrame({"price": prices}, index=idx)
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * n_days}, index=idx)
    return dfs


def check_registration_v2_used() -> None:
    result, _ = run_screening(_dfs(regime_amplitude=0.0, seed=4))
    assert result["provenance"]["registration"]["family"] == "trend-etf-v2"
    assert result["provenance"]["config_hash"], "Config-Hash muss gesetzt sein"
    print("run_screening (v2) nutzt REGISTRATION_V2: OK")


def run_consistency_check() -> None:
    check_registration_v2_used()

    result, per_day = run_screening(_dfs(regime_amplitude=0.0, seed=4))
    assert sorted(result["summary"]) == VARIANT_NAMES
    expected_cols = set(VARIANT_NAMES) | {"matched_long", "bench_spy_bh", "bench_60_40"}
    assert set(per_day.columns) == expected_cols
    assert not per_day.isna().any().any(), "Alle Serien muessen denselben Index vollstaendig fuellen"
    json.dumps(result)

    for name in VARIANT_NAMES:
        s = result["summary"][name]
        for key in ("stats", "excess_bootstrap", "excess_bootstrap_sensitivity", "permutation",
                    "gates", "cost_ladder", "breakeven_cost_multiplier", "max_daily_gross"):
            assert key in s, f"{name}: {key} fehlt"
    assert "research_variant" in result["summary"]["mom63_long_short"]["labels"]

    # NOTIZ FUER DEN IMPLEMENTIERER: falls (0.0, seed=4) hier nicht
    # zuverlaessig candidate=None liefert (das 7-Symbol-Fixture mit
    # sleeve-gewichtetem drift-Multiplikator ist NICHT dasselbe Fixture wie
    # v1s -- andere Symbolzahl, andere Gewichte), seed anpassen und den
    # Grund inline dokumentieren -- exakt wie v1s Task 9 das fuer
    # trend_strength/regime_amplitude bereits getan hat. NICHT die Gates
    # aufweichen, um einen bestimmten Seed zum Laufen zu bringen.
    assert result["candidate"] is None, "Regimefreie Daten duerfen keine Kandidatin liefern"
    print("run_screening (v2, Null-Zweig, 7-Sleeve-Fixture): OK")

    # NOTIZ: dieselbe Escape-Hatch-Regel gilt fuer den Kandidaten-Zweig --
    # Amplitude/Seed sind ein Startpunkt, keine garantierte Konstante.
    result2, _ = run_screening(_dfs(regime_amplitude=0.0022, seed=1))
    assert result2["candidate"] in VARIANT_NAMES, f"Erwartete Kandidatin, bekam {result2['candidate']}"
    winning = result2["summary"][result2["candidate"]]
    if winning["loo_excess_compounds"]:
        sleeve_keys = [k for k in winning["loo_excess_compounds"] if k.startswith("loo_sleeve_")]
        assert len(sleeve_keys) == len(REGISTRATION_V2["sleeves"]), (
            f"Erwartete {len(REGISTRATION_V2['sleeves'])} Sleeve-LOO-Eintraege, bekam {len(sleeve_keys)}"
        )
    print("run_screening (v2, Kandidaten-Zweig, Sleeve-LOO ueber alle 7 Sleeves): OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_run_trend_baseline_v2.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'factor_lab.run_trend_baseline_v2'`

- [ ] **Step 3: Implement `factor_lab/run_trend_baseline_v2.py`**

```python
"""
run_trend_baseline_v2.py — Screening der Familie trend-etf-v2 (erweitertes
19-Instrumenten-Universum, Spec docs/superpowers/specs/2026-09-01-daily-
factor-lab-trend-v2-universe-design.md): 8 Varianten + matched_long +
Kontext-Benchmarks auf IDENTISCHEM Evaluationsfenster, Kostenleiter,
Stationary-Block-Inferenz auf monatlichen Log-Mehrertraegen, Gates A-D
(LOO-Reruns fuer A-C-Passer, 19 Instrument- + 7 Sleeve-Reruns statt v1s
12+4), candidate.json. Das Screening ist ausdruecklich KEINE Bestaetigung.

Bewusst eine fast vollstaendige Kopie von run_trend_baseline.py (trend-etf-
v1) statt einer parametrisierten Wiederverwendung -- der bereits gemergte,
hart geprueft v1-Code (inkl. des irreversiblen Holdout-Pfads) darf durch
diese Erweiterung in keinem Fall veraendert werden. Wiederverwendet
signals.py/portfolio.py/stats.py unveraendert sowie costs.py (rein additiv
erweitert).

Laufzeit-Hinweis: >20 min moeglich (LOO-Reruns, mehr Instrumente/Sleeves als
v1) -> detacht starten (Start-Process + PID-Datei + Monitor, siehe
Tech-Stack-Memory).

Ausfuehren: py -3.12 factor_lab/run_trend_baseline_v2.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from factor_lab.signals import momentum_sign, combo_signal
from factor_lab.costs import COST_BP, COST_LADDER
from factor_lab.portfolio import (
    ewma_annualized_vol, month_end_dates, run_lagged_backtest,
    trend_weight_provider, fixed_mix_provider,
)
from factor_lab.stats import (
    monthly_log_returns, stationary_block_bootstrap, monthly_sign_flip_pvalue,
    annualized_stats, full_year_excess, monthly_excess_sharpe,
    evaluate_screening_gates, screening_verdict,
)
from factor_lab.registration import config_hash, file_sha256
from factor_lab.registration_v2 import REGISTRATION_V2, write_candidate_v2, trend_snapshot_path_v2
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.build_trend_snapshot_v2 import resolve_dev_end

LOOKBACKS = {"mom63": 63, "mom126": 126, "mom252": 252}
VARIANT_NAMES = sorted(f"{sig}_{mode}" for sig in [*LOOKBACKS, "combo"] for mode in ["long_short", "long_flat"])
WARMUP_DAYS = REGISTRATION_V2["warmup_days"]


def prepare_inputs(dfs: dict) -> dict:
    symbols = sorted(s for s in dfs if s != "IRX")
    common = dfs[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(dfs[s].index)
    common = common.sort_values()
    prices = pd.DataFrame({s: dfs[s].loc[common, "price"] for s in symbols})
    returns = prices.pct_change().dropna()
    cash_daily = (dfs["IRX"]["rate_pa_pct"].reindex(common).ffill().bfill() / 100.0 / 252.0).loc[returns.index]

    per_lb = {name: pd.DataFrame({s: momentum_sign(prices[s], lb) for s in symbols})
              for name, lb in LOOKBACKS.items()}
    signal_frames = {**per_lb, "combo": pd.DataFrame({
        s: combo_signal([per_lb[n][s] for n in LOOKBACKS]) for s in symbols})}
    signal_frames = {k: v.loc[returns.index] for k, v in signal_frames.items()}
    vols = ewma_annualized_vol(returns)

    warmup_end = returns.index[WARMUP_DAYS]
    ends = month_end_dates(returns.index)
    eval_decisions = ends[ends >= warmup_end]
    return {"symbols": symbols, "returns": returns, "cash_daily": cash_daily,
            "signal_frames": signal_frames, "vols": vols, "eval_decisions": eval_decisions,
            "dev_end": resolve_dev_end(dfs)}


def run_variant(inputs: dict, variant: str, cost_multiplier: float = 1.0,
                exclude: set = frozenset(), borrow_bp_pa: float = None) -> tuple[pd.Series, dict]:
    """borrow_bp_pa: Override fuer die Borrow-Sensitivitaet (Spec Abschnitt 8:
    25/50/100bp p.a.); Default None -> REGISTRATION_V2-Standard (50bp).
    Wirkungslos fuer long_flat-Varianten, die nie shorten."""
    symbols = [s for s in inputs["symbols"] if s not in exclude]
    returns = inputs["returns"][symbols]
    vols = inputs["vols"][symbols]
    if variant == "matched_long":
        signals = pd.DataFrame(1.0, index=returns.index, columns=symbols)
        mode = "long_flat"
    else:
        sig_name = variant.rsplit("_", 2)[0]
        mode = "_".join(variant.rsplit("_", 2)[1:])
        signals = inputs["signal_frames"][sig_name][symbols]
    provider = trend_weight_provider(returns, signals, vols, mode,
                                     vol_cap=REGISTRATION_V2["vol_cap"], vol_window=63)
    effective_borrow = REGISTRATION_V2["borrow_bp_pa"] if borrow_bp_pa is None else borrow_bp_pa
    return run_lagged_backtest(returns, inputs["cash_daily"], inputs["eval_decisions"],
                               provider, {s: COST_BP.get(s, 3.0) for s in symbols},
                               cost_multiplier, effective_borrow)


def _breakeven_multiplier(excess_by_mult: dict[float, float]) -> float:
    """Linear interpolierter Kostensatz-Multiplikator, bei dem der
    compoundierte Mehrertrag 0 wird; inf, wenn selbst 5x positiv bleibt,
    0, wenn schon 1x negativ ist."""
    mults = sorted(excess_by_mult)
    if excess_by_mult[mults[0]] <= 0:
        return 0.0
    for lo, hi in zip(mults, mults[1:]):
        e_lo, e_hi = excess_by_mult[lo], excess_by_mult[hi]
        if e_hi <= 0 < e_lo:
            return float(lo + (hi - lo) * e_lo / (e_lo - e_hi))
    return float("inf")


def run_screening(dfs: dict, dev_end=None) -> tuple[dict, pd.DataFrame]:
    """dev_end: von main() uebergeben, weil dort die Daten bereits auf das
    Entwicklungsfenster geschnitten sind -- die 80%-Regel darf nicht ein
    zweites Mal auf die geschnittenen Daten angewendet werden."""
    inputs = prepare_inputs(dfs)
    if dev_end is None:
        dev_end = inputs["dev_end"]
    cash = inputs["cash_daily"]

    matched = {m: run_variant(inputs, "matched_long", cost_multiplier=m) for m in COST_LADDER}
    per_day = {"matched_long": matched[1.0][0]}

    summary, excess_sharpes, gates_by_variant = {}, {}, {}
    for variant in VARIANT_NAMES:
        runs = {m: run_variant(inputs, variant, cost_multiplier=m) for m in COST_LADDER}
        net, info = runs[1.0]
        per_day[variant] = net

        excess = np.log1p(net) - np.log1p(matched[1.0][0])
        monthly_excess = (monthly_log_returns(net) - monthly_log_returns(matched[1.0][0])).dropna()
        boot = stationary_block_bootstrap(monthly_excess.to_numpy(),
                                          REGISTRATION_V2["bootstrap"]["expected_block_len_months"],
                                          REGISTRATION_V2["bootstrap"]["n_boot"],
                                          REGISTRATION_V2["bootstrap"]["seed"])
        sensitivity = {str(L): stationary_block_bootstrap(
            monthly_excess.to_numpy(), L, REGISTRATION_V2["bootstrap"]["n_boot"],
            REGISTRATION_V2["bootstrap"]["seed"])["ann_geom_lower_1s95"]
            for L in REGISTRATION_V2["bootstrap"]["sensitivity_block_lens"]}
        perm = monthly_sign_flip_pvalue(monthly_excess.to_numpy(), REGISTRATION_V2["permutation_n"],
                                        REGISTRATION_V2["bootstrap"]["seed"])

        stats = annualized_stats(net, cash)
        stats_stress = annualized_stats(runs[2.0][0], cash)
        ladder = {}
        excess_by_mult = {}
        for m in COST_LADDER:
            ladder[str(m)] = annualized_stats(runs[m][0], cash)
            m_excess = (monthly_log_returns(runs[m][0]) - monthly_log_returns(matched[m][0])).dropna()
            excess_by_mult[m] = float(np.expm1(m_excess.sum()))

        pre = evaluate_screening_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                       stats_stress["cagr"],
                                       full_year_excess(excess),
                                       loo_excess_compounds={"pending": 1.0},
                                       dd_cap=REGISTRATION_V2["dd_cap"],
                                       gate_c_floor=REGISTRATION_V2["gate_c_floor"])
        loo = {}
        if pre["gate_a_excess_ci"] and pre["gate_b_drawdown"] and pre["gate_c_stressed_floor"]:
            loo_configs = {f"loo_{s}": {s} for s in inputs["symbols"]}
            loo_configs.update({f"loo_sleeve_{k}": set(v) for k, v in REGISTRATION_V2["sleeves"].items()})
            for loo_name, excl in loo_configs.items():
                v_net, _ = run_variant(inputs, variant, exclude=excl)
                m_net, _ = run_variant(inputs, "matched_long", exclude=excl)
                diff = (monthly_log_returns(v_net) - monthly_log_returns(m_net)).dropna()
                loo[loo_name] = float(np.expm1(diff.sum()))
        gates = evaluate_screening_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                         stats_stress["cagr"], full_year_excess(excess), loo,
                                         REGISTRATION_V2["dd_cap"], REGISTRATION_V2["gate_c_floor"])
        gates_by_variant[variant] = gates
        excess_sharpes[variant] = monthly_excess_sharpe(monthly_excess)
        summary[variant] = {
            "stats": stats,
            "excess_bootstrap": boot,
            "excess_bootstrap_sensitivity": sensitivity,
            "permutation": perm,
            "gates": gates,
            "cost_ladder": ladder,
            "breakeven_cost_multiplier": _breakeven_multiplier(excess_by_mult),
            "loo_excess_compounds": loo,
            "yearly_excess_full_years": {str(k): v for k, v in full_year_excess(excess).items()},
            "max_daily_gross": info["max_daily_gross"],
            "total_turnover": info["total_turnover"],
            "instrument_contributions": info["instrument_contributions"],
            "labels": (["research_variant"] if variant.endswith("long_short") else []),
        }

    spy_provider = fixed_mix_provider({"SPY": 1.0}) if "SPY" in inputs["symbols"] else fixed_mix_provider({inputs["symbols"][0]: 1.0})
    spy_bh, _ = run_lagged_backtest(inputs["returns"], cash,
                                    pd.DatetimeIndex([inputs["eval_decisions"][0]]),
                                    spy_provider, COST_BP if "SPY" in inputs["symbols"] else {s: 1.5 for s in inputs["symbols"]})
    mix_symbols = ["SPY", "TLT"] if set(["SPY", "TLT"]) <= set(inputs["symbols"]) else inputs["symbols"][:2]
    mix, _ = run_lagged_backtest(inputs["returns"], cash, inputs["eval_decisions"],
                                 fixed_mix_provider({mix_symbols[0]: 0.6, mix_symbols[1]: 0.4}),
                                 {s: COST_BP.get(s, 1.5) for s in inputs["symbols"]})
    per_day["bench_spy_bh"] = spy_bh
    per_day["bench_60_40"] = mix

    verdict, candidate = screening_verdict(gates_by_variant, excess_sharpes)
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git_sha = "unbekannt"
    provenance = {
        "registration": REGISTRATION_V2,
        "config_hash": config_hash(REGISTRATION_V2),
        "snapshot_content_sha256": snapshot_content_sha256(dfs),
        "git_sha": git_sha,
        "dev_end": str(dev_end),
        "eval_start": str(per_day["matched_long"].index.min()),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    result = {
        "summary": summary,
        "benchmarks": {"matched_long": annualized_stats(per_day["matched_long"], cash),
                       "bench_spy_bh": annualized_stats(spy_bh, cash),
                       "bench_60_40": annualized_stats(mix, cash)},
        "verdict": verdict,
        "candidate": candidate,
        "provenance": provenance,
    }
    return result, pd.DataFrame(per_day)


def main() -> None:
    dfs = load_trend_snapshot(path=trend_snapshot_path_v2())
    full_snapshot_sha256 = snapshot_content_sha256(dfs)
    inputs_probe = prepare_inputs(dfs)
    dev_end = inputs_probe["dev_end"]
    dev_dfs = {name: df.loc[df.index <= dev_end] for name, df in dfs.items()}
    result, per_day = run_screening(dev_dfs, dev_end=dev_end)
    result["provenance"]["snapshot_content_sha256"] = full_snapshot_sha256

    print(f"\n{result['verdict']}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "logs", f"trend_screening_v2_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    per_day.to_csv(os.path.join(out_dir, "daily_net_returns.csv"))
    results_path = os.path.join(out_dir, "screening_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if result["candidate"] is not None:
        write_candidate_v2(os.path.join(out_dir, "candidate.json"),
                           variant=result["candidate"],
                           snapshot_sha256=result["provenance"]["snapshot_content_sha256"],
                           git_sha=result["provenance"]["git_sha"],
                           dev_end=result["provenance"]["dev_end"],
                           results_sha256=file_sha256(results_path))
    print(f"Ergebnisse gespeichert: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_run_trend_baseline_v2.py`
Expected: PASS. If the null-branch (`regime_amplitude=0.0, seed=4`) or candidate-branch (`regime_amplitude=0.0022, seed=1`) don't reliably produce the intended `candidate is None` / `candidate in VARIANT_NAMES` outcome for this 7-symbol fixture, tune `seed`/`regime_amplitude` (both are just synthetic-data-generation parameters, not gate logic) and document the reasoning inline as a code comment, following the exact precedent set in v1's `test_run_trend_baseline.py`. Do not weaken any gate threshold to make a test pass.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/run_trend_baseline_v2.py factor_lab/tests/test_run_trend_baseline_v2.py
git commit -m "feat(factor_lab): add trend-etf-v2 screening runner (19-instrument universe, 7 sleeves)"
```

---

### Task 5: run_trend_holdout_v2.py — one-shot holdout runner

**Files:**
- Create: `factor_lab/run_trend_holdout_v2.py`
- Create: `factor_lab/tests/test_run_trend_holdout_v2.py`

**Interfaces:**
- Consumes: `COST_BP`, `COST_LADDER` from `factor_lab.costs`. `REGISTRATION_V2`, `read_and_verify_candidate_v2`, `assert_no_tombstone_v2`, `write_tombstone_v2`, `trend_snapshot_path_v2` from `factor_lab.registration_v2` (Task 2). `load_trend_snapshot`, `snapshot_content_sha256` from `factor_lab.data_snapshot`. `prepare_inputs`, `run_variant`, `VARIANT_NAMES`, `_breakeven_multiplier` from `factor_lab.run_trend_baseline_v2` (Task 4). `month_end_dates`, `run_lagged_backtest`, `trend_weight_provider` from `factor_lab.portfolio`. `monthly_log_returns`, `stationary_block_bootstrap`, `monthly_sign_flip_pvalue`, `annualized_stats`, `evaluate_holdout_gates` from `factor_lab.stats`.
- Produces: `run_holdout(dfs, candidate) -> dict` (pure function, same shape as v1's including `cost_ladder`/`breakeven_cost_multiplier`/`borrow_sensitivity` from day one). `main() -> None` (no arguments; globs `trend_screening_v2_*/candidate.json` specifically, NOT v1's bare `trend_screening_*`; writes to `factor_lab/logs/trend_holdout_v2_<runid>/`).

- [ ] **Step 1: Write the failing test**

```python
"""
test_run_trend_holdout_v2.py — Holdout der Familie trend-etf-v2: startet in
Cash NACH dev_end, wertet nur Holdout-Tage, volle Kostenleiter + Breakeven +
Borrow-Sensitivitaet (von Anfang an Teil der Familie, siehe die v1-
Nachbesserung dazu), verweigert ohne gueltiges Siegel und beim zweiten Mal --
unabhaengig von v1s eigenem Tombstone.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile
import numpy as np
import pandas as pd

from factor_lab import registration_v2
from factor_lab.run_trend_holdout_v2 import run_holdout
from factor_lab.run_trend_baseline_v2 import prepare_inputs

_TEST_SYMBOLS = ["SPY", "EFA", "TLT", "EMB", "GLD", "UUP", "USO"]


def _dfs(n_days: int = 1400, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    dfs = {s: pd.DataFrame({"price": 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.008, n_days))}, index=idx)
           for s in _TEST_SYMBOLS}
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * n_days}, index=idx)
    return dfs


def check_cost_ladder_and_breakeven(dfs: dict, dev_end, variant: str) -> dict:
    candidate = {"variant": variant, "dev_end": str(dev_end)}
    result = run_holdout(dfs, candidate)
    s = result["summary"]
    assert set(s["cost_ladder"]) == {"1.0", "2.0", "5.0"}, f"Erwartete volle Kostenleiter, bekam {sorted(s['cost_ladder'])}"
    assert "breakeven_cost_multiplier" in s
    return result


def check_borrow_sensitivity_long_short(dfs: dict, dev_end) -> None:
    result = check_cost_ladder_and_breakeven(dfs, dev_end, "mom63_long_short")
    sens = result["summary"]["borrow_sensitivity"]
    assert set(sens) == {"25.0", "50.0", "100.0"}, f"Erwartete 3 Borrow-Saetze, bekam {sorted(sens)}"
    cagrs = [sens[k]["cagr"] for k in ("25.0", "50.0", "100.0")]
    assert cagrs[0] >= cagrs[1] >= cagrs[2], f"Hoeherer Borrow-Satz darf CAGR nicht erhoehen, bekam {cagrs}"
    print("run_holdout_v2 Kostenleiter+Breakeven+Borrow-Sensitivitaet (long_short): OK")


def check_borrow_sensitivity_skipped_long_flat(dfs: dict, dev_end) -> None:
    result = check_cost_ladder_and_breakeven(dfs, dev_end, "mom63_long_flat")
    assert result["summary"]["borrow_sensitivity"] is None
    print("run_holdout_v2 Kostenleiter+Breakeven (long_flat, Borrow-Sensitivitaet uebersprungen): OK")


def run_consistency_check() -> None:
    dfs = _dfs()
    dev_end = prepare_inputs(dfs)["dev_end"]
    candidate = {"variant": "mom63_long_flat", "dev_end": str(dev_end)}
    result = run_holdout(dfs, candidate)

    first_day = pd.Timestamp(result["provenance"]["holdout_start"])
    assert first_day > dev_end
    assert result["summary"]["holdout_entry_cost"] > 0, "Positionsaufbau muss im Holdout bezahlt werden"
    assert "gate_d_no_single_driver" not in result["summary"]["gates"], "Holdout hat nur Gates A-C"
    assert set(result["summary"]["attribution_days"]) == {"holdout"}, "Attribution nur aus Holdout-Tagen"

    check_borrow_sensitivity_long_short(dfs, dev_end)
    check_borrow_sensitivity_skipped_long_flat(dfs, dev_end)

    with tempfile.TemporaryDirectory() as tmp:
        registration_v2.assert_no_tombstone_v2(tmp)
        registration_v2.write_tombstone_v2(tmp, "x.json")
        try:
            registration_v2.assert_no_tombstone_v2(tmp)
            raise AssertionError("Zweiter Holdout-Zugriff muss verweigert werden")
        except ValueError:
            pass
    print("run_holdout_v2: OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_run_trend_holdout_v2.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'factor_lab.run_trend_holdout_v2'`

- [ ] **Step 3: Implement `factor_lab/run_trend_holdout_v2.py`**

```python
"""
run_trend_holdout_v2.py — EINMALIGE Holdout-Bestaetigung der versiegelten
Kandidatin der Familie trend-etf-v2. Nimmt KEINE Argumente: liest das
NEUESTE candidate.json unter trend_screening_v2_*/ (familien-gescoptes
Glob-Muster, NICHT v1s trend_screening_*/), verifiziert Familie/Config-Hash/
Snapshot-Hash ueber registration_v2, prueft den familien-eigenen Tombstone
und schreibt ihn nach dem Lauf. Startet in CASH nach dev_end -- der volle
Positionsaufbau inkl. Kosten gehoert zum Holdout-Ergebnis. Volle Kostenleiter
+ Breakeven + Borrow-Sensitivitaet von Anfang an Teil dieser Familie (Spec
Abschnitt 8 -- v1 bekam das erst nachtraeglich per Amendment).

Bewusst eine fast vollstaendige Kopie von run_trend_holdout.py (trend-etf-
v1) -- siehe run_trend_baseline_v2.py fuer die Isolations-Begruendung.

Ausfuehren: py -3.12 factor_lab/run_trend_holdout_v2.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from factor_lab.costs import COST_BP, COST_LADDER
from factor_lab.portfolio import month_end_dates, run_lagged_backtest, trend_weight_provider
from factor_lab.stats import (
    monthly_log_returns, stationary_block_bootstrap, monthly_sign_flip_pvalue,
    annualized_stats, evaluate_holdout_gates,
)
from factor_lab.registration_v2 import (
    REGISTRATION_V2, read_and_verify_candidate_v2, assert_no_tombstone_v2, write_tombstone_v2,
    trend_snapshot_path_v2,
)
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.run_trend_baseline_v2 import prepare_inputs, run_variant, VARIANT_NAMES, _breakeven_multiplier

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
BORROW_SENSITIVITY_BP_PA = (25.0, 50.0, 100.0)


def _holdout_run(inputs: dict, variant: str, dev_end: pd.Timestamp, cost_multiplier: float = 1.0,
                 borrow_bp_pa: float = None):
    """Wie run_variant, aber Entscheidungen NUR nach dev_end -> der Lauf
    beginnt flach in Cash und baut die Position im Holdout auf."""
    holdout_decisions = inputs["eval_decisions"][inputs["eval_decisions"] > dev_end]
    saved = inputs["eval_decisions"]
    inputs["eval_decisions"] = holdout_decisions
    try:
        return run_variant(inputs, variant, cost_multiplier=cost_multiplier, borrow_bp_pa=borrow_bp_pa)
    finally:
        inputs["eval_decisions"] = saved


def run_holdout(dfs: dict, candidate: dict) -> dict:
    variant = candidate["variant"]
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unbekannte Variante im Siegel: {variant}")
    dev_end = pd.Timestamp(candidate["dev_end"])
    inputs = prepare_inputs(dfs)

    runs = {m: _holdout_run(inputs, variant, dev_end, cost_multiplier=m) for m in COST_LADDER}
    matched_runs = {m: _holdout_run(inputs, "matched_long", dev_end, cost_multiplier=m) for m in COST_LADDER}
    net, info = runs[1.0]
    net_stress = runs[2.0][0]
    matched = matched_runs[1.0][0]
    cash = inputs["cash_daily"].loc[net.index]

    monthly_excess = (monthly_log_returns(net) - monthly_log_returns(matched)).dropna()
    boot = stationary_block_bootstrap(monthly_excess.to_numpy(),
                                      REGISTRATION_V2["bootstrap"]["expected_block_len_months"],
                                      REGISTRATION_V2["bootstrap"]["n_boot"],
                                      REGISTRATION_V2["bootstrap"]["seed"])
    stats = annualized_stats(net, cash)
    stats_stress = annualized_stats(net_stress, cash)
    gates = evaluate_holdout_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                   stats_stress["cagr"], REGISTRATION_V2["dd_cap"],
                                   REGISTRATION_V2["gate_c_floor"])

    ladder = {str(m): annualized_stats(runs[m][0], cash) for m in COST_LADDER}
    excess_by_mult = {}
    for m in COST_LADDER:
        m_excess = (monthly_log_returns(runs[m][0]) - monthly_log_returns(matched_runs[m][0])).dropna()
        excess_by_mult[m] = float(np.expm1(m_excess.sum()))
    breakeven = _breakeven_multiplier(excess_by_mult)

    borrow_sensitivity = None
    if variant.endswith("long_short"):
        borrow_sensitivity = {}
        for borrow_bp in BORROW_SENSITIVITY_BP_PA:
            b_net, _ = _holdout_run(inputs, variant, dev_end, borrow_bp_pa=borrow_bp)
            borrow_sensitivity[str(borrow_bp)] = annualized_stats(b_net, cash)

    return {
        "variant": variant,
        "summary": {
            "stats": stats,
            "stats_2x": stats_stress,
            "excess_bootstrap": boot,
            "permutation": monthly_sign_flip_pvalue(monthly_excess.to_numpy(),
                                                    REGISTRATION_V2["permutation_n"],
                                                    REGISTRATION_V2["bootstrap"]["seed"]),
            "gates": gates,
            "cost_ladder": ladder,
            "breakeven_cost_multiplier": breakeven,
            "borrow_sensitivity": borrow_sensitivity,
            "holdout_entry_cost": float(info["per_day"]["trade_cost"].iloc[:2].sum()),
            "instrument_contributions": info["instrument_contributions"],
            "attribution_days": ["holdout"],
            "matched_long_stats": annualized_stats(matched, cash),
        },
        "provenance": {
            "holdout_start": str(net.index.min()),
            "holdout_end": str(net.index.max()),
            "dev_end": str(dev_end),
            "snapshot_content_sha256": snapshot_content_sha256(dfs),
        },
    }


def main() -> None:
    print("*** VERSIEGELTES ONE-SHOT-HOLDOUT trend-etf-v2 (keine Argumente, keine zweite Chance) ***")
    assert_no_tombstone_v2(LOGS_DIR)
    dfs = load_trend_snapshot(path=trend_snapshot_path_v2())
    candidates = sorted(glob.glob(os.path.join(LOGS_DIR, "trend_screening_v2_*", "candidate.json")))
    if not candidates:
        raise ValueError("Kein candidate.json der Familie trend-etf-v2 gefunden -- erst das Screening muss eine Kandidatin versiegeln")
    candidate = read_and_verify_candidate_v2(candidates[-1], snapshot_content_sha256(dfs))
    result = run_holdout(dfs, candidate)

    g = result["summary"]["gates"]
    print(f"\n{result['variant']}: holdout_gates={'PASS' if g['passed_all'] else 'FAIL'}  "
          f"excess_lower95={result['summary']['excess_bootstrap']['ann_geom_lower_1s95']:+.4f}  "
          f"breakeven_x={result['summary']['breakeven_cost_multiplier']:.2f}")
    if result["summary"]["borrow_sensitivity"] is not None:
        for bp, s in result["summary"]["borrow_sensitivity"].items():
            print(f"  Borrow {bp}bp p.a.: cagr={s['cagr']:+.2%}")
    if not g["passed_all"]:
        print("Familie trend-etf-v2 ist damit BEENDET (Spec-Abschnitt 10 Punkt 5) -- keine Runner-up-Variante.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LOGS_DIR, f"trend_holdout_v2_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "holdout_summary.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_tombstone_v2(LOGS_DIR, result_path)
    print(f"Ergebnis gespeichert: {result_path} (Tombstone geschrieben)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_run_trend_holdout_v2.py`
Expected: PASS.

- [ ] **Step 5: Run the FULL factor_lab test suite (v1 + v2, all files) to confirm zero regressions**

Run each of: `test_signals.py`, `test_costs.py`, `test_portfolio.py`, `test_stats.py`, `test_registration.py`, `test_registration_v2.py`, `test_data_snapshot.py`, `test_build_trend_snapshot.py`, `test_build_trend_snapshot_v2.py`, `test_run_trend_baseline.py`, `test_run_trend_baseline_v2.py`, `test_run_trend_holdout.py`, `test_run_trend_holdout_v2.py`.
Expected: all 13 PASS.

- [ ] **Step 6: Commit**

```bash
git add factor_lab/run_trend_holdout_v2.py factor_lab/tests/test_run_trend_holdout_v2.py
git commit -m "feat(factor_lab): add trend-etf-v2 one-shot holdout runner (full cost ladder from day one)"
```

---

### Task 6: Real execution

- [ ] **Step 1: Build the v2 snapshot:** `py -3.12 factor_lab/build_trend_snapshot_v2.py` — prints the content hash and resolved DEV_END for the 19-instrument universe. If `sanity_check_snapshot` raises on `min_common_days`, STOP and report the actual common-day count to the user rather than silently lowering the threshold further — that would be exactly the kind of post-hoc parameter adjustment the pre-registration methodology exists to prevent.
- [ ] **Step 2: Seal the manifest (BEFORE any screening run):**

```bash
git add -f factor_lab/data_snapshots/*.manifest.json
git commit -m "docs(factor_lab): seal trend-etf-v2 snapshot manifest (19-instrument universe, content hash)"
```

- [ ] **Step 3: Start screening detached** (LOO reruns over 19 instruments + 7 sleeves will take noticeably longer than v1's — expect this to run well over 20 minutes; use the detached Start-Process + PID-file + Monitor pattern from the Tech-Stack memory, `-u` for unbuffered output): `py -3.12 factor_lab/run_trend_baseline_v2.py`
- [ ] **Step 4: Report results** — table of all 8 variants (Gate A-D status, excess CI, cost ladder, breakeven), the 3 benchmarks, the verdict, and whether a candidate was sealed. **Do NOT run the holdout** (`run_trend_holdout_v2.py`) even if a candidate was sealed — that decision is the user's alone, exactly as with `trend-etf-v1`.
