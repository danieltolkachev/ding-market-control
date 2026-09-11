"""
test_audit_runner.py — eine Audit-Replikation gegen die ECHTE
v2-Screening-Pipeline. Wichtigste Zusicherung neben der Record-Struktur:
der Audit versiegelt nichts und schreibt nicht nach factor_lab/logs/.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from factor_lab.audit.audit_runner import run_one_replication
from factor_lab.run_trend_baseline_v2 import VARIANT_NAMES

_LOGS = os.path.join(os.path.dirname(__file__), "..", "logs")


def _small_params(n_days: int = 900) -> dict:
    """Kleines Panel: genug fuer warmup(315)+Monatsenden, aber schnell."""
    symbols = ["SPY", "TLT", "GLD", "EFA", "EEM", "UUP", "USO"]
    calendar = [d.strftime("%Y-%m-%d")
                for d in pd.date_range("2010-01-01", periods=n_days, freq="B")]
    garch = {"omega": 1e-6, "alpha": 0.08, "beta": 0.90, "loglik": 0.0}
    return {
        "symbols": symbols,
        "calendar": calendar,
        "n1": {"transition": [[0.95, 0.05], [0.10, 0.90]],
               "state_vol": {s: [0.006, 0.018] for s in symbols}},
        "n2": {s: dict(garch) for s in symbols},
        "n3": {"loadings": [[0.01, 0.004]] * len(symbols),
               "idio_vol": [0.008] * len(symbols),
               "factor_garch": [dict(garch), dict(garch)]},
    }


def check_record_structure() -> None:
    record = run_one_replication("n2", _small_params(), seed=0)
    assert record["world"] == "n2" and record["seed"] == 0
    assert record["candidate"] is None or record["candidate"] in VARIANT_NAMES
    assert sorted(record["gate_a"]) == VARIANT_NAMES
    assert sorted(record["passed_all"]) == VARIANT_NAMES
    assert sorted(record["excess_lower95"]) == VARIANT_NAMES
    assert all(isinstance(v, bool) for v in record["gate_a"].values())
    assert record["elapsed_s"] > 0
    print("run_one_replication: Record-Struktur: OK")


def check_writes_nothing_to_logs() -> None:
    before = set(os.listdir(_LOGS)) if os.path.isdir(_LOGS) else set()
    run_one_replication("n1", _small_params(), seed=1)
    after = set(os.listdir(_LOGS)) if os.path.isdir(_LOGS) else set()
    assert before == after, f"Audit hat nach logs/ geschrieben: {after - before}"
    print("run_one_replication: schreibt nichts nach factor_lab/logs/: OK")


def check_candidate_consistent_with_gates() -> None:
    """Eine Kandidatin darf nur gemeldet werden, wenn sie auch alle Gates
    besteht -- sonst misst FA_pipeline etwas anderes als gemeint."""
    for seed in range(3):
        record = run_one_replication("n3", _small_params(), seed=seed)
        if record["candidate"] is not None:
            assert record["passed_all"][record["candidate"]], (
                f"Kandidatin {record['candidate']} gemeldet, besteht aber nicht alle Gates"
            )
    print("run_one_replication: Kandidatin konsistent mit Gates: OK")


def run_consistency_check() -> None:
    check_record_structure()
    check_writes_nothing_to_logs()
    check_candidate_consistent_with_gates()
    print("\nAlle audit_runner-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
