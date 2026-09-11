"""
audit_runner.py — eine Replikation des Falsifikations-Audits.

Ruft die ECHTE Screening-Pipeline auf (run_trend_baseline_v2.run_screening),
keine Nachbildung. Repliziert dabei die Trimmung auf dev_end genau so, wie
run_trend_baseline_v2.main() es tut -- sonst wuerde der Audit auf einer
laengeren Stichprobe messen als der echte Lauf und damit eine andere
CI-Breite und Fehlalarmrate.

Der Audit versiegelt bewusst NICHTS: kein candidate.json, kein Tombstone,
kein Schreiben nach factor_lab/logs/.
"""
from __future__ import annotations

import time

from factor_lab.audit.null_worlds import generate_null_panel
from factor_lab.run_trend_baseline_v2 import prepare_inputs, run_screening


def run_one_replication(world: str, params: dict, seed: int) -> dict:
    started = time.perf_counter()
    dfs = generate_null_panel(world, params, seed)
    dev_end = prepare_inputs(dfs)["dev_end"]
    dev_dfs = {name: df.loc[df.index <= dev_end] for name, df in dfs.items()}
    result, _ = run_screening(dev_dfs, dev_end=dev_end)
    summary = result["summary"]
    return {
        "world": world,
        "seed": seed,
        "candidate": result["candidate"],
        "gate_a": {v: bool(s["gates"]["gate_a_excess_ci"]) for v, s in summary.items()},
        "passed_all": {v: bool(s["gates"]["passed_all"]) for v, s in summary.items()},
        "excess_lower95": {v: float(s["excess_bootstrap"]["ann_geom_lower_1s95"])
                           for v, s in summary.items()},
        "dev_end": str(dev_end),
        "elapsed_s": time.perf_counter() - started,
    }
