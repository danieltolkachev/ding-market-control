"""
evaluation.py — Fehlalarmraten und die VORAB festgelegten Kriterien
(Spec Abschnitt 3 und 7).

Zwei getrennte Masse, weil sie verschiedene Dinge messen:
  FA_variant  -- schlaegt Gate A pro Einzeltest zu oft an? (Nominal 5%)
  FA_pipeline -- versiegelt die "Beste von 8"-Regel zu oft? (kein Nominalwert)
Die DIFFERENZ beider Zahlen ist die Selektionsinflation der Auswahlregel.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

WORKING_SEEDS = tuple(range(200))
CONTROL_SEEDS = tuple(range(10000, 10050))

FA_VARIANT_LOWER = 0.025
FA_VARIANT_UPPER = 0.10
FA_PIPELINE_LIMIT = 0.15


def _wilson_ci95(hits: int, total: int) -> tuple[float, float]:
    """Wilson-Intervall: bei kleinen Raten deutlich ehrlicher als die
    Normalapproximation, und bei hits=0 nicht entartet."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = hits / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def false_alarm_rates(records: list[dict]) -> dict:
    n_replications = len(records)
    variant_hits = sum(sum(r["gate_a"].values()) for r in records)
    n_variant_tests = sum(len(r["gate_a"]) for r in records)
    pipeline_hits = sum(r["candidate"] is not None for r in records)
    return {
        "n_replications": n_replications,
        "n_variant_tests": n_variant_tests,
        "fa_variant": variant_hits / n_variant_tests if n_variant_tests else 0.0,
        "fa_variant_ci95": _wilson_ci95(variant_hits, n_variant_tests),
        "fa_pipeline": pipeline_hits / n_replications if n_replications else 0.0,
        "fa_pipeline_ci95": _wilson_ci95(pipeline_hits, n_replications),
    }


def verdict(rates: dict) -> dict:
    fa_v, fa_p = rates["fa_variant"], rates["fa_pipeline"]
    calibrated = FA_VARIANT_LOWER <= fa_v <= FA_VARIANT_UPPER
    inflated = fa_p > FA_PIPELINE_LIMIT
    if not calibrated and fa_v > FA_VARIANT_UPPER:
        text = ("SCHWERWIEGEND: Bootstrap-Inferenz selbst fehlkalibriert "
                f"(FA_variant={fa_v:.1%} > {FA_VARIANT_UPPER:.0%}). Alle bisherigen "
                "Gate-A-Aussagen inklusive v2s Screening-Pass sind nicht interpretierbar.")
    elif not calibrated:
        text = (f"Test uebermaessig konservativ (FA_variant={fa_v:.1%} < "
                f"{FA_VARIANT_LOWER:.1%}). Echte Effekte wuerden uebersehen.")
    elif inflated:
        text = (f"Einzeltest kalibriert (FA_variant={fa_v:.1%}), aber die Auswahlregel "
                f"ist inflationaer (FA_pipeline={fa_p:.1%} > {FA_PIPELINE_LIMIT:.0%}). "
                "Mehrfachtest-Korrektur noetig, bevor eine kuenftige Familie versiegelt wird.")
    else:
        text = (f"Methodik-Pruefung bestanden (FA_variant={fa_v:.1%}, "
                f"FA_pipeline={fa_p:.1%}). KEIN Edge-Nachweis.")
    return {"calibrated": calibrated, "selection_inflated": inflated, "text": text}


def seal_control_seeds(path: str) -> str:
    payload = json.dumps({"control_seeds": list(CONTROL_SEEDS),
                          "working_seeds": list(WORKING_SEEDS)},
                         indent=2, sort_keys=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
