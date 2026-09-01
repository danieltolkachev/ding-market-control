"""
data_snapshot.py — Load-only-Zugriff auf den versiegelten Snapshot
(Spec v2 Abschnitt 4). FETCHT NIE: Auswertungen laden nur, verifizieren
den Content-SHA256 gegen das committete Manifest und brechen bei
Abweichung ab. Der Build ist ein eigener Schritt (build_trend_snapshot.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market_control_system", "data_layer"))

import pandas as pd

from frozen_snapshot import snapshot_content_sha256, write_snapshot_manifest  # noqa: F401 (Re-Export fuer Build+Tests)
from factor_lab.registration import REGISTRATION

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "data_snapshots")


def trend_snapshot_path() -> str:
    key = (f"{sorted(REGISTRATION['universe'])}|{REGISTRATION['cash_series']}"
           f"|{REGISTRATION['snapshot_start']}|{REGISTRATION['snapshot_end_exclusive']}")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(SNAPSHOT_DIR, f"trend_snapshot_{digest}.pkl")


def sanity_check_snapshot(dfs: dict[str, pd.DataFrame], min_common_days: int = 4800) -> None:
    """Fail-closed-Pruefungen (Spec v2 §4). Wirft ValueError statt zu warnen."""
    common = None
    for name, df in dfs.items():
        if df.index.has_duplicates:
            raise ValueError(f"{name}: Duplikate im Index")
        if df.isna().any().any():
            raise ValueError(f"{name}: NaN-Werte im Snapshot")
        if name == "IRX":
            rates = df["rate_pa_pct"]
            if ((rates < -1.0) | (rates > 25.0)).any():
                raise ValueError("IRX: Rendite ausserhalb [-1, 25] Prozent p.a.")
            continue
        if (df["price"] <= 0).any():
            raise ValueError(f"{name}: nichtpositive Preise")
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < min_common_days:
        raise ValueError(f"Gemeinsamer Kalender zu kurz: {0 if common is None else len(common)} < {min_common_days}")


def load_trend_snapshot(path: str | None = None) -> dict[str, pd.DataFrame]:
    """Laedt den versiegelten Snapshot und verifiziert den Content-Hash
    gegen das Manifest. Kein Fetch-Fallback -- fehlender Snapshot ist ein
    Fehler (erst build_trend_snapshot.py ausfuehren und Manifest committen)."""
    path = path or trend_snapshot_path()
    manifest_path = path + ".manifest.json"
    if not os.path.exists(path) or not os.path.exists(manifest_path):
        raise ValueError(f"Snapshot oder Manifest fehlt ({path}) -- zuerst build_trend_snapshot.py ausfuehren")
    with open(path, "rb") as f:
        dfs = pickle.load(f)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    actual = snapshot_content_sha256(dfs)
    if actual != manifest["content_sha256"]:
        raise ValueError(f"Snapshot-Content-Hash {actual[:16]}... weicht vom Manifest ab -- Daten wurden veraendert")
    return dfs
