"""
frozen_snapshot.py
=====================

Laedt das Cross-Sectional-Universum EINMAL und friert es in einer
lokalen Datei ein, statt bei jedem Auswertungslauf erneut relativ zu
datetime.now() zu fetchen (siehe Design-Spec docs/superpowers/specs/
2026-08-31-cross-sectional-signal-diagnostics-design.md, Punkt 5):
ohne das laufen verschiedene Varianten/Wiederholungen nicht garantiert
auf identischen Bars, was Vergleiche zwischen ihnen verfaelscht.

Snapshot-Datei: Pickle eines dict[str, pd.DataFrame] (ein DataFrame pro
Symbol, Schema wie fetch_historical_bars_approximate()). Der Dateiname
enthaelt einen Hash der Parameter (Universum, Zeitraum), damit ein
Snapshot mit anderen Parametern nicht versehentlich wiederverwendet wird.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca_client import fetch_historical_bars_approximate

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "data_snapshots")


def _snapshot_hash(universe: list[str], lookback_days: int) -> str:
    key = f"{sorted(universe)}|{lookback_days}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def snapshot_content_sha256(dfs: dict[str, pd.DataFrame]) -> str:
    """SHA-256 ueber den TATSAECHLICHEN INHALT des Snapshots (Werte, Index,
    Spalten, Symbolnamen), nicht nur ueber die Fetch-Parameter wie
    _snapshot_hash(): build_snapshot() fetcht relativ zu datetime.now(),
    derselbe Parameter-Hash kann nach Loeschen+Neubau also auf ANDERE
    Daten zeigen. Dieser Hash pinnt in der Provenance eines Laufs fest,
    auf welchen Bars er wirklich gerechnet hat. Unabhaengig von der
    Dict-Einfuegereihenfolge (Symbole werden sortiert)."""
    hasher = hashlib.sha256()
    for symbol in sorted(dfs):
        df = dfs[symbol]
        hasher.update(symbol.encode("utf-8"))
        hasher.update("|".join(str(c) for c in df.columns).encode("utf-8"))
        hasher.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return hasher.hexdigest()


def snapshot_path(universe: list[str], lookback_days: int) -> str:
    return os.path.join(SNAPSHOT_DIR, f"snapshot_{_snapshot_hash(universe, lookback_days)}.pkl")


def build_snapshot(universe: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Laedt jedes Symbol aus dem Universum EINMAL per
    fetch_historical_bars_approximate()."""
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=lookback_days)
    dfs = {}
    for symbol in universe:
        print(f"  Snapshot: lade {symbol}...")
        dfs[symbol] = fetch_historical_bars_approximate(symbol, start, end)
        print(f"    {dfs[symbol].shape[0]} Bars")
    return dfs


def load_or_build_snapshot(universe: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Laedt den eingefrorenen Snapshot von Platte, falls vorhanden --
    sonst wird er einmalig gebaut und gespeichert. Alle spaeteren Aufrufe
    mit denselben Parametern nutzen exakt denselben Datensatz."""
    path = snapshot_path(universe, lookback_days)
    if os.path.exists(path):
        print(f"  Snapshot gefunden: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  Kein Snapshot gefunden, baue neu: {path}")
    dfs = build_snapshot(universe, lookback_days)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dfs, f)
    print(f"  Snapshot gespeichert: {path}")
    return dfs
