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
