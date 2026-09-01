"""
build_trend_snapshot.py — EIGENER, einmaliger Build-Schritt (Spec v2 §4):
fetcht 12 ETFs + ^IRX via yfinance im fixen Fenster, prueft fail-closed,
friert ein, schreibt das Manifest und druckt Content-Hash + aufgeloestes
DEV_END. Das Manifest MUSS committet werden, BEVOR ein Screening laeuft.

Ausfuehren: py -3.12 factor_lab/build_trend_snapshot.py
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from factor_lab.registration import REGISTRATION
from factor_lab.data_snapshot import (
    SNAPSHOT_DIR, trend_snapshot_path, sanity_check_snapshot,
    snapshot_content_sha256, write_snapshot_manifest,
)
from factor_lab.portfolio import month_end_dates


def _download(symbol: str, column: str) -> pd.Series:
    import yfinance as yf
    raw = yf.download(symbol, start=REGISTRATION["snapshot_start"],
                      end=REGISTRATION["snapshot_end_exclusive"], auto_adjust=True, progress=False)
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
    for symbol in REGISTRATION["universe"]:
        print(f"  lade {symbol}...")
        dfs[symbol] = _download(symbol, "price").to_frame()
    print("  lade ^IRX (T-Bill-Cash-Naeherung)...")
    irx = _download(REGISTRATION["cash_series"], "rate_pa_pct").to_frame()
    dfs["IRX"] = irx.dropna()
    return dfs


def resolve_dev_end(dfs: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """DEV_END-Regel (praeregistriert): letzter Monatsultimo des gemeinsamen
    ETF-Kalenders <= dem 80%-Quantil-Datum."""
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
    print(f"=== Versiegelter Snapshot-Build ({REGISTRATION['snapshot_start']} bis "
          f"exkl. {REGISTRATION['snapshot_end_exclusive']}) ===")
    dfs = fetch_all()
    sanity_check_snapshot(dfs)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = trend_snapshot_path()
    with open(path, "wb") as f:
        pickle.dump(dfs, f)
    write_snapshot_manifest(dfs, path + ".manifest.json")
    print(f"  Snapshot: {path}")
    print(f"  Content-SHA256: {snapshot_content_sha256(dfs)}")
    print(f"  DEV_END (aufgeloest): {resolve_dev_end(dfs).date()}")
    print("\nJETZT das Manifest committen (git add -f ...), DANN erst run_trend_baseline.py.")


if __name__ == "__main__":
    main()
