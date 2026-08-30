"""
test_consistency.py
====================

Validiert, dass FeaturePipeline (Batch/pandas) und LiveFeatureEngine
(Streaming/inkrementell) auf identischen Rohdaten identische Feature-Werte
produzieren. Divergenz hier bedeutet: Training und Live-Inferenz sehen
unterschiedliche Feature-Verteilungen -> stiller, schwer zu findender Bug.

Deckt inzwischen ALLE 7 Features ab (log_return, realized_vol,
orderbook_imbalance, spread_norm, vwap_deviation, trade_intensity,
mid_price_return) -- vorherige Version testete nur 4, weil
LiveFeatureEngine vwap_deviation/trade_intensity/mid_price_return noch
nicht implementierte. Diese Luecke war ein Blocker fuer den Live-Betrieb
(SequenceBuffer.push() haette bei fehlenden Keys sofort eine KeyError
geworfen, sobald das 7-Feature-Modell live angeschlossen wird).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))

import numpy as np
import pandas as pd

from feature_pipeline import (
    FeaturePipeline,
    FeatureConfig,
    LiveFeatureEngine,
    _generate_synthetic_market_data,
)

COMPARABLE_COLUMNS = [
    "log_return",
    "realized_vol",
    "orderbook_imbalance",
    "spread_norm",
    "vwap_deviation",
    "trade_intensity",
    "mid_price_return",
]


def run_consistency_check(n_rows: int = 500, tolerance: float = 1e-6) -> None:
    cfg = FeatureConfig()
    df = _generate_synthetic_market_data(n=n_rows, seed=7)

    # Batch-Pfad
    batch_features = FeaturePipeline(cfg).transform(df)

    # Streaming-Pfad: Tick fuer Tick durchlaufen, Ergebnisse sammeln
    live_engine = LiveFeatureEngine(cfg)
    live_rows = {}
    for idx, row in df.iterrows():
        result = live_engine.update(row.to_dict())
        if result is not None:
            live_rows[idx] = result
    live_features = pd.DataFrame.from_dict(live_rows, orient="index")

    # Vergleich nur auf dem Index, den beide Pfade gemeinsam abdecken
    common_idx = batch_features.dropna().index.intersection(live_features.index)
    assert len(common_idx) > 0, "Kein gemeinsamer Index -> Warm-up-Logik pruefen"

    print(f"Vergleiche {len(common_idx)} gemeinsame Zeitpunkte auf {COMPARABLE_COLUMNS}")

    max_diffs = {}
    for col in COMPARABLE_COLUMNS:
        diff = (batch_features.loc[common_idx, col] - live_features.loc[common_idx, col]).abs()
        max_diffs[col] = diff.max()

    print("\nMaximale absolute Abweichung pro Feature:")
    all_ok = True
    for col, max_diff in max_diffs.items():
        status = "OK" if max_diff < tolerance else "ABWEICHUNG"
        if max_diff >= tolerance:
            all_ok = False
        print(f"  {col:22s}: {max_diff:.2e}  [{status}]")

    if all_ok:
        print("\nBatch- und Streaming-Pfad sind konsistent (alle 7 Features).")
    else:
        print("\nDivergenz gefunden -- Ursache vor Weiterbau klaeren!")
        raise AssertionError("Batch/Streaming Feature-Divergenz")


if __name__ == "__main__":
    run_consistency_check()
