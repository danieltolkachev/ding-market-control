"""
market_relative.py
====================

Marktrelatives Feature: wie stark bewegt sich das Symbol relativ zum
breiten Markt (Referenzindex, Standard SPY)?

    market_relative_return_t = log_return_symbol_t - log_return_reference_t

Positiv = Symbol schlaegt den Markt in diesem Bar, negativ = Symbol
haengt hinter dem Markt zurueck. Klassisches Quant-Feature (idiosynkratische
vs. systematische Bewegung) -- aus der urspruenglichen Systemvision bereits
als "Index- und Sektorbewegungen" genannt.

Additiv wie das News-Feature (siehe data_layer/news_client.py):
FeaturePipeline/LiveFeatureEngine bleiben unangetastet, das Feature wird
extern berechnet und angehaengt (siehe feature_pipeline.
MARKET_RELATIVE_FEATURE_NAMES).

Keine neuen Credentials noetig -- nutzt denselben Alpaca-Bars-Endpunkt wie
alles andere, nur fuer ein zweites Symbol (den Referenzindex).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_relative_features(symbol_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        symbol_df, reference_df: DataFrames mit Spalte "price" (Bar-Close),
            DatetimeIndex (wie von fetch_historical_bars_approximate() /
            fetch_historical_market_data()). reference_df ist typischerweise
            SPY oder ein anderer breiter Marktindex-ETF.

    Returns:
        DataFrame, indiziert wie symbol_df, mit Spalte "market_relative_return".
        Bars ohne (zeitlich nahen) Referenzwert werden auf 0.0 gesetzt
        (neutral) statt NaN -- verhindert NaN-Propagation in den Sequence-
        Buffer bei Randfaellen (z.B. minimale Zeitversatz zwischen den
        Handelskalendern der beiden Symbole).
    """
    symbol_log_return = np.log(symbol_df["price"]).diff()

    # Referenz-Log-Return auf den Zeitindex des Symbols ausrichten (kausal:
    # nur der zuletzt bekannte Referenzpreis VOR/AN diesem Zeitpunkt zaehlt).
    reference_log_price = np.log(reference_df["price"])
    reference_aligned = reference_log_price.reindex(symbol_df.index, method="ffill")
    reference_log_return = reference_aligned.diff()

    relative = (symbol_log_return - reference_log_return).fillna(0.0)
    return pd.DataFrame({"market_relative_return": relative}, index=symbol_df.index)


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from datetime import datetime, timedelta, timezone
    from alpaca_client import fetch_historical_bars_approximate

    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=5)

    aapl = fetch_historical_bars_approximate("AAPL", start, end)
    spy = fetch_historical_bars_approximate("SPY", start, end)
    print(f"AAPL: {aapl.shape[0]} Bars, SPY: {spy.shape[0]} Bars")

    rel = compute_market_relative_features(aapl, spy)
    print(rel.describe())
    print("\nLetzte 5 Zeilen:")
    print(rel.tail())
