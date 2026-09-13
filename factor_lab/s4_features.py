"""Featurevektor der S4-Mechanik, Spec Abschnitt 7.

60 abgeschlossene Bars bis einschliesslich Event t, zehn Kanaele in fester
Reihenfolge, dazu vier Kontextwerte. Historische Rolling-Features bleiben auf
ihrem damaligen Stand; die erst bei p bekannten Setup-Anker stehen
ausschliesslich im Kontext, nicht rueckwirkend in der Sequenz.

Fehlt ein benoetigtes Datum, wird ValueError ausgeloest. Das Event ist dann
als FEATURE_INVALID zu protokollieren und weder zu trainieren noch zu handeln
-- es wird NICHT imputiert und NICHT nachtraeglich nach seinem Ertrag
ausgewaehlt.
"""
import numpy as np

from factor_lab.s4_indicators import SEQUENCE

CHANNELS = 10
CONTEXT = 4


def build_features(ind, bars, p, t):
    """Sequenz [60,10] und Kontext [4] fuer ein Event bei t mit Vorstufe p."""
    if not (0 <= p < t < len(bars)):
        raise ValueError('need 0 <= p < t < len(bars)')
    first = t - SEQUENCE + 1
    if first - 20 < 0 or p - 20 < 0:
        raise ValueError('insufficient history for the S4 feature window')
    close = bars['close'].to_numpy(dtype=float)
    high = bars['high'].to_numpy(dtype=float)
    low = bars['low'].to_numpy(dtype=float)

    sequence = np.empty((SEQUENCE, CHANNELS), dtype=float)
    for offset, j in enumerate(range(first, t + 1)):
        a, s = ind['atr20'][j - 1], ind['sigma20'][j - 1]
        if not np.isfinite([a, s]).all() or a <= 0 or s <= 0:
            raise ValueError(f'undefined normalizer at bar {j}')
        span = high[j] - low[j]
        sequence[offset] = (
            ind['logret'][j] / s,
            np.log(close[j] / close[j - 5]) / (s * np.sqrt(5)),
            np.log(close[j] / close[j - 20]) / (s * np.sqrt(20)),
            (close[j] - ind['ema20'][j - 1]) / a,
            (close[j] - ind['ema50'][j - 1]) / a,
            (ind['ema50'][j - 1] - ind['ema50'][j - 11]) / a,
            ind['tr'][j] / a,
            ind['sigma5'][j - 1] / s,
            ind['er20'][j],
            (close[j] - low[j]) / span if span > 0 else 0.5,
        )
    if not np.isfinite(sequence).all():
        raise ValueError('nonfinite value in the S4 feature sequence')

    anchor = ind['atr20'][p - 1]
    entry_atr = ind['atr20'][t - 1]
    if not np.isfinite([anchor, entry_atr]).all() or anchor <= 0 or entry_atr <= 0:
        raise ValueError('undefined ATR anchor for the S4 event context')
    # Kontext 2 darf negativ sein, wenn der Kurs das alte Hoch nie unterschreitet.
    context = np.array([
        (t - p) / 3.0,
        (high[p - 20:p].max() - low[p:t + 1].min()) / anchor,
        (close[t] - high[t - 1]) / entry_atr,
        entry_atr / close[t],
    ], dtype=float)
    if not np.isfinite(context).all():
        raise ValueError('nonfinite value in the S4 event context')
    return sequence, context
