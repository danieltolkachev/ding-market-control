"""Kausale Indikatoren der S4-Mechanik, Spec Abschnitt 3.

Alles wird pro zusammenhaengendem gueltigem Lauf berechnet: ein Datenbruch
setzt die Indikatorinitialisierung zurueck, der EMA-Zustand wird also NICHT
ueber eine Luecke fortgeschrieben. Undefinierte Stellen bleiben nan bzw.
False; ein epsilon ersetzt hier keine fehlende Datenqualitaet.

ATR ist ein einfacher gleitender Durchschnitt des True Range, NICHT der
Wilder-ATR. ATR und Volatilitaet sind verschiedene Groessen.
"""
import numpy as np
import pandas as pd

WARMUP = 260
SEQUENCE = 60
COLUMNS = ('open', 'high', 'low', 'close')


def _validated(bars):
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError('bars must be a nonempty DataFrame')
    if tuple(bars.columns) != COLUMNS:
        raise ValueError(f'bars must have exactly the columns {COLUMNS}')
    if (not bars.index.is_monotonic_increasing or bars.index.has_duplicates
            or bars.index.hasnans):
        raise ValueError('bars need unique chronological timestamps')
    return bars


def _valid_mask(bars):
    values = bars.to_numpy(dtype=float)
    ok = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    return ok & (bars['high'].to_numpy(dtype=float) >= bars['low'].to_numpy(dtype=float))


def _runs(valid):
    """Start- und Endindex (exklusiv) jedes zusammenhaengenden gueltigen Laufs."""
    out, start = [], None
    for i, flag in enumerate(valid):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(valid)))
    return out


def _ema(closes, n):
    """alpha=2/(n+1), Initialwert = SMA der ersten n gueltigen Closes."""
    out = np.full(len(closes), np.nan)
    if len(closes) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    out[n - 1] = closes[:n].mean()
    for i in range(n, len(closes)):
        out[i] = alpha * closes[i] + (1.0 - alpha) * out[i - 1]
    return out


def _trailing(values, n, function):
    """function auf jedem Fenster [i-n+1 : i], beide Enden einschliesslich."""
    out = np.full(len(values), np.nan)
    for i in range(n - 1, len(values)):
        window = values[i - n + 1:i + 1]
        if np.isfinite(window).all():
            out[i] = function(window)
    return out


def compute_indicators(bars):
    """Indikatorarrays in der Laenge von bars, laufweise zurueckgesetzt."""
    bars = _validated(bars)
    size = len(bars)
    valid = _valid_mask(bars)
    result = {name: np.full(size, np.nan) for name in
              ('logret', 'tr', 'atr20', 'sigma5', 'sigma20', 'er20',
               'ema20', 'ema50', 'ema200')}
    result['valid'] = valid
    result['run_length'] = np.zeros(size, dtype=int)
    result['up'] = np.zeros(size, dtype=bool)

    high = bars['high'].to_numpy(dtype=float)
    low = bars['low'].to_numpy(dtype=float)
    close = bars['close'].to_numpy(dtype=float)

    for start, stop in _runs(valid):
        result['run_length'][start:stop] = np.arange(1, stop - start + 1)
        c = close[start:stop]
        h, l = high[start:stop], low[start:stop]
        if len(c) < 2:
            continue
        logret = np.full(len(c), np.nan)
        logret[1:] = np.log(c[1:] / c[:-1])
        tr = np.full(len(c), np.nan)
        previous = c[:-1]
        tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - previous),
                                    np.abs(l[1:] - previous)])
        atr20 = _trailing(tr, 20, np.mean)
        sigma5 = _trailing(logret, 5, lambda w: np.std(w, ddof=1))
        sigma20 = _trailing(logret, 20, lambda w: np.std(w, ddof=1))
        er20 = np.full(len(c), np.nan)
        for i in range(20, len(c)):
            denominator = np.abs(np.diff(c[i - 20:i + 1])).sum()
            if denominator > 0:
                er20[i] = abs(c[i] - c[i - 20]) / denominator
        ema20, ema50, ema200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
        up = np.zeros(len(c), dtype=bool)
        for i in range(10, len(c)):
            if np.isfinite([ema50[i], ema200[i], ema50[i - 10]]).all():
                up[i] = (ema50[i] > ema200[i] and ema50[i] > ema50[i - 10]
                         and c[i] > ema50[i])
        for name, values in (('logret', logret), ('tr', tr), ('atr20', atr20),
                             ('sigma5', sigma5), ('sigma20', sigma20), ('er20', er20),
                             ('ema20', ema20), ('ema50', ema50), ('ema200', ema200)):
            result[name][start:stop] = values
        result['up'][start:stop] = up
    return result
