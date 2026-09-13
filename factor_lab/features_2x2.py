"""Kausale Kanaele fuer das 2x2-Raster: bestehende fuenf plus fuenf erweiterte.

Alle Kanaele nutzen ausschliesslich Daten bis einschliesslich t und sind rein
preisabgeleitet — der Snapshot enthaelt weder Volumen noch OHLC noch Makro.
Beide Merkmalsbloecke werden auf **demselben** Zeilensatz zurueckgegeben (dem
der erweiterten Kanaele), damit die Feature-Achse nicht mit einer Aenderung der
Trainingsstichprobe vermischt wird.
"""
import numpy as np
import pandas as pd

BASE_HORIZONS = (1, 5, 21, 63, 126)
EXTENDED_NAMES = ('skip_momentum', 'cross_rank', 'vol_ratio', 'drawdown', 'spy_corr')
SEQUENCE = 20


def _validated(prices):
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError('prices must be a nonempty DataFrame')
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('Prices need unique chronological dates')
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Prices must be finite and positive')
    return prices


def extended_channels(prices, market='SPY'):
    """Die fuenf zusaetzlichen Kanaele E1-E5 der Spec, Abschnitt 3."""
    prices = _validated(prices)
    if market not in prices.columns:
        raise ValueError(f'market column {market!r} missing')
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(63, min_periods=63).std().clip(lower=1e-6)

    # E1: 126-Tage-Rendite endend vor 21 Tagen; meidet kurzfristige Umkehr.
    skip = (prices.shift(21) / prices.shift(147) - 1) / vol / np.sqrt(126)

    # E2: tagesweiser Rang von E1 ueber die Instrumente, linear auf [-1, +1].
    counts = skip.notna().sum(axis=1)
    ranks = skip.rank(axis=1, method='average')
    cross_rank = (2 * (ranks - 1).div(counts - 1, axis=0) - 1).where(counts > 1)

    # E3: Vola-Regimewechsel ohne separates Regimemodell.
    vol_short = returns.rolling(21, min_periods=21).std().clip(lower=1e-6)
    vol_long = returns.rolling(126, min_periods=126).std().clip(lower=1e-6)
    vol_ratio = np.log(vol_short / vol_long)

    # E4: Abstand zum 252-Tage-Hoch als Marktzustand.
    peak = prices.rolling(252, min_periods=252).max()
    drawdown = (prices / peak - 1) / vol / np.sqrt(252)

    # E5: Marktkopplung. Fuer das Marktinstrument selbst konstant 1 — die
    # Rolling-Korrelation kann numerisch minimal ueber 1 laufen, daher Clip.
    spy_corr = returns.rolling(63, min_periods=63).corr(returns[market]).clip(-1.0, 1.0)

    channels = {'skip_momentum': skip, 'cross_rank': cross_rank, 'vol_ratio': vol_ratio,
                'drawdown': drawdown, 'spy_corr': spy_corr}
    return {name: channels[name].astype(float) for name in EXTENDED_NAMES}


def build_2x2_dataset(prices, market='SPY'):
    """Ausgerichteter Datensatz mit Basis- und erweitertem Merkmalsblock.

    Label und Basiskanaele sind identisch zu ``daily_comparison.build_dataset``;
    nur der behaltene Zeilensatz ist enger, weil E1/E4 laengeren Vorlauf haben.
    """
    prices = _validated(prices)
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(63, min_periods=63).std().clip(lower=1e-6)
    base = [(prices.pct_change(h, fill_method=None) / vol / np.sqrt(h)).to_numpy()
            for h in BASE_HORIZONS]
    extra = [frame.to_numpy() for frame in extended_channels(prices, market).values()]
    features = np.stack(base + extra, axis=-1)
    last = SEQUENCE - 1
    positions = [d for d in range(last, len(prices))
                 if np.isfinite(features[d - last:d + 1]).all()]
    if not positions:
        raise ValueError('No complete 20-day extended feature sequences')
    window = np.stack([features[d - last:d + 1].transpose(1, 0, 2) for d in positions])
    window = window.astype('float32')
    y = (returns.shift(-2) / vol).clip(-10, 10).iloc[positions].to_numpy(dtype='float32')
    dates = prices.index[positions]
    return {'dates': dates, 'X_base': window[..., :len(BASE_HORIZONS)].copy(),
            'X_ext': window, 'y': y, 'vol': vol.loc[dates], 'returns': returns,
            'columns': list(prices.columns)}
