"""Forward holding-period labels on the supplied trading-row calendar."""
import numpy as np
import pandas as pd


def horizon_labels(prices, vol, horizon) -> pd.DataFrame:
    """Return clipped, volatility-normalized close t+1 to t+H+1 returns.

    Prices must contain consecutive observations of the caller's trading
    calendar; calendar-day gaps (weekends/holidays) are allowed. Volatility
    must cover a contiguous chronological slice of that index, with matching
    asset columns. The final H+1 price rows have unavailable labels (NaN).
    """
    if (isinstance(horizon, (bool, np.bool_))
            or not isinstance(horizon, (int, np.integer)) or horizon < 1):
        raise ValueError('horizon must be a positive integer')
    horizon = int(horizon)
    for name, frame in (('prices', prices), ('vol', vol)):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f'{name} must be a nonempty DataFrame')
        if (not frame.index.is_monotonic_increasing or frame.index.has_duplicates
                or frame.index.hasnans or frame.columns.has_duplicates):
            raise ValueError(f'{name} must have unique chronological rows and asset columns')
        if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
            raise ValueError(f'{name} must be finite and positive')
    if not prices.columns.equals(vol.columns):
        raise ValueError('prices and vol must have matching asset columns')
    positions = prices.index.get_indexer(vol.index)
    if (positions < 0).any() or (np.diff(positions) != 1).any():
        raise ValueError('vol index must be a contiguous slice of prices index')
    returns = prices.shift(-(horizon + 1)) / prices.shift(-1) - 1
    return (returns.loc[vol.index] / (vol * np.sqrt(horizon))).clip(-10, 10)
