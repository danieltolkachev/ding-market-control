"""Causal daily features and self-financing long/cash portfolio accounting."""
import numpy as np
import pandas as pd


def build_dataset(prices):
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('Prices need unique chronological dates')
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Prices must be finite and positive')
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(63, min_periods=63).std().clip(lower=1e-6)
    features = np.stack([(prices.pct_change(h, fill_method=None)/vol/np.sqrt(h)).to_numpy()
                         for h in (1,5,21,63,126)], axis=-1)
    positions = [d for d in range(19,len(prices)) if np.isfinite(features[d-19:d+1]).all()]
    if not positions:
        raise ValueError('No complete 20-day feature sequences')
    X = np.stack([features[d-19:d+1].transpose(1,0,2) for d in positions]).astype('float32')
    y = (returns.shift(-2)/vol).clip(-10,10).iloc[positions].to_numpy(dtype='float32')
    dates = prices.index[positions]
    return {'dates':dates, 'X':X, 'y':y, 'vol':vol.loc[dates],
            'returns':returns, 'reference':(prices.pct_change(126)>0).loc[dates]}


def make_targets(predictions, vol, returns):
    """All assets contribute to normalizer; negative forecasts leave cash."""
    inv = 1/vol
    weights = inv.div(inv.sum(axis=1), axis=0)*(predictions > 0)
    result = weights.copy()
    for date in weights.index:
        trailing = returns.loc[:date].tail(63).to_numpy()
        risk = np.std(trailing @ weights.loc[date].to_numpy(), ddof=1)*np.sqrt(252)
        result.loc[date] *= min(1., .10/max(risk,1e-12))
    return result


def simulate(returns, cash, targets, cost_bp, decision_mask=None):
    """Decision t fills at close t+1; first market P&L at t+2.

    Weights refer to post-fee NAV. Solve trading cost implicitly so a fully
    invested position never borrows to pay its fee. All reported amounts
    use the previous close NAV as denominator. Optional boolean decision_mask
    selects decision closes; all other days leave holdings to drift.
    """
    if not returns.index.equals(targets.index) or not cash.index.equals(returns.index):
        raise ValueError('Mismatched calendars')
    if not returns.columns.equals(targets.columns):
        raise ValueError('Mismatched asset columns')
    if decision_mask is None:
        decision_mask = pd.Series(True,index=returns.index)
    if (not isinstance(decision_mask,pd.Series)
        or not decision_mask.index.equals(returns.index)
        or decision_mask.dtype != bool or decision_mask.isna().any()):
        raise ValueError('Decision mask must be a matching boolean series')
    if (not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any()
        or not np.isfinite(cash.to_numpy()).all() or (cash <= -1).any()
        or not np.isfinite(targets.to_numpy()).all()
        or (targets < 0).any().any() or (targets.sum(axis=1)>1+1e-10).any()
        or not 0 <= cost_bp < 10000):
        raise ValueError('Invalid returns, cash, weights or costs')
    weights = np.zeros(returns.shape[1])
    rows = []
    fee_rate = cost_bp/10000
    r_values, t_values = returns.to_numpy(), targets.to_numpy()
    for i in range(len(returns)):
        holdings = weights*(1+r_values[i])
        cash_value = max(0.,1-weights.sum())*(1+cash.iloc[i])
        growth = holdings.sum()+cash_value
        fee, turnover = 0.,0.
        if i and decision_mask.iloc[i-1]:
            target = t_values[i-1]
            # Monotone bounded root: fee = c * sum(abs(target*(growth-fee)-holdings)).
            lo, hi = 0., fee_rate*(growth+holdings.sum())
            for _ in range(40):
                fee = (lo+hi)/2
                due = fee_rate*np.abs(target*(growth-fee)-holdings).sum()
                if fee > due:
                    hi = fee
                else:
                    lo = fee
            fee = (lo+hi)/2
            turnover = np.abs(target*(growth-fee)-holdings).sum()
            weights = target.copy()
        else:
            weights = holdings/growth
        rows.append((growth-fee-1,fee,turnover,float(weights.sum())))
    detail = pd.DataFrame(rows,index=returns.index,columns=['net','cost','turnover','gross'])
    return detail['net'], detail
