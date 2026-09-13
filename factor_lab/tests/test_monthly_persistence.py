"""Prueft Monatsbestaetigung, Kalenderluecken und Kausalitaet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from factor_lab.signal_persistence import apply_signal_persistence, daily_persistent_signals


def run_consistency_check():
    idx = pd.to_datetime(['2020-01-31','2020-02-28','2020-03-31','2020-04-30','2020-05-29','2020-06-30'])
    raw = pd.DataFrame({'A': [1,-1,1,-1,-1,-.3], 'B':[1,0,0,np.nan,-1,1]}, index=idx)
    out = apply_signal_persistence(raw)
    assert out.A.tolist() == [1,1,1,1,-1,-.3]
    np.testing.assert_allclose(out.B, [1,1,0,np.nan,-1,-1], equal_nan=True)
    gap = raw.iloc[[0,1,3,4]].copy()
    gap['A'] = [1,-1,-1,-1]
    assert apply_signal_persistence(gap).A.tolist() == [1,1,1,-1]
    for bad in [raw.iloc[::-1], pd.concat([raw, raw.iloc[-1:]]), raw.replace(1, np.inf)]:
        try:
            apply_signal_persistence(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Ungueltiger Input akzeptiert')
    dates = pd.bdate_range('2020-01-01', '2020-04-15')
    daily = pd.DataFrame(1., index=dates, columns=['A'])
    daily.loc['2020-02-03':] = -1.
    result = daily_persistent_signals(daily)
    assert result.loc['2020-01-30'].isna().all()
    assert result.loc['2020-02-28','A'] == 1.
    assert result.loc['2020-03-30','A'] == 1.
    assert result.loc['2020-03-31','A'] == -1.
    altered = daily.copy()
    altered.loc['2020-04-01':] = 1.
    pd.testing.assert_frame_equal(result.loc[:'2020-03-31'], daily_persistent_signals(altered).loc[:'2020-03-31'])
    assert daily_persistent_signals(daily.loc[:'2020-03-15']).iloc[-1,0] == 1.
    print('Monatsfilter: Bestaetigung, Null, NaN, Luecke, Validierung, Zukunft und Teilmonat OK')


if __name__ == '__main__':
    run_consistency_check()
