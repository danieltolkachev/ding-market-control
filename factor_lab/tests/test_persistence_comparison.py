"""Gepaarter Lauf: gleicher Zeitraum, cash-Start und unveraenderte Inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from factor_lab.run_persistence_comparison import compare_period
from factor_lab.portfolio import month_end_dates


def run_consistency_check():
    idx = pd.bdate_range('2020-01-01','2021-12-31')
    returns = pd.DataFrame({'SPY': np.sin(np.arange(len(idx))) * .005 + .0002}, index=idx)
    signals = pd.DataFrame(1., index=idx, columns=['SPY'])
    inputs = {'symbols':['SPY'], 'returns':returns, 'cash_daily':pd.Series(.00001,index=idx),
              'vols':pd.DataFrame(.15,index=idx,columns=['SPY']),
              'signal_frames':{'combo':signals}, 'eval_decisions':month_end_dates(idx)[3:]}
    summary, daily = compare_period(inputs, idx[0], idx[-1], ['combo_long_flat'], [1.], n_boot=100)
    pd.testing.assert_series_equal(daily['combo_long_flat/raw/1.0'], daily['combo_long_flat/persistent/1.0'], check_names=False)
    assert summary['variants']['combo_long_flat']['paired_delta']['ann_geom'] == 0
    assert summary['variants']['combo_long_flat']['raw']['1.0']['trade_cost_sum'] > 0
    assert summary['variants']['combo_long_flat']['raw']['1.0']['stats']['n_days'] == len(daily)
    pd.testing.assert_frame_equal(inputs['signal_frames']['combo'], signals)
    assert daily.index[0] > inputs['eval_decisions'][0]
    # Der Split-Monatsultimo bleibt als kausale Entscheidung fuer den ersten
    # Fill der spaeteren Periode erhalten, ohne alte Positionen mitzunehmen.
    _, late = compare_period(inputs, pd.Timestamp('2021-01-01'), idx[-1], ['combo_long_flat'], [1.], n_boot=100)
    assert late.index[0] == pd.Timestamp('2021-01-01'), 'Erster Monat nach Split darf nicht verschwinden'
    assert late.iloc[0]['combo_long_flat/raw/1.0'] < 0, 'Cash-Start muss erste Einstiegskosten tragen'
    print('Gepaarter Lauf: Identitaet bei konstantem Signal, Kosten, Index und Inputschutz OK')


if __name__ == '__main__':
    run_consistency_check()
