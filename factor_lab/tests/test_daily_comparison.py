import unittest
import numpy as np
import pandas as pd
from factor_lab.daily_comparison import build_dataset, make_targets, simulate


class DailyComparisonTests(unittest.TestCase):
    def test_cash_only_and_zero_cost_rebalance(self):
        index = pd.date_range('2020-01-01', periods=3)
        returns = pd.DataFrame({'A':[0., .2, .1]},index=index)
        cash = pd.Series(.001,index=index)
        zeros = pd.DataFrame(0.,index=index,columns=['A'])
        net, _ = simulate(returns,cash,zeros,15.)
        np.testing.assert_allclose(net,.001)
        net, _ = simulate(returns,cash,zeros+1,0.)
        np.testing.assert_allclose(net,[.001,.001,.1])

    def test_excluded_assets_leave_cash(self):
        index = pd.bdate_range('2020-01-01',periods=70)
        returns = pd.DataFrame({'A':.001*np.sin(np.arange(70)), 'B':.001*np.cos(np.arange(70))},index=index)
        vol = pd.DataFrame(.01,index=index[-2:],columns=returns.columns)
        predictions = pd.DataFrame([[1,-1],[-1,-1]],index=vol.index,columns=vol.columns)
        target = make_targets(predictions,vol,returns)
        np.testing.assert_allclose(target.to_numpy(),[[.5,0],[0,0]])

    def test_next_close_execution_and_actual_drift_cost(self):
        index = pd.date_range('2020-01-01', periods=4)
        returns = pd.DataFrame({'A':[0., .1, .2, -.1]}, index=index)
        targets = pd.DataFrame({'A':[1., 0., 0., 0.]}, index=index)
        net, detail = simulate(returns, pd.Series(0., index=index), targets, 10.)
        self.assertAlmostEqual(net.iloc[0], 0.)
        self.assertAlmostEqual(net.iloc[1], -.001/1.001)
        self.assertAlmostEqual(net.iloc[2], .1988)
        self.assertAlmostEqual(net.iloc[3], 0.)
        self.assertAlmostEqual(detail['gross'].max(), 1.)

    def test_features_and_labels_respect_information_dates(self):
        index = pd.bdate_range('2010-01-01', periods=200)
        prices = pd.DataFrame({'A':100*np.exp(np.cumsum(.001+.01*np.sin(np.arange(200))))},index=index)
        original = build_dataset(prices)
        changed = prices.copy()
        changed.iloc[180:] *= 2
        poisoned = build_dataset(changed)
        earlier = original['dates'] < index[180]
        np.testing.assert_array_equal(original['X'][earlier], poisoned['X'][earlier])
        mature = original['dates'] < index[178]
        np.testing.assert_array_equal(original['y'][mature], poisoned['y'][mature])
        day = original['dates'][0]
        pos = prices.index.get_loc(day)
        expected = prices.pct_change().iloc[pos+2,0]/original['vol'].iloc[0,0]
        self.assertAlmostEqual(original['y'][0,0], np.clip(expected,-10,10))


if __name__ == '__main__':
    unittest.main()
