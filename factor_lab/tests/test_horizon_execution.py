import unittest
import numpy as np
import pandas as pd
from factor_lab.daily_comparison import simulate


class HorizonExecutionTests(unittest.TestCase):
    def inputs(self):
        index = pd.bdate_range('2020-01-01',periods=6)
        returns = pd.DataFrame({'A':[0.,0.,.1,.2,.1,-.1]},index=index)
        targets = pd.DataFrame(.5,index=index,columns=['A'])
        cash = pd.Series(0.,index=index)
        decisions = pd.Series([True,False,False,True,False,False],index=index)
        return returns,cash,targets,decisions

    def test_holdings_drift_without_unscheduled_rebalancing(self):
        returns,cash,targets,decisions = self.inputs()
        net,detail = simulate(returns,cash,targets,0.,decision_mask=decisions)
        self.assertAlmostEqual((1+net.iloc[:4]).prod(),1.16)
        self.assertAlmostEqual(detail['gross'].iloc[2],.55/1.05)
        self.assertAlmostEqual(detail['gross'].iloc[3],.66/1.16)
        self.assertAlmostEqual(detail['gross'].iloc[4],.5)
        self.assertEqual(detail['turnover'].iloc[2],0.)
        self.assertEqual(detail['turnover'].iloc[3],0.)

    def test_costs_only_at_fills_and_nondecision_targets_ignored(self):
        returns,cash,targets,decisions = self.inputs()
        original,detail = simulate(returns,cash,targets,15.,decision_mask=decisions)
        changed = targets.copy()
        changed.loc[~decisions] = 1.
        poisoned,_ = simulate(returns,cash,changed,15.,decision_mask=decisions)
        np.testing.assert_array_equal(original,poisoned)
        np.testing.assert_array_equal(detail['cost'].iloc[[0,2,3,5]],0.)
        self.assertGreater(detail['cost'].iloc[1],0.)
        self.assertGreater(detail['cost'].iloc[4],0.)
        self.assertLessEqual(detail['gross'].max(),1.)

    def test_daily_default_identical_to_explicit_daily(self):
        returns,cash,targets,decisions = self.inputs()
        a,da = simulate(returns,cash,targets,3.)
        b,db = simulate(returns,cash,targets,3.,decision_mask=decisions|True)
        pd.testing.assert_frame_equal(da,db)
        pd.testing.assert_series_equal(a,b)

    def test_mismatched_schedule_rejected(self):
        returns,cash,targets,decisions = self.inputs()
        with self.assertRaises(ValueError):
            simulate(returns,cash,targets,3.,decision_mask=decisions.iloc[1:])

    def test_full_holding_return_matches_executable_label(self):
        returns,cash,targets,decisions = self.inputs()
        prices = pd.Series([100.,101.,110.,120.,130.,140.],index=returns.index)
        returns['A'] = prices.pct_change().fillna(0.)
        targets.iloc[:] = 0.
        targets.iloc[0] = 1.
        net,_ = simulate(returns,cash,targets,0.,decision_mask=decisions)
        # H=3: enter at close1; earn returns2,3,4; exit at close4.
        self.assertAlmostEqual((1+net).prod()-1,130/101-1)


if __name__=='__main__':
    unittest.main()
