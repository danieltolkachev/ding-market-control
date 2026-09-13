import unittest
import pandas as pd
from factor_lab.run_horizon_comparison import assert_reproduction


class ReproductionTests(unittest.TestCase):
    def test_timestamp_units_differ_without_changing_dates(self):
        a = pd.DataFrame({'A':[.01,.02]},index=pd.date_range('2020-01-01',periods=2).as_unit('s'))
        b = a.copy()
        b.index = b.index.as_unit('us')
        assert_reproduction(a,b)
        assert_reproduction(a['A'],b['A'])

    def test_changed_dates_or_values_still_fail(self):
        a = pd.DataFrame({'A':[.01,.02]},index=pd.date_range('2020-01-01',periods=2))
        b = a*2
        with self.assertRaises(AssertionError):
            assert_reproduction(a,b)
        b = a.copy()
        b.index += pd.Timedelta(days=1)
        with self.assertRaises(AssertionError):
            assert_reproduction(a,b)


if __name__=='__main__':
    unittest.main()
