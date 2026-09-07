# Daily model comparison: completed exploratory run

The fixed daily experiment does **not** reach the 12% annual return research target. Online updating improves portfolio return over the frozen LSTM in this run, but the paired interval includes zero. It does not establish an online-learning edge. The simple momentum reference remains ahead.

Period: 2015-01-02 through 2026-08-31, 2,932 trading days, 19 ETFs. Initial training: 30,875 pooled examples from 1,625 mature decision-days. Features known at close t; fill at close t+1; first earned market return at t+2. All comparisons use the same costs, cash rates and long/cash risk allocation. This is not a fresh holdout. Results are net of assumed per-notional fees, before taxes; they do not model EUR conversion, whole shares or minimum broker fees.

| Model | CAGR, 3bp | Maximum drawdown | CAGR, 6bp | CAGR, 15bp |
|---|---:|---:|---:|---:|
| Positive 126-day momentum reference | 3.68% | -7.03% | 3.35% | 2.37% |
| Ridge regression | 0.58% | -16.70% | -2.52% | -11.25% |
| Frozen small LSTM | 1.83% | -10.20% | 0.31% | -4.11% |
| Online small LSTM | 2.51% | -12.02% | 1.35% | -2.03% |

At 3bp, a hypothetical 10,000 units become 13,340 with online LSTM over the **entire 11.6-year period**, not one year. The reference becomes 15,219. Geometric annual averages are not promised annual payouts.

## What the experiment establishes

- Online versus frozen relative geometric growth: +0.66% p.a., descriptive 95% stationary-block-bootstrap interval [-0.44%, +1.68%]. No clear improvement; no independent confirmation.
- Online versus reference: -1.12% p.a., interval [-2.64%, +0.35%]. The point estimate favors the reference.
- Online direction accuracy is 51.20%, versus 51.65% for always predicting up. Normalized MSE is 1.14480 versus frozen 1.14439. Portfolio improvement does not imply better forecast accuracy by every measure.
- Online turnover totals439.52 times NAV over the period, versus122.79 for the reference. Doubling costs reduces online CAGR from2.51% to1.35%. Frequent decisions create a substantial cost burden.
- Historical realized volatility is only3.85–4.53% p.a.; the10% volatility target is a ceiling, not a mandate to lever up. Gross allocation is capped at1.0, below the1.25 limit. The15% drawdown criterion is violated by ridge, not the other three at base costs. None meets the joint return/drawdown target.
- This comparison uses different daily accounting and allocation from the prior monthly trend experiment; do not interpret the cross-study CAGR difference as a direct algorithm comparison.

The defensible next hypothesis is whether a longer prediction/holding horizon can retain useful information while reducing turnover. That would need a separately fixed experiment, including a simple reference and an unconditional long/cash baseline. These results do not justify increasing leverage or tuning this run until it crosses12%.

## Reproduction and preservation

Command from the existing worktree:

```powershell
py -3.12 -m factor_lab.run_daily_comparison --snapshot C:\Users\Daniel\Desktop\Ding\research_archive\trend_v2_preserved_20260907\data_snapshots\trend_snapshot_a654e3a4d7368cf2.pkl --output-root C:\Users\Daniel\Desktop\Ding\research_archive\daily_model_comparisons
```

Completed artifact directory: `C:\Users\Daniel\Desktop\Ding\research_archive\daily_model_comparisons\daily_20260907_220548_528118` (UTC directory timestamp). Contains configuration, source snapshot, predictions, targets, daily accounting, labels, annual results, paired statistics, report, SHA256 manifest and COMPLETE marker. Preserved input content SHA256: `36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89`.

Validation: 12 unit tests passed, covering exact fee-funded execution, cash allocation, feature and label causality, maturity boundaries, initial frozen/online equality, online252-day window, deterministic repeatability, ridge objective and intercept. Two independent scoped reviews approved accounting and model integration after corrections, before the full run. Full run exited0. All69 archived artifact hashes verified; all12 return series share2,932 finite daily observations; first21 online/frozen prediction rows exactly identical. No parameter changes after results, no purchases, no live orders.
