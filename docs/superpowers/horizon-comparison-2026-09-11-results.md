# Longer forecast and holding horizons — fixed exploratory comparison

The portfolio calculations show that less frequent execution improves the online LSTM's net return in this experiment, while predicting the21-day holding return does not improve it over holding daily forecasts for21days. No tested configuration reaches12% CAGR. Existing history remains exploratory, not independent confirmation.

Period:2015-01-02 through2026-08-31, 2,932 trading days,19 ETFs. Model recipes are unchanged except the fixed target horizon and its required maturity delay. H is the prediction horizon; holding interval controls actual next-close fills. Between fills, positions drift without trades. All values below are net of stated fees, before tax, FX, whole-share or broker minimum-fee effects.

| Model | H | Holding interval | CAGR3bp | MaxDD3bp | CAGR6bp | CAGR15bp | Turnover total3bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Online LSTM |1|1|2.51%|-12.02%|1.35%|-2.03%|439.52|
| Online LSTM |1|5|2.74%|-13.89%|2.38%|1.31%|135.96|
| Online LSTM |1|21|4.24%|-13.77%|4.12%|3.76%|44.69|
| Online LSTM |5|5|3.16%|-10.36%|2.81%|1.75%|133.95|
| Online LSTM |21|21|3.43%|-12.95%|3.33%|3.03%|37.96|
| Unconditional inverse-volatility long control |n/a|21|4.01%|-15.72%|3.97%|3.87%|13.03|

These rows explain the predeclared online comparison; the complete machine report includes all25 model/case rows and all75 costed return paths, plus cash. The H1/hold21 result is a descriptive observation, not a selected/validated trading candidate. Its near90% turnover reduction versus daily trading supports the cost hypothesis, but changing decision dates also changes exposures; the whole gain must not be attributed to fees alone. At15bp the unconditional control has higher CAGR than H1/hold21 online.

## Reproduction

Completed paired analysis at3bp (annual relative geometric growth; descriptive95% stationary-block-bootstrap intervals):

- Hold daily online forecasts for21days versus daily execution: +1.69% [+0.37%, +3.11%].
- Retrain for21-day targets versus hold the daily forecasts21days: -0.78% [-2.08%, +0.52%].
- Daily online forecasts held21days versus unconditional long at the same cadence: +0.22% [-1.60%, +2.34%].
- The same versus the momentum reference: +0.45% [-0.58%, +1.66%].

The first interval supports a within-experiment implementation improvement. The benchmark intervals do not establish a predictive edge. All intervals use140 monthly observations and are unadjusted for the141 reported comparisons; they are not an independent confirmation or a corrected significance claim.

Final verification: full run exited0; all156 artifact hashes verified;76 finite daily return series share2,932 dates;141 paired comparisons cover47 relationships at each3/6/15bp cost level. The previous three learned prediction sets and all12 original daily return paths reproduce within the predeclared numerical tolerance. No cost/configuration satisfies12% CAGR with the15% historical drawdown criterion. All26 tests passed across the24-test main suite and two timestamp-reproduction regressions. Independent review approved the horizon changes, matching-cost comparisons and timestamp fix. No post-result model tuning was performed.

```powershell
py -3.12 -m factor_lab.run_horizon_comparison --snapshot C:\Users\Daniel\Desktop\Ding\research_archive\trend_v2_preserved_20260907\data_snapshots\trend_snapshot_a654e3a4d7368cf2.pkl --daily-control C:\Users\Daniel\Desktop\Ding\research_archive\daily_model_comparisons\daily_20260907_220548_528118 --output-root C:\Users\Daniel\Desktop\Ding\research_archive\horizon_comparisons
```

Artifact directory: `C:\Users\Daniel\Desktop\Ding\research_archive\horizon_comparisons\horizon_20260911_144532_481498`.

The earlier partial attempt `horizon_20260911_144357_712803` stopped at CSV timestamp precision comparison and is marked INCOMPLETE. The correction normalizes timestamps without relaxing date/value comparisons; it does not change the strategy. Source and original partial files are retained.

The external ML4T source inspection is recorded separately in `ml4t-repository-findings-2026-09-11.md`. None of its proposed new features, models or trade filters changed this already fixed experiment.
