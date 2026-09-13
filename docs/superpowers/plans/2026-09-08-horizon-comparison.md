# Longer horizon comparison — fixed protocol and execution ledger

User approved the proposed comparison of longer prediction and holding periods. This is a bounded extension of the completed daily research spike, not a new independent holdout. The existing isolated worktree and archived snapshot are retained.

## Question and fixed design (before results)

Can reduced turnover improve net results, and does predicting the actual holding-period return add value beyond merely trading daily forecasts less often?

- Main experiment:21 trading days. Secondary sensitivity:5 days. Daily1-day reproduction is a control. Run only these (forecast horizon, decision interval) combinations: (1,1), (1,5), (1,21), (5,5), (21,21). No post-result tuning, selection or phase search.
- Identical preserved19-ETF data, input sequences, scaler, models, seeds and train/evaluation dates as daily experiment. Forecasts computed each day; trades only every H evaluation decisions, anchored at the first evaluation date. Forecasting still uses all mature daily examples, including overlapping multi-day labels; descriptive inference uses portfolio monthly blocks, not independent sample assumptions.
- Label for decision t and forecast horizon H: (price[t+H+1]/price[t+1]-1)/(vol[t]*sqrt(H)), clipped to[-10,10]. Thus the full label first becomes observable at t+H+1. Initial and online fitting include only indices <=decision-(H+1). Features at t, execution at close t+1. Holdings earn t+2 through t+H+1 before next scheduled fill. Labels without full future observations remain NaN.
- Ridge and frozen/online LSTM use the previous fixed recipe, including seed7; online updates every21 daily decisions on the latest252 mature days. Frozen and online start with exactly identical weights per horizon. Holding interval never changes training or predictions.
- Signal reference: positive126-day momentum. Additional unconditional-long control: same inverse-volatility allocation and risk cap with every asset enabled. A cash-only path is also reported. Compare all learned models to both controls at the same holding interval and costs; online also to frozen. These controls identify returns obtainable without forecast timing.
- No intervening rebalancing on non-fill days: holdings and cash drift naturally. Fees are charged only at scheduled fills using the existing exact self-financing cost equation. Last incomplete holding period is marked to market, not artificially liquidated. Cash rate remains previous-close IRX. Gross <=1.0, within1.25 ceiling; trailing63-day10% annualized volatility cap only recalculated at decisions. No guaranteed15% future drawdown limit.
- Costs3/6/15bp per traded notional, no taxes/FX/whole-share/minimum fees. Target remains12% annualized geometric return after modeled costs and historical maxDD no worse than-15%. Report all configurations, CAGR/DD/turnover, annual returns, and paired monthly relative-growth stationary bootstrap6-month blocks,10,000 draws seed7, descriptive95% intervals without multiple-testing correction.
- Archive configuration, this protocol, source, predictions, labels, schedules/targets, daily returns, summary/report and hashes outside all worktrees. Explicitly retain earlier results. Daily control must reproduce previous archived results within numerical tolerance before treating study as complete.

## Work and verification

- [x] Model worker: configurable maturity delay and horizon-label helper; tests of hand-calculated labels, unavailable labels and backward compatibility.
- [x] Controller: sparse execution schedule with drift; tests distinguishing actual holding from stale-target daily rebalancing; runner and paired analysis.
- [x] Independent review and final tests before data run.
- [x] Run fixed study, verify daily reproduction and artifact hashes, document findings, commit source locally.

Do not reinterpret statistically inconclusive results as proof of either success or impossibility. Any better historical outcome remains exploratory.

Execution resumed2026-09-11 after reviewer usage-limit interruption. Anscombe reviewed and approved after requiring paired output for every cost level. First run stopped at differing timestamp storage units in CSV/source; normalized timestamps only, retained strict value/date comparisons, added two regressions and obtained scoped approval.26 tests passed in total. Completed run `horizon_20260911_144532_481498`,156 verified hashes,141 cost-matched paired comparisons, daily control reproduced. See `../horizon-comparison-2026-09-11-results.md`. External repository research did not modify this protocol or its models.
