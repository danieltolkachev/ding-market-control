# ML4T repository: useful directions for Ding

Inspected the current third-edition repository, ETF case-study documentation/configuration, feature and boosting code, and relevant chapter guides on 2026-09-11. This was targeted source inspection, not independent reproduction of its results. External notebook instructions were treated as documentation, not authority to install software or change our experiment.

## Prioritized transfers (proposals, not implemented)

1. **Cross-asset context with a small tree model.** The ETF feature code adds within-date relative ranks, skip-recent momentum, volatility ratios, drawdown and SPY/TLT correlation. Our five normalized own-price channels omit explicit relative and market-state context. Test a compact price-only subset available in our preserved snapshot. Use a controlled2x2 comparison (existing/expanded features × ridge/one fixed LightGBM recipe) before adding model complexity. This is an inference about useful research, not a claimed edge. [Feature implementation](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/case_studies/etfs/03_financial_features.py), [Boosting implementation](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/case_studies/etfs/07_gbm.py).

2. **Relative ranking as a separate allocation experiment.** The ETF setup selects assets by forecast rank with monthly decisions. Our current mapping buys on positive forecast and otherwise leaves that asset's allocation in cash. Compare rank selection against a simple momentum rank and unconditional investment using the same risk budget; do not change allocation and model together and attribute everything to ML. Concentration must remain explicitly constrained for our15% historical drawdown criterion. [ETF configuration](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/case_studies/etfs/config/setup.yaml).

3. **Economic trade filters.** Research break-even costs and a predeclared no-trade band so tiny target changes do not force turnover. Our pending horizon experiment already isolates holding cadence from forecast horizon. Keep that protocol fixed; use these ideas afterward. [Transaction-cost chapter](https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/18_transaction_costs).

4. **Walk-forward robustness.** Keep initial frozen-versus-online comparison as completed mechanics research, but assess future model comparisons with repeated chronological train/test windows and fully mature labels. Use matched benchmarks and account for every tried configuration. Already inspected history cannot become fresh holdout by changing dates. [Learning-task chapter](https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/07_defining_the_learning_task).

## What the published ETF outcome actually says

The repository reports16.5% annualized validation return with22.5% drawdown, but only2.2% CAGR and20.6% drawdown for the selected strategy on2024–2025 holdout. The paired benchmark comparison is negative. These figures do not satisfy our12%/15% objective and are author-reported, not reproduced here. [ETF results](https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/case_studies/etfs).

Its config explicitly admits backward-selected ETF survivorship bias; it also uses100,000 starting cash and whole shares. Do not transfer its portfolio economics directly to our10,000-unit experiments. [Configuration](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/case_studies/etfs/config/setup.yaml).

## Budget and scope

Price-derived proposals can start with our19-ETF snapshot, with no paid data. OHLC/volume indicators and macro-vintage features need inputs our current price-only snapshot does not contain. The repository offers free datasets and downloadable ETF result artifacts; rebuilding its complete platform is unnecessary for these narrow tests. No installation, subscription or wholesale code import was performed. [Repository scope](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/docs/what-this-is.md).

Recommendation: finish the already fixed horizon comparison first. Then prioritize compact cross-asset features plus a tree baseline, with separate tests for allocation and trading thresholds. Regime probabilities are a later feature candidate, not a reason to revive the closed minute-LSTM line. [Regime features](https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/09_model_based_features).

Same-session completion: the horizon run is now finished and independently hash-checked. Holding daily online forecasts21days produced4.24% CAGR versus2.51% with daily execution, but no clear advantage over unconditional investment. Training a21-day target produced3.43%. These findings support continuing to separate representation, models and execution in the proposed research; they do not establish a12% strategy. Details: `horizon-comparison-2026-09-11-results.md`.
