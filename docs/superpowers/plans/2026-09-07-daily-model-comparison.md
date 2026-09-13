# Daily model comparison implementation plan

**Goal:** Explore a path toward >=12% annual geometric net return before taxes, with a 15% historical drawdown evaluation limit and gross exposure <=1.25. Neither return nor future drawdown is guaranteed.

**Design approved:** User approved daily reference / simple learner / small frozen LSTM / online LSTM comparison. This document fixes the recipe before results. All history is exploratory, previously observed. No fresh holdout claim, purchases, or live orders.

**Architecture:** Use the verified preserved 19-ETF daily snapshot. Predict next executable close-to-close return in units of trailing daily volatility. Features known at close t; execution at close t+1; label is return t+2. Equal per-asset risk allocation, positive forecast means long, otherwise cash. Reference is positive 126-day momentum. No leverage in this first comparison (1.0 <= 1.25).

**Frozen protocol:** Five features per asset: daily return, 5/21/63/126-day momentum divided by daily volatility times sqrt(horizon). Volatility is trailing 63-day sample standard deviation. Clip standardized features to [-10,10]. Sequences 20 days. Labels clipped to [-10,10] after dividing by volatility at decision time. Pool instruments without asset IDs. Initial training on all fully matured examples before 2015-01-01; evaluate from 2015 through snapshot end. Training inputs scaled using training-only mean/std, fixed thereafter. Ridge alpha=100 on flattened sequences; LSTM hidden16, one layer, scalar regression MSE, Adam lr=.001, 5 epochs, batch256, seed7. Frozen LSTM never updates after initial fit. Online LSTM copies identical initial weights and updates each 21 decisions using last252 matured decision-days, one epoch at lr=.0001; first update after21 predictions. No parameter selection.

**Execution:** Daily target inverse-volatility weights, normalized over all19 assets before applying long masks (cash for excluded assets); scale to trailing63-day portfolio volatility <=10% annualized using information at decision time. Next-close fills: decision t earns first return t+2. Deduct cost on actual post-return drift-to-target traded notional, 3bp per side and stress6/15bp. Uninvested cash earns prior-day IRX/100/252. No shorting or financing. Report actual gross, turnover, CAGR, volatility, drawdown, annual returns, prediction MSE/sign accuracy and paired monthly growth differences with stationary block bootstrap. Daily rebalancing and costs intentionally differ from prior monthly experiment; compare within this study only.

**Tasks:**
- [x] Model worker: deterministic ridge/LSTM implementations plus tests for frozen identity and causal online updates.
- [x] Controller: causal dataset, exact daily accounting, runner, provenance, tests for time alignment and costs.
- [x] Independent review; fix important defects before full run.
- [x] Full run on preserved snapshot; archive configuration/source hashes/predictions/equity/results outside worktrees; report honestly and commit source.

**Files:** factor_lab/daily_models.py, daily_comparison.py, run_daily_comparison.py; matching tests under factor_lab/tests; results document adjacent to this plan's directory.

**Validation:** Toy accounting with known two-close delay; future-price poisoning cannot change earlier features/training labels/predictions; initial online/frozen equality; repeatability; finite results and identical evaluation indices. Python 3.12 with installed numpy/pandas/torch, no new dependencies.

**Ruling:** Reuse clean existing isolated trend-etf-v3-hysteresis worktree per worktree skill; preserve all archives. This is a bounded research spike, not production trading or validation of an investable strategy.

**Pre-run clarification (2026-09-08, no real-data results yet):** Ridge minimizes SUM squared residuals +100 times squared slope norm, with an unpenalized intercept. This gives the simple learner a learned unconditional mean like the LSTM bias. Five feature-channel means/stds use only mature initial examples and stay frozen. The final initial training label is observable at the first evaluation decision close (index first_test-2), not strictly before that calendar day. This is safe because trading occurs at the following close. Daily refers to predictions; online updates occur every21 decisions. No model recipe changes will be made after this run's results.

**Execution record:** Bacon implemented models and8 tests; controller implemented data/execution/runner and4 tests. Fermat approved accounting; Descartes approved final model integration. Removed unintended pre-standardization clipping before run. Full run completed with exit0 and69 verified artifact hashes. Results in `docs/superpowers/daily-model-comparison-2026-09-08-results.md`. No family is declared universally disproven; this fixed daily recipe did not meet the research target.
