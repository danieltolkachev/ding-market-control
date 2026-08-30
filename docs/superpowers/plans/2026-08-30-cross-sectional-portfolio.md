# Cross-Sectional Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a market-neutral long/short cross-sectional portfolio over 12 sector-diverse large caps, ranking each symbol's existing per-symbol LSTM edge against the others instead of trading each symbol independently long-only.

**Architecture:** Reuse the existing per-symbol offline-pretrain + LSTM forecast machinery unchanged. Add a new stateful ranking/hysteresis layer (`controller/cross_sectional_portfolio.py`) that replaces the single-symbol `ExposureController`/`RiskOverlay`, and a leaner `SymbolForecaster` (forecast + online-learning only, no position/execution) that replaces `ControlLoop` for this path. A new synchronized multi-symbol orchestration script drives all 12 forecasters bar-by-bar over their common timestamps, feeds the ranking layer, and executes each symbol's resulting weight through the existing `PaperExecutionEngine`.

**Tech Stack:** Python 3.12, PyTorch, pandas/numpy, Alpaca (`alpaca-py`) historical bars. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-30-cross-sectional-portfolio-design.md`

## Global Constraints

- Universe is fixed at 12 sector-diverse large caps for this iteration (no dynamic selection) — exact list: AAPL, MSFT, GOOGL, NVDA, JPM, JNJ, XOM, PG, HD, DIS, KO, CAT.
- No changes to per-symbol feature set, model architecture, or training hyperparameters — all three were already ruled out as the limiting factor this session.
- No risk-factor models beyond simple long/short dollar-neutral equal weighting.
- No result from a single backtest run is trusted as a decision — every parameter question goes through the multi-seed methodology already built this session (`backtest_stats.summarize_seed_distribution` / `paired_comparison`).
- Codebase convention: no pytest anywhere in this project. Tests are plain scripts under `tests/` with `assert` statements and a `run_..._check()` function invoked from `if __name__ == "__main__":`, run via `py -3.12 tests/test_xxx.py`. Follow this exactly — do not introduce pytest.
- Always run scripts with the `py -3.12` launcher (see project environment notes), not bare `python`.

## Two deliberate decisions this plan locks in (spec left them open)

1. **New `SymbolForecaster` class instead of reusing `ControlLoop`.** `ControlLoop` bakes together forecasting AND single-symbol position/execution — for cross-sectional, position comes from the ranking layer instead. Rather than bolting a "position override" onto the already-validated `ControlLoop`, Task 3 extracts a leaner sibling class that duplicates only the forecast+online-training code path, and proves it's behaviorally identical to `ControlLoop` on that path via a regression test. `ControlLoop` itself is not touched anywhere in this plan.
2. **Bar alignment via inner join, not mark-to-market-with-gaps.** The spec allowed a simpler fallback for the first iteration given the universe is 12 liquid large caps where timestamp gaps are expected to be rare. Task 4 aligns all 12 symbols' bars via intersection of their timestamp indices and only replays over that common index — simpler than per-symbol gap-handling, and the spec already flagged this as an acceptable simplification, not a compliance gap.

---

## Task 1: Shared universe constants + trivial baseline comparison

No dependency on Tasks 2/3 — can run in parallel with them.

**Files:**
- Create: `market_control_system/orchestration/cross_sectional_universe.py`
- Create: `market_control_system/orchestration/run_cross_sectional_baseline.py`

**Interfaces:**
- Produces: `UNIVERSE: list[str]`, `BACKTEST_LOOKBACK_DAYS: int`, `PRETRAIN_FRACTION: float`, `TIMESTEPS: int`, `HORIZON: int` — imported by Tasks 4 and 5.

- [ ] **Step 1: Create the shared constants module**

`market_control_system/orchestration/cross_sectional_universe.py`:

```python
"""
cross_sectional_universe.py
==============================

Gemeinsames Symbol-Universum und Basis-Konstanten fuer alle Cross-
Sectional-Skripte (Baseline, Haupt-Backtest, Multi-Seed-Vergleich) -- ein
Ort, damit alle drei garantiert denselben Zeitraum/dasselbe Universum
verwenden (sonst waeren Vergleiche zwischen ihnen nicht aussagekraeftig).

Bewusst sektoruebergreifend gestreut (nicht 12x Tech), siehe Design-Spec:
die sqrt(N)-Diversifikationsannahme hinter dem Cross-Sectional-Ansatz
setzt vergleichsweise unkorrelierte Symbole voraus.
"""
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "JNJ", "XOM", "PG", "HD", "DIS", "KO", "CAT"]
BACKTEST_LOOKBACK_DAYS = 365
PRETRAIN_FRACTION = 0.25
TIMESTEPS = 20
HORIZON = 5
```

- [ ] **Step 2: Create the baseline script**

`market_control_system/orchestration/run_cross_sectional_baseline.py`:

```python
"""
run_cross_sectional_baseline.py
==================================

Trivialer Vergleichs-Massstab fuer run_cross_sectional_backtest.py: ein
gleichgewichtetes Long-Only-Portfolio ueber cross_sectional_universe.
UNIVERSE, OHNE Modell/Prognose/Ranking -- reine Durchschnittsbildung der
Log-Returns. Kein Transaktionskosten-Modell (wird nie umgeschichtet), das
ist bewusst eine GUENSTIGERE Referenz als das echte Portfolio bekommt --
nur wenn das Ranking-Portfolio DAS hier schlaegt, traegt das Ranking
selbst etwas bei (siehe Design-Spec, "Testen/Validierung", Punkt 4).

Nutzt denselben Zeitraum/Split wie run_cross_sectional_backtest.py
(gleiche UNIVERSE/BACKTEST_LOOKBACK_DAYS/PRETRAIN_FRACTION-Konstanten),
damit der Vergleich auf demselben Out-of-Sample-Fenster steht.

Ausfuehren: py -3.12 orchestration/run_cross_sectional_baseline.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import numpy as np
import pandas as pd

from alpaca_client import fetch_historical_bars_approximate
from backtest_stats import compute_period_statistics, format_statistics_report
from cross_sectional_universe import UNIVERSE, BACKTEST_LOOKBACK_DAYS, PRETRAIN_FRACTION

CAPITAL = 10_000.0


def main():
    print(f"=== Lade {len(UNIVERSE)} Symbole ({BACKTEST_LOOKBACK_DAYS} Tage) ===")
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=BACKTEST_LOOKBACK_DAYS)
    dfs = {}
    for symbol in UNIVERSE:
        dfs[symbol] = fetch_historical_bars_approximate(symbol, start, end)
        print(f"  {symbol}: {dfs[symbol].shape[0]} Bars")

    aligned_index = dfs[UNIVERSE[0]].index
    for symbol in UNIVERSE[1:]:
        aligned_index = aligned_index.intersection(dfs[symbol].index)
    aligned_index = aligned_index.sort_values()

    split_idx = int(len(aligned_index) * PRETRAIN_FRACTION)
    cutoff = aligned_index[split_idx]
    replay_index = aligned_index[aligned_index >= cutoff]
    print(f"\nGemeinsamer Zeitindex: {len(aligned_index)} Bars, Cutoff: {cutoff}, "
          f"Replay-Fenster: {len(replay_index)} Bars")

    log_returns = pd.DataFrame({
        symbol: np.log(dfs[symbol].loc[replay_index, "price"]).diff()
        for symbol in UNIVERSE
    })
    equal_weight_return = log_returns.mean(axis=1).dropna()

    period_stats = compute_period_statistics(
        pd.DatetimeIndex(equal_weight_return.index), equal_weight_return.to_numpy(), period="W"
    )
    print(f"\n=== Baseline: gleichgewichtetes Long-Only-Portfolio ({len(UNIVERSE)} Symbole) ===")
    print(format_statistics_report(period_stats, period_label="Woche", capital=CAPITAL))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"cross_sectional_baseline_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    summary = {
        "run_id": run_id,
        "universe": UNIVERSE,
        "n_periods": period_stats.n_periods,
        "mean_period_return": period_stats.mean_return,
        "std_period_return": period_stats.std_return,
        "t_statistic": period_stats.t_statistic,
        "sharpe_like": period_stats.sharpe_like,
        "win_rate": period_stats.win_rate,
        "cumulative_return": period_stats.cumulative_return,
        "max_drawdown": period_stats.max_drawdown,
    }
    summary_path = os.path.join(results_dir, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it to verify it works end-to-end**

Run: `py -3.12 orchestration/run_cross_sectional_baseline.py` (from `market_control_system/`)
Expected: prints per-symbol bar counts, the aligned index size/cutoff, a `backtest_stats` report, and writes `logs/cross_sectional_baseline_<run_id>/baseline_summary.json`. No pytest involved — this is a real network-calling script, validated by running it, matching this codebase's existing convention for orchestration scripts (see `run_backtest.py`, none of which have unit tests either).

- [ ] **Step 4: Commit**

```bash
git add market_control_system/orchestration/cross_sectional_universe.py market_control_system/orchestration/run_cross_sectional_baseline.py
git commit -m "feat: add cross-sectional universe constants and trivial baseline"
```

---

## Task 2: Cross-sectional ranking + hysteresis + position sizing

No dependency on Task 1 or Task 3 — can run in parallel.

**Files:**
- Create: `market_control_system/controller/cross_sectional_portfolio.py`
- Test: `market_control_system/tests/test_cross_sectional_portfolio.py`

**Interfaces:**
- Produces: `CrossSectionalPortfolioConfig` (dataclass: `n_long: int = 3`, `n_short: int = 3`, `hysteresis_zone: int = 5`, `gross_exposure: float = 1.0`, `epsilon: float = 1e-6`), `CrossSectionalPortfolio(universe: list[str], config: CrossSectionalPortfolioConfig)` with `.step(mus: dict[str, float], sigmas: dict[str, float]) -> dict[str, float]`, and the pure functions `compute_edges`, `rank_and_select`, `compute_target_weights` — consumed by Task 4.

- [ ] **Step 1: Write the failing test**

`market_control_system/tests/test_cross_sectional_portfolio.py`:

```python
"""
test_cross_sectional_portfolio.py
====================================

Prueft die Ranking-/Hysterese-Logik in controller/cross_sectional_
portfolio.py mit synthetischen edge-Werten (kein Modell/keine echten
Daten noetig -- reine Zustandsmaschinen-Logik).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))

from cross_sectional_portfolio import (
    CrossSectionalPortfolioConfig,
    CrossSectionalPortfolio,
    compute_target_weights,
)


def check_config_validation() -> None:
    # hysteresis_zone < n_long muss ablehnen
    try:
        CrossSectionalPortfolioConfig(n_long=3, n_short=2, hysteresis_zone=2)
        raise AssertionError("Erwartete ValueError fuer hysteresis_zone < n_long, keine geworfen")
    except ValueError:
        pass

    # 2*hysteresis_zone > Universumsgroesse muss ablehnen
    try:
        CrossSectionalPortfolio(
            universe=["A", "B", "C"],
            config=CrossSectionalPortfolioConfig(n_long=1, n_short=1, hysteresis_zone=2),
        )
        raise AssertionError("Erwartete ValueError fuer 2*hysteresis_zone > Universum, keine geworfen")
    except ValueError:
        pass

    print("Config-Validierung: OK")


def check_target_weights() -> None:
    weights = compute_target_weights(
        longs={"X", "Y"}, shorts={"Z"}, universe=["X", "Y", "Z", "W"], gross_exposure=1.0,
    )
    assert abs(weights["X"] - 0.25) < 1e-9, weights
    assert abs(weights["Y"] - 0.25) < 1e-9, weights
    assert abs(weights["Z"] - (-0.5)) < 1e-9, weights
    assert weights["W"] == 0.0, weights
    assert abs(sum(abs(w) for w in weights.values()) - 1.0) < 1e-9, weights
    print("compute_target_weights: OK")


def check_ranking_entry_hysteresis_and_exit() -> None:
    """Drei-Schritt-Szenario ueber 6 Symbole (A..F), n_long=2, n_short=2,
    hysteresis_zone=3:
      Schritt 1: A,B gehen long, E,F gehen short (klarer Fall).
      Schritt 2: C ueberholt B im Rang, B faellt auf Rang 3 -- B bleibt
                 dank Hysterese-Zone (Top 3) trotzdem long, C kommt neu
                 dazu -> Long-Leg waechst voruebergehend auf 3.
      Schritt 3: B faellt auf Rang 4 (ausserhalb der Top-3-Zone) -- B
                 wird jetzt korrekt ausgeschlossen.
    """
    portfolio = CrossSectionalPortfolio(
        universe=["A", "B", "C", "D", "E", "F"],
        config=CrossSectionalPortfolioConfig(n_long=2, n_short=2, hysteresis_zone=3, gross_exposure=1.0),
    )

    # Schritt 1
    mus = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
    sigmas = {s: 1.0 for s in mus}
    weights = portfolio.step(mus, sigmas)
    assert portfolio.current_longs == {"A", "B"}, portfolio.current_longs
    assert portfolio.current_shorts == {"E", "F"}, portfolio.current_shorts
    assert weights["A"] == weights["B"] == 0.25, weights
    assert weights["E"] == weights["F"] == -0.25, weights

    # Schritt 2: C ueberholt B (Rang: A, C, B, D, E, F)
    mus = {"A": 5, "C": 4.5, "B": 4, "D": 2, "E": 1, "F": 0}
    weights = portfolio.step(mus, sigmas)
    assert portfolio.current_longs == {"A", "B", "C"}, (
        f"B sollte dank Hysterese noch gehalten werden, C neu dazu: {portfolio.current_longs}"
    )
    assert abs(weights["A"] - 1 / 6) < 1e-9, weights

    # Schritt 3: B faellt auf Rang 4 (ausserhalb Top-3: A, C, D, B, E, F)
    mus = {"A": 5, "C": 4.5, "D": 4, "B": 1, "E": 0.5, "F": 0}
    weights = portfolio.step(mus, sigmas)
    assert portfolio.current_longs == {"A", "C"}, (
        f"B sollte jetzt ausgeschlossen sein (ausserhalb Hysterese-Zone): {portfolio.current_longs}"
    )
    assert weights["B"] == 0.0, weights

    print("Ranking/Hysterese-Szenario (Entry/Retention/Exit): OK")


def run_consistency_check() -> None:
    check_config_validation()
    check_target_weights()
    check_ranking_entry_hysteresis_and_exit()
    print("\nAlle cross_sectional_portfolio-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.12 tests/test_cross_sectional_portfolio.py` (from `market_control_system/`)
Expected: `ModuleNotFoundError: No module named 'cross_sectional_portfolio'` — the module doesn't exist yet.

- [ ] **Step 3: Implement the module**

`market_control_system/controller/cross_sectional_portfolio.py`:

```python
"""
cross_sectional_portfolio.py
==============================

Layer "Controller" fuer das Cross-Sectional-Portfolio (siehe Design-Spec
docs/superpowers/specs/2026-08-30-cross-sectional-portfolio-design.md):
rankt die edge-Werte (mu/sigma^2) mehrerer Symbole zueinander und leitet
daraus ein marktneutrales Long/Short-Buch ab, statt wie ExposureController
eine einzelne Position fuer EIN Symbol zu berechnen.

Hysterese-Prinzip (Analogon zu RiskOverlayConfig.min_rebalance_threshold
im Single-Symbol-System): ein Symbol MUSS in die strengere Top-/Bottom-N-
Zone fallen, um NEU aufgenommen zu werden, darf aber in der breiteren
Hysterese-Zone bleiben, ohne sofort wieder ausgetauscht zu werden. Ohne
das wuerde jedes kleine Rausch-Wackeln am Rang N/N+1 dieselbe Art von
Kosten-Turnover erzeugen, die im Single-Symbol-System vor dem Deadband-Fix
beobachtet wurde (siehe backtest_20260824_223957-Befund).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossSectionalPortfolioConfig:
    n_long: int = 3
    n_short: int = 3
    hysteresis_zone: int = 5    # muss >= n_long und >= n_short sein
    gross_exposure: float = 1.0  # Summe |Gewicht| ueber alle Positionen (Long+Short zusammen)
    epsilon: float = 1e-6        # numerische Untergrenze im edge-Nenner, wie ExposureController

    def __post_init__(self):
        if self.hysteresis_zone < self.n_long or self.hysteresis_zone < self.n_short:
            raise ValueError(
                f"hysteresis_zone ({self.hysteresis_zone}) muss >= n_long ({self.n_long}) "
                f"und >= n_short ({self.n_short}) sein -- sonst kann eine gehaltene "
                f"Position nie in der Hysterese-Zone liegen."
            )


def compute_edges(mus: dict[str, float], sigmas: dict[str, float], epsilon: float = 1e-6) -> dict[str, float]:
    """edge = mu / (sigma^2 + epsilon), pro Symbol -- dieselbe Formel wie
    ExposureController.compute_target_position(), hier aber fuer das
    Ranking ueber mehrere Symbole statt fuer eine Einzelposition."""
    return {symbol: mus[symbol] / (sigmas[symbol] ** 2 + epsilon) for symbol in mus}


def rank_and_select(
    edges: dict[str, float],
    current_longs: set[str],
    current_shorts: set[str],
    n_long: int,
    n_short: int,
    hysteresis_zone: int,
) -> tuple[set[str], set[str]]:
    """
    Bestimmt die neuen Long-/Short-Mengen fuer diesen Zeitschritt.

    Regel: ein Symbol, das NICHT bereits gehalten wird, muss in die
    strikte Top-n_long- (bzw. Bottom-n_short-)Zone fallen, um neu
    aufgenommen zu werden. Ein Symbol, das BEREITS long/short gehalten
    wird, bleibt darin, solange es innerhalb der breiteren
    hysteresis_zone bleibt. Erst beim Verlassen dieser Zone wird die
    Position geschlossen.
    """
    ranked = sorted(edges, key=lambda s: edges[s], reverse=True)

    top_entry = set(ranked[:n_long])
    bottom_entry = set(ranked[-n_short:]) if n_short > 0 else set()
    top_zone = set(ranked[:hysteresis_zone])
    bottom_zone = set(ranked[-hysteresis_zone:]) if hysteresis_zone > 0 else set()

    new_longs = (current_longs & top_zone) | top_entry
    new_shorts = (current_shorts & bottom_zone) | bottom_entry

    # Randfall bei sehr kleinem Universum/grossen Zonen: ein Symbol darf
    # nicht gleichzeitig long und short sein. Bestehende Long-Position hat
    # Vorrang vor einem neu eintretenden Short (und umgekehrt).
    overlap = new_longs & new_shorts
    for symbol in overlap:
        if symbol in current_longs:
            new_shorts.discard(symbol)
        else:
            new_longs.discard(symbol)

    return new_longs, new_shorts


def compute_target_weights(
    longs: set[str], shorts: set[str], universe: list[str], gross_exposure: float = 1.0,
) -> dict[str, float]:
    """Gleichgewichtet innerhalb jedes Legs, dollar-neutral: Summe der
    Long-Gewichte = +gross_exposure/2, Summe der Short-Gewichte =
    -gross_exposure/2 (zusammen also gross_exposure an eingesetztem
    Kapital, netto markt-neutral)."""
    weights = {symbol: 0.0 for symbol in universe}
    if longs:
        w_long = (gross_exposure / 2.0) / len(longs)
        for symbol in longs:
            weights[symbol] = w_long
    if shorts:
        w_short = -(gross_exposure / 2.0) / len(shorts)
        for symbol in shorts:
            weights[symbol] = w_short
    return weights


class CrossSectionalPortfolio:
    """Haelt den Long-/Short-Zustand ueber die Zeit (Hysterese braucht
    Gedaechtnis an die vorherige Zusammensetzung)."""

    def __init__(self, universe: list[str], config: CrossSectionalPortfolioConfig = CrossSectionalPortfolioConfig()):
        if config.hysteresis_zone * 2 > len(universe):
            raise ValueError(
                f"2 * hysteresis_zone ({config.hysteresis_zone * 2}) darf das Universum "
                f"({len(universe)} Symbole) nicht ueberschreiten -- sonst ueberlappen "
                f"sich Long- und Short-Zone."
            )
        self.universe = list(universe)
        self.cfg = config
        self.current_longs: set[str] = set()
        self.current_shorts: set[str] = set()

    def step(self, mus: dict[str, float], sigmas: dict[str, float]) -> dict[str, float]:
        edges = compute_edges(mus, sigmas, epsilon=self.cfg.epsilon)
        self.current_longs, self.current_shorts = rank_and_select(
            edges, self.current_longs, self.current_shorts,
            self.cfg.n_long, self.cfg.n_short, self.cfg.hysteresis_zone,
        )
        return compute_target_weights(self.current_longs, self.current_shorts, self.universe, self.cfg.gross_exposure)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -3.12 tests/test_cross_sectional_portfolio.py` (from `market_control_system/`)
Expected: prints `Config-Validierung: OK`, `compute_target_weights: OK`, `Ranking/Hysterese-Szenario (Entry/Retention/Exit): OK`, then `Alle cross_sectional_portfolio-Checks bestanden.`

- [ ] **Step 5: Commit**

```bash
git add market_control_system/controller/cross_sectional_portfolio.py market_control_system/tests/test_cross_sectional_portfolio.py
git commit -m "feat: add cross-sectional ranking/hysteresis portfolio module"
```

---

## Task 3: SymbolForecaster (forecast + online-learning only, no position/execution)

No dependency on Task 1 or Task 2 — can run in parallel.

**Files:**
- Create: `market_control_system/orchestration/symbol_forecaster.py`
- Test: `market_control_system/tests/test_symbol_forecaster_consistency.py`

**Interfaces:**
- Consumes: `LSTMForecaster` (from `models/lstm_forecaster_torch.py`, unmodified), `OnlineTrainerConfig` (from `training/online_trainer.py`, unmodified).
- Produces: `ForecastResult` (dataclass: `mu: float`, `sigma: float`, `p_up: float`, `train_stats: dict | None`), `SymbolForecaster(timesteps, model, feature_config=..., horizon=5, feedback_min_batch_size=64, online_trainer_config=..., feature_names=None, scale_features=True, zscore_window=100)` with `.step(raw_event: dict) -> ForecastResult | None` — consumed by Task 4.

- [ ] **Step 1: Write the failing test**

`market_control_system/tests/test_symbol_forecaster_consistency.py`:

```python
"""
test_symbol_forecaster_consistency.py
========================================

Prueft, dass SymbolForecaster (Cross-Sectional-Pfad, orchestration/
symbol_forecaster.py) bei identischem Modell/Config/Daten exakt dieselbe
mu/sigma/p_up/train_stats-Sequenz produziert wie ControlLoop (Single-
Symbol-Pfad, orchestration/control_loop.py) -- SymbolForecaster ist als
Teilmenge von ControlLoop.step() gebaut (Feature/Sequence/Prognose/
Feedback/Online-Training identisch, nur Position/Risk/Ausfuehrung fehlt).
Divergenz hier wuerde bedeuten, dass die Extraktion versehentlich das
Prognose- oder Trainingsverhalten veraendert hat.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orchestration"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "execution"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feedback"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import numpy as np
import torch

from feature_pipeline import FEATURE_NAMES, _generate_synthetic_market_data
from lstm_forecaster_torch import LSTMForecaster
from online_trainer import OnlineTrainerConfig
from control_loop import ControlLoop
from exposure_controller import ControllerConfig
from risk_overlay import RiskOverlayConfig
from paper_execution import ExecutionConfig
from symbol_forecaster import SymbolForecaster

SEED = 123
N_BARS = 1500
TIMESTEPS = 20
HORIZON = 5


def run_control_loop_trace(df) -> list:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=16, num_layers=1)
    loop = ControlLoop(
        timesteps=TIMESTEPS, model=model, horizon=HORIZON,
        controller_config=ControllerConfig(k=0.01, max_position=1.0),
        risk_config=RiskOverlayConfig(max_step_change=0.1, max_sigma=0.08, drawdown_limit=-0.03, cooldown_steps=15, min_rebalance_threshold=0.30),
        execution_config=ExecutionConfig(slippage_bps=0.5),
        online_trainer_config=OnlineTrainerConfig(retrain_every=100, batch_size=32, max_training_window=256),
        recalibrate_k_on_retrain=False,   # kein Aequivalent in SymbolForecaster -- fuer den Vergleich deaktiviert
    )
    trace = []
    for _, row in df.iterrows():
        result = loop.step(row.to_dict())
        if result is None:
            trace.append(None)
        else:
            trace.append((result.mu, result.sigma, result.p_up,
                          result.train_stats["n_samples"] if result.train_stats else None))
    return trace


def run_symbol_forecaster_trace(df) -> list:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=16, num_layers=1)
    forecaster = SymbolForecaster(
        timesteps=TIMESTEPS, model=model, horizon=HORIZON,
        online_trainer_config=OnlineTrainerConfig(retrain_every=100, batch_size=32, max_training_window=256),
    )
    trace = []
    for _, row in df.iterrows():
        result = forecaster.step(row.to_dict())
        if result is None:
            trace.append(None)
        else:
            trace.append((result.mu, result.sigma, result.p_up,
                          result.train_stats["n_samples"] if result.train_stats else None))
    return trace


def run_consistency_check() -> None:
    df = _generate_synthetic_market_data(n=N_BARS, seed=7)

    control_loop_trace = run_control_loop_trace(df)
    forecaster_trace = run_symbol_forecaster_trace(df)

    assert len(control_loop_trace) == len(forecaster_trace) == N_BARS

    n_compared = 0
    for i, (a, b) in enumerate(zip(control_loop_trace, forecaster_trace)):
        if a is None or b is None:
            assert a is None and b is None, f"Warm-up-Divergenz bei Bar {i}: {a} vs {b}"
            continue
        mu_a, sigma_a, p_up_a, n_samples_a = a
        mu_b, sigma_b, p_up_b, n_samples_b = b
        assert abs(mu_a - mu_b) < 1e-9, f"mu weicht bei Bar {i} ab: {mu_a} vs {mu_b}"
        assert abs(sigma_a - sigma_b) < 1e-9, f"sigma weicht bei Bar {i} ab: {sigma_a} vs {sigma_b}"
        assert abs(p_up_a - p_up_b) < 1e-9, f"p_up weicht bei Bar {i} ab: {p_up_a} vs {p_up_b}"
        assert n_samples_a == n_samples_b, f"Online-Training-Zeitpunkt weicht bei Bar {i} ab: {n_samples_a} vs {n_samples_b}"
        n_compared += 1

    assert n_compared > 0, "Kein einziger vergleichbarer Schritt -- Warm-up-Logik pruefen"
    print(f"SymbolForecaster und ControlLoop stimmen ueber {n_compared} Schritte exakt ueberein "
          f"(mu, sigma, p_up, Online-Training-Zeitpunkte).")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.12 tests/test_symbol_forecaster_consistency.py` (from `market_control_system/`)
Expected: `ModuleNotFoundError: No module named 'symbol_forecaster'`.

- [ ] **Step 3: Implement SymbolForecaster**

`market_control_system/orchestration/symbol_forecaster.py`:

```python
"""
symbol_forecaster.py
======================

Layer 2-4 + 7-8 des Regelkreises (Feature-Engine -> Sequence-Buffer ->
LSTM-Prognose -> Feedback -> Online-Korrektur) fuer EIN Symbol, OHNE
Positions-/Risk-/Ausfuehrungslogik (Layer 5-6) -- die uebernimmt beim
Cross-Sectional-Portfolio die Ranking-Ebene (controller/cross_sectional_
portfolio.py) stattdessen, nicht ein Einzelsymbol-Controller.

Bewusst eine eigene, schlanke Klasse statt ControlLoop mit einem
"Positions-Override"-Parameter zu verbiegen: ControlLoop ist bereits
production-validiert (mehrere Backtest-Runs, siehe Projekt-Notizen) --
diese Klasse dupliziert nur den Forecast+Online-Training-Teil, statt das
bestehende, funktionierende Single-Symbol-System anzufassen. Die
Konsistenz beider Pfade (identische mu/sigma/train_stats-Sequenz bei
identischem Modell/Config/Daten) wird in
tests/test_symbol_forecaster_consistency.py explizit geprueft.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feedback"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from feature_pipeline import FeatureConfig, LiveFeatureEngine, FEATURE_NAMES, IncrementalZScoreScaler
from sequence_buffer import SequenceBuffer
from lstm_forecaster_torch import LSTMForecaster
from feedback_buffer import FeedbackBuffer
from online_trainer import OnlineTrainer, OnlineTrainerConfig


@dataclass
class ForecastResult:
    mu: float
    sigma: float
    p_up: float
    train_stats: dict | None


class SymbolForecaster:
    """Wie ControlLoop, aber ohne ExposureController/RiskOverlay/
    PaperExecutionEngine -- reine Prognose + Selbstlern-Schleife."""

    def __init__(
        self,
        timesteps: int,
        model: LSTMForecaster,
        feature_config: FeatureConfig = FeatureConfig(),
        horizon: int = 5,
        feedback_min_batch_size: int = 64,
        online_trainer_config: OnlineTrainerConfig = OnlineTrainerConfig(),
        feature_names: list[str] | None = None,
        scale_features: bool = True,
        zscore_window: int = 100,
    ):
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        self.extra_feature_names = [n for n in self.feature_names if n not in FEATURE_NAMES]
        self.scaler = IncrementalZScoreScaler(self.feature_names, window=zscore_window) if scale_features else None

        self.feature_engine = LiveFeatureEngine(feature_config)
        self.sequence_buffer = SequenceBuffer(timesteps=timesteps, feature_names=self.feature_names)
        self.model = model
        self.feedback_buffer = FeedbackBuffer(horizon=horizon, min_batch_size=feedback_min_batch_size)
        self.online_trainer = OnlineTrainer(model, online_trainer_config)

    def step(self, raw_event: dict) -> ForecastResult | None:
        price = raw_event["price"]
        self.feedback_buffer.resolve(price)

        result = None
        features = self.feature_engine.update(raw_event)
        if features is not None:
            for name in self.extra_feature_names:
                features[name] = raw_event[name]
            if self.scaler is not None:
                features = self.scaler.transform(features)
            self.sequence_buffer.push(features)
            if self.sequence_buffer.is_ready():
                X = self.sequence_buffer.get_batch_window()
                forecast = self.model.predict(X)
                mu = float(forecast.expected_return[0])
                sigma = float(forecast.expected_volatility[0])
                p_up = float(forecast.probability_up[0])

                self.feedback_buffer.record_window(X[0], entry_price=price)
                train_stats = self.online_trainer.maybe_update(self.feedback_buffer)

                result = ForecastResult(mu=mu, sigma=sigma, p_up=p_up, train_stats=train_stats)

        self.feedback_buffer.tick()
        return result
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -3.12 tests/test_symbol_forecaster_consistency.py` (from `market_control_system/`)
Expected: `SymbolForecaster und ControlLoop stimmen ueber <N> Schritte exakt ueberein (mu, sigma, p_up, Online-Training-Zeitpunkte).`

If this fails with a numeric mismatch: check that both `run_control_loop_trace` and `run_symbol_forecaster_trace` re-seed (`torch.manual_seed(SEED); np.random.seed(SEED)`) immediately before constructing their model, and that no other code runs between the seed call and model construction in either path — any extra random draw in between will desync the two traces.

- [ ] **Step 5: Commit**

```bash
git add market_control_system/orchestration/symbol_forecaster.py market_control_system/tests/test_symbol_forecaster_consistency.py
git commit -m "feat: add SymbolForecaster (forecast+online-learning, no position/execution)"
```

---

## Task 4: Synchronized multi-symbol backtest orchestration

Depends on Task 1 (universe constants), Task 2 (`CrossSectionalPortfolio`), Task 3 (`SymbolForecaster`).

**Files:**
- Create: `market_control_system/orchestration/run_cross_sectional_backtest.py`

**Interfaces:**
- Consumes: `UNIVERSE, BACKTEST_LOOKBACK_DAYS, PRETRAIN_FRACTION, TIMESTEPS, HORIZON` (Task 1), `CrossSectionalPortfolio, CrossSectionalPortfolioConfig` (Task 2), `SymbolForecaster` (Task 3), `fetch_historical_bars_approximate` (existing `data_layer/alpaca_client.py`), `build_scaled_features_and_target`, `FEATURE_NAMES` (existing `feature_engineering/feature_pipeline.py`), `SequenceWindowBuilder` (existing `feature_engineering/sequence_buffer.py`), `LSTMForecaster, train_epoch` (existing `models/lstm_forecaster_torch.py`), `PaperExecutionEngine, ExecutionConfig` (existing `execution/paper_execution.py`), `compute_period_statistics, format_statistics_report` (existing `training/backtest_stats.py`).
- Produces: `run_backtest(seed: int | None = None, portfolio_config: CrossSectionalPortfolioConfig | None = None) -> dict` — consumed by Task 5.

- [ ] **Step 1: Implement the orchestration script**

`market_control_system/orchestration/run_cross_sectional_backtest.py`:

```python
"""
run_cross_sectional_backtest.py
=================================

Hauptskript des Cross-Sectional-Ansatzes (siehe Design-Spec
docs/superpowers/specs/2026-08-30-cross-sectional-portfolio-design.md):
laedt alle Symbole aus cross_sectional_universe.UNIVERSE, trainiert pro
Symbol ein unabhaengiges LSTM (wie run_backtest.py), und spielt dann einen
SYNCHRONISIERTEN Replay ueber alle Symbole, bei dem CrossSectionalPortfolio
pro gemeinsamem Zeitschritt long/short/flat pro Symbol entscheidet.

Ausrichtung ueber Symbole: nur Zeitstempel, die ALLE Symbole gemeinsam
haben (Inner-Join), gehen in den Replay ein -- vereinfacht gegenueber dem
Design-Spec-Vorschlag (Mark-to-Market bei fehlenden Symbolen), aber bei
12 liquiden Large Caps wird der Datenverlust dadurch als gering erwartet
und die Vereinfachung ist fuer die erste Iteration bewusst in Kauf
genommen (siehe Spec, "Nicht-Ziel").

Ausfuehren: py -3.12 orchestration/run_cross_sectional_backtest.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "execution"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import numpy as np
import pandas as pd
import torch

from feature_pipeline import FEATURE_NAMES, build_scaled_features_and_target
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import LSTMForecaster, train_epoch
from paper_execution import PaperExecutionEngine, ExecutionConfig
from alpaca_client import fetch_historical_bars_approximate
from backtest_stats import compute_period_statistics, format_statistics_report
from cross_sectional_portfolio import CrossSectionalPortfolio, CrossSectionalPortfolioConfig
from symbol_forecaster import SymbolForecaster
from cross_sectional_universe import UNIVERSE, BACKTEST_LOOKBACK_DAYS, PRETRAIN_FRACTION, TIMESTEPS, HORIZON

CAPITAL = 10_000.0


def fetch_all_symbols() -> dict:
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=BACKTEST_LOOKBACK_DAYS)
    dfs = {}
    for symbol in UNIVERSE:
        print(f"  Lade {symbol}...")
        dfs[symbol] = fetch_historical_bars_approximate(symbol, start, end)
        print(f"    {dfs[symbol].shape[0]} Bars")
    return dfs


def align_and_split(dfs: dict):
    """Inner-Join der Zeitindizes ueber alle Symbole, dann Cutoff-Zeitpunkt
    fuer den Pretrain/Replay-Split (PRETRAIN_FRACTION in den gemeinsamen
    Index hinein)."""
    aligned_index = dfs[UNIVERSE[0]].index
    for symbol in UNIVERSE[1:]:
        aligned_index = aligned_index.intersection(dfs[symbol].index)
    aligned_index = aligned_index.sort_values()

    if len(aligned_index) == 0:
        raise ValueError("Kein gemeinsamer Zeitindex ueber alle Symbole -- Universum/Zeitraum pruefen")

    split_idx = int(len(aligned_index) * PRETRAIN_FRACTION)
    cutoff = aligned_index[split_idx]
    print(f"  Gemeinsamer Zeitindex: {len(aligned_index)} Bars "
          f"({aligned_index.min()} bis {aligned_index.max()})")
    print(f"  Cutoff (Vortraining/Replay-Grenze): {cutoff}")
    return aligned_index, cutoff


def pretrain_symbol(symbol: str, df: pd.DataFrame, cutoff) -> LSTMForecaster:
    pretrain_df = df[df.index < cutoff]
    features, target = build_scaled_features_and_target(pretrain_df, horizon=HORIZON)
    builder = SequenceWindowBuilder(timesteps=TIMESTEPS, feature_names=list(FEATURE_NAMES))
    X, y, _ = builder.build(features, target)

    print(f"  {symbol}: {pretrain_df.shape[0]} Vortrainings-Bars, {X.shape[0]} Sequenzen")
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        stats = train_epoch(model, optimizer, X, y, batch_size=64)
        print(f"    {symbol} Epoch {epoch+1}: mean_loss={stats['mean_loss']:.4f}")
    return model


def run_backtest(seed: int | None = None, portfolio_config: CrossSectionalPortfolioConfig | None = None) -> dict:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"=== Lade {len(UNIVERSE)} Symbole ({BACKTEST_LOOKBACK_DAYS} Tage) ===")
    dfs = fetch_all_symbols()

    print("\n=== Zeitindizes ausrichten ===")
    aligned_index, cutoff = align_and_split(dfs)
    replay_index = aligned_index[aligned_index >= cutoff]

    print(f"\n=== Offline-Vortraining pro Symbol ===")
    forecasters = {}
    for symbol in UNIVERSE:
        model = pretrain_symbol(symbol, dfs[symbol], cutoff)
        forecasters[symbol] = SymbolForecaster(timesteps=TIMESTEPS, model=model, horizon=HORIZON)

    portfolio = CrossSectionalPortfolio(UNIVERSE, portfolio_config or CrossSectionalPortfolioConfig())
    executions = {symbol: PaperExecutionEngine(ExecutionConfig(slippage_bps=0.5)) for symbol in UNIVERSE}

    print(f"\n=== Synchronisierter Replay ({len(replay_index)} gemeinsame Bars) ===")
    timestamps_list = []
    returns_list = []
    n_active_steps = 0

    for timestamp in replay_index:
        raw_events = {symbol: dfs[symbol].loc[timestamp].to_dict() for symbol in UNIVERSE}

        mus, sigmas = {}, {}
        all_ready = True
        for symbol in UNIVERSE:
            result = forecasters[symbol].step(raw_events[symbol])
            if result is None:
                all_ready = False
                continue
            mus[symbol] = result.mu
            sigmas[symbol] = result.sigma

        if not all_ready:
            continue  # noch nicht alle Symbole warmgelaufen (SequenceBuffer-Fuellphase)

        weights = portfolio.step(mus, sigmas)

        bar_return = 0.0
        for symbol in UNIVERSE:
            fill = executions[symbol].execute(weights[symbol], raw_events[symbol])
            bar_return += fill.realized_return

        n_active_steps += 1
        timestamps_list.append(timestamp)
        returns_list.append(bar_return)

        if n_active_steps % 5000 == 0:
            print(f"  {n_active_steps}/{len(replay_index)} Bars verarbeitet ({timestamp.date()}) "
                  f"-- longs={sorted(portfolio.current_longs)} shorts={sorted(portfolio.current_shorts)}")

    print(f"\n=== Replay abgeschlossen: {n_active_steps} aktive Bars ===")

    period_stats = compute_period_statistics(
        pd.DatetimeIndex(timestamps_list), np.array(returns_list), period="W"
    )
    print(format_statistics_report(period_stats, period_label="Woche", capital=CAPITAL))

    return {
        "seed": seed,
        "universe": UNIVERSE,
        "n_active_steps": n_active_steps,
        "n_periods": period_stats.n_periods,
        "mean_period_return": period_stats.mean_return,
        "std_period_return": period_stats.std_return,
        "t_statistic": period_stats.t_statistic,
        "sharpe_like": period_stats.sharpe_like,
        "win_rate": period_stats.win_rate,
        "cumulative_return": period_stats.cumulative_return,
        "max_drawdown": period_stats.max_drawdown,
        "period_returns": period_stats.period_returns.tolist(),
        "period_timestamps": [str(t) for t in
                               pd.Series(returns_list, index=pd.DatetimeIndex(timestamps_list))
                               .resample("W").sum().index],
    }


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"cross_sectional_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Cross-Sectional-Backtest-Run-ID: {run_id}")

    summary = run_backtest()
    summary["run_id"] = run_id

    summary_path = os.path.join(results_dir, "portfolio_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
```

Deliberate simplification for this iteration (documented, not an oversight): no per-bar CSV log of per-symbol edges/weights (unlike `run_backtest.py`'s per-symbol CSV). The periodic `longs=... shorts=...` print plus the final JSON summary are enough to validate the design; add a CSV later only if debugging a specific run requires it.

- [ ] **Step 2: Smoke-test at small scale before trusting a real run**

Same caution as every other orchestration script built this session — verify end-to-end plumbing on a tiny slice before a ~365-day/12-symbol run. Run this from `market_control_system/`:

```bash
py -3.12 -c "
import sys
sys.path.insert(0, 'orchestration')
sys.path.insert(0, 'feature_engineering')
sys.path.insert(0, 'models')
sys.path.insert(0, 'controller')
sys.path.insert(0, 'execution')
sys.path.insert(0, 'training')
sys.path.insert(0, 'data_layer')
sys.path.insert(0, 'config')
import cross_sectional_universe as u
u.UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'JPM']
u.BACKTEST_LOOKBACK_DAYS = 3
u.PRETRAIN_FRACTION = 0.5
import run_cross_sectional_backtest as m
m.UNIVERSE = u.UNIVERSE
m.BACKTEST_LOOKBACK_DAYS = u.BACKTEST_LOOKBACK_DAYS
m.PRETRAIN_FRACTION = u.PRETRAIN_FRACTION
summary = m.run_backtest(seed=0)
print('SMOKE TEST OK:', summary['n_active_steps'], 'aktive Bars,', summary['cumulative_return'], 'cum. Return')
"
```

Expected: runs end-to-end without exceptions, prints `SMOKE TEST OK: <N> aktive Bars, <x> cum. Return` with a finite number (not NaN/inf). If it raises `KeyError` on `dfs[symbol].loc[timestamp]`, double check that `align_and_split` is intersecting `.index` (DatetimeIndex), not something else. If `n_active_steps` is 0, the replay window was too short relative to `TIMESTEPS` warm-up — increase `BACKTEST_LOOKBACK_DAYS` in the smoke test.

Delete any log directories the smoke test created under `logs/cross_sectional_*` afterward (throwaway, same as every other smoke test this session).

- [ ] **Step 3: Launch the real full-scale run**

```bash
py -3.12 orchestration/run_cross_sectional_backtest.py
```

This is a 12-symbol, ~365-day run — expect it to take roughly 12x a single-symbol run's data-fetch time plus 12 independent offline-pretrain cycles (based on this session's measured ~7.3 min per 3-symbol full run, budget order-of-magnitude ~30-45 min, but confirm empirically rather than trusting this estimate — this session's own estimates have been off before in both directions). Run it with `run_in_background` if using an agentic harness that supports it.

- [ ] **Step 4: Commit**

```bash
git add market_control_system/orchestration/run_cross_sectional_backtest.py
git commit -m "feat: add synchronized cross-sectional backtest orchestration"
```

---

## Task 5: Multi-seed variant comparison

Depends on Task 4.

**Files:**
- Create: `market_control_system/orchestration/run_cross_sectional_multi_seed.py`

**Interfaces:**
- Consumes: `run_backtest(seed, portfolio_config)` (Task 4), `CrossSectionalPortfolioConfig` (Task 2), `summarize_seed_distribution, format_seed_distribution_report, paired_comparison, format_paired_comparison_report` (existing `training/backtest_stats.py`, built earlier this session).

- [ ] **Step 1: Implement the multi-seed comparison script**

`market_control_system/orchestration/run_cross_sectional_multi_seed.py`:

```python
"""
run_cross_sectional_multi_seed.py
====================================

Multi-Seed-Vergleich fuer CrossSectionalPortfolioConfig-Varianten
(n_long/n_short/hysteresis_zone), nach demselben Muster wie
orchestration/run_multi_seed_comparison.py fuer das Single-Symbol-System:
ein einzelner Lauf ist wegen Seed-Streuung keine verlaessliche Grundlage
fuer eine Entscheidung (siehe Projekt-Notizen zum Overnight-Backtest).

Ausfuehren: py -3.12 orchestration/run_cross_sectional_multi_seed.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from cross_sectional_portfolio import CrossSectionalPortfolioConfig
from backtest_stats import (
    summarize_seed_distribution, format_seed_distribution_report,
    paired_comparison, format_paired_comparison_report,
)
import run_cross_sectional_backtest as backtest_module

SEEDS = [0, 1, 2, 3, 4]
COMPARISON_METRICS = ["cumulative_return", "t_statistic", "sharpe_like", "win_rate", "max_drawdown"]

VARIANTS = {
    "default_3x3_h5": CrossSectionalPortfolioConfig(n_long=3, n_short=3, hysteresis_zone=5),
    "wider_5x5_h7": CrossSectionalPortfolioConfig(n_long=5, n_short=5, hysteresis_zone=7),
}


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"cross_sectional_multiseed_{run_id}")
    os.makedirs(base_dir, exist_ok=True)
    print(f"Cross-Sectional-Multi-Seed-Run-ID: {run_id}")
    print(f"Varianten: {list(VARIANTS)}, Seeds: {SEEDS}")

    all_distributions = {}
    for variant_name, config in VARIANTS.items():
        summaries = []
        for seed in SEEDS:
            print(f"\n{'#'*70}\n# Variante '{variant_name}', Seed {seed}\n{'#'*70}")
            summary = backtest_module.run_backtest(seed=seed, portfolio_config=config)
            summary_path = os.path.join(base_dir, f"{variant_name}_seed{seed}_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            summaries.append(summary)

        print(f"\n=== Seed-Verteilung: {variant_name} (n={len(summaries)} Seeds) ===")
        distributions = {}
        for metric in COMPARISON_METRICS:
            values = [s[metric] for s in summaries]
            dist = summarize_seed_distribution(metric, values)
            print(format_seed_distribution_report(dist))
            distributions[metric] = dist.values.tolist()
        all_distributions[variant_name] = distributions

        dist_path = os.path.join(base_dir, f"{variant_name}_seed_distribution.json")
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(distributions, f, indent=2)

    variant_names = list(all_distributions)
    for i in range(len(variant_names)):
        for j in range(i + 1, len(variant_names)):
            a, b = variant_names[i], variant_names[j]
            print(f"\n=== Vergleich: {b} vs. {a} (gepaart) ===")
            comparisons = {}
            for metric in COMPARISON_METRICS:
                cmp = paired_comparison(all_distributions[a][metric], all_distributions[b][metric], metric_name=metric)
                print(format_paired_comparison_report(cmp))
                comparisons[metric] = {
                    "mean_diff": cmp.mean_diff, "t_statistic": cmp.t_statistic, "b_wins_rate": cmp.b_wins_rate,
                }
            cmp_path = os.path.join(base_dir, f"comparison_{a}_vs_{b}.json")
            with open(cmp_path, "w", encoding="utf-8") as f:
                json.dump(comparisons, f, indent=2)

    print(f"\nAlle Ergebnisse in: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import wiring before the full multi-hour run**

```bash
py -3.12 -c "
import sys
sys.path.insert(0, 'orchestration')
sys.path.insert(0, 'controller')
sys.path.insert(0, 'training')
import run_cross_sectional_multi_seed as m
print('VARIANTS:', list(m.VARIANTS))
print('SEEDS:', m.SEEDS)
"
```

Expected: prints the two variant names and the 5 seeds without import errors.

- [ ] **Step 3: Launch the real multi-seed comparison**

```bash
py -3.12 orchestration/run_cross_sectional_multi_seed.py
```

2 variants x 5 seeds = 10 full runs of Task 4's backtest — budget accordingly based on Task 4's Step 3 measured runtime (10x that single-run time). Run with `run_in_background` if available. This is the run whose `comparison_default_3x3_h5_vs_wider_5x5_h7.json` output actually answers whether the cross-sectional approach shows a real, multi-seed-validated edge — nothing before this point should be treated as a final result.

- [ ] **Step 4: Commit**

```bash
git add market_control_system/orchestration/run_cross_sectional_multi_seed.py
git commit -m "feat: add multi-seed cross-sectional variant comparison"
```
