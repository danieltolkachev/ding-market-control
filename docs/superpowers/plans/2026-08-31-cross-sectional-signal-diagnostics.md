# Cross-Sectional Signal Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controller-free, cost-free diagnostic pipeline that measures whether the 12-symbol cross-sectional universe has ANY real Rank-IC signal, using walk-forward folds on a frozen data snapshot, before any further controller/deadband/hysteresis tuning.

**Architecture:** A frozen local data snapshot (fetched once, reused by every fold) feeds a cross-sectional walk-forward loop (per fold: 12 freshly-trained models, no online learning) that computes several score variants per symbol per bar and measures Rank-IC / gross top-minus-bottom spread against each variant, alongside random/momentum/reversal baselines in the same fold. A reserved holdout window is never touched by the automatic fold loop. No `ControlLoop`, `PaperExecutionEngine`, or `CrossSectionalPortfolio` is used anywhere in this plan — this is pure signal measurement.

**Tech Stack:** Python 3.12, PyTorch, pandas/numpy. No new dependencies (Spearman correlation implemented via `pandas.Series.rank()` + Pearson correlation, not scipy).

**Spec:** `docs/superpowers/specs/2026-08-31-cross-sectional-signal-diagnostics-design.md`

## Global Constraints

- No use of `orchestration/control_loop.py`, `execution/paper_execution.py`, or `controller/cross_sectional_portfolio.py` anywhere in this plan's code — this pipeline measures signal quality only, with no simulated positions, fills, or costs.
- No online training (`training/online_trainer.py`) — each walk-forward fold trains a fresh model and evaluates it frozen, matching `training/walk_forward.py`'s existing single-symbol convention, generalized to 12 symbols.
- Rank-IC and gross-spread evaluation use `horizon=1` (next-bar return) specifically, to eliminate any decision/fill-timing ambiguity by construction — do not reuse the project's other default of `horizon=5` for this pipeline.
- Codebase convention: no pytest. Tests are plain scripts under `tests/` using `assert` + a `run_..._check()` function invoked from `if __name__ == "__main__":`, run via `py -3.12 tests/test_xxx.py`.
- Same per-symbol model hyperparameters used everywhere else in this project: `hidden_size=32, num_layers=2, lr=1e-3, batch_size=64`.
- Always use `py -3.12`, never bare `python`.
- The holdout window (last `WALKFORWARD_FRACTION`-complement of the frozen snapshot) must never be read by the automatic fold loop — only by a separate, manually-invoked script (Task 5).

---

## Task 1: Frozen data snapshot

No dependency on other tasks.

**Files:**
- Create: `market_control_system/data_layer/frozen_snapshot.py`
- Test: `market_control_system/tests/test_frozen_snapshot.py`

**Interfaces:**
- Produces: `load_or_build_snapshot(universe: list[str], lookback_days: int) -> dict[str, pd.DataFrame]`, `snapshot_path(universe: list[str], lookback_days: int) -> str` — consumed by Task 4.

- [ ] **Step 1: Write the failing test**

`market_control_system/tests/test_frozen_snapshot.py`:

```python
"""
test_frozen_snapshot.py
==========================

Prueft, dass load_or_build_snapshot() beim ersten Aufruf eine Datei
anlegt und bei jedem weiteren Aufruf mit denselben Parametern GENAU
dieselben Daten zurueckgibt, ohne erneut zu fetchen (kein Netzwerkzugriff
noetig fuer diesen Test -- build_snapshot() wird durch eine Stub-Funktion
ersetzt, die deterministische synthetische Daten liefert).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))

import pandas as pd

import frozen_snapshot


def _fake_build_snapshot(universe, lookback_days):
    """Deterministischer Ersatz fuer build_snapshot() -- keine echten
    Netzwerkaufrufe im Test."""
    return {
        symbol: pd.DataFrame({"price": [100.0 + i for i in range(5)]})
        for symbol in universe
    }


def run_consistency_check() -> None:
    universe = ["AAA", "BBB"]
    lookback_days = 7

    path = frozen_snapshot.snapshot_path(universe, lookback_days)
    if os.path.exists(path):
        os.remove(path)

    original_build = frozen_snapshot.build_snapshot
    frozen_snapshot.build_snapshot = _fake_build_snapshot
    try:
        first = frozen_snapshot.load_or_build_snapshot(universe, lookback_days)
        assert os.path.exists(path), f"Snapshot-Datei wurde nicht angelegt: {path}"

        second = frozen_snapshot.load_or_build_snapshot(universe, lookback_days)
        for symbol in universe:
            pd.testing.assert_frame_equal(first[symbol], second[symbol])

        # Andere lookback_days -> anderer Pfad, kein Ueberschreiben
        other_path = frozen_snapshot.snapshot_path(universe, lookback_days + 1)
        assert other_path != path, "Unterschiedliche Parameter muessen unterschiedliche Snapshot-Pfade ergeben"

        print("frozen_snapshot: Datei angelegt, wiederholter Load liefert identische Daten, "
              "unterschiedliche Parameter -> unterschiedlicher Pfad -- OK")
    finally:
        frozen_snapshot.build_snapshot = original_build
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.12 tests/test_frozen_snapshot.py` (from `market_control_system/`)
Expected: `ModuleNotFoundError: No module named 'frozen_snapshot'`.

- [ ] **Step 3: Implement the module**

`market_control_system/data_layer/frozen_snapshot.py`:

```python
"""
frozen_snapshot.py
=====================

Laedt das Cross-Sectional-Universum EINMAL und friert es in einer
lokalen Datei ein, statt bei jedem Auswertungslauf erneut relativ zu
datetime.now() zu fetchen (siehe Design-Spec docs/superpowers/specs/
2026-08-31-cross-sectional-signal-diagnostics-design.md, Punkt 5):
ohne das laufen verschiedene Varianten/Wiederholungen nicht garantiert
auf identischen Bars, was Vergleiche zwischen ihnen verfaelscht.

Snapshot-Datei: Pickle eines dict[str, pd.DataFrame] (ein DataFrame pro
Symbol, Schema wie fetch_historical_bars_approximate()). Der Dateiname
enthaelt einen Hash der Parameter (Universum, Zeitraum), damit ein
Snapshot mit anderen Parametern nicht versehentlich wiederverwendet wird.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca_client import fetch_historical_bars_approximate

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "data_snapshots")


def _snapshot_hash(universe: list[str], lookback_days: int) -> str:
    key = f"{sorted(universe)}|{lookback_days}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def snapshot_path(universe: list[str], lookback_days: int) -> str:
    return os.path.join(SNAPSHOT_DIR, f"snapshot_{_snapshot_hash(universe, lookback_days)}.pkl")


def build_snapshot(universe: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Laedt jedes Symbol aus dem Universum EINMAL per
    fetch_historical_bars_approximate()."""
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=lookback_days)
    dfs = {}
    for symbol in universe:
        print(f"  Snapshot: lade {symbol}...")
        dfs[symbol] = fetch_historical_bars_approximate(symbol, start, end)
        print(f"    {dfs[symbol].shape[0]} Bars")
    return dfs


def load_or_build_snapshot(universe: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Laedt den eingefrorenen Snapshot von Platte, falls vorhanden --
    sonst wird er einmalig gebaut und gespeichert. Alle spaeteren Aufrufe
    mit denselben Parametern nutzen exakt denselben Datensatz."""
    path = snapshot_path(universe, lookback_days)
    if os.path.exists(path):
        print(f"  Snapshot gefunden: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  Kein Snapshot gefunden, baue neu: {path}")
    dfs = build_snapshot(universe, lookback_days)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dfs, f)
    print(f"  Snapshot gespeichert: {path}")
    return dfs
```

Also add the snapshot directory to `.gitignore` (it holds large, regenerable pickled market data — same treatment as `logs/`):

`market_control_system/.gitignore` — check the existing file first (`cat market_control_system/.gitignore`) and append a line for `data_snapshots/` if a project-local `.gitignore` exists there, otherwise add `market_control_system/data_snapshots/` to the root `.gitignore` (same file that already ignores `market_control_system/logs/`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -3.12 tests/test_frozen_snapshot.py` (from `market_control_system/`)
Expected: `frozen_snapshot: Datei angelegt, wiederholter Load liefert identische Daten, unterschiedliche Parameter -> unterschiedlicher Pfad -- OK`

- [ ] **Step 5: Commit**

```bash
git add market_control_system/data_layer/frozen_snapshot.py market_control_system/tests/test_frozen_snapshot.py .gitignore
git commit -m "feat: add frozen data snapshot for cross-sectional signal diagnostics"
```

---

## Task 2: Signal metrics (Rank-IC, gross spread, compounding, baselines)

No dependency on other tasks.

**Files:**
- Create: `market_control_system/controller/cross_sectional_signal_metrics.py`
- Test: `market_control_system/tests/test_cross_sectional_signal_metrics.py`

**Interfaces:**
- Produces: `compute_rank_ic(scores, forward_returns) -> float`, `compute_gross_spread(scores, forward_returns, n_long, n_short) -> float`, `compute_breakeven_cost(gross_spread_series, turnover_series) -> float`, `compound_return(returns) -> float`, `equity_curve(returns, initial=1.0) -> np.ndarray`, `max_drawdown_from_returns(returns) -> float`, `rolling_percentile_score(history, window) -> float`, `random_ranking_scores(symbols, seed) -> dict[str, float]`, `momentum_scores(prices, lookback_bars) -> dict[str, float]`, `reversal_scores(prices, lookback_bars) -> dict[str, float]` — all consumed by Task 4.

- [ ] **Step 1: Write the failing test**

`market_control_system/tests/test_cross_sectional_signal_metrics.py`:

```python
"""
test_cross_sectional_signal_metrics.py
=========================================

Prueft die reinen Statistikfunktionen in controller/cross_sectional_
signal_metrics.py mit synthetischen Werten (kein Modell/keine echten
Daten noetig).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))

import numpy as np

from cross_sectional_signal_metrics import (
    compute_rank_ic,
    compute_gross_spread,
    compute_breakeven_cost,
    compound_return,
    equity_curve,
    max_drawdown_from_returns,
    rolling_percentile_score,
    random_ranking_scores,
    momentum_scores,
    reversal_scores,
)


def check_rank_ic() -> None:
    scores = {"A": 1.0, "B": 2.0, "C": 3.0}
    perfect_positive = {"A": 0.01, "B": 0.02, "C": 0.03}
    ic = compute_rank_ic(scores, perfect_positive)
    assert abs(ic - 1.0) < 1e-9, f"Erwartete Rank-IC=1.0 bei perfekter positiver Korrelation, bekam {ic}"

    perfect_negative = {"A": 0.03, "B": 0.02, "C": 0.01}
    ic = compute_rank_ic(scores, perfect_negative)
    assert abs(ic - (-1.0)) < 1e-9, f"Erwartete Rank-IC=-1.0 bei perfekter negativer Korrelation, bekam {ic}"

    too_few = compute_rank_ic({"A": 1.0, "B": 2.0}, {"A": 0.1, "B": 0.2})
    assert np.isnan(too_few), "Bei < 3 gemeinsamen Symbolen sollte NaN zurueckkommen"

    print("compute_rank_ic: OK")


def check_gross_spread() -> None:
    scores = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
    forward_returns = {"A": 0.02, "B": 0.01, "C": -0.01, "D": -0.02}
    spread = compute_gross_spread(scores, forward_returns, n_long=1, n_short=1)
    # Long A (+0.02), Short D (-0.02) -> Spread = 0.02 - (-0.02) = 0.04
    assert abs(spread - 0.04) < 1e-9, f"Erwarteter Spread 0.04, bekam {spread}"
    print("compute_gross_spread: OK")


def check_breakeven_cost() -> None:
    gross = [0.01, 0.02, 0.0]
    turnover = [0.5, 0.5, 0.5]
    breakeven = compute_breakeven_cost(gross, turnover)
    expected = (sum(gross) / len(gross)) / (sum(turnover) / len(turnover))
    assert abs(breakeven - expected) < 1e-9
    print("compute_breakeven_cost: OK")


def check_compounding() -> None:
    returns = [0.1, 0.1]
    compounded = compound_return(returns)
    simple_sum = sum(returns)
    assert abs(compounded - 0.21) < 1e-9, f"Erwartetes Compound-Ergebnis 0.21, bekam {compounded}"
    assert abs(compounded - simple_sum) > 1e-6, (
        "Compounding und einfache Summe muessen bei diesen Werten unterschiedlich sein -- "
        "sonst wird faelschlich additiv gerechnet"
    )
    print("compound_return: OK (0.21, weicht bewusst von additiver Summe 0.20 ab)")


def check_drawdown() -> None:
    returns = [0.1, -0.2, 0.05]
    dd = max_drawdown_from_returns(returns)
    # equity: 1.1 -> 0.88 -> 0.924; running_max bleibt bei 1.1; Tiefpunkt 0.88/1.1-1=-0.2
    assert abs(dd - (-0.2)) < 1e-9, f"Erwarteter Max-Drawdown -0.2, bekam {dd}"
    print("max_drawdown_from_returns: OK")


def check_rolling_percentile() -> None:
    history = [1.0, 2.0, 3.0, 4.0, 100.0]  # letzter Wert ist der hoechste je gesehene
    pct = rolling_percentile_score(history, window=10)
    assert abs(pct - 1.0) < 1e-9, f"Hoechster je gesehener Wert sollte Perzentil 1.0 ergeben, bekam {pct}"

    too_short = rolling_percentile_score([1.0], window=10)
    assert too_short == 0.5, "Bei zu wenig Historie soll ein neutraler Default (0.5) zurueckkommen"
    print("rolling_percentile_score: OK")


def check_baselines() -> None:
    symbols = ["A", "B", "C"]
    scores_a = random_ranking_scores(symbols, seed=0)
    scores_b = random_ranking_scores(symbols, seed=0)
    scores_c = random_ranking_scores(symbols, seed=1)
    assert scores_a == scores_b, "Gleicher Seed muss identische Scores liefern"
    assert scores_a != scores_c, "Unterschiedlicher Seed sollte (fast sicher) unterschiedliche Scores liefern"
    assert set(scores_a) == set(symbols)

    prices = {
        "A": [100.0, 101.0, 102.0, 110.0],
        "B": [100.0, 99.0, 98.0, 90.0],
    }
    mom = momentum_scores(prices, lookback_bars=3)
    assert mom["A"] > 0, "A ist gestiegen -- Momentum-Score muss positiv sein"
    assert mom["B"] < 0, "B ist gefallen -- Momentum-Score muss negativ sein"

    rev = reversal_scores(prices, lookback_bars=3)
    assert rev["A"] < 0 and rev["B"] > 0, "Reversal muss exakt das Vorzeichen von Momentum umkehren"
    print("random_ranking_scores/momentum_scores/reversal_scores: OK")


def run_consistency_check() -> None:
    check_rank_ic()
    check_gross_spread()
    check_breakeven_cost()
    check_compounding()
    check_drawdown()
    check_rolling_percentile()
    check_baselines()
    print("\nAlle cross_sectional_signal_metrics-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.12 tests/test_cross_sectional_signal_metrics.py` (from `market_control_system/`)
Expected: `ModuleNotFoundError: No module named 'cross_sectional_signal_metrics'`.

- [ ] **Step 3: Implement the module**

`market_control_system/controller/cross_sectional_signal_metrics.py`:

```python
"""
cross_sectional_signal_metrics.py
====================================

Reine Statistikfunktionen fuer die Cross-Sectional-Signal-Diagnostik
(siehe Design-Spec docs/superpowers/specs/2026-08-31-cross-sectional-
signal-diagnostics-design.md). Bewusst OHNE jede Abhaengigkeit von
ControlLoop/PaperExecutionEngine/CrossSectionalPortfolio -- das hier
misst Signalqualitaet, nicht Handelsverhalten.

Spearman-Rangkorrelation wird ueber pandas' .rank() + Pearson-Korrelation
auf den Raengen selbst implementiert (Pearson-Korrelation der Raenge IST
per Definition die Spearman-Korrelation) -- keine scipy-Abhaengigkeit
noetig, die dieses Projekt bisher nicht hat (siehe requirements.txt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rank_ic(scores: dict[str, float], forward_returns: dict[str, float]) -> float:
    """Spearman-Rangkorrelation zwischen Score und tatsaechlichem
    Vorwaerts-Return ueber die gemeinsamen Symbole zu EINEM Zeitpunkt.
    NaN, wenn weniger als 3 gemeinsame Symbole vorliegen (Korrelation
    bei so wenigen Punkten nicht aussagekraeftig)."""
    common = sorted(set(scores) & set(forward_returns))
    if len(common) < 3:
        return float("nan")
    score_ranks = pd.Series([scores[s] for s in common]).rank()
    return_ranks = pd.Series([forward_returns[s] for s in common]).rank()
    return float(score_ranks.corr(return_ranks))


def compute_gross_spread(
    scores: dict[str, float], forward_returns: dict[str, float], n_long: int = 3, n_short: int = 3,
) -> float:
    """Long die hoechsten n_long Scores, Short die niedrigsten n_short,
    gleichgewichtet, OHNE Kosten, OHNE Deadband -- reiner Signal-Spread
    fuer EINEN Zeitpunkt. NaN, wenn nicht genug gemeinsame Symbole."""
    common = sorted(set(scores) & set(forward_returns))
    if len(common) < n_long + n_short:
        return float("nan")
    ranked = sorted(common, key=lambda s: scores[s], reverse=True)
    longs = ranked[:n_long]
    shorts = ranked[-n_short:]
    long_return = sum(forward_returns[s] for s in longs) / n_long
    short_return = sum(forward_returns[s] for s in shorts) / n_short
    return long_return - short_return


def compute_breakeven_cost(gross_spread_series: list[float], turnover_series: list[float]) -> float:
    """Welcher Pro-Einheit-Turnover-Kostensatz wuerde den mittleren
    Brutto-Spread auf 0 druecken. NaN, wenn mittlerer Turnover 0 ist."""
    mean_spread = float(np.mean(gross_spread_series))
    mean_turnover = float(np.mean(turnover_series))
    if mean_turnover == 0:
        return float("nan")
    return mean_spread / mean_turnover


def compound_return(returns: list[float]) -> float:
    """Echtes Compounding: prod(1+r) - 1, NICHT sum(r) -- siehe Design-
    Spec Review-Punkt 6 (additive Summe war der Fehler im bestehenden
    backtest_stats.py)."""
    result = 1.0
    for r in returns:
        result *= (1.0 + r)
    return result - 1.0


def equity_curve(returns: list[float], initial: float = 1.0) -> np.ndarray:
    """Echte Equity-Kurve: equity_t = equity_{t-1} * (1 + r_t)."""
    equity = [initial]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    return np.array(equity[1:])


def max_drawdown_from_returns(returns: list[float]) -> float:
    """Max Drawdown auf der ECHTEN (compoundenden) Equity-Kurve, nicht
    auf einer additiven cumsum-Kurve."""
    equity = equity_curve(returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def rolling_percentile_score(history: list[float], window: int = 500) -> float:
    """Perzentil-Rang des LETZTEN Werts in history relativ zu den
    VORHERIGEN `window` Werten (kausal -- der aktuelle Wert selbst geht
    nicht in sein eigenes Perzentil ein, und es werden nie zukuenftige
    Werte verwendet). Neutraler Default 0.5 bei zu wenig Historie."""
    if len(history) < 2:
        return 0.5
    lookback = history[max(0, len(history) - 1 - window):-1]
    if not lookback:
        return 0.5
    current = history[-1]
    return sum(1 for v in lookback if v <= current) / len(lookback)


def random_ranking_scores(symbols: list[str], seed: int) -> dict[str, float]:
    """Zufaellige Scores fuer die Random-Ranking-Baseline -- deterministisch
    pro Seed, damit Wiederholungen ueber mehrere Seeds gemittelt werden
    koennen."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=len(symbols))
    return {symbol: float(v) for symbol, v in zip(symbols, values)}


def momentum_scores(prices: dict[str, list[float]], lookback_bars: int) -> dict[str, float]:
    """Score = Return der letzten lookback_bars Bars. Symbole mit zu
    kurzer Preishistorie werden ausgelassen (kein kuenstlicher Default)."""
    scores = {}
    for symbol, series in prices.items():
        if len(series) <= lookback_bars:
            continue
        past = series[-1 - lookback_bars]
        current = series[-1]
        scores[symbol] = (current - past) / past
    return scores


def reversal_scores(prices: dict[str, list[float]], lookback_bars: int) -> dict[str, float]:
    """Exaktes Gegenteil von momentum_scores() -- dieselben Symbole,
    negiertes Vorzeichen."""
    return {symbol: -score for symbol, score in momentum_scores(prices, lookback_bars).items()}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -3.12 tests/test_cross_sectional_signal_metrics.py` (from `market_control_system/`)
Expected: all 7 check functions print `OK`, then `Alle cross_sectional_signal_metrics-Checks bestanden.`

- [ ] **Step 5: Commit**

```bash
git add market_control_system/controller/cross_sectional_signal_metrics.py market_control_system/tests/test_cross_sectional_signal_metrics.py
git commit -m "feat: add cross-sectional signal metrics (rank-IC, gross spread, compounding, baselines)"
```

---

## Task 3: Cross-sectional fold training helpers

No dependency on other tasks.

**Files:**
- Create: `market_control_system/training/cross_sectional_fold_training.py`
- Test: `market_control_system/tests/test_cross_sectional_fold_training.py`

**Interfaces:**
- Consumes: `WalkForwardConfig`, `generate_fold_slices` (existing, `training/walk_forward.py`, unmodified), `FEATURE_NAMES`, `build_scaled_features_and_target` (existing, `feature_engineering/feature_pipeline.py`), `SequenceWindowBuilder` (existing, `feature_engineering/sequence_buffer.py`), `LSTMForecaster`, `train_epoch` (existing, `models/lstm_forecaster_torch.py`).
- Produces: `build_symbol_sequences(df: pd.DataFrame, cfg: WalkForwardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (X, y, end_idx), `train_fold_model(X_train: np.ndarray, y_train: np.ndarray, cfg: WalkForwardConfig) -> LSTMForecaster` — consumed by Task 4.

- [ ] **Step 1: Write the failing test**

`market_control_system/tests/test_cross_sectional_fold_training.py`:

```python
"""
test_cross_sectional_fold_training.py
========================================

Prueft build_symbol_sequences()/train_fold_model() auf synthetischen
Daten -- kein Netzwerkzugriff, keine echten Marktdaten noetig. Verifiziert
nur, dass die Formen stimmen und ein trainiertes Modell tatsaechlich
predict() ohne Fehler ausfuehren kann, NICHT die Vorhersagequalitaet
(die ist auf synthetischen Random-Walk-Daten erwartungsgemaess nahe
Zufall, siehe training/walk_forward.py).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from feature_pipeline import FEATURE_NAMES, _generate_synthetic_market_data
from walk_forward import WalkForwardConfig
from cross_sectional_fold_training import build_symbol_sequences, train_fold_model


def run_consistency_check() -> None:
    cfg = WalkForwardConfig(horizon=1, timesteps=20, train_size=200, test_size=50, epochs_per_fold=2, seed=0)
    df = _generate_synthetic_market_data(n=500, seed=7)

    X, y, end_idx = build_symbol_sequences(df, cfg)
    assert X.shape[0] == y.shape[0] == end_idx.shape[0], "X/y/end_idx muessen gleich viele Zeilen haben"
    assert X.shape[1] == cfg.timesteps, f"Erwartete {cfg.timesteps} Zeitschritte pro Sequenz, bekam {X.shape[1]}"
    assert X.shape[2] == len(FEATURE_NAMES), f"Erwartete {len(FEATURE_NAMES)} Features, bekam {X.shape[2]}"
    assert X.shape[0] > cfg.train_size, "Zu wenig Sequenzen fuer den Testaufbau -- n in _generate_synthetic_market_data erhoehen"

    X_train, y_train = X[:cfg.train_size], y[:cfg.train_size]
    model = train_fold_model(X_train, y_train, cfg)

    X_test = X[cfg.train_size:cfg.train_size + cfg.test_size]
    forecast = model.predict(X_test)
    assert forecast.expected_return.shape[0] == X_test.shape[0]
    assert forecast.expected_volatility.shape[0] == X_test.shape[0]
    assert (forecast.expected_volatility > 0).all(), "Sigma muss ueberall positiv sein"

    print(f"build_symbol_sequences: {X.shape[0]} Sequenzen, Form {X.shape} -- OK")
    print(f"train_fold_model: trainiertes Modell liefert Vorhersagen fuer {X_test.shape[0]} Test-Sequenzen -- OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.12 tests/test_cross_sectional_fold_training.py` (from `market_control_system/`)
Expected: `ModuleNotFoundError: No module named 'cross_sectional_fold_training'`.

- [ ] **Step 3: Implement the module**

`market_control_system/training/cross_sectional_fold_training.py`:

```python
"""
cross_sectional_fold_training.py
===================================

Wiederverwendbare Trainings-Bausteine fuer die Cross-Sectional-Walk-
Forward-Diagnostik (siehe Design-Spec docs/superpowers/specs/2026-08-31-
cross-sectional-signal-diagnostics-design.md). Nutzt dieselbe Fold-
Erzeugung wie training/walk_forward.py (generate_fold_slices, dort
unveraendert) -- neu ist hier nur, dass Training/Vorhersage als
wiederverwendbare Funktionen statt inline in einem Skript vorliegen,
weil run_cross_sectional_signal_diagnostics.py sie fuer 12 Symbole
gleichzeitig pro Fold aufrufen muss.

KEIN Online-Training hier -- jedes Fold trainiert ein frisches Modell
und wertet es eingefroren aus, exakt wie training/walk_forward.py es
fuer ein Symbol tut, hier verallgemeinert auf viele Symbole.
"""
from __future__ import annotations

import numpy as np
import torch

from feature_pipeline import FEATURE_NAMES, build_scaled_features_and_target
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import LSTMForecaster, train_epoch
from walk_forward import WalkForwardConfig


def build_symbol_sequences(df, cfg: WalkForwardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baut skalierte Features + Zielvariable (horizon=cfg.horizon, siehe
    Global Constraints: hier IMMER 1, nicht der sonst im Projekt uebliche
    Default 5) und daraus Sequenz-Fenster fuer EIN Symbol."""
    features, target = build_scaled_features_and_target(df, horizon=cfg.horizon)
    builder = SequenceWindowBuilder(timesteps=cfg.timesteps, feature_names=list(FEATURE_NAMES))
    X, y, end_idx = builder.build(features, target)
    return X, y, end_idx


def train_fold_model(X_train: np.ndarray, y_train: np.ndarray, cfg: WalkForwardConfig) -> LSTMForecaster:
    """Trainiert ein FRISCHES Modell auf dem Trainingsfenster eines Folds.
    Gibt das Modell im eval()-Modus zurueck (kein weiteres Training danach
    -- Vorhersagen im Testfenster sind eingefroren)."""
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs_per_fold):
        train_epoch(model, optimizer, X_train, y_train, batch_size=cfg.batch_size)
    model.eval()
    return model
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -3.12 tests/test_cross_sectional_fold_training.py` (from `market_control_system/`)
Expected: both `-- OK` lines print without error.

- [ ] **Step 5: Commit**

```bash
git add market_control_system/training/cross_sectional_fold_training.py market_control_system/tests/test_cross_sectional_fold_training.py
git commit -m "feat: add cross-sectional fold training helpers"
```

---

## Task 4: Orchestration — walk-forward signal diagnostics

Depends on Task 1 (`frozen_snapshot`), Task 2 (`cross_sectional_signal_metrics`), Task 3 (`cross_sectional_fold_training`).

**Files:**
- Create: `market_control_system/orchestration/run_cross_sectional_signal_diagnostics.py`

**Interfaces:**
- Consumes: `load_or_build_snapshot` (Task 1), all of Task 2's metrics functions, `build_symbol_sequences`/`train_fold_model` (Task 3), `WalkForwardConfig`/`generate_fold_slices` (existing `training/walk_forward.py`), `calibrate_k` (existing `controller/exposure_controller.py`), `UNIVERSE`, `BACKTEST_LOOKBACK_DAYS` (existing `orchestration/cross_sectional_universe.py`).
- Produces: `run_diagnostics(walkforward_index_only: bool = True) -> dict` — a function, so Task 5's holdout script can import and call it with a flag rather than duplicating the pipeline.

- [ ] **Step 1: Implement the orchestration script**

`market_control_system/orchestration/run_cross_sectional_signal_diagnostics.py`:

```python
"""
run_cross_sectional_signal_diagnostics.py
============================================

Cross-Sectional-Walk-Forward-Signaldiagnostik OHNE Controller/Ausfuehrung/
Kosten (siehe Design-Spec docs/superpowers/specs/2026-08-31-cross-
sectional-signal-diagnostics-design.md). Misst Rank-IC und Brutto-Top-
minus-Bottom-Spread ueber mehrere Score-Varianten und vergleicht gegen
Random/Momentum/Reversal-Baselines -- bevor irgendeine weitere Arbeit an
Positionsgroessen, Deadbands oder Hysterese investiert wird.

Nutzt NIRGENDS ControlLoop/PaperExecutionEngine/CrossSectionalPortfolio.
Nutzt horizon=1 (naechster Bar), nicht den sonst ueblichen Default 5 --
das eliminiert jede Entscheidungs-/Fuellzeitpunkt-Ambiguitaet durch
Konstruktion (siehe Global Constraints im Plan).

Das Walk-Forward-Fenster (WALKFORWARD_FRACTION des eingefrorenen
Snapshots) wird hier automatisch in Folds durchlaufen. Der Rest
(Holdout) wird NIE von diesem Skript gelesen -- nur von
run_cross_sectional_holdout_eval.py (Task 5), manuell ausgeloest.

Ausfuehren: py -3.12 orchestration/run_cross_sectional_signal_diagnostics.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import numpy as np
import torch

from frozen_snapshot import load_or_build_snapshot
from cross_sectional_signal_metrics import (
    compute_rank_ic, compute_gross_spread, compute_breakeven_cost,
    compound_return, max_drawdown_from_returns, rolling_percentile_score,
    random_ranking_scores, momentum_scores, reversal_scores,
)
from cross_sectional_fold_training import build_symbol_sequences, train_fold_model
from walk_forward import WalkForwardConfig, generate_fold_slices
from exposure_controller import calibrate_k
from cross_sectional_universe import UNIVERSE, BACKTEST_LOOKBACK_DAYS

WALKFORWARD_FRACTION = 0.8
N_LONG = 3
N_SHORT = 3
ROLLING_PERCENTILE_WINDOW = 500
MOMENTUM_LOOKBACK_BARS = 20

SCORE_VARIANT_NAMES = ["mu", "mu_over_sigma", "kelly_edge", "p_up", "mu_percentile"]
BASELINE_NAMES = ["random", "momentum", "reversal"]


def split_walkforward_and_holdout(aligned_index):
    split_idx = int(len(aligned_index) * WALKFORWARD_FRACTION)
    return aligned_index[:split_idx], aligned_index[split_idx:]


def build_aligned_index(dfs: dict) -> "pd.DatetimeIndex":
    aligned_index = dfs[UNIVERSE[0]].index
    for symbol in UNIVERSE[1:]:
        aligned_index = aligned_index.intersection(dfs[symbol].index)
    return aligned_index.sort_values()


def run_diagnostics(use_holdout: bool = False) -> dict:
    print(f"=== Lade eingefrorenen Snapshot ({len(UNIVERSE)} Symbole, {BACKTEST_LOOKBACK_DAYS} Tage) ===")
    dfs = load_or_build_snapshot(UNIVERSE, BACKTEST_LOOKBACK_DAYS)
    aligned_index = build_aligned_index(dfs)
    walkforward_index, holdout_index = split_walkforward_and_holdout(aligned_index)
    print(f"  Gemeinsamer Zeitindex: {len(aligned_index)} Bars")
    print(f"  Walk-Forward-Fenster: {len(walkforward_index)} Bars, Holdout: {len(holdout_index)} Bars (nie automatisch gelesen)")

    eval_index = holdout_index if use_holdout else walkforward_index
    if use_holdout:
        print("  *** HOLDOUT-MODUS: dieser Lauf liest das reservierte Fenster. Nur manuell, nach positivem Walk-Forward-Ergebnis. ***")

    cfg = WalkForwardConfig(horizon=1, seed=0)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    sequences = {}
    for symbol in UNIVERSE:
        df_eval = dfs[symbol].loc[eval_index]
        X, y, end_idx = build_symbol_sequences(df_eval, cfg)
        prices = df_eval["price"].to_numpy()
        sequences[symbol] = (X, y, end_idx, prices)

    n_samples = min(len(sequences[s][0]) for s in UNIVERSE)
    folds = generate_fold_slices(n_samples, cfg)
    if not folds:
        raise ValueError(f"Nicht genug Sequenzen ({n_samples}) fuer train_size={cfg.train_size}+test_size={cfg.test_size}")
    print(f"  {n_samples} gemeinsame Sequenzen, {len(folds)} Folds")

    results = {name: {"rank_ic": [], "gross_spread": [], "turnover": []} for name in SCORE_VARIANT_NAMES + BASELINE_NAMES}

    for fold_i, (train_slice, test_slice) in enumerate(folds):
        print(f"\n=== Fold {fold_i+1}/{len(folds)} ===")
        models, k_calibrated, mu_histories = {}, {}, {}
        for symbol in UNIVERSE:
            X, y, _, _ = sequences[symbol]
            X_train, y_train = X[train_slice], y[train_slice]
            model = train_fold_model(X_train, y_train, cfg)
            models[symbol] = model
            forecast_train = model.predict(X_train)
            k_calibrated[symbol] = calibrate_k(
                forecast_train.expected_return, forecast_train.expected_volatility,
                max_position=1.0, target_utilization=0.5, percentile=95.0,
            )
            mu_histories[symbol] = list(forecast_train.expected_return[-ROLLING_PERCENTILE_WINDOW:])

        forecasts, test_len = {}, None
        for symbol in UNIVERSE:
            X, y, end_idx, prices = sequences[symbol]
            X_test, y_test = X[test_slice], y[test_slice]
            forecast = models[symbol].predict(X_test)
            forecasts[symbol] = (forecast, y_test, end_idx[test_slice], prices)
            test_len = len(y_test) if test_len is None else min(test_len, len(y_test))

        previous_weights = {name: {s: 0.0 for s in UNIVERSE} for name in SCORE_VARIANT_NAMES + BASELINE_NAMES}

        for t in range(test_len):
            mus = {s: float(forecasts[s][0].expected_return[t]) for s in UNIVERSE}
            sigmas = {s: float(forecasts[s][0].expected_volatility[t]) for s in UNIVERSE}
            p_ups = {s: float(forecasts[s][0].probability_up[t]) for s in UNIVERSE}
            forward_returns = {s: float(forecasts[s][1][t]) for s in UNIVERSE}

            for symbol in UNIVERSE:
                mu_histories[symbol].append(mus[symbol])

            price_history = {}
            for symbol in UNIVERSE:
                _, _, end_idx_test, prices = forecasts[symbol]
                row = end_idx_test[t]
                price_history[symbol] = list(prices[: row + 1])

            score_variants = {
                "mu": mus,
                "mu_over_sigma": {s: mus[s] / (sigmas[s] + 1e-6) for s in UNIVERSE},
                "kelly_edge": {s: k_calibrated[s] * mus[s] / (sigmas[s] ** 2 + 1e-6) for s in UNIVERSE},
                "p_up": p_ups,
                "mu_percentile": {s: rolling_percentile_score(mu_histories[s], ROLLING_PERCENTILE_WINDOW) for s in UNIVERSE},
                # Ein Zufalls-Seed PRO (Fold, Zeitschritt) statt mehrerer
                # gemittelter Wiederholungen PRO Zeitschritt -- ueber die
                # tausenden Zeitschritte aller Folds hinweg mittelt sich das
                # Ergebnis ohnehin zu einer stabilen Zufalls-Baseline, ohne
                # den N-fachen Rechenaufwand pro Schritt (YAGNI).
                "random": random_ranking_scores(UNIVERSE, seed=fold_i * 10_000 + t),
                "momentum": momentum_scores(price_history, MOMENTUM_LOOKBACK_BARS),
                "reversal": reversal_scores(price_history, MOMENTUM_LOOKBACK_BARS),
            }

            for name, scores in score_variants.items():
                if len(scores) < N_LONG + N_SHORT:
                    continue
                ic = compute_rank_ic(scores, forward_returns)
                spread = compute_gross_spread(scores, forward_returns, N_LONG, N_SHORT)
                ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
                new_weights = {s: 0.0 for s in UNIVERSE}
                for s in ranked[:N_LONG]:
                    new_weights[s] = 1.0 / N_LONG
                for s in ranked[-N_SHORT:]:
                    new_weights[s] = -1.0 / N_SHORT
                turnover = sum(abs(new_weights[s] - previous_weights[name][s]) for s in UNIVERSE)
                previous_weights[name] = new_weights

                if not np.isnan(ic):
                    results[name]["rank_ic"].append(ic)
                if not np.isnan(spread):
                    results[name]["gross_spread"].append(spread)
                results[name]["turnover"].append(turnover)

        print(f"  Fold {fold_i+1} abgeschlossen ({test_len} Test-Bars)")

    summary = {}
    for name in SCORE_VARIANT_NAMES + BASELINE_NAMES:
        rank_ics = results[name]["rank_ic"]
        spreads = results[name]["gross_spread"]
        turnovers = results[name]["turnover"]
        summary[name] = {
            "mean_rank_ic": float(np.mean(rank_ics)) if rank_ics else float("nan"),
            "n_observations": len(rank_ics),
            "mean_gross_spread_per_bar": float(np.mean(spreads)) if spreads else float("nan"),
            "compounded_gross_return": compound_return(spreads) if spreads else float("nan"),
            "max_drawdown": max_drawdown_from_returns(spreads) if spreads else float("nan"),
            "breakeven_cost": compute_breakeven_cost(spreads, turnovers) if spreads and turnovers else float("nan"),
        }

    return {"use_holdout": use_holdout, "n_folds": len(folds), "summary": summary}


def main():
    result = run_diagnostics(use_holdout=False)

    print(f"\n{'='*70}\n=== Zusammenfassung ueber {result['n_folds']} Walk-Forward-Folds ===\n{'='*70}")
    for name, stats in result["summary"].items():
        print(f"{name:>15}: mean_rank_ic={stats['mean_rank_ic']:+.4f}  "
              f"n={stats['n_observations']}  "
              f"compounded_gross_return={stats['compounded_gross_return']:+.4%}  "
              f"max_drawdown={stats['max_drawdown']:+.4%}  "
              f"breakeven_cost={stats['breakeven_cost']:+.6f}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"signal_diagnostics_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "diagnostics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSummary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test at small scale before trusting a real run**

Run this from `market_control_system/` — overrides the universe/lookback/fold-size to something tiny so it runs in well under a minute:

```bash
py -3.12 -c "
import sys
sys.path.insert(0, 'orchestration')
sys.path.insert(0, 'data_layer')
sys.path.insert(0, 'controller')
sys.path.insert(0, 'training')
sys.path.insert(0, 'feature_engineering')
sys.path.insert(0, 'models')
import cross_sectional_universe as u
u.UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'XOM']
u.BACKTEST_LOOKBACK_DAYS = 5
import run_cross_sectional_signal_diagnostics as m
m.UNIVERSE = u.UNIVERSE
m.BACKTEST_LOOKBACK_DAYS = u.BACKTEST_LOOKBACK_DAYS
import walk_forward
m.WalkForwardConfig = walk_forward.WalkForwardConfig
result = m.run_diagnostics(use_holdout=False)
print('SMOKE TEST OK:', result['n_folds'], 'folds')
for name, stats in result['summary'].items():
    print(' ', name, stats['n_observations'], 'obs, rank_ic', stats['mean_rank_ic'])
"
```

Note: the small smoke-scale run uses `WalkForwardConfig`'s defaults for `train_size`/`test_size` (2000/400), which need enough sequences to produce at least one fold — 5 days of 1-min bars for 6 symbols gives roughly 1900 bars per symbol, likely producing 0 folds. If `result['n_folds']` is 0 or the run raises the `ValueError` from `run_diagnostics`, increase `BACKTEST_LOOKBACK_DAYS` in the smoke test (try 20-30) until at least one fold is produced — the goal of this step is confirming the wiring works end-to-end, not a statistically meaningful result.

Expected: runs without exceptions, prints `SMOKE TEST OK: <N> folds` and per-variant observation counts. Delete any `logs/signal_diagnostics_*` directories the smoke test created afterward (throwaway).

- [ ] **Step 3: Launch the real walk-forward run**

```bash
py -3.12 orchestration/run_cross_sectional_signal_diagnostics.py
```

First run builds the frozen snapshot (one fetch of 12 symbols × 365 days, cached under `market_control_system/data_snapshots/` for all future runs — including Task 5's holdout run). Then walks forward across however many folds the full year produces, training 12 fresh models per fold. Budget accordingly — this trains many more models than a single backtest run (12 symbols × N folds), so expect longer than any single prior run this session; confirm empirically rather than estimating.

- [ ] **Step 4: Commit**

```bash
git add market_control_system/orchestration/run_cross_sectional_signal_diagnostics.py
git commit -m "feat: add cross-sectional walk-forward signal diagnostics orchestration"
```

---

## Task 5: Holdout evaluation (separate, manual-trigger only)

Depends on Task 4.

**Files:**
- Create: `market_control_system/orchestration/run_cross_sectional_holdout_eval.py`

**Interfaces:**
- Consumes: `run_diagnostics(use_holdout: bool)` (Task 4).

- [ ] **Step 1: Implement the holdout script**

`market_control_system/orchestration/run_cross_sectional_holdout_eval.py`:

```python
"""
run_cross_sectional_holdout_eval.py
======================================

Wertet das RESERVIERTE Holdout-Fenster aus (siehe Design-Spec
docs/superpowers/specs/2026-08-31-cross-sectional-signal-diagnostics-
design.md) -- der letzte WALKFORWARD_FRACTION-Komplementanteil des
eingefrorenen Snapshots, der von run_cross_sectional_signal_diagnostics.py
NIE gelesen wird.

ABSICHTLICH ein separates Skript, kein Kommandozeilen-Flag auf dem
Walk-Forward-Skript: das macht es unmoeglich, das Holdout versehentlich
durch wiederholtes Ausfuehren desselben Befehls "anzupeeken". Nur einmalig
ausfuehren, NACHDEM die Walk-Forward-Folds einen robusten (im Mittel
positiven, in der Mehrheit der Folds positiven) Rank-IC gezeigt haben --
sonst ist das Ergebnis wertlos (siehe Design-Spec, Nicht-Ziel-Abschnitt
und Review-Punkt 5).

Ausfuehren: py -3.12 orchestration/run_cross_sectional_holdout_eval.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from run_cross_sectional_signal_diagnostics import run_diagnostics


def main():
    print("=== HOLDOUT-AUSWERTUNG ===")
    print("Dieses Skript liest das reservierte Holdout-Fenster. Nur ausfuehren, ")
    print("wenn die Walk-Forward-Folds (run_cross_sectional_signal_diagnostics.py) ")
    print("bereits einen robusten, positiven Rank-IC gezeigt haben.\n")

    result = run_diagnostics(use_holdout=True)

    print(f"\n{'='*70}\n=== Holdout-Ergebnis ===\n{'='*70}")
    for name, stats in result["summary"].items():
        print(f"{name:>15}: mean_rank_ic={stats['mean_rank_ic']:+.4f}  "
              f"n={stats['n_observations']}  "
              f"compounded_gross_return={stats['compounded_gross_return']:+.4%}  "
              f"max_drawdown={stats['max_drawdown']:+.4%}  "
              f"breakeven_cost={stats['breakeven_cost']:+.6f}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"holdout_eval_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "holdout_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSummary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import wiring (do NOT run the real holdout yet)**

```bash
py -3.12 -c "
import sys
sys.path.insert(0, 'orchestration')
import run_cross_sectional_holdout_eval as m
print('Import OK:', m.run_diagnostics)
"
```

Expected: prints `Import OK: <function run_diagnostics ...>` with no errors. Do NOT run the real holdout script as part of this task — per the Global Constraints, the holdout is only evaluated once, manually, after Task 4's real walk-forward run shows a robust positive Rank-IC. Report that result to the user and let them decide when to trigger Task 5's real run.

- [ ] **Step 3: Commit**

```bash
git add market_control_system/orchestration/run_cross_sectional_holdout_eval.py
git commit -m "feat: add manual-trigger holdout evaluation script"
```
