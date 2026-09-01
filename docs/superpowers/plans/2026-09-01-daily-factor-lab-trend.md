# Daily-Factor-Lab Trend-Leg Implementation Plan

> **SUPERSEDED (2026-09-01):** Dieser Plan v1 wurde durch das externe Review
> vom 2026-09-01 invalidiert (Same-Close-Ausführung, variantenabhängige
> Zeitfenster, fehlender Matched-Benchmark, kontaminiertes Holdout,
> Gross-Drift, unabhängige Monats-Blöcke — siehe Spec v2, Änderungshistorie).
> NICHT ausführen. Die Neufassung entsteht nach Freigabe der Spec v2 aus
> `docs/superpowers/specs/2026-09-01-daily-factor-lab-trend-design.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ein präregistriertes Baseline-Lab, das misst, ob Time-Series-Trend auf 12 Cross-Asset-ETFs nach Kosten einen belastbaren Baustein Richtung 12.7 % Netto-CAGR bei max. 15 % Drawdown liefert.

**Architecture:** Neues Top-Level-Package `factor_lab/` neben `market_control_system/`. Reine, einzeln getestete Funktionsmodule (signals/costs/portfolio/stats/data_snapshot) plus zwei Orchestrierungs-Skripte (Entwicklungsfenster automatisch, Holdout nur manuell). Inferenz- und Snapshot-Bausteine werden aus `market_control_system` importiert, nicht dupliziert.

**Tech Stack:** Python 3.12 (Store-Python via `py -3.12`), pandas, numpy, yfinance (nur Snapshot-Build). Kein Torch, kein ML.

**Spec:** `docs/superpowers/specs/2026-09-01-daily-factor-lab-trend-design.md`

## Global Constraints

- ALLES mit `py -3.12` ausführen (nie bare `python`) — Launcher-Quirk dieser Maschine.
- Test-Konvention wie im Bestand: reine Skripte mit `check_*`-Funktionen, `run_consistency_check()`, `if __name__ == "__main__":` — Ausführung `py -3.12 factor_lab/tests/test_x.py`. Kein pytest.
- Docstrings/Kommentare auf Deutsch (ASCII-Umschreibung ue/oe/ae wie im Bestand).
- Alle Equity-Kennzahlen MULTIPLIKATIV (prod(1+r)−1, echte Equity-Kurve).
- Exakt 8 präregistrierte Varianten: Signale {mom63, mom126, mom252, combo} × Modi {long_short, long_flat}. Keine weiteren, kein Mechanismus dafür.
- Parameter verbatim aus der Spec: Lookbacks {63, 126, 252}; EWMA-Span 63; Ziel-Vol 0.10 p.a.; Gross-Cap 1.0 (nie hebeln); Rebalance monatlich am letzten gemeinsamen Handelstag; Kosten 1.5 bp (SPY, QQQ, IWM, TLT, IEF, GLD) / 3.0 bp (EFA, EEM, LQD, SLV, DBC, VNQ); Borrow 50 bp p.a. auf Short-Nominal; Stresstest = alle Kosten ×2; DD-Cap 0.15; Entwicklungsfenster = erste 80 % der gemeinsamen Handelstage; Universum-Start 2007-01-01; Bootstrap n=2000, Seed 0, MONATS-Blöcke.
- Import aus `market_control_system` per `sys.path.insert`-Muster (Bestands-Konvention), niemals Code kopieren.
- `factor_lab/logs/` und `factor_lab/data_snapshots/` sind gitignored; das Snapshot-Manifest wird per `git add -f` committet (Muster aus PR #2).
- Commits nach jedem Task; Commit-Messages mit `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Package-Skelett + signals.py

**Files:**
- Create: `factor_lab/__init__.py` (leer), `factor_lab/tests/__init__.py` (leer)
- Create: `factor_lab/signals.py`
- Test: `factor_lab/tests/test_signals.py`

**Interfaces:**
- Produces: `momentum_sign(prices: pd.Series, lookback: int) -> pd.Series` — Werte in {−1.0, 0.0, +1.0}, NaN für die ersten `lookback` Einträge. `combo_signal(signals: list[pd.Series]) -> pd.Series` — elementweises Mittel (NaN, wo irgendein Input NaN ist).

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_signals.py — prueft die reinen Signalfunktionen mit handgerechneten Werten.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.signals import momentum_sign, combo_signal


def check_momentum_sign() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    prices = pd.Series([100.0, 101.0, 99.0, 99.0, 102.0, 98.0], index=idx)
    sig = momentum_sign(prices, lookback=2)
    # Erste 2 Eintraege: keine Historie -> NaN
    assert sig.iloc[:2].isna().all(), f"Erste lookback Eintraege muessen NaN sein, bekam {sig.iloc[:2].tolist()}"
    # t=2: 99/100-1 < 0 -> -1 ; t=3: 99/101-1 < 0 -> -1 ; t=4: 102/99-1 > 0 -> +1
    assert sig.iloc[2] == -1.0 and sig.iloc[3] == -1.0 and sig.iloc[4] == 1.0
    # Exakt-Null-Return -> Signal 0 (kein kuenstliches Vorzeichen)
    flat = pd.Series([100.0, 100.0, 100.0], index=pd.date_range("2020-01-01", periods=3, freq="B"))
    assert momentum_sign(flat, lookback=1).iloc[2] == 0.0
    print("momentum_sign: OK")


def check_combo_signal() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    s1 = pd.Series([1.0, 1.0, -1.0], index=idx)
    s2 = pd.Series([1.0, -1.0, -1.0], index=idx)
    s3 = pd.Series([np.nan, 1.0, -1.0], index=idx)
    combo = combo_signal([s1, s2, s3])
    assert np.isnan(combo.iloc[0]), "NaN in irgendeinem Input muss NaN im Combo ergeben"
    assert abs(combo.iloc[1] - (1.0 / 3.0)) < 1e-12
    assert combo.iloc[2] == -1.0
    print("combo_signal: OK")


def run_consistency_check() -> None:
    check_momentum_sign()
    check_combo_signal()
    print("\nAlle signals-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen, Fehlschlag verifizieren**

Run: `py -3.12 factor_lab/tests/test_signals.py`
Expected: FAIL mit `ModuleNotFoundError: No module named 'factor_lab.signals'` (o.ä.)

- [ ] **Step 3: Minimale Implementierung**

```python
"""
signals.py — praeregistrierte Trend-Signale (Spec Abschnitt 5).

Reine Funktionen, keine Zustaende: sign(Return ueber lookback Handelstage)
je Instrument, plus Gleichgewichts-Kombination der drei Lookbacks.
Es gibt BEWUSST keinen Mechanismus fuer weitere Varianten — die 8
praeregistrierten Varianten sind in der Design-Spec fixiert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_sign(prices: pd.Series, lookback: int) -> pd.Series:
    """Vorzeichen des Returns ueber `lookback` Handelstage. NaN fuer die
    ersten `lookback` Eintraege (keine Historie); exakter Null-Return -> 0."""
    returns = prices / prices.shift(lookback) - 1.0
    return np.sign(returns)


def combo_signal(signals: list[pd.Series]) -> pd.Series:
    """Gleichgewichts-Mittel mehrerer Signal-Serien; NaN, wo irgendein
    Input NaN ist (Signal erst gueltig, wenn ALLE Lookbacks Historie haben)."""
    frame = pd.concat(signals, axis=1)
    return frame.mean(axis=1).where(frame.notna().all(axis=1))
```

- [ ] **Step 4: Test ausführen, Bestehen verifizieren**

Run: `py -3.12 factor_lab/tests/test_signals.py`
Expected: PASS ("Alle signals-Checks bestanden.")

- [ ] **Step 5: Commit**

```bash
git add factor_lab/__init__.py factor_lab/tests/__init__.py factor_lab/signals.py factor_lab/tests/test_signals.py
git commit -m "feat(factor_lab): add pre-registered trend signals (mom63/126/252, combo)"
```

---

### Task 2: costs.py

**Files:**
- Create: `factor_lab/costs.py`
- Test: `factor_lab/tests/test_costs.py`

**Interfaces:**
- Produces: `COST_BP: dict[str, float]` (12 Symbole), `BORROW_BP_PA: float = 50.0`, `TRADING_DAYS_PA: int = 252`, `trade_cost_fraction(weight_deltas: pd.Series, cost_bp: dict[str, float], cost_multiplier: float = 1.0) -> float`, `daily_borrow_cost_fraction(weights: pd.Series, cost_multiplier: float = 1.0) -> float`. Beide Rückgaben sind Anteile am Portfoliowert (dimensionslos, positiv = Kosten).

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_costs.py — prueft das bp-Kostenmodell mit handgerechneten Werten.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from factor_lab.costs import COST_BP, trade_cost_fraction, daily_borrow_cost_fraction


def check_trade_cost() -> None:
    # SPY (1.5bp) 0.5 Gewichtseinheiten gehandelt + EEM (3.0bp) 0.2 gehandelt:
    # 0.5*1.5/10000 + 0.2*3.0/10000 = 0.000075 + 0.00006 = 0.000135
    deltas = pd.Series({"SPY": 0.5, "EEM": -0.2})
    cost = trade_cost_fraction(deltas, COST_BP)
    assert abs(cost - 0.000135) < 1e-12, f"Erwartete 0.000135, bekam {cost}"
    # Stress x2 verdoppelt exakt
    assert abs(trade_cost_fraction(deltas, COST_BP, cost_multiplier=2.0) - 0.00027) < 1e-12
    print("trade_cost_fraction: OK")


def check_borrow_cost() -> None:
    # Short 0.5 in TLT: 0.5 * 50bp / 252 = 0.5 * 0.005 / 252; Longs kosten nichts.
    weights = pd.Series({"TLT": -0.5, "SPY": 0.5})
    expected = 0.5 * 0.005 / 252
    got = daily_borrow_cost_fraction(weights)
    assert abs(got - expected) < 1e-15, f"Erwartete {expected}, bekam {got}"
    assert daily_borrow_cost_fraction(pd.Series({"SPY": 1.0})) == 0.0
    assert abs(daily_borrow_cost_fraction(weights, cost_multiplier=2.0) - 2 * expected) < 1e-15
    print("daily_borrow_cost_fraction: OK")


def check_cost_table() -> None:
    assert sorted(COST_BP) == sorted(["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"])
    assert COST_BP["SPY"] == 1.5 and COST_BP["EEM"] == 3.0
    print("COST_BP-Tabelle: OK")


def run_consistency_check() -> None:
    check_cost_table()
    check_trade_cost()
    check_borrow_cost()
    print("\nAlle costs-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen, Fehlschlag verifizieren**

Run: `py -3.12 factor_lab/tests/test_costs.py`
Expected: FAIL mit ModuleNotFoundError

- [ ] **Step 3: Minimale Implementierung**

```python
"""
costs.py — Kostenmodell des Trend-Legs (Spec Abschnitt 7).

Pro Trade: Half-Spread + Slippage-Puffer als Basispunkte vom gehandelten
Nominal, Kommission 0. Im long/short-Modus zusaetzlich pauschal 50 bp p.a.
Borrow auf das Short-Nominal, taeglich anteilig. Der Stresstest laeuft
ueber cost_multiplier=2.0 auf ALLEN Kosten (Spec: "alles x2").
ETF-Verwaltungsgebuehren (TER) stecken bereits in den adjustierten Preisen.
"""
from __future__ import annotations

import pandas as pd

COST_BP: dict[str, float] = {
    "SPY": 1.5, "QQQ": 1.5, "IWM": 1.5, "TLT": 1.5, "IEF": 1.5, "GLD": 1.5,
    "EFA": 3.0, "EEM": 3.0, "LQD": 3.0, "SLV": 3.0, "DBC": 3.0, "VNQ": 3.0,
}
BORROW_BP_PA: float = 50.0
TRADING_DAYS_PA: int = 252


def trade_cost_fraction(weight_deltas: pd.Series, cost_bp: dict[str, float], cost_multiplier: float = 1.0) -> float:
    """Kosten eines Rebalance als Anteil am Portfoliowert:
    Summe |Delta-Gewicht_i| * bp_i / 10000, skaliert mit cost_multiplier."""
    total = 0.0
    for symbol, delta in weight_deltas.items():
        total += abs(float(delta)) * cost_bp[symbol] / 10_000.0
    return total * cost_multiplier


def daily_borrow_cost_fraction(weights: pd.Series, cost_multiplier: float = 1.0) -> float:
    """Taegliche Borrow-Kosten als Anteil am Portfoliowert: Short-Nominal
    (Summe der negativen Gewichte, betragsmaessig) * 50bp p.a. / 252."""
    short_nominal = float(weights[weights < 0].abs().sum())
    return short_nominal * (BORROW_BP_PA / 10_000.0) / TRADING_DAYS_PA * cost_multiplier
```

- [ ] **Step 4: Test ausführen, Bestehen verifizieren**

Run: `py -3.12 factor_lab/tests/test_costs.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add factor_lab/costs.py factor_lab/tests/test_costs.py
git commit -m "feat(factor_lab): add bp cost model with borrow and x2 stress"
```

---

### Task 3: portfolio.py Teil 1 — Vol-Schätzung + Rebalance-Gewichte

**Files:**
- Create: `factor_lab/portfolio.py`
- Test: `factor_lab/tests/test_portfolio.py`

**Interfaces:**
- Consumes: nichts aus anderen Tasks (reine pandas/numpy-Funktionen).
- Produces: `ewma_annualized_vol(returns: pd.DataFrame, span: int = 63) -> pd.DataFrame` (kausal, min_periods=span, annualisiert mit sqrt(252)); `rebalance_weights(signal_row: pd.Series, vol_row: pd.Series, trailing_returns: pd.DataFrame, mode: str, target_vol: float = 0.10) -> pd.Series` — Gewichte mit Gross ≤ 1.0, `mode` ∈ {"long_short", "long_flat"}.

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_portfolio.py — prueft Vol-Schaetzung, Rebalance-Gewichte und (ab
Task 4) den taeglichen Backtest-Loop mit handgerechneten Faellen.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.portfolio import ewma_annualized_vol, rebalance_weights


def check_ewma_vol() -> None:
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    rets = pd.DataFrame({"A": [0.01, -0.01] * 40}, index=idx)
    vol = ewma_annualized_vol(rets, span=63)
    assert vol["A"].iloc[:62].isna().all(), "Vor min_periods muss NaN stehen"
    # Alternierende +-1%-Returns: Tages-Std nahe 0.01 -> annualisiert nahe 0.159
    assert abs(vol["A"].iloc[-1] - 0.01 * np.sqrt(252)) < 0.02
    print("ewma_annualized_vol: OK")


def check_rebalance_weights() -> None:
    signal = pd.Series({"A": 1.0, "B": -1.0})
    vol = pd.Series({"A": 0.2, "B": 0.1})
    # trailing-Returns alle 0 -> Portfolio-Vol 0 -> Skalierung bleibt 1 (nie hebeln)
    trailing = pd.DataFrame(0.0, index=range(63), columns=["A", "B"])

    # long_short: raw = [1/0.2, -1/0.1] = [5, -10], gross 15 -> base [1/3, -2/3]
    w = rebalance_weights(signal, vol, trailing, mode="long_short", target_vol=0.10)
    assert abs(w["A"] - 1.0 / 3.0) < 1e-12 and abs(w["B"] + 2.0 / 3.0) < 1e-12
    assert abs(w.abs().sum() - 1.0) < 1e-12, "Gross muss exakt 1.0 sein (vor Vol-Skalierung nach unten)"

    # long_flat: negatives Signal -> 0 -> alles in A
    w = rebalance_weights(signal, vol, trailing, mode="long_flat", target_vol=0.10)
    assert abs(w["A"] - 1.0) < 1e-12 and w["B"] == 0.0

    # Vol-Targeting skaliert HERUNTER: trailing = A-Returns mit hoher Vol
    rng = np.random.default_rng(0)
    hot = pd.DataFrame({"A": rng.normal(0, 0.02, 63), "B": 0.0})
    w = rebalance_weights(pd.Series({"A": 1.0, "B": 0.0}), vol, hot, mode="long_flat", target_vol=0.10)
    assert w["A"] < 1.0, "Bei Portfolio-Vol > Ziel muss heruntergeskaliert werden"
    assert w["A"] > 0.0

    # NaN-Vol -> Gewicht 0 (Instrument ohne stabile Vol-Schaetzung wird ausgelassen)
    w = rebalance_weights(pd.Series({"A": 1.0, "B": 1.0}), pd.Series({"A": 0.2, "B": np.nan}), trailing, mode="long_flat")
    assert w["B"] == 0.0 and abs(w["A"] - 1.0) < 1e-12

    # Alles-Null-Signal -> alles Cash
    w = rebalance_weights(pd.Series({"A": 0.0, "B": 0.0}), vol, trailing, mode="long_flat")
    assert (w == 0.0).all()
    print("rebalance_weights: OK")


def run_consistency_check() -> None:
    check_ewma_vol()
    check_rebalance_weights()
    print("\nAlle portfolio-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen, Fehlschlag verifizieren**

Run: `py -3.12 factor_lab/tests/test_portfolio.py`
Expected: FAIL mit ModuleNotFoundError

- [ ] **Step 3: Minimale Implementierung**

```python
"""
portfolio.py — Portfolio-Konstruktion des Trend-Legs (Spec Abschnitt 6).

Inverse-Vol-Gewichtung auf Gross 1.0, Vol-Targeting auf 10% p.a. ueber die
realisierte Vol des Kandidaten-Portfolios der letzten 63 Tage (kausal:
heutige Gewichte auf vergangene Returns anzuwenden nutzt keine Zukunft).
Vol-Targeting kann Exposure NUR SENKEN — Gross-Cap 1.0, es wird nie gehebelt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PA = 252


def ewma_annualized_vol(returns: pd.DataFrame, span: int = 63) -> pd.DataFrame:
    """Kausale EWMA-Tagesvol (min_periods=span), annualisiert mit sqrt(252)."""
    return returns.ewm(span=span, min_periods=span).std() * np.sqrt(TRADING_DAYS_PA)


def rebalance_weights(
    signal_row: pd.Series,
    vol_row: pd.Series,
    trailing_returns: pd.DataFrame,
    mode: str,
    target_vol: float = 0.10,
) -> pd.Series:
    """Zielgewichte fuer EINEN Rebalance-Zeitpunkt.

    mode="long_flat": negative Signale werden auf 0 gesetzt (Cash statt
    Short — realistisch fuer ein kleines Konto, keine Borrow-Kosten).
    mode="long_short": klassisches TSMOM, Vorzeichen wird gehandelt."""
    if mode not in ("long_short", "long_flat"):
        raise ValueError(f"Unbekannter Modus: {mode}")
    signal = signal_row.copy().astype(float)
    if mode == "long_flat":
        signal = signal.clip(lower=0.0)

    valid_vol = vol_row.notna() & (vol_row > 0)
    raw = (signal / vol_row).where(valid_vol, 0.0).fillna(0.0)
    gross = float(raw.abs().sum())
    if gross == 0.0:
        return pd.Series(0.0, index=signal_row.index)
    base = raw / gross  # Gross exakt 1.0

    portfolio_returns = (trailing_returns[base.index] * base).sum(axis=1)
    realized_vol = float(portfolio_returns.std()) * np.sqrt(TRADING_DAYS_PA)
    # Nur herunterskalieren, nie hebeln (Gross-Cap 1.0, Spec Abschnitt 6).
    scale = min(1.0, target_vol / realized_vol) if realized_vol > 0 else 1.0
    return base * scale
```

- [ ] **Step 4: Test ausführen, Bestehen verifizieren**

Run: `py -3.12 factor_lab/tests/test_portfolio.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add factor_lab/portfolio.py factor_lab/tests/test_portfolio.py
git commit -m "feat(factor_lab): add inverse-vol weights with vol targeting and gross cap 1.0"
```

---

### Task 4: portfolio.py Teil 2 — täglicher Backtest-Loop + Benchmarks

**Files:**
- Modify: `factor_lab/portfolio.py` (Funktionen anhängen)
- Modify: `factor_lab/tests/test_portfolio.py` (Checks anhängen)

**Interfaces:**
- Consumes: `rebalance_weights`, `trade_cost_fraction`, `daily_borrow_cost_fraction` (Task 2/3, exakte Signaturen siehe dort).
- Produces:
  - `run_daily_backtest(returns: pd.DataFrame, signals: pd.DataFrame, vols: pd.DataFrame, mode: str, cost_bp: dict[str, float], cost_multiplier: float = 1.0, target_vol: float = 0.10, vol_window: int = 63) -> tuple[pd.Series, dict]` — Netto-Tagesreturns (Index = Teilmenge von returns.index ab erstem gültigen Rebalance) und Info-Dict mit Schlüsseln `"total_trade_cost"`, `"total_borrow_cost"`, `"total_turnover"`, `"n_rebalances"`, `"instrument_contributions"` (dict Symbol → additiver P&L-Beitrag).
  - `month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex` — letzter Handelstag je (Jahr, Monat).
  - `benchmark_buy_and_hold(returns: pd.Series, cost_bp_value: float, cost_multiplier: float = 1.0) -> pd.Series`.
  - `benchmark_fixed_mix(returns: pd.DataFrame, target_weights: dict[str, float], cost_bp: dict[str, float], cost_multiplier: float = 1.0) -> pd.Series` — monatliches Rebalancing auf feste Gewichte (für 60/40 SPY/TLT).

- [ ] **Step 1: Failing Checks anhängen**

```python
# In test_portfolio.py ergaenzen (Imports erweitern):
from factor_lab.portfolio import (
    run_daily_backtest, month_end_dates, benchmark_buy_and_hold, benchmark_fixed_mix,
)


def check_month_end_dates() -> None:
    idx = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2020-03-02"])
    ends = month_end_dates(idx)
    assert list(ends) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-02")]
    print("month_end_dates: OK")


def check_run_daily_backtest() -> None:
    # Ein Instrument mit deterministischem Muster, eines flach mit Null-Signal:
    # Nach Warmup ist w_A = 1.0 (target_vol=1.0 -> keine Herunterskalierung),
    # Nettoreturn = r_A, am ersten Rebalance-Tag zusaetzlich 1.5bp Kosten
    # fuer den Aufbau |0 -> 1|. Ein-Instrument-Drift haelt w_A exakt bei 1.
    idx = pd.date_range("2020-01-01", periods=130, freq="B")
    pattern = [0.01, -0.005] * 65
    returns = pd.DataFrame({"SPY": pattern, "IEF": [0.0] * 130}, index=idx)
    signals = pd.DataFrame({"SPY": 1.0, "IEF": 0.0}, index=idx)
    vols = pd.DataFrame({"SPY": 0.1, "IEF": 0.1}, index=idx)

    net, info = run_daily_backtest(
        returns, signals, vols, mode="long_flat",
        cost_bp={"SPY": 1.5, "IEF": 1.5}, target_vol=1.0, vol_window=63,
    )
    rebalances = month_end_dates(net.index)
    first_reb = rebalances[0]
    day_after = net.index[net.index.get_loc(first_reb) + 1]
    # Am ersten Rebalance-Tag selbst ist noch keine Position offen (Aufbau
    # zum Schlusskurs) -> Return = 0 - Kosten
    assert abs(net.loc[first_reb] - (0.0 - 1.0 * 1.5 / 10_000.0)) < 1e-12, (
        f"Erster Rebalance-Tag: erwartete reine Kosten, bekam {net.loc[first_reb]}"
    )
    # Tag danach: volle Position, Return = r_A ohne Kosten
    assert abs(net.loc[day_after] - returns.loc[day_after, "SPY"]) < 1e-12
    assert info["n_rebalances"] >= 2
    assert info["total_turnover"] >= 1.0  # mindestens der Aufbau
    assert abs(info["instrument_contributions"]["IEF"]) < 1e-15
    print("run_daily_backtest (long_flat, deterministisch): OK")

    # Borrow-Kosten im long_short-Modus: Signal -1 -> w = -1, taeglich Borrow.
    signals_short = pd.DataFrame({"SPY": -1.0, "IEF": 0.0}, index=idx)
    net_s, info_s = run_daily_backtest(
        returns, signals_short, vols, mode="long_short",
        cost_bp={"SPY": 1.5, "IEF": 1.5}, target_vol=1.0, vol_window=63,
    )
    expected_daily_borrow = 1.0 * 0.005 / 252
    assert abs(net_s.loc[day_after] - (-returns.loc[day_after, "SPY"] - expected_daily_borrow)) < 1e-9, (
        "Short-Tag muss -r_A minus taegliche Borrow-Kosten liefern"
    )
    assert info_s["total_borrow_cost"] > 0
    print("run_daily_backtest (long_short, Borrow): OK")


def check_benchmarks() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    spy = pd.Series([0.01, 0.0, -0.01, 0.02, 0.0], index=idx)
    bh = benchmark_buy_and_hold(spy, cost_bp_value=1.5)
    # Tag 1: Kaufkosten 1.5bp, danach Roh-Returns
    assert abs(bh.iloc[0] - (0.01 - 1.5 / 10_000.0)) < 1e-12
    assert abs(bh.iloc[1] - 0.0) < 1e-12

    tlt = pd.Series([0.0] * 5, index=idx)
    mix = benchmark_fixed_mix(
        pd.DataFrame({"SPY": spy, "TLT": tlt}),
        target_weights={"SPY": 0.6, "TLT": 0.4},
        cost_bp={"SPY": 1.5, "TLT": 1.5},
    )
    # Tag 1: Aufbau (Turnover 1.0 -> 1.5bp Kosten), noch keine Position offen
    assert abs(mix.iloc[0] - (0.0 - 1.0 * 1.5 / 10_000.0)) < 1e-12
    # Tag 2: 0.6 * r_SPY
    assert abs(mix.iloc[1] - 0.6 * spy.iloc[1]) < 1e-12
    print("benchmark_buy_and_hold / benchmark_fixed_mix: OK")
```

Und in `run_consistency_check()` ergänzen: `check_month_end_dates()`, `check_run_daily_backtest()`, `check_benchmarks()`.

- [ ] **Step 2: Test ausführen, Fehlschlag verifizieren**

Run: `py -3.12 factor_lab/tests/test_portfolio.py`
Expected: FAIL mit ImportError (run_daily_backtest fehlt)

- [ ] **Step 3: Implementierung anhängen**

```python
def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Letzter vorhandener Handelstag je (Jahr, Monat) des Index."""
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.groupby([index.year, index.month]).max().sort_values().values)


def run_daily_backtest(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    vols: pd.DataFrame,
    mode: str,
    cost_bp: dict[str, float],
    cost_multiplier: float = 1.0,
    target_vol: float = 0.10,
    vol_window: int = 63,
) -> tuple[pd.Series, dict]:
    """Taeglicher Loop: Positionen werden am Rebalance-SCHLUSS aufgebaut
    (wirken also ab dem Folgetag), zwischen Rebalances driften die Gewichte
    mit den Preisen, Kosten fallen am Rebalance-Tag an, Borrow taeglich.

    Konvention Tag t: erst wirkt die Position von gestern auf r_t (plus
    Borrow auf die gestrige Position), DANN wird ggf. rebalanced. Damit ist
    jede Information, die in die Gewichte eingeht (Signal/Vol bis t), erst
    ab t+1 wirksam — kein Look-Ahead."""
    from factor_lab.costs import trade_cost_fraction, daily_borrow_cost_fraction

    symbols = list(returns.columns)
    # Erst ab dem Tag starten, an dem ALLE Signale und Vols vorliegen und
    # genug trailing-Returns fuer das Vol-Targeting existieren.
    valid = signals.notna().all(axis=1) & vols.notna().all(axis=1)
    valid_dates = valid[valid].index
    if len(valid_dates) == 0:
        raise ValueError("Keine gueltigen Tage — Warmup laenger als Datenfenster")
    start = valid_dates[0]
    eligible = returns.loc[start:].index
    if len(eligible) <= vol_window:
        raise ValueError("Zu wenig Tage nach Warmup fuer einen Backtest")
    rebalances = set(month_end_dates(eligible))

    weights = pd.Series(0.0, index=symbols)
    net, out_index = [], []
    contributions = {s: 0.0 for s in symbols}
    total_trade_cost = total_borrow_cost = total_turnover = 0.0
    n_rebalances = 0

    for t in eligible:
        r_t = returns.loc[t]
        gross_pnl = float((weights * r_t).sum())
        for s in symbols:
            contributions[s] += float(weights[s] * r_t[s])
        borrow = daily_borrow_cost_fraction(weights, cost_multiplier)
        total_borrow_cost += borrow
        day_net = gross_pnl - borrow

        # Drift: Gewichte bewegen sich mit den Preisen relativ zur Equity.
        equity_growth = 1.0 + gross_pnl
        if equity_growth > 0:
            weights = weights * (1.0 + r_t) / equity_growth

        if t in rebalances:
            trailing = returns.loc[:t].tail(vol_window)
            target = rebalance_weights(signals.loc[t], vols.loc[t], trailing, mode, target_vol)
            deltas = target - weights
            cost = trade_cost_fraction(deltas, cost_bp, cost_multiplier)
            day_net -= cost
            total_trade_cost += cost
            total_turnover += float(deltas.abs().sum())
            n_rebalances += 1
            weights = target

        net.append(day_net)
        out_index.append(t)

    info = {
        "total_trade_cost": total_trade_cost,
        "total_borrow_cost": total_borrow_cost,
        "total_turnover": total_turnover,
        "n_rebalances": n_rebalances,
        "instrument_contributions": contributions,
    }
    return pd.Series(net, index=pd.DatetimeIndex(out_index)), info


def benchmark_buy_and_hold(returns: pd.Series, cost_bp_value: float, cost_multiplier: float = 1.0) -> pd.Series:
    """Einmal kaufen (Kosten am ersten Tag), dann halten."""
    net = returns.copy().astype(float)
    net.iloc[0] = net.iloc[0] - cost_bp_value / 10_000.0 * cost_multiplier
    return net


def benchmark_fixed_mix(
    returns: pd.DataFrame,
    target_weights: dict[str, float],
    cost_bp: dict[str, float],
    cost_multiplier: float = 1.0,
) -> pd.Series:
    """Feste Zielgewichte (z.B. 60/40 SPY/TLT), monatlich rebalanced —
    gleiche Tageskonvention und Kostenlogik wie run_daily_backtest."""
    from factor_lab.costs import trade_cost_fraction

    symbols = list(target_weights)
    target = pd.Series(target_weights, dtype=float)
    rebalances = set(month_end_dates(returns.index))
    # Der erste Tag wirkt wie ein Rebalance (Aufbau), egal ob Monatsende.
    first_day = returns.index[0]
    rebalances.add(first_day)

    weights = pd.Series(0.0, index=symbols)
    net = []
    for t in returns.index:
        r_t = returns.loc[t, symbols]
        gross_pnl = float((weights * r_t).sum())
        day_net = gross_pnl
        equity_growth = 1.0 + gross_pnl
        if equity_growth > 0:
            weights = weights * (1.0 + r_t) / equity_growth
        if t in rebalances:
            deltas = target - weights
            day_net -= trade_cost_fraction(deltas, cost_bp, cost_multiplier)
            weights = target.copy()
        net.append(day_net)
    return pd.Series(net, index=returns.index)
```

- [ ] **Step 4: Test ausführen, Bestehen verifizieren**

Run: `py -3.12 factor_lab/tests/test_portfolio.py`
Expected: PASS (alle Checks inkl. Task-3-Checks)

- [ ] **Step 5: Commit**

```bash
git add factor_lab/portfolio.py factor_lab/tests/test_portfolio.py
git commit -m "feat(factor_lab): add daily backtest loop with drift, costs, and benchmarks"
```

---

### Task 5: stats.py — Monats-Block-Inferenz, Kennzahlen, Gates, Verdikt

**Files:**
- Create: `factor_lab/stats.py`
- Test: `factor_lab/tests/test_stats.py`

**Interfaces:**
- Consumes: `day_block_bootstrap`, `day_sign_flip_pvalue`, `compound_return`, `max_drawdown_from_returns` aus `market_control_system/controller/cross_sectional_signal_metrics.py` (via sys.path).
- Produces:
  - `month_block_bootstrap(values, timestamps, n_boot=2000, seed=0) -> dict` und `month_sign_flip_pvalue(values, timestamps, n_perm=2000, seed=0) -> dict` — gleiche Rückgabe-Schlüssel wie die Tages-Versionen, aber Kalendermonat = Block.
  - `annualized_stats(net_returns: pd.Series) -> dict` mit Schlüsseln `"cagr"`, `"vol_pa"`, `"sharpe"`, `"max_drawdown"`, `"n_days"`.
  - `yearly_returns(net_returns: pd.Series) -> dict[int, float]` (compoundiert je Kalenderjahr).
  - `evaluate_gates(stats: dict, bootstrap: dict, stressed_cagr: float, yearly: dict[int, float], instrument_contributions: dict[str, float], dd_cap: float = 0.15) -> dict` mit `"gate_a_ci_positive"`, `"gate_b_drawdown"`, `"gate_c_stressed_costs"`, `"gate_d_no_single_driver"`, `"passed_all"` (alles bool).
  - `verdict_string(gates_by_variant: dict[str, dict], sharpe_by_variant: dict[str, float]) -> str` — nennt Bestehende + die Holdout-Kandidatin (höchster Sharpe unter Bestehenden) oder den Nullbefund.

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_stats.py — prueft Monats-Block-Inferenz, Kennzahlen und Gates.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.stats import (
    month_block_bootstrap, month_sign_flip_pvalue, annualized_stats,
    yearly_returns, evaluate_gates, verdict_string,
)


def check_month_blocks() -> None:
    # Monat 1 konstant +1, Monat 2 konstant -1: echtes MONATS-Block-
    # Resampling kann nur Mittel aus {-1, 0, +1} erzeugen.
    idx = pd.DatetimeIndex(
        [f"2020-01-{d:02d}" for d in range(2, 22)] + [f"2020-02-{d:02d}" for d in range(3, 23)]
    )
    values = [1.0] * 20 + [-1.0] * 20
    result = month_block_bootstrap(values, idx, n_boot=300, seed=0)
    unique = set(np.round(result["bootstrap_means"], 12))
    assert unique <= {-1.0, 0.0, 1.0}, f"Monats-Bloecke verletzt: {sorted(unique)}"
    assert result["n_days"] == 2, "Zwei Monatsbloecke erwartet (n_days-Feld der Basisfunktion)"

    perm = month_sign_flip_pvalue(values, idx, n_perm=300, seed=0)
    assert perm["n_days"] == 2
    print("month_block_bootstrap / month_sign_flip_pvalue: OK")


def check_annualized_stats_and_years() -> None:
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    net = pd.Series([0.001] * 252, index=idx)
    stats = annualized_stats(net)
    assert abs(stats["cagr"] - (1.001 ** 252 - 1)) < 1e-9
    assert stats["max_drawdown"] == 0.0
    assert stats["n_days"] == 252

    two_years = pd.Series(
        [0.001] * 100 + [-0.001] * 100,
        index=list(pd.date_range("2020-06-01", periods=100, freq="B"))
        + list(pd.date_range("2021-06-01", periods=100, freq="B")),
    )
    ys = yearly_returns(two_years)
    assert abs(ys[2020] - (1.001 ** 100 - 1)) < 1e-9
    assert ys[2021] < 0
    print("annualized_stats / yearly_returns: OK")


def check_gates_and_verdict() -> None:
    good_stats = {"cagr": 0.05, "max_drawdown": -0.10, "sharpe": 0.8}
    good_boot = {"ci_low_95": 0.0001, "ci_high_95": 0.001}
    yearly = {2020: 0.02, 2021: 0.03, 2022: 0.01}
    contrib = {"SPY": 0.03, "TLT": 0.02, "GLD": 0.01}
    gates = evaluate_gates(good_stats, good_boot, stressed_cagr=0.02, yearly=yearly,
                           instrument_contributions=contrib, dd_cap=0.15)
    assert gates["passed_all"], f"Alle Gates muessten bestehen: {gates}"

    # Gate b: Drawdown -20% bei Cap 15% -> durchgefallen
    bad = evaluate_gates({**good_stats, "max_drawdown": -0.20}, good_boot, 0.02, yearly, contrib)
    assert not bad["gate_b_drawdown"] and not bad["passed_all"]

    # Gate d: bestes Jahr entfernt -> Rest negativ
    one_year = {2020: 0.50, 2021: -0.01, 2022: -0.01}
    bad = evaluate_gates(good_stats, good_boot, 0.02, one_year, contrib)
    assert not bad["gate_d_no_single_driver"]

    # Gate d: bestes Instrument abgezogen -> Rest negativ
    one_instr = {"SPY": 0.10, "TLT": -0.02, "GLD": -0.03}
    bad = evaluate_gates(good_stats, good_boot, 0.02, yearly, one_instr)
    assert not bad["gate_d_no_single_driver"]

    verdict = verdict_string(
        {"mom252_long_flat": gates, "mom63_long_short": bad},
        {"mom252_long_flat": 0.8, "mom63_long_short": 0.9},
    )
    assert "mom252_long_flat" in verdict, f"Bestehende Variante muss genannt werden: {verdict}"

    none_pass = verdict_string({"mom63_long_short": bad}, {"mom63_long_short": 0.9})
    assert "keine" in none_pass.lower()
    print("evaluate_gates / verdict_string: OK")


def run_consistency_check() -> None:
    check_month_blocks()
    check_annualized_stats_and_years()
    check_gates_and_verdict()
    print("\nAlle stats-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen, Fehlschlag verifizieren**

Run: `py -3.12 factor_lab/tests/test_stats.py`
Expected: FAIL mit ModuleNotFoundError

- [ ] **Step 3: Minimale Implementierung**

```python
"""
stats.py — Kennzahlen, Monats-Block-Inferenz und praeregistrierte Gates
(Spec Abschnitte 8+9).

Re-Use statt Duplikat: die getesteten Inferenz-Funktionen aus
market_control_system/controller/cross_sectional_signal_metrics.py werden
importiert; fuer MONATS-Bloecke wird der Timestamp-Index vor dem Aufruf auf
den jeweiligen Monatsanfang normalisiert (die Basisfunktionen gruppieren
nach index.date — jeder Kalendermonat wird so EIN Block).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market_control_system", "controller"))

import numpy as np
import pandas as pd

from cross_sectional_signal_metrics import (
    day_block_bootstrap, day_sign_flip_pvalue, compound_return, max_drawdown_from_returns,
)

TRADING_DAYS_PA = 252


def _month_normalized(timestamps) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(timestamps)
    return idx.to_period("M").to_timestamp()


def month_block_bootstrap(values, timestamps, n_boot: int = 2000, seed: int = 0) -> dict:
    return day_block_bootstrap(list(values), _month_normalized(timestamps), n_boot=n_boot, seed=seed)


def month_sign_flip_pvalue(values, timestamps, n_perm: int = 2000, seed: int = 0) -> dict:
    return day_sign_flip_pvalue(list(values), _month_normalized(timestamps), n_perm=n_perm, seed=seed)


def annualized_stats(net_returns: pd.Series) -> dict:
    """CAGR/Vol/Sharpe/Max-DD, alles multiplikativ auf der echten Equity-Kurve."""
    r = net_returns.astype(float)
    n = len(r)
    total = compound_return(r.tolist())
    years = n / TRADING_DAYS_PA
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    vol_pa = float(r.std()) * np.sqrt(TRADING_DAYS_PA)
    sharpe = (float(r.mean()) * TRADING_DAYS_PA) / vol_pa if vol_pa > 0 else 0.0
    return {
        "cagr": float(cagr),
        "vol_pa": float(vol_pa),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown_from_returns(r.tolist()),
        "n_days": n,
    }


def yearly_returns(net_returns: pd.Series) -> dict[int, float]:
    return {
        int(year): compound_return(group.tolist())
        for year, group in net_returns.groupby(net_returns.index.year)
    }


def evaluate_gates(
    stats: dict,
    bootstrap: dict,
    stressed_cagr: float,
    yearly: dict[int, float],
    instrument_contributions: dict[str, float],
    dd_cap: float = 0.15,
) -> dict:
    """Die vier praeregistrierten Gates aus Spec Abschnitt 9. Gate d nutzt
    fuer Instrumente den ADDITIVEN Beitragsabzug (definierte Naeherung,
    KEINE Portfolio-Neuberechnung ohne das Instrument)."""
    gate_a = bootstrap["ci_low_95"] > 0
    gate_b = abs(stats["max_drawdown"]) <= dd_cap
    gate_c = stressed_cagr > 0

    without_best_year = [v for k, v in yearly.items() if k != max(yearly, key=yearly.get)]
    rest_years_positive = compound_return(without_best_year) > 0 if without_best_year else False
    total_contrib = sum(instrument_contributions.values())
    best_contrib = max(instrument_contributions.values()) if instrument_contributions else 0.0
    rest_instruments_positive = (total_contrib - best_contrib) > 0
    gate_d = rest_years_positive and rest_instruments_positive

    return {
        "gate_a_ci_positive": bool(gate_a),
        "gate_b_drawdown": bool(gate_b),
        "gate_c_stressed_costs": bool(gate_c),
        "gate_d_no_single_driver": bool(gate_d),
        "passed_all": bool(gate_a and gate_b and gate_c and gate_d),
    }


def verdict_string(gates_by_variant: dict[str, dict], sharpe_by_variant: dict[str, float]) -> str:
    passed = [name for name, g in gates_by_variant.items() if g["passed_all"]]
    if not passed:
        return (
            "VERDICT: keine Variante besteht alle vier Gates im Entwicklungsfenster -- "
            "das Holdout wird NICHT angefasst, Ergebnis als Nullbefund dokumentieren."
        )
    candidate = max(passed, key=lambda name: sharpe_by_variant[name])
    return (
        f"VERDICT: {len(passed)} Variante(n) bestehen alle Gates ({', '.join(sorted(passed))}). "
        f"Holdout-Kandidatin nach vorab fixierter Regel (hoechster Netto-Sharpe): {candidate}. "
        f"Einmalige manuelle Evaluation via run_trend_holdout.py."
    )
```

- [ ] **Step 4: Test ausführen, Bestehen verifizieren**

Run: `py -3.12 factor_lab/tests/test_stats.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add factor_lab/stats.py factor_lab/tests/test_stats.py
git commit -m "feat(factor_lab): add month-block inference, annualized stats, and pre-registered gates"
```

---

### Task 6: data_snapshot.py

**Files:**
- Create: `factor_lab/data_snapshot.py`
- Test: `factor_lab/tests/test_data_snapshot.py`
- Modify: `.gitignore` (Zeilen `factor_lab/logs/` und `factor_lab/data_snapshots/` ergänzen)
- Modify: `requirements.txt` bzw. `market_control_system/requirements.txt` — wo immer die bestehenden Dependencies stehen (`yfinance` ergänzen); zusätzlich `py -3.12 -m pip install yfinance`

**Interfaces:**
- Consumes: `snapshot_content_sha256`, `write_snapshot_manifest` aus `market_control_system/data_layer/frozen_snapshot.py` (via sys.path).
- Produces: `TREND_UNIVERSE: list[str]` (12 Symbole), `SNAPSHOT_START = "2007-01-01"`, `fetch_trend_data(symbols: list[str], start: str) -> dict[str, pd.DataFrame]` (je Symbol DataFrame mit Spalte `"price"` = adjustierter Schlusskurs, DatetimeIndex), `load_or_build_trend_snapshot() -> dict[str, pd.DataFrame]` (friert nach `factor_lab/data_snapshots/trend_snapshot_<parameterhash>.pkl` ein, schreibt daneben `<datei>.manifest.json`).

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_data_snapshot.py — prueft Freeze/Reload/Manifest mit gestubbtem
Fetch (kein Netzwerkzugriff, Muster aus test_frozen_snapshot.py).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pandas as pd

from factor_lab import data_snapshot


def _fake_fetch(symbols, start):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    return {s: pd.DataFrame({"price": [100.0 + i for i in range(5)]}, index=idx) for s in symbols}


def run_consistency_check() -> None:
    original_fetch = data_snapshot.fetch_trend_data
    original_universe = data_snapshot.TREND_UNIVERSE
    data_snapshot.fetch_trend_data = _fake_fetch
    data_snapshot.TREND_UNIVERSE = ["AAA", "BBB"]
    path = data_snapshot.trend_snapshot_path(["AAA", "BBB"], data_snapshot.SNAPSHOT_START)
    manifest_path = path + ".manifest.json"
    try:
        for p in (path, manifest_path):
            if os.path.exists(p):
                os.remove(p)

        first = data_snapshot.load_or_build_trend_snapshot()
        assert os.path.exists(path), "Snapshot-Pickle wurde nicht angelegt"
        assert os.path.exists(manifest_path), "Manifest wurde nicht angelegt"

        second = data_snapshot.load_or_build_trend_snapshot()
        for s in ["AAA", "BBB"]:
            pd.testing.assert_frame_equal(first[s], second[s])

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["symbols"]["AAA"]["rows"] == 5
        assert "content_sha256" in manifest
        print("data_snapshot: Freeze, identischer Reload, Manifest -- OK")
    finally:
        data_snapshot.fetch_trend_data = original_fetch
        data_snapshot.TREND_UNIVERSE = original_universe
        for p in (path, manifest_path):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen, Fehlschlag verifizieren**

Run: `py -3.12 factor_lab/tests/test_data_snapshot.py`
Expected: FAIL mit ModuleNotFoundError

- [ ] **Step 3: Minimale Implementierung**

```python
"""
data_snapshot.py — Frozen Snapshot der ETF-Tagesdaten (Spec Abschnitt 4).

Einmalig via yfinance fetchen (auto_adjust=True: Splits + Dividenden in
den Preisen -> Returns aus "price" sind Total-Return-Naeherungen), dann
einfrieren. Alle Auswertungen laufen gegen den eingefrorenen Snapshot;
Content-Hash + Manifest kommen aus market_control_system (Re-Use).
yfinance wird erst IM Fetch importiert — Tests und Auswertungen brauchen
die Dependency nicht.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market_control_system", "data_layer"))

import pandas as pd

from frozen_snapshot import snapshot_content_sha256, write_snapshot_manifest

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "data_snapshots")
SNAPSHOT_START = "2007-01-01"
TREND_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"]


def trend_snapshot_path(symbols: list[str], start: str) -> str:
    key = f"{sorted(symbols)}|{start}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(SNAPSHOT_DIR, f"trend_snapshot_{digest}.pkl")


def fetch_trend_data(symbols: list[str], start: str) -> dict[str, pd.DataFrame]:
    """Laedt Tagesdaten je Symbol via yfinance (nur beim Snapshot-Build)."""
    import yfinance as yf

    dfs = {}
    for symbol in symbols:
        print(f"  Snapshot: lade {symbol}...")
        raw = yf.download(symbol, start=start, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"{symbol}: yfinance lieferte keine Daten")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # yfinance liefert je nach Version MultiIndex-Spalten
            close = close.iloc[:, 0]
        df = pd.DataFrame({"price": close.astype(float)})
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        dfs[symbol] = df
        print(f"    {df.shape[0]} Tage ({df.index.min().date()} bis {df.index.max().date()})")
    return dfs


def load_or_build_trend_snapshot() -> dict[str, pd.DataFrame]:
    """Laedt den eingefrorenen Snapshot, baut ihn sonst einmalig und friert
    ihn samt Manifest ein. Wiederholte Aufrufe liefern exakt dieselben Daten."""
    path = trend_snapshot_path(TREND_UNIVERSE, SNAPSHOT_START)
    if os.path.exists(path):
        print(f"  Snapshot gefunden: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  Kein Snapshot gefunden, baue neu: {path}")
    dfs = fetch_trend_data(TREND_UNIVERSE, SNAPSHOT_START)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dfs, f)
    write_snapshot_manifest(dfs, path + ".manifest.json")
    print(f"  Snapshot + Manifest gespeichert (sha256: {snapshot_content_sha256(dfs)[:16]}...)")
    return dfs
```

- [ ] **Step 4: Test ausführen, Bestehen verifizieren**

Run: `py -3.12 factor_lab/tests/test_data_snapshot.py`
Expected: PASS. Danach `.gitignore` ergänzen (`factor_lab/logs/`, `factor_lab/data_snapshots/`), `yfinance` in die requirements-Datei des Repos eintragen und installieren: `py -3.12 -m pip install yfinance`

- [ ] **Step 5: Commit**

```bash
git add factor_lab/data_snapshot.py factor_lab/tests/test_data_snapshot.py .gitignore
git add -u
git commit -m "feat(factor_lab): add frozen yfinance ETF snapshot with content hash and manifest"
```

---

### Task 7: run_trend_baseline.py + Integrationstest

**Files:**
- Create: `factor_lab/run_trend_baseline.py`
- Test: `factor_lab/tests/test_run_trend_baseline.py`

**Interfaces:**
- Consumes: alles aus Task 1–6 (exakte Signaturen siehe dort).
- Produces: `DEV_FRACTION = 0.8`, `VARIANT_NAMES: list[str]` (genau die 8: `mom63_long_short`, `mom63_long_flat`, `mom126_long_short`, `mom126_long_flat`, `mom252_long_short`, `mom252_long_flat`, `combo_long_short`, `combo_long_flat`), `run_baseline(dfs: dict[str, pd.DataFrame], cost_bp: dict[str, float]) -> tuple[dict, pd.DataFrame]` — (JSON-serialisierbares Ergebnis-Dict, per-Tag-Netto-Return-Frame mit einer Spalte je Variante + `bench_spy_bh` + `bench_60_40`), `main()` (lädt echten Snapshot, speichert JSON + CSV nach `factor_lab/logs/trend_baseline_<runid>/`).

Das Ergebnis-Dict enthält: `"summary"` (je Variante: annualized_stats + `"bootstrap"` + `"permutation"` + `"gates"` + `"yearly_returns"` + `"instrument_contributions"` + `"stressed_cagr"` + `"total_turnover"` + `"total_trade_cost"` + `"total_borrow_cost"`), `"benchmarks"` (je Benchmark: annualized_stats), `"verdict"`, `"provenance"` (Universum, Start, DEV_FRACTION, dev_start/dev_end als Strings, Kostentabelle, target_vol, Lookbacks, Bootstrap-Parameter, `snapshot_content_sha256`).

- [ ] **Step 1: Failing Integrationstest schreiben**

```python
"""
test_run_trend_baseline.py — Integrationstest der kompletten Pipeline auf
synthetischen Daten (3 Symbole, ~700 Tage, kein Netzwerk, kein Holdout-Zugriff).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
import pandas as pd

from factor_lab.run_trend_baseline import run_baseline, VARIANT_NAMES


def _synthetic_dfs() -> dict:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=700, freq="B")
    dfs = {}
    for i, symbol in enumerate(["SPY", "TLT", "GLD"]):
        drift = [0.0003, 0.0001, 0.0002][i]
        prices = 100.0 * np.cumprod(1.0 + rng.normal(drift, 0.01, 700))
        dfs[symbol] = pd.DataFrame({"price": prices}, index=idx)
    return dfs


def run_consistency_check() -> None:
    dfs = _synthetic_dfs()
    cost_bp = {"SPY": 1.5, "TLT": 1.5, "GLD": 1.5}
    result, per_day = run_baseline(dfs, cost_bp)

    assert sorted(result["summary"]) == sorted(VARIANT_NAMES), (
        f"Genau die 8 praeregistrierten Varianten erwartet, bekam {sorted(result['summary'])}"
    )
    for name in VARIANT_NAMES:
        s = result["summary"][name]
        for key in ("cagr", "max_drawdown", "bootstrap", "permutation", "gates",
                    "yearly_returns", "instrument_contributions", "stressed_cagr"):
            assert key in s, f"{name}: Schluessel {key} fehlt"
    assert "bench_spy_bh" in result["benchmarks"] and "bench_60_40" in result["benchmarks"]
    assert "VERDICT" in result["verdict"]

    # Holdout-Schutz: Entwicklungsfenster endet VOR dem letzten Datum
    dev_end = pd.Timestamp(result["provenance"]["dev_end"])
    assert dev_end < dfs["SPY"].index.max(), "Entwicklungsfenster darf nicht bis zum Datenende reichen"

    # per-Tag-Frame: eine Spalte je Variante + 2 Benchmarks, JSON-serialisierbar
    assert set(per_day.columns) == set(VARIANT_NAMES) | {"bench_spy_bh", "bench_60_40"}
    json.dumps(result)  # wirft, falls nicht serialisierbar
    print("run_baseline (synthetisch, 3 Symbole): OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen, Fehlschlag verifizieren**

Run: `py -3.12 factor_lab/tests/test_run_trend_baseline.py`
Expected: FAIL mit ModuleNotFoundError

- [ ] **Step 3: Implementierung**

```python
"""
run_trend_baseline.py — Entwicklungsfenster-Lauf des Trend-Legs
(Spec Abschnitte 5-9): 8 praeregistrierte Varianten + 2 Benchmarks,
Monats-Block-Inferenz, Gates, Verdikt. Das Holdout (letzte 20% der
gemeinsamen Handelstage) wird von diesem Skript NIE gelesen — nur von
run_trend_holdout.py, manuell.

Ausfuehren: py -3.12 factor_lab/run_trend_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from factor_lab.signals import momentum_sign, combo_signal
from factor_lab.costs import COST_BP
from factor_lab.portfolio import (
    ewma_annualized_vol, run_daily_backtest, benchmark_buy_and_hold, benchmark_fixed_mix,
)
from factor_lab.stats import (
    month_block_bootstrap, month_sign_flip_pvalue, annualized_stats,
    yearly_returns, evaluate_gates, verdict_string,
)
from factor_lab.data_snapshot import (
    load_or_build_trend_snapshot, snapshot_content_sha256, TREND_UNIVERSE, SNAPSHOT_START,
)

DEV_FRACTION = 0.8
TARGET_VOL = 0.10
LOOKBACKS = {"mom63": 63, "mom126": 126, "mom252": 252}
MODES = ["long_short", "long_flat"]
VARIANT_NAMES = [f"{sig}_{mode}" for sig in [*LOOKBACKS, "combo"] for mode in MODES]
DD_CAP = 0.15


def _build_signal_frames(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Signal-Frames je Signalvariante (Spalten = Symbole)."""
    per_lookback = {
        name: pd.DataFrame({s: momentum_sign(prices[s], lb) for s in prices.columns})
        for name, lb in LOOKBACKS.items()
    }
    combo = pd.DataFrame({
        s: combo_signal([per_lookback[name][s] for name in LOOKBACKS])
        for s in prices.columns
    })
    return {**per_lookback, "combo": combo}


def run_baseline(dfs: dict[str, pd.DataFrame], cost_bp: dict[str, float]) -> tuple[dict, pd.DataFrame]:
    symbols = sorted(dfs)
    # Gemeinsamer Kalender = Schnittmenge aller Handelstage.
    common = dfs[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(dfs[s].index)
    common = common.sort_values()
    prices = pd.DataFrame({s: dfs[s].loc[common, "price"] for s in symbols})

    # Holdout-Schnitt VOR jeder Auswertung: nur die ersten 80% werden benutzt.
    split = int(len(common) * DEV_FRACTION)
    dev_prices = prices.iloc[:split]
    returns = dev_prices.pct_change().dropna()

    signal_frames = _build_signal_frames(dev_prices)
    vols = ewma_annualized_vol(returns)

    summary, per_day_columns = {}, {}
    for sig_name, signals in signal_frames.items():
        signals_aligned = signals.loc[returns.index]
        for mode in MODES:
            variant = f"{sig_name}_{mode}"
            net, info = run_daily_backtest(
                returns, signals_aligned, vols, mode=mode,
                cost_bp=cost_bp, target_vol=TARGET_VOL,
            )
            stressed_net, _ = run_daily_backtest(
                returns, signals_aligned, vols, mode=mode,
                cost_bp=cost_bp, cost_multiplier=2.0, target_vol=TARGET_VOL,
            )
            stats = annualized_stats(net)
            boot = month_block_bootstrap(net.tolist(), net.index)
            boot.pop("bootstrap_means")
            perm = month_sign_flip_pvalue(net.tolist(), net.index)
            years = yearly_returns(net)
            gates = evaluate_gates(
                stats, boot, annualized_stats(stressed_net)["cagr"],
                years, info["instrument_contributions"], dd_cap=DD_CAP,
            )
            summary[variant] = {
                **stats,
                "bootstrap": boot,
                "permutation": perm,
                "gates": gates,
                "yearly_returns": {str(k): v for k, v in years.items()},
                "instrument_contributions": info["instrument_contributions"],
                "stressed_cagr": annualized_stats(stressed_net)["cagr"],
                "total_turnover": info["total_turnover"],
                "total_trade_cost": info["total_trade_cost"],
                "total_borrow_cost": info["total_borrow_cost"],
            }
            per_day_columns[variant] = net

    bench_spy = benchmark_buy_and_hold(returns[symbols[0] if "SPY" not in symbols else "SPY"], cost_bp.get("SPY", 1.5))
    mix_symbols = ["SPY", "TLT"] if "SPY" in symbols and "TLT" in symbols else symbols[:2]
    bench_mix = benchmark_fixed_mix(
        returns[mix_symbols],
        target_weights={mix_symbols[0]: 0.6, mix_symbols[1]: 0.4},
        cost_bp={k: cost_bp[k] for k in mix_symbols},
    )
    per_day_columns["bench_spy_bh"] = bench_spy
    per_day_columns["bench_60_40"] = bench_mix
    benchmarks = {
        "bench_spy_bh": annualized_stats(bench_spy),
        "bench_60_40": annualized_stats(bench_mix),
    }

    verdict = verdict_string(
        {v: summary[v]["gates"] for v in summary},
        {v: summary[v]["sharpe"] for v in summary},
    )
    provenance = {
        "universe": symbols,
        "snapshot_start": SNAPSHOT_START,
        "dev_fraction": DEV_FRACTION,
        "dev_start": str(returns.index.min()),
        "dev_end": str(returns.index.max()),
        "target_vol": TARGET_VOL,
        "dd_cap": DD_CAP,
        "lookbacks": LOOKBACKS,
        "cost_bp": cost_bp,
        "bootstrap": {"n_boot": 2000, "seed": 0, "blocks": "Kalendermonate"},
        "snapshot_content_sha256": snapshot_content_sha256(dfs),
    }
    result = {"summary": summary, "benchmarks": benchmarks, "verdict": verdict, "provenance": provenance}
    per_day = pd.DataFrame(per_day_columns)
    return result, per_day


def main() -> None:
    print(f"=== Trend-Baseline: lade Snapshot ({len(TREND_UNIVERSE)} ETFs ab {SNAPSHOT_START}) ===")
    dfs = load_or_build_trend_snapshot()
    result, per_day = run_baseline(dfs, COST_BP)

    print(f"\n{'='*70}\n=== Entwicklungsfenster: {result['provenance']['dev_start']} bis {result['provenance']['dev_end']} ===\n{'='*70}")
    for name in VARIANT_NAMES:
        s = result["summary"][name]
        b = s["bootstrap"]
        g = s["gates"]
        print(f"{name:>22}: cagr={s['cagr']:+.2%}  vol={s['vol_pa']:.2%}  sharpe={s['sharpe']:+.2f}  "
              f"max_dd={s['max_drawdown']:+.2%}  ci95=[{b['ci_low_95']:+.6f},{b['ci_high_95']:+.6f}]  "
              f"p_perm={s['permutation']['p_two_sided']:.3f}  stressed_cagr={s['stressed_cagr']:+.2%}  "
              f"gates={'PASS' if g['passed_all'] else 'fail'}")
    for name, s in result["benchmarks"].items():
        print(f"{name:>22}: cagr={s['cagr']:+.2%}  vol={s['vol_pa']:.2%}  sharpe={s['sharpe']:+.2f}  max_dd={s['max_drawdown']:+.2%}")
    print(f"\n{result['verdict']}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "logs", f"trend_baseline_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    per_day.to_csv(os.path.join(out_dir, "daily_net_returns.csv"))
    with open(os.path.join(out_dir, "baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nErgebnisse gespeichert: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test ausführen, Bestehen verifizieren**

Run: `py -3.12 factor_lab/tests/test_run_trend_baseline.py`
Expected: PASS (dauert wegen 16 Backtest-Läufen auf 700 Tagen ein paar Sekunden)

- [ ] **Step 5: Alle factor_lab-Tests + Bestands-Tests laufen lassen**

Run: alle `py -3.12 factor_lab/tests/test_*.py` einzeln + die 7 bestehenden `market_control_system/tests/test_*.py`
Expected: alles PASS (Bestand darf nicht brechen — stats.py importiert nur, ändert nichts)

- [ ] **Step 6: Commit**

```bash
git add factor_lab/run_trend_baseline.py factor_lab/tests/test_run_trend_baseline.py
git commit -m "feat(factor_lab): add development-window baseline runner with gates and verdict"
```

---

### Task 8: run_trend_holdout.py + echter Lauf

**Files:**
- Create: `factor_lab/run_trend_holdout.py`
- Test: `factor_lab/tests/test_run_trend_holdout.py`

**Interfaces:**
- Consumes: `run_baseline`-Bausteine (Task 7), `VARIANT_NAMES`, `DEV_FRACTION`.
- Produces: `run_holdout(dfs: dict, cost_bp: dict, variant: str) -> dict` — evaluiert GENAU EINE benannte Variante auf dem Holdout-Fenster (Signale dürfen Preise VOR dem Holdout sehen — kausal; Returns werden nur im Holdout gewertet); `main()` verlangt den Variantennamen als `sys.argv[1]`.

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_run_trend_holdout.py — prueft, dass die Holdout-Auswertung genau eine
Variante rechnet, nur Holdout-Tage wertet und unbekannte Namen ablehnt.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.run_trend_holdout import run_holdout
from factor_lab.run_trend_baseline import DEV_FRACTION


def _synthetic_dfs() -> dict:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=700, freq="B")
    return {
        s: pd.DataFrame({"price": 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, 700))}, index=idx)
        for s in ["SPY", "TLT", "GLD"]
    }


def run_consistency_check() -> None:
    dfs = _synthetic_dfs()
    cost_bp = {"SPY": 1.5, "TLT": 1.5, "GLD": 1.5}
    result = run_holdout(dfs, cost_bp, variant="mom63_long_flat")

    assert result["variant"] == "mom63_long_flat"
    assert "cagr" in result["summary"] and "gates" in result["summary"]
    # Alle gewerteten Tage liegen NACH dem Entwicklungsfenster
    holdout_start = pd.Timestamp(result["provenance"]["holdout_start"])
    common_len = 700
    assert holdout_start >= dfs["SPY"].index[int(common_len * DEV_FRACTION)]

    try:
        run_holdout(dfs, cost_bp, variant="mom999_long_flat")
        raise AssertionError("Unbekannte Variante muss ValueError ausloesen")
    except ValueError:
        pass
    print("run_holdout: OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen, Fehlschlag verifizieren**

Run: `py -3.12 factor_lab/tests/test_run_trend_holdout.py`
Expected: FAIL mit ModuleNotFoundError

- [ ] **Step 3: Implementierung**

```python
"""
run_trend_holdout.py — EINMALIGE manuelle Holdout-Auswertung GENAU EINER
Variante (Spec Abschnitt 9). Absichtlich separates Skript mit Pflicht-
Argument, damit das Holdout nicht versehentlich "mitgerechnet" wird.

Nur ausfuehren, wenn run_trend_baseline.py mindestens eine Variante durch
alle vier Gates gebracht hat, und nur fuer die im Verdikt benannte
Kandidatin (hoechster Netto-Sharpe unter den Bestehenden).

Ausfuehren: py -3.12 factor_lab/run_trend_holdout.py <variantenname>
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from factor_lab.costs import COST_BP
from factor_lab.portfolio import ewma_annualized_vol, run_daily_backtest
from factor_lab.stats import (
    month_block_bootstrap, month_sign_flip_pvalue, annualized_stats,
    yearly_returns, evaluate_gates,
)
from factor_lab.run_trend_baseline import (
    DEV_FRACTION, TARGET_VOL, DD_CAP, VARIANT_NAMES, _build_signal_frames,
)
from factor_lab.data_snapshot import load_or_build_trend_snapshot, snapshot_content_sha256


def run_holdout(dfs: dict[str, pd.DataFrame], cost_bp: dict[str, float], variant: str) -> dict:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unbekannte Variante: {variant} (erlaubt: {VARIANT_NAMES})")
    sig_name, mode = variant.rsplit("_", 2)[0], "_".join(variant.rsplit("_", 2)[1:])

    symbols = sorted(dfs)
    common = dfs[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(dfs[s].index)
    common = common.sort_values()
    prices = pd.DataFrame({s: dfs[s].loc[common, "price"] for s in symbols})

    # Signale/Vols auf der VOLLEN Historie (kausal — Preise vor dem Holdout
    # zu sehen ist erlaubt und noetig fuer die Lookbacks), GEWERTET wird
    # ausschliesslich im Holdout-Fenster.
    returns = prices.pct_change().dropna()
    signals = _build_signal_frames(prices)[sig_name].loc[returns.index]
    vols = ewma_annualized_vol(returns)

    split = int(len(common) * DEV_FRACTION)
    holdout_index = returns.index[returns.index >= common[split]]

    net_full, info = run_daily_backtest(returns, signals, vols, mode=mode, cost_bp=cost_bp, target_vol=TARGET_VOL)
    stressed_full, _ = run_daily_backtest(returns, signals, vols, mode=mode, cost_bp=cost_bp, cost_multiplier=2.0, target_vol=TARGET_VOL)
    net = net_full.loc[net_full.index.isin(holdout_index)]
    stressed = stressed_full.loc[stressed_full.index.isin(holdout_index)]

    stats = annualized_stats(net)
    boot = month_block_bootstrap(net.tolist(), net.index)
    boot.pop("bootstrap_means")
    gates = evaluate_gates(
        stats, boot, annualized_stats(stressed)["cagr"],
        yearly_returns(net), info["instrument_contributions"], dd_cap=DD_CAP,
    )
    return {
        "variant": variant,
        "summary": {**stats, "bootstrap": boot,
                    "permutation": month_sign_flip_pvalue(net.tolist(), net.index),
                    "gates": gates},
        "provenance": {
            "holdout_start": str(net.index.min()),
            "holdout_end": str(net.index.max()),
            "dev_fraction": DEV_FRACTION,
            "snapshot_content_sha256": snapshot_content_sha256(dfs),
        },
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Nutzung: py -3.12 factor_lab/run_trend_holdout.py <variantenname>")
        print(f"Erlaubte Varianten: {VARIANT_NAMES}")
        sys.exit(1)
    print("*** HOLDOUT-AUSWERTUNG: einmalig, nur nach bestandenen Gates im Entwicklungsfenster. ***")
    dfs = load_or_build_trend_snapshot()
    result = run_holdout(dfs, COST_BP, sys.argv[1])
    s = result["summary"]
    print(f"\n{result['variant']}: cagr={s['cagr']:+.2%}  sharpe={s['sharpe']:+.2f}  "
          f"max_dd={s['max_drawdown']:+.2%}  ci95=[{s['bootstrap']['ci_low_95']:+.6f},{s['bootstrap']['ci_high_95']:+.6f}]  "
          f"gates={'PASS' if s['gates']['passed_all'] else 'fail'}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "logs", f"trend_holdout_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "holdout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nErgebnis gespeichert: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test ausführen, Bestehen verifizieren**

Run: `py -3.12 factor_lab/tests/test_run_trend_holdout.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add factor_lab/run_trend_holdout.py factor_lab/tests/test_run_trend_holdout.py
git commit -m "feat(factor_lab): add one-shot manual holdout evaluation script"
```

- [ ] **Step 6: Echten Snapshot bauen + echten Baseline-Lauf ausführen**

Run: `py -3.12 factor_lab/run_trend_baseline.py` (baut beim ersten Mal den yfinance-Snapshot; Gesamtlaufzeit erwartet < 10 min — falls länger, detacht starten nach dem Muster aus dem Tech-Stack-Memory). Danach das Snapshot-Manifest committen:

```bash
git add -f factor_lab/data_snapshots/*.manifest.json
git commit -m "docs(factor_lab): commit trend snapshot manifest"
```

- [ ] **Step 7: Ergebnisse berichten**

Dem User die Tabelle (8 Varianten + 2 Benchmarks), das Verdikt und die Gate-Übersicht zeigen. KEINE Holdout-Ausführung ohne explizite User-Freigabe, selbst wenn Varianten bestehen.
