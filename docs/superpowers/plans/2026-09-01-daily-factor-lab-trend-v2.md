# Daily-Factor-Lab Trend-Leg Implementation Plan v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Das präregistrierte Trend-Baseline-Lab aus Spec v2 bauen: 8 Varianten gegen einen Matched-No-Signal-Benchmark, mit ausführbarer Lag-Konvention, versiegeltem One-Shot-Holdout und Stationary-Block-Inferenz.

**Architecture:** Top-Level-Package `factor_lab/` mit reinen Funktionsmodulen (signals/costs/portfolio/stats/registration/data_snapshot), einem gemeinsamen täglichen Loop mit Weight-Provider-Callback (Trend, matched_long, Fixed-Mix-Benchmarks teilen sich exakt dieselbe Ausführungs- und Kostenlogik), einem separaten versiegelten Snapshot-Build und zwei Orchestrierungs-Skripten (Screening automatisch, Holdout nur über candidate.json).

**Tech Stack:** Python 3.12 (`py -3.12`), pandas, numpy, yfinance (nur Build). Kein Torch.

**Spec:** `docs/superpowers/specs/2026-09-01-daily-factor-lab-trend-design.md` (v2). Plan v1 (`2026-09-01-daily-factor-lab-trend.md`) ist superseded — NICHT von dort kopieren.

## Global Constraints

- ALLES mit `py -3.12` ausführen (nie bare `python`).
- Test-Konvention: reine Skripte, `check_*`-Funktionen, `run_consistency_check()`, `__main__`; Ausführung `py -3.12 factor_lab/tests/test_x.py`. Kein pytest.
- Docstrings/Kommentare Deutsch (ASCII-Umschreibung ue/oe/ae).
- Alle Equity-Kennzahlen multiplikativ; Drawdown-Equity-Kurve INKLUSIVE Startwert 1.0 (der Helper in `market_control_system` ist auf diesem Branch bereits gefixt).
- Exakt 8 präregistrierte Varianten: {mom63, mom126, mom252, combo} × {long_short, long_flat}. long_short ist Forschungsvariante (Label in Outputs).
- Parameter verbatim aus Spec v2: Lookbacks {63, 126, 252}; EWMA-Span 63 (min_periods 63); Vol-OBERGRENZE 0.10 über einfache 63-Tage-Std des Kandidatenportfolios; Gross-Cap 1.0 nur für Zielgewichte, tägliches Max-Gross wird reportet; Entscheidung am Monatsultimo t, Fill zu Close(t+1), Position wirkt ab Return t+1→t+2; Kosten 1.5 bp (SPY, QQQ, IWM, TLT, IEF, GLD) / 3.0 bp (EFA, EEM, LQD, SLV, DBC, VNQ), Kommission 0; Kostenleiter 1×/2×/5×; Borrow 50 bp p.a. (Sensitivität 25/100); Cash-Gewicht `max(0, 1 − Σ|w_i|)`, verzinst mit `^IRX`/100/252; `SNAPSHOT_START=2007-01-01`, `SNAPSHOT_END=2026-09-01` (exklusiv), beide im Parameter-Hash; Warmup 252+63 Handelstage; gemeinsames Evaluationsfenster für ALLE Varianten und Benchmarks; `DEV_END` = letzter Monatsultimo ≤ 80%-Quantil-Datum des gemeinsamen Kalenders; Bootstrap: Stationary-Block auf monatlichen Log-(Mehr-)Erträgen, erwartete Blocklänge 6 Monate (Sensitivität 3/12), B=10.000, p=(extreme+1)/(B+1), identische Ziehungen über Varianten; Permutation 10.000; Gates A–D und Holdout-Regeln aus Spec §10; DD-Cap 0.15; Gate-C-Floor +2.0 % p.a. bei 2×-Kosten.
- Import aus `market_control_system` per `sys.path.insert`, nie kopieren.
- `factor_lab/logs/` und `factor_lab/data_snapshots/` gitignored; Manifest per `git add -f` committen, BEVOR ein Screening-Lauf startet.
- Läufe > 20 min detacht starten (Start-Process + PID-Datei + Monitor, siehe Tech-Stack-Memory).
- Commit nach jedem Task, Messages mit `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Package-Skelett + signals.py

**Files:**
- Create: `factor_lab/__init__.py` (leer), `factor_lab/tests/__init__.py` (leer)
- Create: `factor_lab/signals.py`
- Test: `factor_lab/tests/test_signals.py`

**Interfaces:**
- Produces: `momentum_sign(prices: pd.Series, lookback: int) -> pd.Series` (Werte {−1.0, 0.0, +1.0}, NaN für erste `lookback` Einträge); `combo_signal(signals: list[pd.Series]) -> pd.Series` (Mittel, NaN wo irgendein Input NaN).

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
    assert sig.iloc[:2].isna().all(), f"Erste lookback Eintraege muessen NaN sein, bekam {sig.iloc[:2].tolist()}"
    # t=2: 99/100-1 < 0 -> -1 ; t=3: 99/101-1 < 0 -> -1 ; t=4: 102/99-1 > 0 -> +1
    assert sig.iloc[2] == -1.0 and sig.iloc[3] == -1.0 and sig.iloc[4] == 1.0
    flat = pd.Series([100.0, 100.0, 100.0], index=pd.date_range("2020-01-01", periods=3, freq="B"))
    assert momentum_sign(flat, lookback=1).iloc[2] == 0.0, "Exakter Null-Return muss Signal 0 geben"
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

- [ ] **Step 2: Test ausführen — FAIL erwartet** (`ModuleNotFoundError`)

Run: `py -3.12 factor_lab/tests/test_signals.py`

- [ ] **Step 3: Implementierung**

```python
"""
signals.py — praeregistrierte Trend-Signale (Spec v2 Abschnitt 5).
Reine Funktionen; BEWUSST kein Mechanismus fuer weitere Varianten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_sign(prices: pd.Series, lookback: int) -> pd.Series:
    """Vorzeichen des Returns ueber `lookback` Handelstage; NaN fuer die
    ersten `lookback` Eintraege; exakter Null-Return -> 0."""
    return np.sign(prices / prices.shift(lookback) - 1.0)


def combo_signal(signals: list[pd.Series]) -> pd.Series:
    """Gleichgewichts-Mittel; NaN, wo irgendein Input NaN ist."""
    frame = pd.concat(signals, axis=1)
    return frame.mean(axis=1).where(frame.notna().all(axis=1))
```

- [ ] **Step 4: Test ausführen — PASS erwartet**
- [ ] **Step 5: Commit**

```bash
git add factor_lab/__init__.py factor_lab/tests/__init__.py factor_lab/signals.py factor_lab/tests/test_signals.py
git commit -m "feat(factor_lab): add pre-registered trend signals"
```

---

### Task 2: costs.py

**Files:**
- Create: `factor_lab/costs.py`
- Test: `factor_lab/tests/test_costs.py`

**Interfaces:**
- Produces: `COST_BP: dict[str, float]`, `BORROW_BP_PA: float = 50.0`, `TRADING_DAYS_PA: int = 252`, `COST_LADDER: tuple = (1.0, 2.0, 5.0)`, `trade_cost_fraction(weight_deltas: pd.Series, cost_bp: dict[str, float], cost_multiplier: float = 1.0) -> float`, `daily_borrow_cost_fraction(weights: pd.Series, cost_multiplier: float = 1.0, borrow_bp_pa: float = 50.0) -> float`. Rückgaben = Anteile am Portfoliowert (positiv = Kosten).

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_costs.py — prueft bp-Kostenmodell, Flip-Turnover und Borrow.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from factor_lab.costs import COST_BP, COST_LADDER, trade_cost_fraction, daily_borrow_cost_fraction


def check_trade_cost_and_flip() -> None:
    deltas = pd.Series({"SPY": 0.5, "EEM": -0.2})
    # 0.5*1.5bp + 0.2*3.0bp = 0.000075 + 0.00006
    assert abs(trade_cost_fraction(deltas, COST_BP) - 0.000135) < 1e-12
    assert abs(trade_cost_fraction(deltas, COST_BP, cost_multiplier=2.0) - 0.00027) < 1e-12

    # Pflichttest aus dem Review: Flip +1 -> -1 ist One-way-Turnover 2
    # und kostet entsprechend 2 * 1.5bp.
    flip = pd.Series({"SPY": -1.0 - 1.0})
    assert abs(flip.abs().sum() - 2.0) < 1e-12
    assert abs(trade_cost_fraction(flip, COST_BP) - 2.0 * 1.5 / 10_000.0) < 1e-12
    print("trade_cost_fraction (inkl. Flip=2): OK")


def check_borrow() -> None:
    weights = pd.Series({"TLT": -0.5, "SPY": 0.5})
    expected = 0.5 * 0.005 / 252
    assert abs(daily_borrow_cost_fraction(weights) - expected) < 1e-15
    assert daily_borrow_cost_fraction(pd.Series({"SPY": 1.0})) == 0.0
    # Borrow-Sensitivitaet ueber den Parameter, Stress ueber den Multiplier
    assert abs(daily_borrow_cost_fraction(weights, borrow_bp_pa=100.0) - 2 * expected) < 1e-15
    assert abs(daily_borrow_cost_fraction(weights, cost_multiplier=2.0) - 2 * expected) < 1e-15
    print("daily_borrow_cost_fraction: OK")


def check_tables() -> None:
    assert sorted(COST_BP) == sorted(["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"])
    assert COST_BP["SPY"] == 1.5 and COST_BP["EEM"] == 3.0
    assert COST_LADDER == (1.0, 2.0, 5.0)
    print("COST_BP / COST_LADDER: OK")


def run_consistency_check() -> None:
    check_tables()
    check_trade_cost_and_flip()
    check_borrow()
    print("\nAlle costs-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen — FAIL erwartet**
- [ ] **Step 3: Implementierung**

```python
"""
costs.py — Kostenmodell (Spec v2 Abschnitt 8). Basiskosten sind bewusst
als UNKALIBRIERTE konservative Schaetzung dokumentiert (Stand 2026-09-01,
keine Quoted-Spread-Studie) — deshalb laeuft die Kostenleiter 1x/2x/5x als
fester Bestandteil jedes Laufs. One-way-Turnover = Sum |Delta w|; ein Flip
+1 -> -1 ist Turnover 2.
"""
from __future__ import annotations

import pandas as pd

COST_BP: dict[str, float] = {
    "SPY": 1.5, "QQQ": 1.5, "IWM": 1.5, "TLT": 1.5, "IEF": 1.5, "GLD": 1.5,
    "EFA": 3.0, "EEM": 3.0, "LQD": 3.0, "SLV": 3.0, "DBC": 3.0, "VNQ": 3.0,
}
BORROW_BP_PA: float = 50.0
TRADING_DAYS_PA: int = 252
COST_LADDER: tuple = (1.0, 2.0, 5.0)


def trade_cost_fraction(weight_deltas: pd.Series, cost_bp: dict[str, float], cost_multiplier: float = 1.0) -> float:
    """Rebalance-Kosten als Anteil am Portfoliowert."""
    total = 0.0
    for symbol, delta in weight_deltas.items():
        total += abs(float(delta)) * cost_bp[symbol] / 10_000.0
    return total * cost_multiplier


def daily_borrow_cost_fraction(weights: pd.Series, cost_multiplier: float = 1.0, borrow_bp_pa: float = BORROW_BP_PA) -> float:
    """Taegliche Borrow-Kosten auf das Short-Nominal."""
    short_nominal = float(weights[weights < 0].abs().sum())
    return short_nominal * (borrow_bp_pa / 10_000.0) / TRADING_DAYS_PA * cost_multiplier
```

- [ ] **Step 4: Test ausführen — PASS erwartet**
- [ ] **Step 5: Commit**

```bash
git add factor_lab/costs.py factor_lab/tests/test_costs.py
git commit -m "feat(factor_lab): add cost model with flip-turnover test, ladder, borrow sensitivity"
```

---

### Task 3: portfolio.py Teil 1 — Vol-Schätzung + Zielgewichte

**Files:**
- Create: `factor_lab/portfolio.py`
- Test: `factor_lab/tests/test_portfolio.py`

**Interfaces:**
- Produces: `ewma_annualized_vol(returns: pd.DataFrame, span: int = 63) -> pd.DataFrame`; `rebalance_weights(signal_row: pd.Series, vol_row: pd.Series, trailing_returns: pd.DataFrame, mode: str, vol_cap: float = 0.10) -> pd.Series` (Gross der Zielgewichte ≤ 1.0; Skalierung `min(1, vol_cap / realisierte 63-Tage-Std des Kandidatenportfolios, annualisiert)`).

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_portfolio.py — Vol-Schaetzung, Zielgewichte und (ab Task 4) der
taegliche Loop mit Lag-Konvention.
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
    assert abs(vol["A"].iloc[-1] - 0.01 * np.sqrt(252)) < 0.02
    print("ewma_annualized_vol: OK")


def check_rebalance_weights() -> None:
    signal = pd.Series({"A": 1.0, "B": -1.0})
    vol = pd.Series({"A": 0.2, "B": 0.1})
    trailing = pd.DataFrame(0.0, index=range(63), columns=["A", "B"])

    # long_short: raw = [5, -10] -> base [1/3, -2/3], Gross exakt 1
    w = rebalance_weights(signal, vol, trailing, mode="long_short", vol_cap=0.10)
    assert abs(w["A"] - 1.0 / 3.0) < 1e-12 and abs(w["B"] + 2.0 / 3.0) < 1e-12
    assert abs(w.abs().sum() - 1.0) < 1e-12

    # long_flat: negatives Signal -> 0
    w = rebalance_weights(signal, vol, trailing, mode="long_flat", vol_cap=0.10)
    assert abs(w["A"] - 1.0) < 1e-12 and w["B"] == 0.0

    # Vol-Cap skaliert nur HERUNTER
    rng = np.random.default_rng(0)
    hot = pd.DataFrame({"A": rng.normal(0, 0.02, 63), "B": 0.0})
    w = rebalance_weights(pd.Series({"A": 1.0, "B": 0.0}), vol, hot, mode="long_flat", vol_cap=0.10)
    assert 0.0 < w["A"] < 1.0

    # NaN-Vol -> Gewicht 0; Alles-Null-Signal -> Cash
    w = rebalance_weights(pd.Series({"A": 1.0, "B": 1.0}), pd.Series({"A": 0.2, "B": np.nan}), trailing, mode="long_flat")
    assert w["B"] == 0.0 and abs(w["A"] - 1.0) < 1e-12
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

- [ ] **Step 2: Test ausführen — FAIL erwartet**
- [ ] **Step 3: Implementierung**

```python
"""
portfolio.py — Sizing, Lag-Konvention und taeglicher Loop (Spec v2
Abschnitte 6-7). Vol-Cap 0.10 ist eine OBERGRENZE: Gross-Cap 1.0 fuer
Zielgewichte, es wird nie gehebelt; zwischen Rebalances driften die
Gewichte (tageweises Max-Gross wird in Task 4 mitgemessen und reportet,
NICHT zwangsdeleveraged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PA = 252


def ewma_annualized_vol(returns: pd.DataFrame, span: int = 63) -> pd.DataFrame:
    """Kausale EWMA-Tagesvol (min_periods=span), annualisiert."""
    return returns.ewm(span=span, min_periods=span).std() * np.sqrt(TRADING_DAYS_PA)


def rebalance_weights(
    signal_row: pd.Series,
    vol_row: pd.Series,
    trailing_returns: pd.DataFrame,
    mode: str,
    vol_cap: float = 0.10,
) -> pd.Series:
    """Zielgewichte fuer EINEN Entscheidungszeitpunkt (Daten bis t)."""
    if mode not in ("long_short", "long_flat"):
        raise ValueError(f"Unbekannter Modus: {mode}")
    signal = signal_row.astype(float).copy()
    if mode == "long_flat":
        signal = signal.clip(lower=0.0)

    valid = vol_row.notna() & (vol_row > 0)
    raw = (signal / vol_row).where(valid, 0.0).fillna(0.0)
    gross = float(raw.abs().sum())
    if gross == 0.0:
        return pd.Series(0.0, index=signal_row.index)
    base = raw / gross  # Zielgewichte: Gross exakt 1.0

    # EINE praeregistrierte Formel (Spec v2 §6): einfache Std der mit den
    # Kandidatengewichten gewichteten letzten 63 Tagesreturns, annualisiert.
    portfolio_returns = (trailing_returns[base.index] * base).sum(axis=1)
    realized_vol = float(portfolio_returns.std()) * np.sqrt(TRADING_DAYS_PA)
    scale = min(1.0, vol_cap / realized_vol) if realized_vol > 0 else 1.0
    return base * scale
```

- [ ] **Step 4: Test ausführen — PASS erwartet**
- [ ] **Step 5: Commit**

```bash
git add factor_lab/portfolio.py factor_lab/tests/test_portfolio.py
git commit -m "feat(factor_lab): add vol estimation and target weights with vol cap"
```

---

### Task 4: portfolio.py Teil 2 — gemeinsamer täglicher Loop mit Lag-Konvention

**Files:**
- Modify: `factor_lab/portfolio.py`
- Modify: `factor_lab/tests/test_portfolio.py`

**Interfaces:**
- Consumes: Task 2 (`trade_cost_fraction`, `daily_borrow_cost_fraction`), Task 3 (`rebalance_weights`).
- Produces:
  - `month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex` — letzter Handelstag je (Jahr, Monat).
  - `run_lagged_backtest(returns: pd.DataFrame, cash_daily: pd.Series, decision_dates: pd.DatetimeIndex, weight_provider, cost_bp: dict[str, float], cost_multiplier: float = 1.0, borrow_bp_pa: float = 50.0) -> tuple[pd.Series, dict]`.
    - `weight_provider(decision_date: pd.Timestamp) -> pd.Series` liefert Zielgewichte auf Basis von Daten bis einschließlich `decision_date`.
    - Lag-Konvention: Entscheidung an t, Kosten und Positionswechsel am Handelstag t+1 (Fill zu Close(t+1)), neue Position wirkt erstmals auf den Return t+1→t+2.
    - PnL-Index = alle Handelstage NACH der ersten Entscheidung (erster PnL-Tag = erster Ausführungstag).
    - Rückgabe-Info-Dict: `"per_day"` (DataFrame mit Spalten `gross_pnl`, `cash_pnl`, `trade_cost`, `borrow_cost` auf demselben Index wie die Netto-Serie), `"total_turnover"`, `"n_rebalances"`, `"instrument_contributions"` (dict Symbol → additiver Brutto-P&L-Beitrag), `"max_daily_gross"`.
  - `trend_weight_provider(returns, signals, vols, mode, vol_cap=0.10, vol_window=63)` → Funktion für `run_lagged_backtest` (Trend UND matched_long: matched_long = Signale konstant +1, mode long_flat).
  - `fixed_mix_provider(target_weights: dict[str, float])` → Funktion (für SPY-B&H mit einer einzigen Entscheidung bzw. 60/40 monatlich).

- [ ] **Step 1: Failing Checks anhängen** (Imports erweitern: `run_lagged_backtest, month_end_dates, trend_weight_provider, fixed_mix_provider`)

```python
def check_month_end_dates() -> None:
    idx = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2020-03-02"])
    ends = month_end_dates(idx)
    assert list(ends) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-02")]
    print("month_end_dates: OK")


def _lag_fixture():
    """5 Tage, 1 Instrument. Entscheidung an d0 setzt Gewicht 1.0.
    Erwartung (Lag-Konvention): d1 = Ausfuehrungstag (nur Kosten, alte
    Position 0 verdient nichts), d2 = erster Tag MIT Position."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    returns = pd.DataFrame({"SPY": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=idx)
    cash = pd.Series(0.0, index=idx)
    return idx, returns, cash


def check_future_poison_and_lag() -> None:
    idx, returns, cash = _lag_fixture()
    provider = fixed_mix_provider({"SPY": 1.0})
    net, info = run_lagged_backtest(
        returns, cash, decision_dates=pd.DatetimeIndex([idx[0]]),
        weight_provider=provider, cost_bp={"SPY": 1.5},
    )
    # PnL-Index beginnt am Ausfuehrungstag d1
    assert net.index[0] == idx[1], f"PnL muss am Ausfuehrungstag beginnen, bekam {net.index[0]}"
    # d1: KEINE Marktposition (Future-Poison-Test: r(d1)=0.02 darf NICHT
    # verdient werden), nur Kaufkosten 1.5bp
    assert abs(net.loc[idx[1]] - (0.0 - 1.5 / 10_000.0)) < 1e-12, (
        f"Ausfuehrungstag darf r(t+1) nicht verdienen, bekam {net.loc[idx[1]]}"
    )
    # d2: erster Tag mit Position -> r(d2)=0.03
    assert abs(net.loc[idx[2]] - 0.03) < 1e-12
    assert info["n_rebalances"] == 1 and abs(info["total_turnover"] - 1.0) < 1e-12
    print("run_lagged_backtest (Future-Poison + Fill-Lag): OK")


def check_reconciliation_and_cash() -> None:
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(3)
    returns = pd.DataFrame({"SPY": rng.normal(0.0005, 0.01, 60), "TLT": rng.normal(0.0, 0.008, 60)}, index=idx)
    cash = pd.Series(0.04 / 252, index=idx)  # 4% p.a. T-Bill
    provider = fixed_mix_provider({"SPY": 0.3, "TLT": 0.3})
    decisions = pd.DatetimeIndex([idx[0], *month_end_dates(idx)])
    net, info = run_lagged_backtest(returns, cash, decisions, provider, cost_bp={"SPY": 1.5, "TLT": 1.5})

    # Pflichttest Review: vollstaendige Netto = Brutto + Cash - Kosten - Borrow, pro Tag.
    per_day = info["per_day"]
    recon = per_day["gross_pnl"] + per_day["cash_pnl"] - per_day["trade_cost"] - per_day["borrow_cost"]
    assert np.allclose(net.to_numpy(), recon.to_numpy(), atol=1e-15), "Tagesgenaue Reconciliation verletzt"
    # Cash-Gewicht max(0, 1-Gross)=0.4 verdient Zins: am 2. Tag nach Aufbau
    day2 = net.index[2]
    assert per_day.loc[day2, "cash_pnl"] > 0
    print("run_lagged_backtest (Reconciliation + verzinstes Cash): OK")


def check_gross_drift_reporting() -> None:
    # +0.5/-0.5, beide Underlyings +10% an einem Tag -> Gross drifted auf 1.1.
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    returns = pd.DataFrame({"SPY": [0.0, 0.0, 0.10, 0.0], "TLT": [0.0, 0.0, -0.10, 0.0]}, index=idx)
    cash = pd.Series(0.0, index=idx)
    provider = fixed_mix_provider({"SPY": 0.5, "TLT": -0.5})
    net, info = run_lagged_backtest(returns, cash, pd.DatetimeIndex([idx[0]]), provider,
                                    cost_bp={"SPY": 1.5, "TLT": 1.5})
    # SPY: 0.5*1.1=0.55; TLT: -0.5*0.9=-0.45 -> bei Netto-PnL 0.5*0.1+(-0.5)*(-0.1)=0.1:
    # Gewichte /1.1 -> 0.5 & -0.409 -> Gross 0.909? Nein: Drift teilt durch (1+gross_pnl).
    # gross_pnl = 0.10 -> w_SPY = 0.55/1.1 = 0.5, w_TLT = -0.45/1.1 = -0.409, Gross 0.909.
    # Der REVIEW-Fall (beide +10%) braucht gleiche Vorzeichen der Returns:
    returns2 = pd.DataFrame({"SPY": [0.0, 0.0, 0.10, 0.0], "TLT": [0.0, 0.0, 0.10, 0.0]}, index=idx)
    net2, info2 = run_lagged_backtest(returns2, cash, pd.DatetimeIndex([idx[0]]), provider,
                                      cost_bp={"SPY": 1.5, "TLT": 1.5})
    # gross_pnl = 0.5*0.1 - 0.5*0.1 = 0 -> w = 0.55/-0.55 -> Gross 1.10
    assert info2["max_daily_gross"] > 1.09, f"Gross-Drift muss gemessen werden, bekam {info2['max_daily_gross']}"
    print("run_lagged_backtest (Max-Gross-Reporting): OK")
```

Und in `run_consistency_check()` ergänzen: `check_month_end_dates()`, `check_future_poison_and_lag()`, `check_reconciliation_and_cash()`, `check_gross_drift_reporting()`.

- [ ] **Step 2: Test ausführen — FAIL erwartet** (ImportError)
- [ ] **Step 3: Implementierung anhängen**

```python
def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Letzter vorhandener Handelstag je (Jahr, Monat)."""
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.groupby([index.year, index.month]).max().sort_values().values)


def trend_weight_provider(returns, signals, vols, mode, vol_cap: float = 0.10, vol_window: int = 63):
    """Weight-Provider fuer Trend UND matched_long (Signale konstant +1):
    nutzt ausschliesslich Daten bis einschliesslich decision_date."""
    def provider(decision_date):
        trailing = returns.loc[:decision_date].tail(vol_window)
        return rebalance_weights(
            signals.loc[decision_date], vols.loc[decision_date], trailing, mode, vol_cap,
        )
    return provider


def fixed_mix_provider(target_weights: dict[str, float]):
    """Weight-Provider fuer feste Zielgewichte (SPY-B&H, 60/40)."""
    target = pd.Series(target_weights, dtype=float)
    def provider(decision_date):
        return target.copy()
    return provider


def run_lagged_backtest(
    returns: pd.DataFrame,
    cash_daily: pd.Series,
    decision_dates: pd.DatetimeIndex,
    weight_provider,
    cost_bp: dict[str, float],
    cost_multiplier: float = 1.0,
    borrow_bp_pa: float = 50.0,
) -> tuple[pd.Series, dict]:
    """Gemeinsamer taeglicher Loop fuer Strategie und ALLE Benchmarks
    (identisches Entry-Timing per Konstruktion, Spec v2 §6/§9):

    Tag s: (1) gestrige Gewichte verdienen r_s, Cash `max(0, 1-Gross)`
    verdient cash_daily_s, Borrow auf gestrige Shorts; (2) Drift;
    (3) wenn s Ausfuehrungstag (= Handelstag nach einer Entscheidung):
    Zielgewichte der GESTRIGEN Entscheidung werden gegen die gedrifteten
    Gewichte gehandelt (Kosten heute), wirken ab morgen. Kein Wert, der in
    die Gewichte eingeht, ist zum Fill unbekannt (Future-Poison-Test)."""
    from factor_lab.costs import trade_cost_fraction, daily_borrow_cost_fraction

    symbols = list(returns.columns)
    index = returns.index
    decisions = set(decision_dates)
    first_decision_pos = index.get_loc(decision_dates.min())
    weights = pd.Series(0.0, index=symbols)
    pending_target = None

    rows, out_index = [], []
    contributions = {s: 0.0 for s in symbols}
    total_turnover = 0.0
    n_rebalances = 0
    max_daily_gross = 0.0

    for pos in range(first_decision_pos, len(index)):
        t = index[pos]
        if pos > first_decision_pos:
            r_t = returns.iloc[pos]
            gross_pnl = float((weights * r_t).sum())
            for s in symbols:
                contributions[s] += float(weights[s] * r_t[s])
            cash_weight = max(0.0, 1.0 - float(weights.abs().sum()))
            cash_pnl = cash_weight * float(cash_daily.loc[t])
            borrow = daily_borrow_cost_fraction(weights, cost_multiplier, borrow_bp_pa)

            equity_growth = 1.0 + gross_pnl + cash_pnl - borrow
            if equity_growth > 0:
                weights = weights * (1.0 + r_t) / equity_growth
            max_daily_gross = max(max_daily_gross, float(weights.abs().sum()))

            trade_cost = 0.0
            if pending_target is not None:
                deltas = pending_target - weights
                trade_cost = trade_cost_fraction(deltas, cost_bp, cost_multiplier)
                total_turnover += float(deltas.abs().sum())
                n_rebalances += 1
                weights = pending_target
                pending_target = None

            rows.append((gross_pnl, cash_pnl, trade_cost, borrow))
            out_index.append(t)

        if t in decisions:
            pending_target = weight_provider(t)

    per_day = pd.DataFrame(rows, index=pd.DatetimeIndex(out_index),
                           columns=["gross_pnl", "cash_pnl", "trade_cost", "borrow_cost"])
    net = per_day["gross_pnl"] + per_day["cash_pnl"] - per_day["trade_cost"] - per_day["borrow_cost"]
    info = {
        "per_day": per_day,
        "total_turnover": total_turnover,
        "n_rebalances": n_rebalances,
        "instrument_contributions": contributions,
        "max_daily_gross": max_daily_gross,
    }
    return net, info
```

- [ ] **Step 4: Test ausführen — PASS erwartet.** Hinweis für den Implementierer: der Future-Poison-Test schlägt fehl, wenn Ausführungstag und Wirkungstag vertauscht sind — genau dafür ist er da.
- [ ] **Step 5: Commit**

```bash
git add factor_lab/portfolio.py factor_lab/tests/test_portfolio.py
git commit -m "feat(factor_lab): add shared lagged daily loop with reconciliation, cash interest, gross reporting"
```

---

### Task 5: stats.py Teil 1 — Monatsaggregation + Stationary-Block-Bootstrap + Permutation

**Files:**
- Create: `factor_lab/stats.py`
- Test: `factor_lab/tests/test_stats.py`

**Interfaces:**
- Consumes: `compound_return`, `max_drawdown_from_returns` aus `market_control_system/controller/cross_sectional_signal_metrics.py` (via sys.path; der DD-Helper ist auf diesem Branch gefixt).
- Produces:
  - `monthly_log_returns(net: pd.Series) -> pd.Series` — Summe `log(1+r)` je Kalendermonat, Index = Monatsanfang.
  - `stationary_bootstrap_indices(n: int, expected_block_len: float, rng: np.random.Generator) -> np.ndarray` — Politis/Romano-Indizes (Länge n, Wrap-around, geometrische Blocklängen).
  - `stationary_block_bootstrap(monthly_values: np.ndarray, expected_block_len: float = 6.0, n_boot: int = 10_000, seed: int = 0) -> dict` mit Schlüsseln `"mean_monthly"`, `"ann_geom"`, `"ann_geom_lower_1s95"` (5. Perzentil der annualisierten Bootstrap-Verteilung, einseitige 95%-Untergrenze), `"ci_low_95"`, `"ci_high_95"` (zweiseitig, annualisiert), `"p_leq_zero"` = `(Anzahl Bootstrap-ann <= 0 + 1) / (B + 1)`, `"n_months"`, `"n_boot"`, `"expected_block_len"`. Annualisierung: `exp(12 * mean_monthly_log) - 1`.
  - `monthly_sign_flip_pvalue(monthly_values: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> dict` mit `"p_two_sided"`, `"p_greater_zero"` — beide `(extreme+1)/(B+1)` — `"n_months"`, `"n_perm"`.

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_stats.py — Monatsaggregation, Stationary-Block-Bootstrap, Permutation
(ab Task 6 auch Kennzahlen/Gates).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.stats import (
    monthly_log_returns, stationary_bootstrap_indices, stationary_block_bootstrap,
    monthly_sign_flip_pvalue,
)


def check_monthly_log_returns() -> None:
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-02-03"])
    net = pd.Series([0.01, 0.01, -0.02], index=idx)
    monthly = monthly_log_returns(net)
    assert len(monthly) == 2
    assert abs(monthly.iloc[0] - 2 * np.log(1.01)) < 1e-12
    assert abs(monthly.iloc[1] - np.log(0.98)) < 1e-12
    print("monthly_log_returns: OK")


def check_stationary_indices() -> None:
    rng = np.random.default_rng(0)
    # Riesige erwartete Blocklaenge -> Indizes praktisch durchgehend
    # zusammenhaengend (mod n). Genau das unterscheidet den Stationary-
    # Bootstrap vom unabhaengigen Monats-Resampling aus Plan v1.
    idx = stationary_bootstrap_indices(n=10, expected_block_len=1000.0, rng=rng)
    diffs = (np.diff(idx) - 1) % 10
    assert (diffs == 0).sum() >= 8, f"Erwartete fast nur zusammenhaengende Schritte, bekam {idx}"

    # Blocklaenge 1 -> im Mittel viele Neustarts (nicht fast alle Schritte +1)
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_indices(n=200, expected_block_len=1.0, rng=rng)
    contiguous = ((np.diff(idx) - 1) % 200 == 0).mean()
    assert contiguous < 0.5, f"Blocklaenge 1 darf nicht ueberwiegend zusammenhaengen ({contiguous})"
    assert len(idx) == 200 and idx.min() >= 0 and idx.max() < 200
    print("stationary_bootstrap_indices: OK")


def check_stationary_bootstrap() -> None:
    rng = np.random.default_rng(2)
    strong = rng.normal(0.01, 0.002, 120)  # 10 Jahre stark positiver Monats-Log-Ertrag
    result = stationary_block_bootstrap(strong, expected_block_len=6.0, n_boot=500, seed=0)
    assert result["ann_geom_lower_1s95"] > 0, "Einseitige Untergrenze muss bei starkem Signal > 0 sein"
    assert result["p_leq_zero"] <= (0 + 1) / (500 + 1) + 1e-12, "p muss (extreme+1)/(B+1) sein, minimal 1/(B+1)"
    assert abs(result["ann_geom"] - (np.exp(12 * strong.mean()) - 1)) < 1e-9
    again = stationary_block_bootstrap(strong, expected_block_len=6.0, n_boot=500, seed=0)
    assert result["ann_geom_lower_1s95"] == again["ann_geom_lower_1s95"], "Gleicher Seed -> identisch"

    noise = rng.normal(0.0, 0.02, 120)
    weak = stationary_block_bootstrap(noise, expected_block_len=6.0, n_boot=500, seed=0)
    assert weak["ci_low_95"] < 0 < weak["ci_high_95"], "Rauschen: CI muss 0 einschliessen"
    print("stationary_block_bootstrap: OK")


def check_sign_flip() -> None:
    strong = np.full(60, 0.01)
    p = monthly_sign_flip_pvalue(strong, n_perm=500, seed=0)
    assert p["p_two_sided"] <= 2 * (0 + 1) / (500 + 1), f"Konstant positives Signal: p minimal, bekam {p}"
    rng = np.random.default_rng(4)
    p = monthly_sign_flip_pvalue(rng.normal(0.0, 0.01, 60), n_perm=500, seed=0)
    assert p["p_two_sided"] > 0.05
    print("monthly_sign_flip_pvalue: OK")


def run_consistency_check() -> None:
    check_monthly_log_returns()
    check_stationary_indices()
    check_stationary_bootstrap()
    check_sign_flip()
    print("\nAlle stats-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen — FAIL erwartet**
- [ ] **Step 3: Implementierung**

```python
"""
stats.py — Monatsaggregation, Stationary-Block-Bootstrap (Politis/Romano),
Sign-Flip-Permutation, Kennzahlen und Gates (Spec v2 Abschnitte 9-10).

Warum Stationary-Block statt unabhaengiger Monats-Bloecke: 63-252-Tage-
Signale erzeugen Persistenz UEBER Monatsgrenzen; unabhaengiges Resampling
einzelner Monate zerstoert genau diese Struktur. Erwartete Blocklaenge 6
Monate ist praeregistriert primaer (3/12 nur Sensitivitaet).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market_control_system", "controller"))

import numpy as np
import pandas as pd

from cross_sectional_signal_metrics import compound_return, max_drawdown_from_returns

TRADING_DAYS_PA = 252
MONTHS_PA = 12


def monthly_log_returns(net: pd.Series) -> pd.Series:
    """Summe log(1+r) je Kalendermonat; Index = Monatsanfang."""
    log_r = np.log1p(net.astype(float))
    grouped = log_r.groupby(net.index.to_period("M")).sum()
    grouped.index = grouped.index.to_timestamp()
    return grouped


def stationary_bootstrap_indices(n: int, expected_block_len: float, rng: np.random.Generator) -> np.ndarray:
    """Politis/Romano: nach jedem Schritt mit Wahrscheinlichkeit 1/L neu
    starten (uniformer Startindex), sonst zum Nachfolger (Wrap-around)."""
    p_restart = 1.0 / expected_block_len
    indices = np.empty(n, dtype=int)
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p_restart:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def _annualize_monthly_log(mean_monthly_log: float) -> float:
    return float(np.exp(MONTHS_PA * mean_monthly_log) - 1.0)


def stationary_block_bootstrap(
    monthly_values: np.ndarray, expected_block_len: float = 6.0, n_boot: int = 10_000, seed: int = 0,
) -> dict:
    """Bootstrap-Verteilung des annualisierten geometrischen Ertrags aus
    monatlichen Log-Ertraegen. Fester Seed -> identische Ziehungsfolge fuer
    alle Varianten (Spec v2 §9)."""
    values = np.asarray(monthly_values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    ann = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, expected_block_len, rng)
        ann[b] = _annualize_monthly_log(float(values[idx].mean()))
    return {
        "mean_monthly": float(values.mean()),
        "ann_geom": _annualize_monthly_log(float(values.mean())),
        "ann_geom_lower_1s95": float(np.percentile(ann, 5.0)),
        "ci_low_95": float(np.percentile(ann, 2.5)),
        "ci_high_95": float(np.percentile(ann, 97.5)),
        "p_leq_zero": float(((ann <= 0).sum() + 1) / (n_boot + 1)),
        "n_months": n,
        "n_boot": n_boot,
        "expected_block_len": expected_block_len,
    }


def monthly_sign_flip_pvalue(monthly_values: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> dict:
    """Sign-Flip-Permutation auf Monatsebene, p = (extreme+1)/(B+1)."""
    values = np.asarray(monthly_values, dtype=float)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(values)))
    perm_means = (signs * values).mean(axis=1)
    return {
        "p_two_sided": float(((np.abs(perm_means) >= abs(observed)).sum() + 1) / (n_perm + 1)),
        "p_greater_zero": float(((perm_means >= observed).sum() + 1) / (n_perm + 1)),
        "n_months": len(values),
        "n_perm": n_perm,
    }
```

- [ ] **Step 4: Test ausführen — PASS erwartet**
- [ ] **Step 5: Commit**

```bash
git add factor_lab/stats.py factor_lab/tests/test_stats.py
git commit -m "feat(factor_lab): add stationary block bootstrap and monthly permutation inference"
```

---

### Task 6: stats.py Teil 2 — Kennzahlen, Gates, Verdikt

**Files:**
- Modify: `factor_lab/stats.py`
- Modify: `factor_lab/tests/test_stats.py`

**Interfaces:**
- Produces:
  - `annualized_stats(net: pd.Series, cash_daily: pd.Series | None = None) -> dict` mit `"cagr"`, `"vol_pa"`, `"sharpe"`, `"excess_sharpe"` (über Cash, NaN wenn cash_daily fehlt), `"max_drawdown"` (inkl. Start-Peak — Pflicht-Assert `[-0.10] → −0.10` im Test), `"n_days"`.
  - `full_year_excess(excess_daily: pd.Series) -> dict[int, float]` — compoundierte Jahres-Mehrerträge NUR vollständiger Kalenderjahre (Jahr zählt als vollständig, wenn es Handelstage in Januar UND Dezember enthält).
  - `evaluate_screening_gates(excess_boot: dict, dd_base: float, dd_stress: float, stressed_cagr: float, yearly_excess: dict[int, float], loo_excess_compounds: dict[str, float], dd_cap: float = 0.15, gate_c_floor: float = 0.02) -> dict` mit `"gate_a_excess_ci"`, `"gate_b_drawdown"`, `"gate_c_stressed_floor"`, `"gate_d_no_single_driver"`, `"passed_all"`. Gate d = (bestes vollständiges Jahr entfernt → Rest-Compound > 0) UND (jeder Eintrag in `loo_excess_compounds` > 0; leeres Dict → False).
  - `evaluate_holdout_gates(excess_boot: dict, dd_base: float, dd_stress: float, stressed_cagr: float, dd_cap: float = 0.15, gate_c_floor: float = 0.02) -> dict` — nur A/B/C (Spec §10 Punkt 4).
  - `screening_verdict(gates_by_variant: dict[str, dict], excess_sharpe_by_variant: dict[str, float]) -> tuple[str, str | None]` — (Verdikt-Text, Kandidatin oder None); Auswahl: höchster Excess-Sharpe unter Bestehenden, Tie-Break alphabetisch erster Name.

- [ ] **Step 1: Failing Checks anhängen** (Imports erweitern)

```python
from factor_lab.stats import (
    annualized_stats, full_year_excess, evaluate_screening_gates,
    evaluate_holdout_gates, screening_verdict,
)


def check_annualized_stats() -> None:
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    stats = annualized_stats(pd.Series([0.001] * 252, index=idx))
    assert abs(stats["cagr"] - (1.001 ** 252 - 1)) < 1e-9
    assert stats["max_drawdown"] == 0.0
    # Pflichttest Review: Start-Equity zaehlt als Peak.
    dd = annualized_stats(pd.Series([-0.10], index=idx[:1]))["max_drawdown"]
    assert abs(dd - (-0.10)) < 1e-12, f"[-0.10] muss -10% Drawdown ergeben, bekam {dd}"
    print("annualized_stats: OK")


def check_full_year_excess() -> None:
    idx = (list(pd.date_range("2019-11-01", "2019-12-31", freq="B"))
           + list(pd.date_range("2020-01-02", "2020-12-30", freq="B"))
           + list(pd.date_range("2021-01-04", "2021-03-31", freq="B")))
    excess = pd.Series(0.001, index=pd.DatetimeIndex(idx))
    years = full_year_excess(excess)
    assert list(years) == [2020], f"Nur 2020 ist vollstaendig (Jan+Dez), bekam {list(years)}"
    print("full_year_excess: OK")


def check_gates_and_verdict() -> None:
    good_boot = {"ann_geom_lower_1s95": 0.005, "ann_geom": 0.03}
    yearly = {2020: 0.02, 2021: 0.03, 2022: 0.01}
    loo = {"loo_SPY": 0.01, "loo_sleeve_bonds": 0.02}
    gates = evaluate_screening_gates(good_boot, dd_base=-0.10, dd_stress=-0.12,
                                     stressed_cagr=0.03, yearly_excess=yearly,
                                     loo_excess_compounds=loo)
    assert gates["passed_all"], f"Muesste bestehen: {gates}"

    # Gate A: Untergrenze <= 0 -> fail
    assert not evaluate_screening_gates({**good_boot, "ann_geom_lower_1s95": -0.001},
                                        -0.10, -0.12, 0.03, yearly, loo)["gate_a_excess_ci"]
    # Gate B: Stress-DD -20% -> fail
    assert not evaluate_screening_gates(good_boot, -0.10, -0.20, 0.03, yearly, loo)["gate_b_drawdown"]
    # Gate C: 1.5% < 2%-Floor -> fail (nicht bloss > 0!)
    assert not evaluate_screening_gates(good_boot, -0.10, -0.12, 0.015, yearly, loo)["gate_c_stressed_floor"]
    # Gate D: ein LOO-Rerun negativ -> fail
    assert not evaluate_screening_gates(good_boot, -0.10, -0.12, 0.03, yearly,
                                        {**loo, "loo_GLD": -0.001})["gate_d_no_single_driver"]
    # Gate D: bestes Jahr traegt alles -> fail
    assert not evaluate_screening_gates(good_boot, -0.10, -0.12, 0.03,
                                        {2020: 0.50, 2021: -0.01, 2022: -0.01}, loo)["gate_d_no_single_driver"]

    hold = evaluate_holdout_gates(good_boot, -0.10, -0.12, 0.03)
    assert hold["passed_all"] and "gate_d_no_single_driver" not in hold

    verdict, candidate = screening_verdict(
        {"combo_long_flat": gates, "mom63_long_flat": gates},
        {"combo_long_flat": 0.5, "mom63_long_flat": 0.5},
    )
    assert candidate == "combo_long_flat", f"Tie-Break alphabetisch, bekam {candidate}"
    verdict, candidate = screening_verdict({"mom63_long_flat": {**gates, "passed_all": False}},
                                           {"mom63_long_flat": 0.5})
    assert candidate is None and "keine" in verdict.lower()
    print("Gates + Verdikt: OK")
```

Und in `run_consistency_check()` ergänzen: `check_annualized_stats()`, `check_full_year_excess()`, `check_gates_and_verdict()`.

- [ ] **Step 2: Test ausführen — FAIL erwartet**
- [ ] **Step 3: Implementierung anhängen**

```python
def annualized_stats(net: pd.Series, cash_daily: pd.Series | None = None) -> dict:
    """Multiplikative Kennzahlen; Drawdown inkl. Start-Equity 1.0 (der
    Helper in market_control_system ist entsprechend gefixt)."""
    r = net.astype(float)
    n = len(r)
    total = compound_return(r.tolist())
    years = n / TRADING_DAYS_PA
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    vol_pa = float(r.std()) * np.sqrt(TRADING_DAYS_PA) if n > 1 else 0.0
    sharpe = (float(r.mean()) * TRADING_DAYS_PA) / vol_pa if vol_pa > 0 else 0.0
    if cash_daily is not None:
        excess = r - cash_daily.loc[r.index]
        ex_vol = float(excess.std()) * np.sqrt(TRADING_DAYS_PA)
        excess_sharpe = (float(excess.mean()) * TRADING_DAYS_PA) / ex_vol if ex_vol > 0 else 0.0
    else:
        excess_sharpe = float("nan")
    return {
        "cagr": float(cagr),
        "vol_pa": vol_pa,
        "sharpe": float(sharpe),
        "excess_sharpe": float(excess_sharpe),
        "max_drawdown": max_drawdown_from_returns(r.tolist()),
        "n_days": n,
    }


def full_year_excess(excess_daily: pd.Series) -> dict[int, float]:
    """Compoundierte Jahres-Mehrertraege NUR vollstaendiger Kalenderjahre
    (Handelstage in Januar UND Dezember vorhanden) — unvollstaendige
    Randjahre verzerren das Bestes-Jahr-Gate sonst (Review-Punkt)."""
    out = {}
    for year, group in excess_daily.groupby(excess_daily.index.year):
        months = set(group.index.month)
        if 1 in months and 12 in months:
            out[int(year)] = compound_return(group.tolist())
    return out


def evaluate_screening_gates(
    excess_boot: dict,
    dd_base: float,
    dd_stress: float,
    stressed_cagr: float,
    yearly_excess: dict[int, float],
    loo_excess_compounds: dict[str, float],
    dd_cap: float = 0.15,
    gate_c_floor: float = 0.02,
) -> dict:
    gate_a = excess_boot["ann_geom_lower_1s95"] > 0
    gate_b = abs(dd_base) <= dd_cap and abs(dd_stress) <= dd_cap
    gate_c = stressed_cagr >= gate_c_floor
    if yearly_excess:
        best = max(yearly_excess, key=yearly_excess.get)
        rest = [v for k, v in yearly_excess.items() if k != best]
        years_ok = compound_return(rest) > 0 if rest else False
    else:
        years_ok = False
    loo_ok = bool(loo_excess_compounds) and all(v > 0 for v in loo_excess_compounds.values())
    gate_d = years_ok and loo_ok
    return {
        "gate_a_excess_ci": bool(gate_a),
        "gate_b_drawdown": bool(gate_b),
        "gate_c_stressed_floor": bool(gate_c),
        "gate_d_no_single_driver": bool(gate_d),
        "passed_all": bool(gate_a and gate_b and gate_c and gate_d),
    }


def evaluate_holdout_gates(
    excess_boot: dict, dd_base: float, dd_stress: float, stressed_cagr: float,
    dd_cap: float = 0.15, gate_c_floor: float = 0.02,
) -> dict:
    gate_a = excess_boot["ann_geom_lower_1s95"] > 0
    gate_b = abs(dd_base) <= dd_cap and abs(dd_stress) <= dd_cap
    gate_c = stressed_cagr >= gate_c_floor
    return {
        "gate_a_excess_ci": bool(gate_a),
        "gate_b_drawdown": bool(gate_b),
        "gate_c_stressed_floor": bool(gate_c),
        "passed_all": bool(gate_a and gate_b and gate_c),
    }


def screening_verdict(gates_by_variant: dict[str, dict], excess_sharpe_by_variant: dict[str, float]) -> tuple[str, str | None]:
    """Screening-Verdikt + versiegelbare Kandidatin. Tie-Break bei gleichem
    Excess-Sharpe: alphabetisch erster Variantenname (praeregistriert)."""
    passed = sorted(n for n, g in gates_by_variant.items() if g["passed_all"])
    if not passed:
        return (
            "VERDICT: keine Variante besteht alle Screening-Gates -- das Holdout wird "
            "NICHT angefasst; Ergebnis als Nullbefund der Familie dokumentieren. "
            "(Screening ist ausdruecklich KEINE Bestaetigung.)",
            None,
        )
    best_sharpe = max(excess_sharpe_by_variant[n] for n in passed)
    candidate = sorted(n for n in passed if excess_sharpe_by_variant[n] == best_sharpe)[0]
    return (
        f"VERDICT: {len(passed)} Variante(n) bestehen das Screening ({', '.join(passed)}). "
        f"Versiegelte Holdout-Kandidatin: {candidate}. Screening ist KEINE Bestaetigung -- "
        f"bestaetigen kann nur der eine Holdout-Lauf (run_trend_holdout.py, liest candidate.json).",
        candidate,
    )
```

- [ ] **Step 4: Test ausführen — PASS erwartet**
- [ ] **Step 5: Commit**

```bash
git add factor_lab/stats.py factor_lab/tests/test_stats.py
git commit -m "feat(factor_lab): add annualized stats, screening/holdout gates, verdict with tie-break"
```

---

### Task 7: registration.py — Config-Hash, Kandidaten-Siegel, Tombstone

**Files:**
- Create: `factor_lab/registration.py`
- Test: `factor_lab/tests/test_registration.py`

**Interfaces:**
- Produces:
  - `FAMILY = "trend-etf-v1"`, `REGISTRATION: dict` (alle präregistrierten Parameter aus den Global Constraints, als ein normalisierbares Dict).
  - `config_hash(registration: dict = REGISTRATION) -> str` — SHA256 über `json.dumps(registration, sort_keys=True)`.
  - `write_candidate(path: str, variant: str, snapshot_sha256: str, git_sha: str, dev_end: str, results_sha256: str) -> None` — schreibt candidate.json inkl. `family`, `config_hash`.
  - `read_and_verify_candidate(path: str, snapshot_sha256: str) -> dict` — liest, prüft `family`, `config_hash` (gegen aktuelles REGISTRATION) und `snapshot_sha256`; ValueError bei Abweichung.
  - `tombstone_path(logs_dir: str) -> str`, `assert_no_tombstone(logs_dir: str) -> None` (ValueError, wenn vorhanden), `write_tombstone(logs_dir: str, holdout_result_path: str) -> None`.
  - `file_sha256(path: str) -> str`.

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_registration.py — Config-Hash-Stabilitaet, Kandidaten-Siegel und
Tombstone-Einmaligkeit.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile

from factor_lab import registration


def run_consistency_check() -> None:
    h1 = registration.config_hash()
    h2 = registration.config_hash()
    assert h1 == h2 and len(h1) == 64, "Config-Hash muss deterministisch sein"
    changed = dict(registration.REGISTRATION)
    changed["gate_c_floor"] = 0.03
    assert registration.config_hash(changed) != h1, "Parameteraenderung muss den Hash aendern"

    with tempfile.TemporaryDirectory() as tmp:
        cand = os.path.join(tmp, "candidate.json")
        registration.write_candidate(cand, variant="combo_long_flat", snapshot_sha256="abc",
                                     git_sha="deadbeef", dev_end="2022-10-31", results_sha256="123")
        loaded = registration.read_and_verify_candidate(cand, snapshot_sha256="abc")
        assert loaded["variant"] == "combo_long_flat" and loaded["family"] == registration.FAMILY

        # Falscher Snapshot-Hash -> Verweigerung
        try:
            registration.read_and_verify_candidate(cand, snapshot_sha256="anders")
            raise AssertionError("Falscher Snapshot-Hash muss ValueError ausloesen")
        except ValueError:
            pass

        # Manipulierte Datei (anderer config_hash) -> Verweigerung
        with open(cand, encoding="utf-8") as f:
            payload = json.load(f)
        payload["config_hash"] = "0" * 64
        with open(cand, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        try:
            registration.read_and_verify_candidate(cand, snapshot_sha256="abc")
            raise AssertionError("Manipulierter config_hash muss ValueError ausloesen")
        except ValueError:
            pass

        # Tombstone: vorher ok, nachher Verweigerung
        registration.assert_no_tombstone(tmp)
        registration.write_tombstone(tmp, holdout_result_path="ergebnis.json")
        try:
            registration.assert_no_tombstone(tmp)
            raise AssertionError("Existierender Tombstone muss ValueError ausloesen")
        except ValueError:
            pass
    print("registration: Config-Hash, Siegel, Tombstone -- OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen — FAIL erwartet**
- [ ] **Step 3: Implementierung**

```python
"""
registration.py — operationale Praeregistrierung der Familie trend-etf-v1
(Spec v2 Abschnitte 10+12): Config-Hash, unveraenderliches candidate.json,
Tombstone-Einmaligkeit fuer das Holdout.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

FAMILY = "trend-etf-v1"

REGISTRATION: dict = {
    "family": FAMILY,
    "universe": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"],
    "cash_series": "^IRX",
    "snapshot_start": "2007-01-01",
    "snapshot_end_exclusive": "2026-09-01",
    "dev_end_rule": "letzter Monatsultimo <= 80%-Quantil-Datum des gemeinsamen Kalenders",
    "warmup_days": 252 + 63,
    "lookbacks": [63, 126, 252],
    "signals": ["mom63", "mom126", "mom252", "combo"],
    "modes": ["long_short", "long_flat"],
    "ewma_span": 63,
    "vol_cap": 0.10,
    "gross_cap_targets_only": True,
    "rebalance": "monatsultimo_entscheid_fill_naechster_close",
    "cost_bp": {"SPY": 1.5, "QQQ": 1.5, "IWM": 1.5, "TLT": 1.5, "IEF": 1.5, "GLD": 1.5,
                "EFA": 3.0, "EEM": 3.0, "LQD": 3.0, "SLV": 3.0, "DBC": 3.0, "VNQ": 3.0},
    "cost_ladder": [1.0, 2.0, 5.0],
    "borrow_bp_pa": 50.0,
    "bootstrap": {"kind": "stationary", "expected_block_len_months": 6.0,
                  "sensitivity_block_lens": [3.0, 12.0], "n_boot": 10000, "seed": 0},
    "permutation_n": 10000,
    "dd_cap": 0.15,
    "gate_c_floor": 0.02,
    "candidate_rule": "hoechster Excess-Sharpe unter Bestehenden, Tie-Break alphabetisch",
    "holdout_gates": ["A", "B", "C"],
}


def config_hash(registration: dict = REGISTRATION) -> str:
    canonical = json.dumps(registration, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def write_candidate(path: str, variant: str, snapshot_sha256: str, git_sha: str,
                    dev_end: str, results_sha256: str) -> None:
    payload = {
        "family": FAMILY,
        "config_hash": config_hash(),
        "variant": variant,
        "snapshot_sha256": snapshot_sha256,
        "git_sha": git_sha,
        "dev_end": dev_end,
        "results_sha256": results_sha256,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_and_verify_candidate(path: str, snapshot_sha256: str) -> dict:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("family") != FAMILY:
        raise ValueError(f"candidate.json gehoert zu Familie {payload.get('family')}, erwartet {FAMILY}")
    if payload.get("config_hash") != config_hash():
        raise ValueError("config_hash weicht ab -- Registrierung wurde nach dem Siegel geaendert (Amendment-Regel: neue Familie noetig)")
    if payload.get("snapshot_sha256") != snapshot_sha256:
        raise ValueError("Snapshot-Hash weicht ab -- das ist nicht der versiegelte Datensatz")
    return payload


def tombstone_path(logs_dir: str) -> str:
    return os.path.join(logs_dir, f"holdout_tombstone_{FAMILY}.json")


def assert_no_tombstone(logs_dir: str) -> None:
    path = tombstone_path(logs_dir)
    if os.path.exists(path):
        raise ValueError(f"Holdout der Familie {FAMILY} wurde bereits ausgefuehrt ({path}) -- kein zweiter Zugriff")


def write_tombstone(logs_dir: str, holdout_result_path: str) -> None:
    with open(tombstone_path(logs_dir), "w", encoding="utf-8") as f:
        json.dump({"family": FAMILY, "holdout_result": holdout_result_path,
                   "executed_utc": datetime.now(timezone.utc).isoformat()}, f, indent=2)
```

- [ ] **Step 4: Test ausführen — PASS erwartet**
- [ ] **Step 5: Commit**

```bash
git add factor_lab/registration.py factor_lab/tests/test_registration.py
git commit -m "feat(factor_lab): add pre-registration config hash, candidate seal, holdout tombstone"
```

---

### Task 8: build_trend_snapshot.py + data_snapshot.py (versiegelter Build, Load-only mit Hash-Verifikation)

**Files:**
- Create: `factor_lab/build_trend_snapshot.py`, `factor_lab/data_snapshot.py`
- Test: `factor_lab/tests/test_data_snapshot.py`
- Modify: `.gitignore` (`factor_lab/logs/`, `factor_lab/data_snapshots/`); yfinance in die requirements-Datei des Repos eintragen (dort, wo die bestehenden Dependencies stehen) und `py -3.12 -m pip install yfinance`

**Interfaces:**
- Consumes: `snapshot_content_sha256`, `write_snapshot_manifest` aus `market_control_system/data_layer/frozen_snapshot.py`; `REGISTRATION` aus Task 7.
- Produces (`data_snapshot.py`): `SNAPSHOT_DIR`, `trend_snapshot_path() -> str` (Parameter-Hash über Universum + `^IRX` + Start + End), `sanity_check_snapshot(dfs: dict[str, pd.DataFrame]) -> None` (fail-closed: Duplikate, NaN, nichtpositive Preise bei ETFs, Serien-Ende, gemeinsamer Kalender ≥ 4800 Tage; `^IRX`-Werte in [−1, 25]), `load_trend_snapshot() -> dict[str, pd.DataFrame]` (lädt Pickle, verifiziert Content-SHA256 gegen `<pfad>.manifest.json`, ValueError bei Abweichung; FETCHT NIE). Schema: je ETF Spalte `"price"`; `^IRX` unter Key `"IRX"` mit Spalte `"rate_pa_pct"`.
- Produces (`build_trend_snapshot.py`): `fetch_all() -> dict`, `main()` — fetcht, prüft, friert ein, schreibt Manifest, druckt Content-Hash + den nach der DEV_END-Regel aufgelösten Termin; läuft als EIGENER Schritt vor jedem Screening.

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_data_snapshot.py — Load-only-Verhalten: Hash-Verifikation gegen das
Manifest, Manipulation -> Fehler, Sanity-Checks fail-closed. Kein Netzwerk.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pickle
import tempfile
import pandas as pd

from factor_lab import data_snapshot


def _fake_dfs():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    dfs = {s: pd.DataFrame({"price": [100.0 + i for i in range(5)]}, index=idx) for s in ["AAA", "BBB"]}
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [4.0] * 5}, index=idx)
    return dfs


def run_consistency_check() -> None:
    dfs = _fake_dfs()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trend_snapshot_test.pkl")
        with open(path, "wb") as f:
            pickle.dump(dfs, f)
        data_snapshot.write_snapshot_manifest(dfs, path + ".manifest.json")

        loaded = data_snapshot.load_trend_snapshot(path=path)
        pd.testing.assert_frame_equal(loaded["AAA"], dfs["AAA"])

        # Pflichttest Review: Manipulation des Pickles -> Hashfehler beim Load
        dfs_tampered = _fake_dfs()
        dfs_tampered["AAA"].iloc[0, 0] += 0.5
        with open(path, "wb") as f:
            pickle.dump(dfs_tampered, f)
        try:
            data_snapshot.load_trend_snapshot(path=path)
            raise AssertionError("Manipulierter Snapshot muss ValueError ausloesen")
        except ValueError:
            pass
    print("load_trend_snapshot (Hash-Verifikation): OK")

    # Sanity-Checks fail-closed
    bad = _fake_dfs()
    bad["AAA"].iloc[2, 0] = -1.0
    try:
        data_snapshot.sanity_check_snapshot(bad, min_common_days=3)
        raise AssertionError("Nichtpositiver Preis muss ValueError ausloesen")
    except ValueError:
        pass
    bad = _fake_dfs()
    bad["BBB"] = bad["BBB"].iloc[:2]  # gemeinsamer Kalender schrumpft unter min_common_days
    try:
        data_snapshot.sanity_check_snapshot(bad, min_common_days=3)
        raise AssertionError("Zu kurzer gemeinsamer Kalender muss ValueError ausloesen")
    except ValueError:
        pass
    data_snapshot.sanity_check_snapshot(_fake_dfs(), min_common_days=3)
    print("sanity_check_snapshot (fail-closed): OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen — FAIL erwartet**
- [ ] **Step 3: Implementierung**

`factor_lab/data_snapshot.py`:

```python
"""
data_snapshot.py — Load-only-Zugriff auf den versiegelten Snapshot
(Spec v2 Abschnitt 4). FETCHT NIE: Auswertungen laden nur, verifizieren
den Content-SHA256 gegen das committete Manifest und brechen bei
Abweichung ab. Der Build ist ein eigener Schritt (build_trend_snapshot.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market_control_system", "data_layer"))

import pandas as pd

from frozen_snapshot import snapshot_content_sha256, write_snapshot_manifest  # noqa: F401 (Re-Export fuer Build+Tests)
from factor_lab.registration import REGISTRATION

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "data_snapshots")


def trend_snapshot_path() -> str:
    key = (f"{sorted(REGISTRATION['universe'])}|{REGISTRATION['cash_series']}"
           f"|{REGISTRATION['snapshot_start']}|{REGISTRATION['snapshot_end_exclusive']}")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(SNAPSHOT_DIR, f"trend_snapshot_{digest}.pkl")


def sanity_check_snapshot(dfs: dict[str, pd.DataFrame], min_common_days: int = 4800) -> None:
    """Fail-closed-Pruefungen (Spec v2 §4). Wirft ValueError statt zu warnen."""
    common = None
    for name, df in dfs.items():
        if df.index.has_duplicates:
            raise ValueError(f"{name}: Duplikate im Index")
        if df.isna().any().any():
            raise ValueError(f"{name}: NaN-Werte im Snapshot")
        if name == "IRX":
            rates = df["rate_pa_pct"]
            if ((rates < -1.0) | (rates > 25.0)).any():
                raise ValueError("IRX: Rendite ausserhalb [-1, 25] Prozent p.a.")
            continue
        if (df["price"] <= 0).any():
            raise ValueError(f"{name}: nichtpositive Preise")
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < min_common_days:
        raise ValueError(f"Gemeinsamer Kalender zu kurz: {0 if common is None else len(common)} < {min_common_days}")


def load_trend_snapshot(path: str | None = None) -> dict[str, pd.DataFrame]:
    """Laedt den versiegelten Snapshot und verifiziert den Content-Hash
    gegen das Manifest. Kein Fetch-Fallback -- fehlender Snapshot ist ein
    Fehler (erst build_trend_snapshot.py ausfuehren und Manifest committen)."""
    path = path or trend_snapshot_path()
    manifest_path = path + ".manifest.json"
    if not os.path.exists(path) or not os.path.exists(manifest_path):
        raise ValueError(f"Snapshot oder Manifest fehlt ({path}) -- zuerst build_trend_snapshot.py ausfuehren")
    with open(path, "rb") as f:
        dfs = pickle.load(f)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    actual = snapshot_content_sha256(dfs)
    if actual != manifest["content_sha256"]:
        raise ValueError(f"Snapshot-Content-Hash {actual[:16]}... weicht vom Manifest ab -- Daten wurden veraendert")
    return dfs
```

`factor_lab/build_trend_snapshot.py`:

```python
"""
build_trend_snapshot.py — EIGENER, einmaliger Build-Schritt (Spec v2 §4):
fetcht 12 ETFs + ^IRX via yfinance im fixen Fenster, prueft fail-closed,
friert ein, schreibt das Manifest und druckt Content-Hash + aufgeloestes
DEV_END. Das Manifest MUSS committet werden, BEVOR ein Screening laeuft.

Ausfuehren: py -3.12 factor_lab/build_trend_snapshot.py
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from factor_lab.registration import REGISTRATION
from factor_lab.data_snapshot import (
    SNAPSHOT_DIR, trend_snapshot_path, sanity_check_snapshot,
    snapshot_content_sha256, write_snapshot_manifest,
)
from factor_lab.portfolio import month_end_dates


def _download(symbol: str, column: str) -> pd.Series:
    import yfinance as yf
    raw = yf.download(symbol, start=REGISTRATION["snapshot_start"],
                      end=REGISTRATION["snapshot_end_exclusive"], auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"{symbol}: yfinance lieferte keine Daten")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    series = close.astype(float)
    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    return series.rename(column)


def fetch_all() -> dict[str, pd.DataFrame]:
    dfs = {}
    for symbol in REGISTRATION["universe"]:
        print(f"  lade {symbol}...")
        dfs[symbol] = _download(symbol, "price").to_frame()
    print("  lade ^IRX (T-Bill-Cash-Naeherung)...")
    irx = _download(REGISTRATION["cash_series"], "rate_pa_pct").to_frame()
    dfs["IRX"] = irx.dropna()
    return dfs


def resolve_dev_end(dfs: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """DEV_END-Regel (praeregistriert): letzter Monatsultimo des gemeinsamen
    ETF-Kalenders <= dem 80%-Quantil-Datum."""
    common = None
    for name, df in dfs.items():
        if name == "IRX":
            continue
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    quantile_date = common[int(len(common) * 0.8) - 1]
    ends = month_end_dates(common)
    return ends[ends <= quantile_date].max()


def main() -> None:
    print(f"=== Versiegelter Snapshot-Build ({REGISTRATION['snapshot_start']} bis "
          f"exkl. {REGISTRATION['snapshot_end_exclusive']}) ===")
    dfs = fetch_all()
    sanity_check_snapshot(dfs)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = trend_snapshot_path()
    with open(path, "wb") as f:
        pickle.dump(dfs, f)
    write_snapshot_manifest(dfs, path + ".manifest.json")
    print(f"  Snapshot: {path}")
    print(f"  Content-SHA256: {snapshot_content_sha256(dfs)}")
    print(f"  DEV_END (aufgeloest): {resolve_dev_end(dfs).date()}")
    print("\nJETZT das Manifest committen (git add -f ...), DANN erst run_trend_baseline.py.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test ausführen — PASS erwartet.** Dann `.gitignore` + requirements ergänzen, `py -3.12 -m pip install yfinance`.
- [ ] **Step 5: Commit**

```bash
git add factor_lab/build_trend_snapshot.py factor_lab/data_snapshot.py factor_lab/tests/test_data_snapshot.py .gitignore
git add -u
git commit -m "feat(factor_lab): add sealed snapshot build and load-only access with hash verification"
```

---

### Task 9: run_trend_baseline.py — Screening

**Files:**
- Create: `factor_lab/run_trend_baseline.py`
- Test: `factor_lab/tests/test_run_trend_baseline.py`

**Interfaces:**
- Consumes: alles aus Task 1–8 (Signaturen siehe dort).
- Produces:
  - `VARIANT_NAMES: list[str]` — genau `["combo_long_flat", "combo_long_short", "mom126_long_flat", "mom126_long_short", "mom252_long_flat", "mom252_long_short", "mom63_long_flat", "mom63_long_short"]` (alphabetisch).
  - `prepare_inputs(dfs: dict) -> dict` — gemeinsamer ETF-Kalender, `returns` (T×12), `cash_daily` (auf den Kalender ffill-aligniert: `rate_pa_pct/100/252`), Signal-Frames, Vols, `eval_decisions` (Monatsultimos ab Warmup 252+63), `dev_end` (per `resolve_dev_end`-Regel aus Task 8).
  - `run_variant(inputs: dict, variant: str, cost_multiplier: float = 1.0, exclude: set[str] = frozenset()) -> tuple[pd.Series, dict]` — kompletter Lauf einer Variante (oder von `matched_long`) auf dem gemeinsamen Fenster; `exclude` entfernt Instrumente (für LOO-Reruns; matched_long wird OHNE dieselben Instrumente neu gerechnet).
  - `run_screening(dfs: dict) -> tuple[dict, pd.DataFrame]` — Ergebnis-Dict (JSON-serialisierbar) + per-Tag-Frame (Netto-Returns aller 8 Varianten + `matched_long` + `bench_spy_bh` + `bench_60_40`, IDENTISCHER Index). Enthält je Variante: Kennzahlen, Kostenleiter-Zeilen (1×/2×/5×), Break-even-bp (Kostensatz-Multiplikator, bei dem der Mehrertrag 0 wird, linear interpoliert aus der Leiter), Bootstrap (primär 6M + Sensitivität 3M/12M), Permutation, Gates (LOO nur für A–C-Passer), max_daily_gross, Verdikt + Kandidatin; Provenance (Registrierung, config_hash, snapshot_sha256, git_sha, Versionen, dev_end, eval_start).
  - `main()` — lädt versiegelten Snapshot (Load-only!), schneidet auf Tage ≤ `dev_end`, ruft `run_screening`, schreibt JSON + CSV + (bei Kandidatin) `candidate.json` nach `factor_lab/logs/trend_screening_<runid>/`.

- [ ] **Step 1: Failing Integrationstest schreiben**

```python
"""
test_run_trend_baseline.py — Integrationstest auf synthetischen Daten:
identischer Index ueber alle Serien, Holdout-Schutz, beide Verdikt-Zweige.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
import pandas as pd

from factor_lab.run_trend_baseline import run_screening, VARIANT_NAMES


def _dfs(regime_amplitude: float, n_days: int = 1400, seed: int = 0) -> dict:
    """regime_amplitude=0: reiner Random Walk (Null-Zweig). Sonst wechselt
    der Drift alle 130 Tage das Vorzeichen: long/flat geht in Abwaerts-
    Regimen in Cash, waehrend matched_long durchgehend long bleibt -- nur
    SO entsteht positiver MEHRertrag (gleichmaessig steigende Preise
    wuerden den Always-Long-Benchmark gerade NICHT schlagen)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    regime = np.sign(np.sin(np.arange(n_days) * np.pi / 130.0) + 1e-9)
    dfs = {}
    for i, symbol in enumerate(["SPY", "TLT", "GLD"]):
        drift = regime_amplitude * regime * [1.0, 0.8, 0.9][i]
        prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.006, n_days) + drift)
        dfs[symbol] = pd.DataFrame({"price": prices}, index=idx)
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * n_days}, index=idx)
    return dfs


def run_consistency_check() -> None:
    result, per_day = run_screening(_dfs(regime_amplitude=0.0))

    assert sorted(result["summary"]) == VARIANT_NAMES
    expected_cols = set(VARIANT_NAMES) | {"matched_long", "bench_spy_bh", "bench_60_40"}
    assert set(per_day.columns) == expected_cols
    # Pflichttest Review: IDENTISCHER Return-Index ueber alle Serien.
    assert not per_day.isna().any().any(), "Alle Serien muessen denselben Index vollstaendig fuellen"
    json.dumps(result)

    for name in VARIANT_NAMES:
        s = result["summary"][name]
        for key in ("stats", "excess_bootstrap", "excess_bootstrap_sensitivity", "permutation",
                    "gates", "cost_ladder", "breakeven_cost_multiplier", "max_daily_gross"):
            assert key in s, f"{name}: {key} fehlt"
    assert "research_variant" in result["summary"]["mom63_long_short"]["labels"]

    # Rauschen ohne Trend: erwartungsgemaess keine Kandidatin.
    assert result["candidate"] is None, "Random-Walk-Daten duerfen keine Kandidatin liefern"
    print("run_screening (Null-Zweig, identischer Index): OK")

    # Regime-wechselnde Daten: Trend schlaegt Always-Long -> Kandidatin gesetzt.
    result2, _ = run_screening(_dfs(regime_amplitude=0.0035, seed=1))
    assert result2["candidate"] in VARIANT_NAMES, f"Erwartete Kandidatin, bekam {result2['candidate']}"
    print("run_screening (Kandidaten-Zweig): OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen — FAIL erwartet**
- [ ] **Step 3: Implementierung** (Kern; Prints/Formatierung analog zu den Bestands-Runnern)

```python
"""
run_trend_baseline.py — Screening der Familie trend-etf-v1 (Spec v2
§5-§10): 8 Varianten + matched_long + Kontext-Benchmarks auf IDENTISCHEM
Evaluationsfenster, Kostenleiter, Stationary-Block-Inferenz auf monatlichen
Log-MEHRertraegen, Gates A-D (LOO-Reruns fuer A-C-Passer), candidate.json.
Das Screening ist ausdruecklich KEINE Bestaetigung (Spec §10).

Laufzeit-Hinweis: >20 min moeglich (LOO-Reruns) -> detacht starten
(Start-Process + PID-Datei + Monitor, siehe Tech-Stack-Memory).

Ausfuehren: py -3.12 factor_lab/run_trend_baseline.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from factor_lab.signals import momentum_sign, combo_signal
from factor_lab.costs import COST_BP, COST_LADDER
from factor_lab.portfolio import (
    ewma_annualized_vol, month_end_dates, run_lagged_backtest,
    trend_weight_provider, fixed_mix_provider,
)
from factor_lab.stats import (
    monthly_log_returns, stationary_block_bootstrap, monthly_sign_flip_pvalue,
    annualized_stats, full_year_excess, evaluate_screening_gates, screening_verdict,
)
from factor_lab.registration import REGISTRATION, config_hash, write_candidate, file_sha256
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.build_trend_snapshot import resolve_dev_end

LOOKBACKS = {"mom63": 63, "mom126": 126, "mom252": 252}
SLEEVES = {
    "us_equity": ["SPY", "QQQ", "IWM"], "intl_equity": ["EFA", "EEM"],
    "bonds": ["TLT", "IEF", "LQD"], "real_assets": ["GLD", "SLV", "DBC", "VNQ"],
}
VARIANT_NAMES = sorted(f"{sig}_{mode}" for sig in [*LOOKBACKS, "combo"] for mode in ["long_short", "long_flat"])
WARMUP_DAYS = REGISTRATION["warmup_days"]


def prepare_inputs(dfs: dict) -> dict:
    symbols = sorted(s for s in dfs if s != "IRX")
    common = dfs[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(dfs[s].index)
    common = common.sort_values()
    prices = pd.DataFrame({s: dfs[s].loc[common, "price"] for s in symbols})
    returns = prices.pct_change().dropna()
    cash_daily = (dfs["IRX"]["rate_pa_pct"].reindex(common).ffill().bfill() / 100.0 / 252.0).loc[returns.index]

    per_lb = {name: pd.DataFrame({s: momentum_sign(prices[s], lb) for s in symbols})
              for name, lb in LOOKBACKS.items()}
    signal_frames = {**per_lb, "combo": pd.DataFrame({
        s: combo_signal([per_lb[n][s] for n in LOOKBACKS]) for s in symbols})}
    signal_frames = {k: v.loc[returns.index] for k, v in signal_frames.items()}
    vols = ewma_annualized_vol(returns)

    warmup_end = returns.index[WARMUP_DAYS]
    ends = month_end_dates(returns.index)
    eval_decisions = ends[ends >= warmup_end]
    return {"symbols": symbols, "returns": returns, "cash_daily": cash_daily,
            "signal_frames": signal_frames, "vols": vols, "eval_decisions": eval_decisions,
            "dev_end": resolve_dev_end(dfs)}


def run_variant(inputs: dict, variant: str, cost_multiplier: float = 1.0,
                exclude: set = frozenset()) -> tuple[pd.Series, dict]:
    symbols = [s for s in inputs["symbols"] if s not in exclude]
    returns = inputs["returns"][symbols]
    vols = inputs["vols"][symbols]
    if variant == "matched_long":
        signals = pd.DataFrame(1.0, index=returns.index, columns=symbols)
        mode = "long_flat"
    else:
        sig_name = variant.rsplit("_", 2)[0]
        mode = "_".join(variant.rsplit("_", 2)[1:])
        signals = inputs["signal_frames"][sig_name][symbols]
    provider = trend_weight_provider(returns, signals, vols, mode,
                                     vol_cap=REGISTRATION["vol_cap"], vol_window=63)
    return run_lagged_backtest(returns, inputs["cash_daily"], inputs["eval_decisions"],
                               provider, {s: COST_BP.get(s, 3.0) for s in symbols},
                               cost_multiplier, REGISTRATION["borrow_bp_pa"])


def _breakeven_multiplier(excess_by_mult: dict[float, float]) -> float:
    """Linear interpolierter Kostensatz-Multiplikator, bei dem der
    compoundierte Mehrertrag 0 wird; inf, wenn selbst 5x positiv bleibt,
    0, wenn schon 1x negativ ist."""
    mults = sorted(excess_by_mult)
    if excess_by_mult[mults[0]] <= 0:
        return 0.0
    for lo, hi in zip(mults, mults[1:]):
        e_lo, e_hi = excess_by_mult[lo], excess_by_mult[hi]
        if e_hi <= 0 < e_lo:
            return float(lo + (hi - lo) * e_lo / (e_lo - e_hi))
    return float("inf")


def run_screening(dfs: dict, dev_end=None) -> tuple[dict, pd.DataFrame]:
    """dev_end: von main() uebergeben, weil dort die Daten bereits auf das
    Entwicklungsfenster geschnitten sind -- die 80%-Regel darf nicht ein
    zweites Mal auf die geschnittenen Daten angewendet werden."""
    inputs = prepare_inputs(dfs)
    if dev_end is None:
        dev_end = inputs["dev_end"]
    cash = inputs["cash_daily"]

    matched = {m: run_variant(inputs, "matched_long", cost_multiplier=m) for m in COST_LADDER}
    per_day = {"matched_long": matched[1.0][0]}

    summary, excess_sharpes, gates_by_variant = {}, {}, {}
    for variant in VARIANT_NAMES:
        runs = {m: run_variant(inputs, variant, cost_multiplier=m) for m in COST_LADDER}
        net, info = runs[1.0]
        per_day[variant] = net

        excess = net - matched[1.0][0]
        # Mehrertrag als Differenz der monatlichen Log-Ertraege (geometrisch sauber):
        monthly_excess = (monthly_log_returns(net) - monthly_log_returns(matched[1.0][0])).dropna()
        boot = stationary_block_bootstrap(monthly_excess.to_numpy(),
                                          REGISTRATION["bootstrap"]["expected_block_len_months"],
                                          REGISTRATION["bootstrap"]["n_boot"],
                                          REGISTRATION["bootstrap"]["seed"])
        sensitivity = {str(L): stationary_block_bootstrap(
            monthly_excess.to_numpy(), L, REGISTRATION["bootstrap"]["n_boot"],
            REGISTRATION["bootstrap"]["seed"])["ann_geom_lower_1s95"]
            for L in REGISTRATION["bootstrap"]["sensitivity_block_lens"]}
        perm = monthly_sign_flip_pvalue(monthly_excess.to_numpy(), REGISTRATION["permutation_n"],
                                        REGISTRATION["bootstrap"]["seed"])

        stats = annualized_stats(net, cash)
        stats_stress = annualized_stats(runs[2.0][0], cash)
        ladder = {}
        excess_by_mult = {}
        for m in COST_LADDER:
            ladder[str(m)] = annualized_stats(runs[m][0], cash)
            m_excess = (monthly_log_returns(runs[m][0]) - monthly_log_returns(matched[m][0])).dropna()
            excess_by_mult[m] = float(np.expm1(m_excess.sum()))

        # Gates A-C zuerst; LOO-Reruns (teuer) nur fuer A-C-Passer.
        pre = evaluate_screening_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                       stats_stress["cagr"],
                                       full_year_excess(excess),
                                       loo_excess_compounds={"pending": 1.0},
                                       dd_cap=REGISTRATION["dd_cap"],
                                       gate_c_floor=REGISTRATION["gate_c_floor"])
        loo = {}
        if pre["gate_a_excess_ci"] and pre["gate_b_drawdown"] and pre["gate_c_stressed_floor"]:
            loo_configs = {f"loo_{s}": {s} for s in inputs["symbols"]}
            loo_configs.update({f"loo_sleeve_{k}": set(v) for k, v in SLEEVES.items()})
            for loo_name, excl in loo_configs.items():
                v_net, _ = run_variant(inputs, variant, exclude=excl)
                m_net, _ = run_variant(inputs, "matched_long", exclude=excl)
                diff = (monthly_log_returns(v_net) - monthly_log_returns(m_net)).dropna()
                loo[loo_name] = float(np.expm1(diff.sum()))
        gates = evaluate_screening_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                         stats_stress["cagr"], full_year_excess(excess), loo,
                                         REGISTRATION["dd_cap"], REGISTRATION["gate_c_floor"])
        gates_by_variant[variant] = gates
        excess_sharpes[variant] = stats["excess_sharpe"]
        summary[variant] = {
            "stats": stats,
            "excess_bootstrap": boot,
            "excess_bootstrap_sensitivity": sensitivity,
            "permutation": perm,
            "gates": gates,
            "cost_ladder": ladder,
            "breakeven_cost_multiplier": _breakeven_multiplier(excess_by_mult),
            "loo_excess_compounds": loo,
            "yearly_excess_full_years": {str(k): v for k, v in full_year_excess(excess).items()},
            "max_daily_gross": info["max_daily_gross"],
            "total_turnover": info["total_turnover"],
            "instrument_contributions": info["instrument_contributions"],
            "labels": (["research_variant"] if variant.endswith("long_short") else []),
        }

    spy_provider = fixed_mix_provider({"SPY": 1.0}) if "SPY" in inputs["symbols"] else fixed_mix_provider({inputs["symbols"][0]: 1.0})
    spy_bh, _ = run_lagged_backtest(inputs["returns"], cash,
                                    pd.DatetimeIndex([inputs["eval_decisions"][0]]),
                                    spy_provider, COST_BP if "SPY" in inputs["symbols"] else {s: 1.5 for s in inputs["symbols"]})
    mix_symbols = ["SPY", "TLT"] if set(["SPY", "TLT"]) <= set(inputs["symbols"]) else inputs["symbols"][:2]
    mix, _ = run_lagged_backtest(inputs["returns"], cash, inputs["eval_decisions"],
                                 fixed_mix_provider({mix_symbols[0]: 0.6, mix_symbols[1]: 0.4}),
                                 {s: COST_BP.get(s, 1.5) for s in mix_symbols})
    per_day["bench_spy_bh"] = spy_bh
    per_day["bench_60_40"] = mix

    verdict, candidate = screening_verdict(gates_by_variant, excess_sharpes)
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git_sha = "unbekannt"
    provenance = {
        "registration": REGISTRATION,
        "config_hash": config_hash(),
        "snapshot_content_sha256": snapshot_content_sha256(dfs),
        "git_sha": git_sha,
        "dev_end": str(dev_end),
        "eval_start": str(per_day["matched_long"].index.min()),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    result = {
        "summary": summary,
        "benchmarks": {"matched_long": annualized_stats(per_day["matched_long"], cash),
                       "bench_spy_bh": annualized_stats(spy_bh, cash),
                       "bench_60_40": annualized_stats(mix, cash)},
        "verdict": verdict,
        "candidate": candidate,
        "provenance": provenance,
    }
    return result, pd.DataFrame(per_day)


def main() -> None:
    dfs = load_trend_snapshot()
    inputs_probe = prepare_inputs(dfs)
    dev_end = inputs_probe["dev_end"]
    dev_dfs = {name: df.loc[df.index <= dev_end] for name, df in dfs.items()}
    result, per_day = run_screening(dev_dfs, dev_end=dev_end)

    print(f"\n{result['verdict']}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "logs", f"trend_screening_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    per_day.to_csv(os.path.join(out_dir, "daily_net_returns.csv"))
    results_path = os.path.join(out_dir, "screening_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if result["candidate"] is not None:
        write_candidate(os.path.join(out_dir, "candidate.json"),
                        variant=result["candidate"],
                        snapshot_sha256=result["provenance"]["snapshot_content_sha256"],
                        git_sha=result["provenance"]["git_sha"],
                        dev_end=result["provenance"]["dev_end"],
                        results_sha256=file_sha256(results_path))
    print(f"Ergebnisse gespeichert: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test ausführen — PASS erwartet** (dauert wegen 8 Varianten × Leiter + LOO im Kandidaten-Zweig einige Minuten). Beide Test-Zweige sind mit festem Seed deterministisch — sollte der Null-Zweig durch Seed-Zufall doch eine Kandidatin liefern oder der Regime-Zweig keine, die Fixture-Parameter (Regime-Amplitude/Seed) anpassen und im Test-Kommentar begründen; NICHT die Gates aufweichen.
- [ ] **Step 5: Alle factor_lab-Tests + die 8 Bestands-Suiten in `market_control_system/tests/` laufen lassen — alles PASS**
- [ ] **Step 6: Commit**

```bash
git add factor_lab/run_trend_baseline.py factor_lab/tests/test_run_trend_baseline.py
git commit -m "feat(factor_lab): add screening runner with matched benchmark, cost ladder, LOO gates, candidate seal"
```

---

### Task 10: run_trend_holdout.py — versiegelter One-Shot

**Files:**
- Create: `factor_lab/run_trend_holdout.py`
- Test: `factor_lab/tests/test_run_trend_holdout.py`

**Interfaces:**
- Consumes: Task 7 (`read_and_verify_candidate`, `assert_no_tombstone`, `write_tombstone`), Task 9 (`prepare_inputs`, `run_variant`-Bausteine).
- Produces: `run_holdout(dfs: dict, candidate: dict) -> dict` — rechnet GENAU die versiegelte Variante: Signale/Vols auf voller Historie (kausal), aber der Lauf STARTET IN CASH mit erster Entscheidung am ersten Monatsultimo NACH `dev_end` (voller Positionsaufbau inkl. Kosten im Holdout — keine geerbte Dev-Position), PnL/Kosten/Attribution stammen ausschließlich aus Holdout-Tagen (per Konstruktion, nicht per Schnitt); Gates A/B/C via `evaluate_holdout_gates`; `main()` nimmt KEINE Argumente, liest das jüngste `candidate.json` aus `factor_lab/logs/trend_screening_*/`, verifiziert Hashes, prüft Tombstone, schreibt Ergebnis + Tombstone.

- [ ] **Step 1: Failing Test schreiben**

```python
"""
test_run_trend_holdout.py — Holdout: startet in Cash NACH dev_end, wertet
nur Holdout-Tage, verweigert ohne gueltiges Siegel und beim zweiten Mal.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile
import numpy as np
import pandas as pd

from factor_lab import registration
from factor_lab.run_trend_holdout import run_holdout
from factor_lab.run_trend_baseline import prepare_inputs


def _dfs(n_days: int = 1400, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    dfs = {s: pd.DataFrame({"price": 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.008, n_days))}, index=idx)
           for s in ["SPY", "TLT", "GLD"]}
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": [2.0] * n_days}, index=idx)
    return dfs


def run_consistency_check() -> None:
    dfs = _dfs()
    dev_end = prepare_inputs(dfs)["dev_end"]
    candidate = {"variant": "mom63_long_flat", "dev_end": str(dev_end)}
    result = run_holdout(dfs, candidate)

    # Alle PnL-Tage liegen NACH dev_end; erste Kosten (Positionsaufbau)
    # fallen im Holdout an -- keine geerbte Dev-Position.
    first_day = pd.Timestamp(result["provenance"]["holdout_start"])
    assert first_day > dev_end
    assert result["summary"]["holdout_entry_cost"] > 0, "Positionsaufbau muss im Holdout bezahlt werden"
    assert "gate_d_no_single_driver" not in result["summary"]["gates"], "Holdout hat nur Gates A-C"
    assert set(result["summary"]["attribution_days"]) == {"holdout"}, "Attribution nur aus Holdout-Tagen"

    # Tombstone-Einmaligkeit (auf Task-7-Funktionen, hier als Integrationspruefung)
    with tempfile.TemporaryDirectory() as tmp:
        registration.assert_no_tombstone(tmp)
        registration.write_tombstone(tmp, "x.json")
        try:
            registration.assert_no_tombstone(tmp)
            raise AssertionError("Zweiter Holdout-Zugriff muss verweigert werden")
        except ValueError:
            pass
    print("run_holdout: OK")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Test ausführen — FAIL erwartet**
- [ ] **Step 3: Implementierung**

```python
"""
run_trend_holdout.py — EINMALIGE Holdout-Bestaetigung der versiegelten
Kandidatin (Spec v2 §10). Nimmt KEINE Argumente: liest candidate.json,
verifiziert Familie/Config-Hash/Snapshot-Hash, prueft den Tombstone und
schreibt ihn nach dem Lauf. Startet in CASH nach dev_end -- der volle
Positionsaufbau inkl. Kosten gehoert zum Holdout-Ergebnis.

Ausfuehren: py -3.12 factor_lab/run_trend_holdout.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from factor_lab.costs import COST_BP
from factor_lab.portfolio import month_end_dates, run_lagged_backtest, trend_weight_provider
from factor_lab.stats import (
    monthly_log_returns, stationary_block_bootstrap, monthly_sign_flip_pvalue,
    annualized_stats, evaluate_holdout_gates,
)
from factor_lab.registration import (
    REGISTRATION, read_and_verify_candidate, assert_no_tombstone, write_tombstone,
)
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.run_trend_baseline import prepare_inputs, run_variant, VARIANT_NAMES

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


def _holdout_run(inputs: dict, variant: str, dev_end: pd.Timestamp, cost_multiplier: float = 1.0):
    """Wie run_variant, aber Entscheidungen NUR nach dev_end -> der Lauf
    beginnt flach in Cash und baut die Position im Holdout auf."""
    holdout_decisions = inputs["eval_decisions"][inputs["eval_decisions"] > dev_end]
    saved = inputs["eval_decisions"]
    inputs["eval_decisions"] = holdout_decisions
    try:
        return run_variant(inputs, variant, cost_multiplier=cost_multiplier)
    finally:
        inputs["eval_decisions"] = saved


def run_holdout(dfs: dict, candidate: dict) -> dict:
    variant = candidate["variant"]
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unbekannte Variante im Siegel: {variant}")
    dev_end = pd.Timestamp(candidate["dev_end"])
    inputs = prepare_inputs(dfs)

    net, info = _holdout_run(inputs, variant, dev_end)
    net_stress, _ = _holdout_run(inputs, variant, dev_end, cost_multiplier=2.0)
    matched, _ = _holdout_run(inputs, "matched_long", dev_end)
    matched_stress, _ = _holdout_run(inputs, "matched_long", dev_end, cost_multiplier=2.0)
    cash = inputs["cash_daily"].loc[net.index]

    monthly_excess = (monthly_log_returns(net) - monthly_log_returns(matched)).dropna()
    boot = stationary_block_bootstrap(monthly_excess.to_numpy(),
                                      REGISTRATION["bootstrap"]["expected_block_len_months"],
                                      REGISTRATION["bootstrap"]["n_boot"],
                                      REGISTRATION["bootstrap"]["seed"])
    stats = annualized_stats(net, cash)
    stats_stress = annualized_stats(net_stress, cash)
    gates = evaluate_holdout_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                   stats_stress["cagr"], REGISTRATION["dd_cap"],
                                   REGISTRATION["gate_c_floor"])
    return {
        "variant": variant,
        "summary": {
            "stats": stats,
            "stats_2x": stats_stress,
            "excess_bootstrap": boot,
            "permutation": monthly_sign_flip_pvalue(monthly_excess.to_numpy(),
                                                    REGISTRATION["permutation_n"],
                                                    REGISTRATION["bootstrap"]["seed"]),
            "gates": gates,
            "holdout_entry_cost": float(info["per_day"]["trade_cost"].iloc[:2].sum()),
            "instrument_contributions": info["instrument_contributions"],
            "attribution_days": ["holdout"],
            "matched_long_stats": annualized_stats(matched, cash),
        },
        "provenance": {
            "holdout_start": str(net.index.min()),
            "holdout_end": str(net.index.max()),
            "dev_end": str(dev_end),
            "snapshot_content_sha256": snapshot_content_sha256(dfs),
        },
    }


def main() -> None:
    print("*** VERSIEGELTES ONE-SHOT-HOLDOUT (keine Argumente, keine zweite Chance) ***")
    assert_no_tombstone(LOGS_DIR)
    dfs = load_trend_snapshot()
    candidates = sorted(glob.glob(os.path.join(LOGS_DIR, "trend_screening_*", "candidate.json")))
    if not candidates:
        raise ValueError("Kein candidate.json gefunden -- erst das Screening muss eine Kandidatin versiegeln")
    candidate = read_and_verify_candidate(candidates[-1], snapshot_content_sha256(dfs))
    result = run_holdout(dfs, candidate)

    g = result["summary"]["gates"]
    print(f"\n{result['variant']}: holdout_gates={'PASS' if g['passed_all'] else 'FAIL'}  "
          f"excess_lower95={result['summary']['excess_bootstrap']['ann_geom_lower_1s95']:+.4f}")
    if not g["passed_all"]:
        print("Familie trend-etf-v1 ist damit BEENDET (Spec v2 §10 Punkt 5) -- keine Runner-up-Variante.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LOGS_DIR, f"trend_holdout_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "holdout_summary.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_tombstone(LOGS_DIR, result_path)
    print(f"Ergebnis gespeichert: {result_path} (Tombstone geschrieben)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test ausführen — PASS erwartet**
- [ ] **Step 5: Commit**

```bash
git add factor_lab/run_trend_holdout.py factor_lab/tests/test_run_trend_holdout.py
git commit -m "feat(factor_lab): add sealed one-shot holdout runner (cash start, tombstone, no arguments)"
```

---

### Task 11: Echter Ablauf

- [ ] **Step 1: Snapshot bauen:** `py -3.12 factor_lab/build_trend_snapshot.py` — druckt Content-Hash + aufgelöstes DEV_END.
- [ ] **Step 2: Manifest committen (VOR dem Screening — Versiegelung):**

```bash
git add -f factor_lab/data_snapshots/*.manifest.json
git commit -m "docs(factor_lab): seal trend snapshot manifest (content hash, per-series ranges)"
```

- [ ] **Step 3: Screening starten** — wegen LOO-Reruns detacht (Start-Process + PID-Datei + persistenter Monitor, Muster aus dem Tech-Stack-Memory), `-u` für ungepuffertes Log.
- [ ] **Step 4: Ergebnisse berichten** — Tabelle (8 Varianten mit Excess-CI, Gates, Kostenleiter, Break-even, Max-Gross; matched_long/SPY/60-40 als Kontext), Verdikt, ob eine Kandidatin versiegelt wurde. **Das Holdout wird NICHT ausgeführt — auch bei versiegelter Kandidatin erst nach expliziter User-Freigabe.**
