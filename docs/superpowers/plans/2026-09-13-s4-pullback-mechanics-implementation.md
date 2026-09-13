# S4 Pullback-Mechanik — Implementierungsplan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Indikatoren, Featurevektor, Eventgenerator und Labelkernel aus Abschnitt 8 der S4-Spec implementieren und isoliert gegen synthetische OHLC-Fixtures pruefen — Zeitpunkte, Gaps, Gebuehren, mehrdeutige Barrieren, Splits, Dividenden, Datenluecken.

**Architecture:** Vier neue, additive Module unter `factor_lab/`, streng nach Verantwortung getrennt: Indikatoren (reine Arrays), Features (reine Abbildung), Eventgenerator (Zustandsautomat) und Labelkernel (Ausfuehrungs- und P&L-Mechanik). Kein bestehendes Modul wird veraendert. Kein Lauf auf echten Daten, kein Modell, keine Parametersuche.

**Tech Stack:** Python 3.12 (`py -3.12`), numpy 2.5.2, pandas 3.0.5. Kein torch, kein lightgbm, kein sklearn in dieser Stufe.

**Spec:** `docs/superpowers/specs/2026-09-13-s4-pullback-event-mechanics-design.md`

## Global Constraints

- Alles mit `py -3.12` ausfuehren. Niemals bare `python`/`pip`.
- Tests im `unittest`-Stil des Bestands, Ausfuehrung `py -3.12 -m unittest factor_lab.tests.<modul> -v`.
- Deutsche Kommentare/Docstrings in ASCII (ue/oe/ae/ss, keine Umlaute).
- **Nichts veraendern** an bestehenden Modulen. Namentlich frozen: `daily_comparison.py`, `daily_models.py`, `stats.py`, `data_snapshot.py`, `horizon_data.py`, `features_2x2.py`, `models_2x2.py`, `evaluate_2x2.py`, `run_feature_model_2x2.py`, alles unter `factor_lab/audit/`, `market_control_system/`, `research_archive/`.
- **Kein Lauf auf echten Daten.** Der versiegelte Snapshot erfuellt den OHLC-Vertrag nicht (Spec Abschnitt 2). Kein Datenkauf, kein Netzabruf in diesem Plan.
- **Alle Testdaten sind synthetisch und als solche zu kennzeichnen** — Modul-Docstring der Testdateien sagt das explizit. Keine synthetischen Daten ausserhalb von `factor_lab/tests/`.
- **Eingefrorene Konventionen, die kein Task veraendern darf:** `WARMUP=260`, `SEQUENCE=60`, EMA 20/50/200 mit SMA-Seed und `alpha=2/(n+1)`, `ATR=SMA(TR,20)`, `TOUCH_BAND=0.25*ATR20[p-1]`, `CONFIRM_MAX=3`, `HOLD_N=10`, `TARGET_Q=1.5`, `R=ATR20[t-1]`, `EVENT_COOLDOWN` bis einschliesslich `t+10`, `PRIMARY_OHLC_TIE=stop_first` mit `ambiguous=true`, Time-Exit am Close von `t+10`.
- **Alle Intervalle `[a:b]` der Spec sind einschliesslich beider Endpunkte.** Python-Slices sind es nicht — jede Umsetzung muss das explizit umrechnen.
- **Keine Parameteroptimierung, kein LSTM, kein Census, kein Nullaudit.** Diese Stufe liefert Mechanik, keine Ergebnisse.
- Nach jedem Task committen; Commit-Nachricht endet mit einer Leerzeile und dann `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Nicht pushen, keinen PR oeffnen.

## Drei Entscheidungen, die die Spec offen laesst

Vorab festgelegt, damit sie nicht waehrend der Umsetzung improvisiert werden:

1. **Zwei Preissichten sind Sache des Adapters, nicht dieser Module.** Spec 8.1: „Der Datenadapter stellt zwei Sichten bereit." Der Eventgenerator konsumiert die **Indikatorsicht** (split-konsistente Historie), der Labelkernel **Roh-Bars plus Kapitalmassnahmen**. Kein Modul hier rechnet Sichten ineinander um. Der Adapter ist eine spaetere Stufe; Fixtures liefern beide Sichten direkt. Dokumentierte Annahme: die Indikatorsicht liegt auf der zum jeweiligen Entscheidungszeitpunkt geltenden Stueckbasis, weshalb `event.R` auf Signal-Stueckbasis steht und der Kernel es per `s[e]` konvertiert. In Fixtures ohne Split vor dem Entry fallen beide Sichten zusammen.

2. **Indikatoren werden pro zusammenhaengendem gueltigem Lauf berechnet.** Spec Abschnitt 6: nach einem Datenbruch beginnt die Indikatorinitialisierung neu. EMA-Zustand wird also bei jedem Laufbeginn zurueckgesetzt, nicht ueber die Luecke fortgeschrieben.

3. **Die Warmup-Zahl 260 ist ausreichend, und zwar knapp.** Nachgerechnet: die bindende Bedingung ist `UP[t-1]`, das `EMA200[t-1]` braucht; EMA200 ist ab lauf-lokalem Index 199 definiert. Die Features reichen ueber `EMA50[j-11]` bei `j=t-59` bis Index `t-70` (EMA50 ab 49) und ueber `sigma20[j-1]`/`C[j-20]` bis `t-80`. Bei einer Laufllaenge von mindestens 260 bis einschliesslich `t` gilt `t-80 >= 179`, `t-70 >= 189` und `t-1 >= 258` — alle erfuellt. Der Gate ist daher: **Bar `t` ist nur verwendbar, wenn die Zahl zusammenhaengender gueltiger Bars bis einschliesslich `t` mindestens 260 betraegt.**

## Dateistruktur

| Datei | Verantwortung |
|---|---|
| `factor_lab/s4_indicators.py` | Abschnitt 3 der Spec: TR, ATR, EMA, sigma, ER, UP, Lauflaenge. Reine Arrays, kennt weder Events noch Portfolios. |
| `factor_lab/s4_features.py` | Abschnitt 7 der Spec: `build_features(p,t)` mit exakt 60x10 Sequenz und 4 Kontextwerten in vorgegebener Reihenfolge. |
| `factor_lab/s4_events.py` | Abschnitt 6 der Spec: Zustandsautomat, Cooldown, Fristen, Emission. |
| `factor_lab/s4_labels.py` | Abschnitt 8 der Spec: Barrier-Aufloesung, Gaps, Splits, Dividenden, Kosten, Netto-R. |
| `factor_lab/tests/test_s4_indicators.py` | Kausalitaet, Laufsegmentierung, EMA-Seed, Warmup-Grenzen. |
| `factor_lab/tests/test_s4_events.py` | Zeitpunkte, Frist, Invalidierung, Cooldown, Datenbruch, keine Emission auf der Beruehrungsbar. |
| `factor_lab/tests/test_s4_labels.py` | Gaps, Gebuehren, mehrdeutige Barrieren, Timeout, Splits, Dividenden, Zensierung. |

---

### Task 1: Indikatoren und Featurevektor

**Files:**
- Create: `factor_lab/s4_indicators.py`
- Create: `factor_lab/s4_features.py`
- Create: `factor_lab/tests/test_s4_indicators.py`

**Interfaces:**
- Consumes: nichts aus frueheren Tasks.
- Produces:
  - `WARMUP = 260`, `SEQUENCE = 60`
  - `compute_indicators(bars: pd.DataFrame) -> dict[str, np.ndarray]` mit den Schluesseln `valid` (bool), `run_length` (int), `logret`, `tr`, `atr20`, `sigma5`, `sigma20`, `er20`, `ema20`, `ema50`, `ema200`, `up` (bool). Alle Arrays haben die Laenge von `bars`; undefinierte Stellen sind `nan` bzw. `False`.
  - `build_features(ind: dict, bars: pd.DataFrame, p: int, t: int) -> tuple[np.ndarray, np.ndarray]` mit Formen `(60, 10)` und `(4,)`, oder `ValueError` bei fehlenden Daten.

`bars` ist ein `pd.DataFrame` mit genau den Spalten `['open','high','low','close']`, monoton steigendem eindeutigem `DatetimeIndex`, eine Zeile je planmaessiger Sitzung. Eine Zeile ist **ungueltig**, wenn ein Wert fehlt, nichtfinit oder nichtpositiv ist oder `high < low`.

- [ ] **Step 1: Write the failing test**

Create `factor_lab/tests/test_s4_indicators.py`:

```python
"""Indikatoren und Featurevektor der S4-Mechanik.

ALLE Daten in dieser Datei sind SYNTHETISCH und dienen ausschliesslich der
Mechanikpruefung. Sie stammen aus keiner Marktquelle und stellen kein
Marktverhalten dar.
"""
import unittest

import numpy as np
import pandas as pd

from factor_lab.s4_features import build_features
from factor_lab.s4_indicators import SEQUENCE, WARMUP, compute_indicators


def synthetic_bars(n=400, seed=1, start='2010-01-04'):
    """Synthetische, streng positive OHLC-Bars auf einem Werktagskalender."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(0.0004 + 0.01 * rng.standard_normal(n)))
    spread = close * 0.01
    high = close + spread * rng.uniform(0.2, 1.0, n)
    low = close - spread * rng.uniform(0.2, 1.0, n)
    open_ = low + (high - low) * rng.uniform(0.0, 1.0, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close},
                        index=pd.bdate_range(start, periods=n))


class IndicatorTests(unittest.TestCase):
    def test_constants_are_frozen(self):
        self.assertEqual(WARMUP, 260)
        self.assertEqual(SEQUENCE, 60)

    def test_true_range_matches_definition(self):
        bars = synthetic_bars(50)
        ind = compute_indicators(bars)
        h, l, c = bars['high'].to_numpy(), bars['low'].to_numpy(), bars['close'].to_numpy()
        for i in range(1, 50):
            expected = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            self.assertAlmostEqual(ind['tr'][i], expected, places=12)
        self.assertTrue(np.isnan(ind['tr'][0]))

    def test_atr_is_a_simple_moving_average_not_wilder(self):
        bars = synthetic_bars(60)
        ind = compute_indicators(bars)
        self.assertAlmostEqual(ind['atr20'][25], float(np.mean(ind['tr'][6:26])), places=12)
        self.assertTrue(np.isnan(ind['atr20'][19]))
        self.assertFalse(np.isnan(ind['atr20'][20]))

    def test_ema_is_seeded_with_the_sma_of_the_first_n_closes(self):
        bars = synthetic_bars(80)
        ind = compute_indicators(bars)
        c = bars['close'].to_numpy()
        self.assertTrue(np.isnan(ind['ema20'][18]))
        self.assertAlmostEqual(ind['ema20'][19], float(np.mean(c[:20])), places=12)
        alpha = 2 / 21
        self.assertAlmostEqual(ind['ema20'][20], alpha * c[20] + (1 - alpha) * ind['ema20'][19],
                               places=12)

    def test_sigma_uses_sample_standard_deviation_of_log_returns(self):
        bars = synthetic_bars(60)
        ind = compute_indicators(bars)
        r = ind['logret']
        self.assertAlmostEqual(ind['sigma20'][30], float(np.std(r[11:31], ddof=1)), places=12)
        self.assertAlmostEqual(ind['sigma5'][30], float(np.std(r[26:31], ddof=1)), places=12)

    def test_efficiency_ratio_matches_definition(self):
        bars = synthetic_bars(60)
        ind = compute_indicators(bars)
        c = bars['close'].to_numpy()
        j = 40
        expected = abs(c[j] - c[j - 20]) / np.abs(np.diff(c[j - 20:j + 1])).sum()
        self.assertAlmostEqual(ind['er20'][j], expected, places=12)

    def test_uptrend_requires_all_three_conditions(self):
        bars = synthetic_bars(400)
        ind = compute_indicators(bars)
        j = 350
        manual = (ind['ema50'][j] > ind['ema200'][j]
                  and ind['ema50'][j] > ind['ema50'][j - 10]
                  and bars['close'].to_numpy()[j] > ind['ema50'][j])
        self.assertEqual(bool(ind['up'][j]), manual)
        # EMA200 ist erst ab lauf-lokalem Index 199 definiert; davor immer False.
        self.assertFalse(bool(ind['up'][198]))

    def test_indicators_are_causal(self):
        """Spaetere Bars aendern fruehere Indikatorwerte nicht."""
        bars = synthetic_bars(300)
        cut = 200
        tampered = bars.copy()
        tampered.iloc[cut + 1:] *= 1.5
        a = compute_indicators(bars)
        b = compute_indicators(tampered)
        for key in ('tr', 'atr20', 'sigma5', 'sigma20', 'er20', 'ema20', 'ema50', 'up'):
            np.testing.assert_allclose(np.nan_to_num(a[key][:cut + 1].astype(float)),
                                       np.nan_to_num(b[key][:cut + 1].astype(float)))


class RunSegmentationTests(unittest.TestCase):
    def test_run_length_counts_contiguous_valid_bars(self):
        bars = synthetic_bars(50)
        bars.iloc[20] = np.nan
        ind = compute_indicators(bars)
        self.assertEqual(int(ind['run_length'][19]), 20)
        self.assertFalse(bool(ind['valid'][20]))
        self.assertEqual(int(ind['run_length'][20]), 0)
        self.assertEqual(int(ind['run_length'][21]), 1)

    def test_ema_state_restarts_after_a_data_break(self):
        bars = synthetic_bars(120)
        bars.iloc[60] = np.nan
        ind = compute_indicators(bars)
        c = bars['close'].to_numpy()
        self.assertTrue(np.isnan(ind['ema20'][79]))
        self.assertAlmostEqual(ind['ema20'][80], float(np.mean(c[61:81])), places=12)

    def test_nonpositive_and_inverted_bars_are_invalid(self):
        bars = synthetic_bars(40)
        bars.iloc[10, bars.columns.get_loc('close')] = 0.0
        bars.iloc[20, bars.columns.get_loc('high')] = bars['low'].iloc[20] - 1.0
        ind = compute_indicators(bars)
        self.assertFalse(bool(ind['valid'][10]))
        self.assertFalse(bool(ind['valid'][20]))

    def test_rejects_malformed_frames(self):
        with self.assertRaises(ValueError):
            compute_indicators(pd.DataFrame({'close': [1.0, 2.0]}))


class FeatureTests(unittest.TestCase):
    def test_shapes_and_exact_channel_order(self):
        bars = synthetic_bars(400)
        ind = compute_indicators(bars)
        t = 399
        p = t - 2
        seq, ctx = build_features(ind, bars, p, t)
        self.assertEqual(seq.shape, (60, 10))
        self.assertEqual(ctx.shape, (4,))
        c = bars['close'].to_numpy()
        j = t
        a, s = ind['atr20'][j - 1], ind['sigma20'][j - 1]
        row = seq[-1]
        self.assertAlmostEqual(row[0], ind['logret'][j] / s, places=12)
        self.assertAlmostEqual(row[1], np.log(c[j] / c[j - 5]) / (s * np.sqrt(5)), places=12)
        self.assertAlmostEqual(row[2], np.log(c[j] / c[j - 20]) / (s * np.sqrt(20)), places=12)
        self.assertAlmostEqual(row[3], (c[j] - ind['ema20'][j - 1]) / a, places=12)
        self.assertAlmostEqual(row[4], (c[j] - ind['ema50'][j - 1]) / a, places=12)
        self.assertAlmostEqual(row[5], (ind['ema50'][j - 1] - ind['ema50'][j - 11]) / a, places=12)
        self.assertAlmostEqual(row[6], ind['tr'][j] / a, places=12)
        self.assertAlmostEqual(row[7], ind['sigma5'][j - 1] / s, places=12)
        self.assertAlmostEqual(row[8], ind['er20'][j], places=12)

    def test_context_values_match_definition(self):
        bars = synthetic_bars(400)
        ind = compute_indicators(bars)
        t, p = 399, 396
        seq, ctx = build_features(ind, bars, p, t)
        h, l, c = bars['high'].to_numpy(), bars['low'].to_numpy(), bars['close'].to_numpy()
        self.assertAlmostEqual(ctx[0], (t - p) / 3, places=12)
        expected = (h[p - 20:p].max() - l[p:t + 1].min()) / ind['atr20'][p - 1]
        self.assertAlmostEqual(ctx[1], expected, places=12)
        self.assertAlmostEqual(ctx[2], (c[t] - h[t - 1]) / ind['atr20'][t - 1], places=12)
        self.assertAlmostEqual(ctx[3], ind['atr20'][t - 1] / c[t], places=12)

    def test_zero_range_bar_uses_the_defined_convention(self):
        bars = synthetic_bars(400)
        j = 380
        for column in ('open', 'high', 'low'):
            bars.iloc[j, bars.columns.get_loc(column)] = bars['close'].iloc[j]
        ind = compute_indicators(bars)
        seq, _ = build_features(ind, bars, 397, 399)
        self.assertAlmostEqual(seq[j - (399 - 59)][9], 0.5, places=12)

    def test_missing_history_raises_rather_than_imputing(self):
        bars = synthetic_bars(400)
        ind = compute_indicators(bars)
        with self.assertRaises(ValueError):
            build_features(ind, bars, 58, 60)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 -m unittest factor_lab.tests.test_s4_indicators -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'factor_lab.s4_indicators'`

- [ ] **Step 3: Write the indicator module**

Create `factor_lab/s4_indicators.py`:

```python
"""Kausale Indikatoren der S4-Mechanik, Spec Abschnitt 3.

Alles wird pro zusammenhaengendem gueltigem Lauf berechnet: ein Datenbruch
setzt die Indikatorinitialisierung zurueck, der EMA-Zustand wird also NICHT
ueber eine Luecke fortgeschrieben. Undefinierte Stellen bleiben nan bzw.
False; ein epsilon ersetzt hier keine fehlende Datenqualitaet.

ATR ist ein einfacher gleitender Durchschnitt des True Range, NICHT der
Wilder-ATR. ATR und Volatilitaet sind verschiedene Groessen.
"""
import numpy as np
import pandas as pd

WARMUP = 260
SEQUENCE = 60
COLUMNS = ('open', 'high', 'low', 'close')


def _validated(bars):
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError('bars must be a nonempty DataFrame')
    if tuple(bars.columns) != COLUMNS:
        raise ValueError(f'bars must have exactly the columns {COLUMNS}')
    if (not bars.index.is_monotonic_increasing or bars.index.has_duplicates
            or bars.index.hasnans):
        raise ValueError('bars need unique chronological timestamps')
    return bars


def _valid_mask(bars):
    values = bars.to_numpy(dtype=float)
    ok = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    return ok & (bars['high'].to_numpy(dtype=float) >= bars['low'].to_numpy(dtype=float))


def _runs(valid):
    """Start- und Endindex (exklusiv) jedes zusammenhaengenden gueltigen Laufs."""
    out, start = [], None
    for i, flag in enumerate(valid):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(valid)))
    return out


def _ema(closes, n):
    """alpha=2/(n+1), Initialwert = SMA der ersten n gueltigen Closes."""
    out = np.full(len(closes), np.nan)
    if len(closes) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    out[n - 1] = closes[:n].mean()
    for i in range(n, len(closes)):
        out[i] = alpha * closes[i] + (1.0 - alpha) * out[i - 1]
    return out


def _trailing(values, n, function):
    """function auf jedem Fenster [i-n+1 : i], beide Enden einschliesslich."""
    out = np.full(len(values), np.nan)
    for i in range(n - 1, len(values)):
        window = values[i - n + 1:i + 1]
        if np.isfinite(window).all():
            out[i] = function(window)
    return out


def compute_indicators(bars):
    """Indikatorarrays in der Laenge von bars, laufweise zurueckgesetzt."""
    bars = _validated(bars)
    size = len(bars)
    valid = _valid_mask(bars)
    result = {name: np.full(size, np.nan) for name in
              ('logret', 'tr', 'atr20', 'sigma5', 'sigma20', 'er20',
               'ema20', 'ema50', 'ema200')}
    result['valid'] = valid
    result['run_length'] = np.zeros(size, dtype=int)
    result['up'] = np.zeros(size, dtype=bool)

    high = bars['high'].to_numpy(dtype=float)
    low = bars['low'].to_numpy(dtype=float)
    close = bars['close'].to_numpy(dtype=float)

    for start, stop in _runs(valid):
        result['run_length'][start:stop] = np.arange(1, stop - start + 1)
        c = close[start:stop]
        h, l = high[start:stop], low[start:stop]
        if len(c) < 2:
            continue
        logret = np.full(len(c), np.nan)
        logret[1:] = np.log(c[1:] / c[:-1])
        tr = np.full(len(c), np.nan)
        previous = c[:-1]
        tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - previous),
                                    np.abs(l[1:] - previous)])
        atr20 = _trailing(tr, 20, np.mean)
        sigma5 = _trailing(logret, 5, lambda w: np.std(w, ddof=1))
        sigma20 = _trailing(logret, 20, lambda w: np.std(w, ddof=1))
        er20 = np.full(len(c), np.nan)
        for i in range(20, len(c)):
            denominator = np.abs(np.diff(c[i - 20:i + 1])).sum()
            if denominator > 0:
                er20[i] = abs(c[i] - c[i - 20]) / denominator
        ema20, ema50, ema200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
        up = np.zeros(len(c), dtype=bool)
        for i in range(10, len(c)):
            if np.isfinite([ema50[i], ema200[i], ema50[i - 10]]).all():
                up[i] = (ema50[i] > ema200[i] and ema50[i] > ema50[i - 10]
                         and c[i] > ema50[i])
        for name, values in (('logret', logret), ('tr', tr), ('atr20', atr20),
                             ('sigma5', sigma5), ('sigma20', sigma20), ('er20', er20),
                             ('ema20', ema20), ('ema50', ema50), ('ema200', ema200)):
            result[name][start:stop] = values
        result['up'][start:stop] = up
    return result
```

- [ ] **Step 4: Write the feature module**

Create `factor_lab/s4_features.py`:

```python
"""Featurevektor der S4-Mechanik, Spec Abschnitt 7.

60 abgeschlossene Bars bis einschliesslich Event t, zehn Kanaele in fester
Reihenfolge, dazu vier Kontextwerte. Historische Rolling-Features bleiben auf
ihrem damaligen Stand; die erst bei p bekannten Setup-Anker stehen
ausschliesslich im Kontext, nicht rueckwirkend in der Sequenz.

Fehlt ein benoetigtes Datum, wird ValueError ausgeloest. Das Event ist dann
als FEATURE_INVALID zu protokollieren und weder zu trainieren noch zu handeln
-- es wird NICHT imputiert und NICHT nachtraeglich nach seinem Ertrag
ausgewaehlt.
"""
import numpy as np

from factor_lab.s4_indicators import SEQUENCE

CHANNELS = 10
CONTEXT = 4


def build_features(ind, bars, p, t):
    """Sequenz [60,10] und Kontext [4] fuer ein Event bei t mit Vorstufe p."""
    if not (0 <= p < t < len(bars)):
        raise ValueError('need 0 <= p < t < len(bars)')
    first = t - SEQUENCE + 1
    if first - 20 < 0 or p - 20 < 0:
        raise ValueError('insufficient history for the S4 feature window')
    close = bars['close'].to_numpy(dtype=float)
    high = bars['high'].to_numpy(dtype=float)
    low = bars['low'].to_numpy(dtype=float)

    sequence = np.empty((SEQUENCE, CHANNELS), dtype=float)
    for offset, j in enumerate(range(first, t + 1)):
        a, s = ind['atr20'][j - 1], ind['sigma20'][j - 1]
        if not np.isfinite([a, s]).all() or a <= 0 or s <= 0:
            raise ValueError(f'undefined normalizer at bar {j}')
        span = high[j] - low[j]
        sequence[offset] = (
            ind['logret'][j] / s,
            np.log(close[j] / close[j - 5]) / (s * np.sqrt(5)),
            np.log(close[j] / close[j - 20]) / (s * np.sqrt(20)),
            (close[j] - ind['ema20'][j - 1]) / a,
            (close[j] - ind['ema50'][j - 1]) / a,
            (ind['ema50'][j - 1] - ind['ema50'][j - 11]) / a,
            ind['tr'][j] / a,
            ind['sigma5'][j - 1] / s,
            ind['er20'][j],
            (close[j] - low[j]) / span if span > 0 else 0.5,
        )
    if not np.isfinite(sequence).all():
        raise ValueError('nonfinite value in the S4 feature sequence')

    anchor = ind['atr20'][p - 1]
    entry_atr = ind['atr20'][t - 1]
    if not np.isfinite([anchor, entry_atr]).all() or anchor <= 0 or entry_atr <= 0:
        raise ValueError('undefined ATR anchor for the S4 event context')
    # Kontext 2 darf negativ sein, wenn der Kurs das alte Hoch nie unterschreitet.
    context = np.array([
        (t - p) / 3.0,
        (high[p - 20:p].max() - low[p:t + 1].min()) / anchor,
        (close[t] - high[t - 1]) / entry_atr,
        entry_atr / close[t],
    ], dtype=float)
    if not np.isfinite(context).all():
        raise ValueError('nonfinite value in the S4 event context')
    return sequence, context
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -3.12 -m unittest factor_lab.tests.test_s4_indicators -v`
Expected: PASS, 16 Tests.

- [ ] **Step 6: Commit**

```bash
git add factor_lab/s4_indicators.py factor_lab/s4_features.py factor_lab/tests/test_s4_indicators.py
git commit -m "feat(factor_lab): causal S4 indicators and fixed-order feature vector"
```

---

### Task 2: Eventgenerator

**Files:**
- Create: `factor_lab/s4_events.py`
- Create: `factor_lab/tests/test_s4_events.py`

**Interfaces:**
- Consumes: `compute_indicators`, `WARMUP`, `SEQUENCE` aus `s4_indicators`; `build_features` aus `s4_features`.
- Produces: `CONFIRM_MAX = 3`, `HOLD_N = 10`, `TARGET_Q = 1.5`, `COOLDOWN_BARS = 10`, `TOUCH_FRACTION = 0.25`, `EVENT_VERSION = 'S4-v1'`; `generate_events(bars, instrument, ind=None) -> list[dict]`. Jedes Event ist ein dict mit `id` (Tupel `(instrument, t, 'S4-v1')`), `t`, `signal_time`, `setup_start`, `side`, `R`, `q`, `N`, `sequence`, `context`, `earliest_entry_bar`, `feature_error` (`None` oder die Fehlermeldung).

Ein Event mit `feature_error is not None` wird **trotzdem emittiert**, traegt `sequence=None` und `context=None` und ist als `FEATURE_INVALID` zu behandeln. Der Eventstrom darf nicht davon abhaengen, ob Features berechenbar waren.

- [ ] **Step 1: Write the failing test**

Create `factor_lab/tests/test_s4_events.py`:

```python
"""Zustandsautomat des S4-Eventgenerators, Spec Abschnitt 6.

ALLE Daten in dieser Datei sind SYNTHETISCH und dienen ausschliesslich der
Mechanikpruefung. Sie stammen aus keiner Marktquelle.
"""
import unittest

import numpy as np
import pandas as pd

from factor_lab.s4_events import (
    COOLDOWN_BARS,
    EVENT_VERSION,
    HOLD_N,
    TARGET_Q,
    generate_events,
)
from factor_lab.s4_indicators import compute_indicators


def trending_bars(n=600, seed=7, drift=0.0012, noise=0.006):
    """Synthetischer Aufwaertstrend mit Ruecksetzern, damit S4 ueberhaupt feuert."""
    rng = np.random.default_rng(seed)
    steps = drift + noise * rng.standard_normal(n)
    steps[::37] -= 4 * noise           # regelmaessige Pullbacks erzwingen
    close = 100.0 * np.exp(np.cumsum(steps))
    width = close * 0.008
    high = close + width * rng.uniform(0.3, 1.0, n)
    low = close - width * rng.uniform(0.3, 1.0, n)
    open_ = low + (high - low) * rng.uniform(0.0, 1.0, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close},
                        index=pd.bdate_range('2008-01-02', periods=n))


class ContractTests(unittest.TestCase):
    def test_constants_are_frozen(self):
        self.assertEqual(COOLDOWN_BARS, 10)
        self.assertEqual(HOLD_N, 10)
        self.assertEqual(TARGET_Q, 1.5)
        self.assertEqual(EVENT_VERSION, 'S4-v1')

    def test_emitted_events_carry_the_frozen_contract(self):
        bars = trending_bars()
        ind = compute_indicators(bars)
        events = generate_events(bars, 'SYNTH', ind)
        self.assertGreater(len(events), 0)
        for event in events:
            self.assertEqual(event['id'], ('SYNTH', event['t'], 'S4-v1'))
            self.assertEqual(event['side'], 1)
            self.assertEqual(event['q'], 1.5)
            self.assertEqual(event['N'], 10)
            self.assertEqual(event['earliest_entry_bar'], event['t'] + 1)
            self.assertAlmostEqual(event['R'], ind['atr20'][event['t'] - 1], places=12)
            self.assertEqual(event['signal_time'], bars.index[event['t']])
            self.assertIn(event['t'] - event['setup_start'], (1, 2, 3))

    def test_no_event_before_the_warmup_is_complete(self):
        bars = trending_bars()
        for event in generate_events(bars, 'SYNTH'):
            self.assertGreaterEqual(event['t'], 259)


class TimingTests(unittest.TestCase):
    def test_events_respect_the_cooldown(self):
        events = generate_events(trending_bars(), 'SYNTH')
        times = [event['t'] for event in events]
        self.assertEqual(times, sorted(times))
        for earlier, later in zip(times, times[1:]):
            self.assertGreater(later, earlier + COOLDOWN_BARS)

    def test_label_windows_never_overlap_per_instrument(self):
        """Entry t+1, letzte gehaltene Bar t+10, naechstes Event fruehestens t+11."""
        events = generate_events(trending_bars(), 'SYNTH')
        for earlier, later in zip(events, events[1:]):
            self.assertGreaterEqual(later['t'] + 1, earlier['t'] + HOLD_N + 1)

    def test_confirmation_window_is_at_most_three_bars(self):
        for event in generate_events(trending_bars(), 'SYNTH'):
            self.assertLessEqual(event['t'], event['setup_start'] + 3)

    def test_generator_is_deterministic(self):
        bars = trending_bars()
        a = [event['t'] for event in generate_events(bars, 'SYNTH')]
        b = [event['t'] for event in generate_events(bars, 'SYNTH')]
        self.assertEqual(a, b)


class StateMachineTests(unittest.TestCase):
    def test_no_event_is_emitted_on_the_touch_bar_itself(self):
        for event in generate_events(trending_bars(), 'SYNTH'):
            self.assertGreater(event['t'], event['setup_start'])

    def test_events_use_only_information_available_at_t(self):
        """Bars nach t veraendern weder Zeitpunkt noch R eines Events."""
        bars = trending_bars()
        events = generate_events(bars, 'SYNTH')
        self.assertGreater(len(events), 2)
        cut = events[1]['t']
        tampered = bars.copy()
        tampered.iloc[cut + 1:] *= 1.7
        after = generate_events(tampered, 'SYNTH')
        early_before = [(e['t'], e['setup_start'], e['R']) for e in events if e['t'] <= cut]
        early_after = [(e['t'], e['setup_start'], e['R']) for e in after if e['t'] <= cut]
        self.assertEqual(early_before, early_after)

    def test_a_data_break_clears_the_pending_setup(self):
        bars = trending_bars()
        events = generate_events(bars, 'SYNTH')
        self.assertGreater(len(events), 0)
        target = events[0]
        broken = bars.copy()
        broken.iloc[target['setup_start'] + 1] = np.nan
        survivors = [e['t'] for e in generate_events(broken, 'SYNTH')]
        self.assertNotIn(target['t'], survivors)

    def test_no_event_inside_the_cooldown_window_after_a_data_break(self):
        """Regressionsschutz, kein isolierter Beweis der Cooldown-Persistenz.

        Die absolute Cooldown-Grenze soll eine Luecke ueberleben. Beobachtbar
        ist das durch die oeffentliche Schnittstelle nur eingeschraenkt, weil
        der Warmup-Gate nach einem Bruch ohnehin rund 260 Bars lang jedes
        Event unterdrueckt. Der Test haelt die Eigenschaft fest, beweist sie
        aber nicht unabhaengig vom Warmup.
        """
        bars = trending_bars()
        events = generate_events(bars, 'SYNTH')
        self.assertGreater(len(events), 1)
        first = events[0]['t']
        broken = bars.copy()
        broken.iloc[first + 2] = np.nan
        for event in generate_events(broken, 'SYNTH'):
            self.assertFalse(first < event['t'] <= first + COOLDOWN_BARS)

    def test_feature_failures_do_not_remove_events_from_the_stream(self):
        bars = trending_bars()
        events = generate_events(bars, 'SYNTH')
        for event in events:
            if event['feature_error'] is None:
                self.assertEqual(event['sequence'].shape, (60, 10))
                self.assertEqual(event['context'].shape, (4,))
            else:
                self.assertIsNone(event['sequence'])
                self.assertIsNone(event['context'])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 -m unittest factor_lab.tests.test_s4_events -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'factor_lab.s4_events'`

- [ ] **Step 3: Write minimal implementation**

Create `factor_lab/s4_events.py`:

```python
"""S4-Eventgenerator, Spec Abschnitt 6.

Long-Events auf bestaetigten Pullbacks in einem quantitativ definierten
Aufwaertstrend. Der Generator laeuft einmal chronologisch und haengt weder
von spaeteren Labels noch von einer spaeteren Modellentscheidung ab: ein
Event wird auch dann emittiert, wenn das Meta-Modell spaeter SKIP sagt oder
wenn die Features nicht berechenbar sind.

Der Cooldown ist unabhaengig davon, ob ein frueherer Trade Stop oder Target
erreicht hat. Sonst koennten Labels und Modellannahmen den zukuenftigen
Eventstrom veraendern. An Train-/Kalibrierungs-/Testgrenzen wird weder der
Setup-Zustand noch der Cooldown zurueckgesetzt.

Der Generator konsumiert die INDIKATORSICHT der Preise (split-konsistente
Historie). Roh-Bars fuer Fills sind Sache des Labelkernels; die beiden
Sichten ineinander umzurechnen ist Aufgabe des Datenadapters, nicht dieses
Moduls.
"""
import numpy as np

from factor_lab.s4_features import build_features
from factor_lab.s4_indicators import WARMUP, compute_indicators

CONFIRM_MAX = 3
HOLD_N = 10
TARGET_Q = 1.5
COOLDOWN_BARS = 10
TOUCH_FRACTION = 0.25
EVENT_VERSION = 'S4-v1'

_IDLE, _WAIT_CONFIRM = 'IDLE', 'WAIT_CONFIRM'


def _usable(ind, t):
    """Warmup-Gate: mindestens WARMUP zusammenhaengende gueltige Bars bis t."""
    if t < 1 or not ind['valid'][t] or ind['run_length'][t] < WARMUP:
        return False
    return np.isfinite([ind['atr20'][t - 1], ind['ema20'][t - 1], ind['ema50'][t - 1]]).all()


def generate_events(bars, instrument, ind=None):
    """Chronologischer Eventstrom eines Instruments; ein Durchlauf, kein Reset."""
    if ind is None:
        ind = compute_indicators(bars)
    close = bars['close'].to_numpy(dtype=float)
    high = bars['high'].to_numpy(dtype=float)
    low = bars['low'].to_numpy(dtype=float)

    events = []
    state, pending_p, blocked_through = _IDLE, None, -1
    for t in range(len(bars)):
        if not _usable(ind, t):
            state, pending_p = _IDLE, None
            continue
        if t <= blocked_through:
            continue

        if state == _WAIT_CONFIRM:
            # Invalidierung VOR Bestaetigung; keine neue Vorstufe am selben Bar.
            if t > pending_p + CONFIRM_MAX or close[t] <= ind['ema50'][t - 1]:
                state, pending_p = _IDLE, None
                continue
            if ind['up'][t] and close[t] > high[t - 1]:
                events.append(_emit(bars, ind, instrument, pending_p, t))
                blocked_through = t + COOLDOWN_BARS
                state, pending_p = _IDLE, None
                continue
            if t == pending_p + CONFIRM_MAX:
                state, pending_p = _IDLE, None
            continue

        # IDLE: nur die erste passende Beruehrung speichern, kein Event hier.
        a = ind['atr20'][t - 1]
        center = ind['ema20'][t - 1]
        touched = (low[t] <= center + TOUCH_FRACTION * a
                   and high[t] >= center - TOUCH_FRACTION * a)
        if (ind['up'][t - 1] and close[t - 1] > center and close[t] < close[t - 1]
                and touched and close[t] > ind['ema50'][t - 1]):
            pending_p, state = t, _WAIT_CONFIRM
    return events


def _emit(bars, ind, instrument, p, t):
    try:
        sequence, context = build_features(ind, bars, p, t)
        error = None
    except ValueError as problem:      # FEATURE_INVALID, Event bleibt im Strom
        sequence, context, error = None, None, str(problem)
    return {
        'id': (instrument, t, EVENT_VERSION),
        't': t,
        'signal_time': bars.index[t],
        'setup_start': p,
        'side': 1,
        'R': float(ind['atr20'][t - 1]),
        'q': TARGET_Q,
        'N': HOLD_N,
        'sequence': sequence,
        'context': context,
        'earliest_entry_bar': t + 1,
        'feature_error': error,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.12 -m unittest factor_lab.tests.test_s4_events -v`
Expected: PASS, 12 Tests.

**Falls `test_emitted_events_carry_the_frozen_contract` mit `len(events) == 0` scheitert**, feuert die synthetische Reihe nie. Dann `drift`, `noise` oder die Pullback-Periode der Fixture anpassen, bis Events entstehen — **niemals die Eventregel**. Die Regel ist praeregistriert; die Fixture ist ein Testhilfsmittel.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/s4_events.py factor_lab/tests/test_s4_events.py
git commit -m "feat(factor_lab): S4 pullback event generator with cooldown and purged state"
```

---

### Task 3: Labelkernel

**Files:**
- Create: `factor_lab/s4_labels.py`
- Create: `factor_lab/tests/test_s4_labels.py`

**Interfaces:**
- Consumes: Events in der Form, die `s4_events.generate_events` liefert (benoetigt werden `t`, `R`, `q`, `N`).
- Produces: `CENSORED`, `DATA_ERROR`, `ACTION_UNSUPPORTED`, `GAP_STOP`, `GAP_TARGET`, `STOP`, `TARGET`, `TIME`; `STOP_FIRST`, `TARGET_FIRST`; `label_event(event, raw_bars, actions, b, f, tie=STOP_FIRST) -> dict`.

`raw_bars` ist die **unveraenderte handelbare** OHLC-Sicht mit denselben Spalten wie in Task 1. `actions` ist ein `pd.DataFrame` mit denselben Zeilen und genau den Spalten `['split','dividend']`: `split` ist `s[k]`, neue Stueckzahl je altem Stueck, wirksam vor Open k, ohne Split `1.0`; `dividend` ist `d[k]`, Anspruch je Stueck auf der nach diesem Split geltenden Basis, sonst `0.0`.

Erfolgsrueckgabe ist ein dict mit `barrier_class`, `meta_label`, `net_R`, `entry`, `exit`, `exit_bar`, `reason`, `ambiguous`, `nonpositive_stop`, `label_available_after`. Misserfolg ist ein dict mit `status` aus den Fehlerkonstanten plus `event_id`.

- [ ] **Step 1: Write the failing test**

Create `factor_lab/tests/test_s4_labels.py`:

```python
"""Labelkernel der S4-Mechanik, Spec Abschnitt 8.

ALLE Daten in dieser Datei sind SYNTHETISCH und handgebaut, damit jeder
Ausgang exakt bestimmbar ist. Sie stammen aus keiner Marktquelle.
"""
import unittest

import numpy as np
import pandas as pd

from factor_lab.s4_labels import (
    ACTION_UNSUPPORTED,
    CENSORED,
    GAP_STOP,
    GAP_TARGET,
    STOP,
    TARGET,
    TARGET_FIRST,
    TIME,
    label_event,
)


def frame(rows):
    """rows: Liste von (open, high, low, close); Index ist ein Werktagskalender."""
    return pd.DataFrame(np.array(rows, dtype=float),
                        index=pd.bdate_range('2020-01-01', periods=len(rows)),
                        columns=['open', 'high', 'low', 'close'])


def plain_actions(bars):
    return pd.DataFrame({'split': 1.0, 'dividend': 0.0}, index=bars.index)


def event(t=0, R=1.0, q=1.5, N=10):
    return {'id': ('SYNTH', t, 'S4-v1'), 't': t, 'R': R, 'q': q, 'N': N}


class BarrierResolutionTests(unittest.TestCase):
    def test_target_hit_gives_plus_one_and_the_expected_net_r(self):
        # Low 99.2 liegt UEBER dem Stop 99 -- sonst waere die Bar mehrdeutig
        # und wuerde nach stop_first als Verlust aufgeloest.
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 99.2, 101.5)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['barrier_class'], 1)
        self.assertEqual(out['reason'], TARGET)
        self.assertAlmostEqual(out['entry'], 100.0)
        self.assertAlmostEqual(out['exit'], 101.5)
        self.assertAlmostEqual(out['net_R'], 1.5)
        self.assertEqual(out['meta_label'], 1)
        self.assertFalse(out['ambiguous'])

    def test_stop_hit_gives_minus_one(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 100.5, 98.9, 99)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['barrier_class'], -1)
        self.assertEqual(out['reason'], STOP)
        self.assertAlmostEqual(out['net_R'], -1.0)
        self.assertEqual(out['meta_label'], 0)

    def test_timeout_closes_at_the_last_held_bar(self):
        rows = [(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.1)] * 10
        bars = frame(rows)
        out = label_event(event(N=10), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['barrier_class'], 0)
        self.assertEqual(out['reason'], TIME)
        self.assertEqual(out['exit_bar'], 10)
        self.assertAlmostEqual(out['exit'], 100.1)
        self.assertEqual(out['label_available_after'], bars.index[10])


class GapTests(unittest.TestCase):
    def test_gap_below_the_stop_fills_at_the_open_not_at_the_stop(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (95, 96, 94, 95)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['reason'], GAP_STOP)
        self.assertAlmostEqual(out['exit'], 95.0)
        self.assertAlmostEqual(out['net_R'], -5.0)   # Gap verliert mehr als 1R
        self.assertFalse(out['ambiguous'])

    def test_gap_above_the_target_fills_at_the_open(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (104, 105, 103, 104)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['reason'], GAP_TARGET)
        self.assertAlmostEqual(out['exit'], 104.0)
        self.assertAlmostEqual(out['net_R'], 4.0)

    def test_the_open_is_resolved_before_high_and_low(self):
        """Eine Bar, die beide Barrieren enthaelt, aber unter dem Stop eroeffnet."""
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (98, 102, 97, 101)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['reason'], GAP_STOP)
        self.assertFalse(out['ambiguous'])


class AmbiguityTests(unittest.TestCase):
    def test_dual_touch_resolves_stop_first_and_is_flagged(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 98.5, 101)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['barrier_class'], -1)
        self.assertEqual(out['reason'], STOP)
        self.assertTrue(out['ambiguous'])

    def test_target_priority_variant_reverses_the_same_bar(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 98.5, 101)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0, tie=TARGET_FIRST)
        self.assertEqual(out['barrier_class'], 1)
        self.assertEqual(out['reason'], TARGET)
        self.assertTrue(out['ambiguous'])


class CostTests(unittest.TestCase):
    def test_haircut_moves_entry_up_and_exit_down(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 200, 99.9, 150)])
        out = label_event(event(), bars, plain_actions(bars), 0.001, 0.0)
        self.assertAlmostEqual(out['entry'], 100.1)
        self.assertAlmostEqual(out['exit'], (100.1 + 1.5) * 0.999, places=9)

    def test_commission_is_charged_on_both_sides(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 99.9, 101.5)])
        free = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        paid = label_event(event(), bars, plain_actions(bars), 0.0, 0.0001)
        self.assertLess(paid['net_R'], free['net_R'])
        expected = free['net_R'] - 0.0001 * (free['entry'] + free['exit'])
        self.assertAlmostEqual(paid['net_R'], expected, places=9)

    def test_costs_can_turn_a_barrier_win_into_a_negative_meta_label(self):
        """Kleines R gegen hohe Kosten: der Target-Treffer deckt sie nicht.

        R=0,1 bei O=100 und b=0,0005: der Eintritts-Haircut betraegt 0,05 und
        liegt damit unter R, der Trade wird also nicht sofort ausgestoppt.
        Die Provision von 10 bp je Seite ist hier bewusst hoch gewaehlt, um den
        Effekt zu zeigen -- sie ist kein Anbieterpreis.
        """
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 100.25, 99.99, 100.2)])
        out = label_event(event(R=0.1), bars, plain_actions(bars), 0.0005, 0.001)
        self.assertEqual(out['barrier_class'], 1)
        self.assertEqual(out['reason'], TARGET)
        self.assertEqual(out['meta_label'], 0)
        self.assertLess(out['net_R'], 0.0)


class CorporateActionTests(unittest.TestCase):
    def test_a_split_before_entry_converts_r_to_the_entry_share_basis(self):
        # R wird zu 0,5, Stop 49,5, Target 50,75. Low 49,6 liegt ueber dem Stop.
        bars = frame([(100, 100, 100, 100), (50, 50, 50, 50), (50, 51, 49.6, 50.8)])
        actions = plain_actions(bars)
        actions.iloc[1, actions.columns.get_loc('split')] = 2.0
        out = label_event(event(R=1.0), bars, actions, 0.0, 0.0)
        self.assertEqual(out['reason'], TARGET)
        self.assertAlmostEqual(out['net_R'], 1.5, places=9)

    def test_a_split_after_entry_leaves_net_r_unchanged(self):
        clean = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                       (100, 102, 99.5, 101.5)])
        out_clean = label_event(event(), clean, plain_actions(clean), 0.0, 0.0)
        split = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                       (50, 51, 49.75, 50.75)])
        actions = plain_actions(split)
        actions.iloc[2, actions.columns.get_loc('split')] = 2.0
        out_split = label_event(event(), split, actions, 0.0, 0.0)
        self.assertAlmostEqual(out_split['net_R'], out_clean['net_R'], places=9)

    def test_a_dividend_after_entry_is_booked_and_not_a_price_gain(self):
        rows = [(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.0)] * 10
        bars = frame(rows)
        actions = plain_actions(bars)
        actions.iloc[3, actions.columns.get_loc('dividend')] = 0.5
        out = label_event(event(N=10), bars, actions, 0.0, 0.0)
        self.assertEqual(out['reason'], TIME)
        self.assertAlmostEqual(out['net_R'], 0.5, places=9)

    def test_no_dividend_claim_when_buying_on_the_ex_day(self):
        rows = [(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.0)] * 10
        bars = frame(rows)
        actions = plain_actions(bars)
        actions.iloc[1, actions.columns.get_loc('dividend')] = 0.5
        out = label_event(event(N=10), bars, actions, 0.0, 0.0)
        self.assertAlmostEqual(out['net_R'], 0.0, places=9)

    def test_nonpositive_split_factor_is_rejected(self):
        bars = frame([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 102, 99, 101.5)])
        actions = plain_actions(bars)
        actions.iloc[2, actions.columns.get_loc('split')] = 0.0
        out = label_event(event(), bars, actions, 0.0, 0.0)
        self.assertEqual(out['status'], ACTION_UNSUPPORTED)


class CensoringTests(unittest.TestCase):
    def test_missing_entry_bar_is_censored_not_zero(self):
        bars = frame([(100, 100, 100, 100)])
        out = label_event(event(), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['status'], CENSORED)

    def test_a_gap_inside_the_holding_window_is_censored(self):
        rows = [(100, 100, 100, 100), (100, 100, 100, 100),
                (np.nan, np.nan, np.nan, np.nan)]
        bars = frame(rows)
        out = label_event(event(N=10), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['status'], CENSORED)
        self.assertEqual(out['event_id'], ('SYNTH', 0, 'S4-v1'))

    def test_truncated_history_before_the_time_exit_is_censored(self):
        rows = [(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.0)] * 5
        bars = frame(rows)
        out = label_event(event(N=10), bars, plain_actions(bars), 0.0, 0.0)
        self.assertEqual(out['status'], CENSORED)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 -m unittest factor_lab.tests.test_s4_labels -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'factor_lab.s4_labels'`

- [ ] **Step 3: Write minimal implementation**

Create `factor_lab/s4_labels.py`:

```python
"""Labelkernel der S4-Mechanik, Spec Abschnitt 8.

OHLC sind beobachtete Handelskurse, keine Bid-/Ask-Pfade. Das Fill-Modell ist
deshalb eine reproduzierbare NAEHERUNG, kein Nachweis erreichbarer Fills.
Ausgaenge sind ausdruecklich market-on-touch, keine garantierten passiven
Limit-Fills; Spread und Slippage werden am Fill beruecksichtigt.

Die Eroeffnungsorder wird nach Signal t vor der naechsten Sitzung aufgegeben.
E wird erst beim simulierten Fill bekannt; es findet keine Auswahl anhand des
spaeter bekannten O[t+1] statt und keine Verwendung spaeterer H/L fuer den
Entry.

Dieses Modul verbraucht die ROH-Sicht der Preise plus Kapitalmassnahmen. Die
Indikatorsicht ist Sache des Eventgenerators; beide ineinander umzurechnen ist
Aufgabe des Datenadapters.

Ein Ein-Stueck-Netto-R ist KEIN wirtschaftlicher 10.000-EUR-Trade:
Mindestgebuehren gehoeren in die mengenabhaengige Portfolio-P&L.
"""
import numpy as np
import pandas as pd

CENSORED = 'CENSORED'
DATA_ERROR = 'DATA_ERROR'
ACTION_UNSUPPORTED = 'ACTION_UNSUPPORTED'

GAP_STOP = 'GAP_STOP'
GAP_TARGET = 'GAP_TARGET'
STOP = 'STOP'
TARGET = 'TARGET'
TIME = 'TIME'

STOP_FIRST = 'stop_first'
TARGET_FIRST = 'target_first'

COLUMNS = ('open', 'high', 'low', 'close')
ACTION_COLUMNS = ('split', 'dividend')


def _failure(status, event):
    return {'status': status, 'event_id': event['id']}


def _bar_invalid(values):
    return not np.isfinite(values).all() or (np.asarray(values) <= 0).any()


def label_event(event, raw_bars, actions, b, f, tie=STOP_FIRST):
    """Barrier-Klasse, Meta-Label und Netto-R eines hypothetischen Long-Trades."""
    if tuple(raw_bars.columns) != COLUMNS or tuple(actions.columns) != ACTION_COLUMNS:
        raise ValueError('unexpected column layout')
    if not raw_bars.index.equals(actions.index):
        raise ValueError('bars and actions must share one calendar')
    if tie not in (STOP_FIRST, TARGET_FIRST):
        raise ValueError('tie must be stop_first or target_first')
    if not (0 <= b < 1) or not (0 <= f < 1):
        raise ValueError('cost parameters out of range')

    e = event['t'] + 1
    last = e + event['N'] - 1
    if e >= len(raw_bars):
        return _failure(CENSORED, event)

    values = raw_bars.to_numpy(dtype=float)
    split = actions['split'].to_numpy(dtype=float)
    dividend = actions['dividend'].to_numpy(dtype=float)
    if _bar_invalid(values[e]):
        return _failure(CENSORED, event)
    if not np.isfinite(split[e]) or split[e] <= 0:
        return _failure(ACTION_UNSUPPORTED, event)

    entry = values[e][0] * (1.0 + b)
    risk = event['R'] / split[e]
    if not np.isfinite([entry, risk]).all() or entry <= 0 or risk <= 0:
        return _failure(DATA_ERROR, event)

    shares, dividends = 1.0, 0.0
    stop, target = entry - risk, entry + event['q'] * risk
    # S<=0 ist kein erreichbarer Stop, aber kein Grund, den Trade zu entfernen.
    nonpositive_stop = stop <= 0

    exit_reference, outcome, reason, ambiguous, exit_bar = None, None, None, False, None
    for k in range(e, last + 1):
        if k >= len(raw_bars) or _bar_invalid(values[k]):
            return _failure(CENSORED, event)
        if not np.isfinite([split[k], dividend[k]]).all() or split[k] <= 0 or dividend[k] < 0:
            return _failure(ACTION_UNSUPPORTED, event)

        if k > e:
            shares *= split[k]
            stop /= split[k]
            target /= split[k]
            dividends += shares * dividend[k]
            # Kein Dividendenanspruch bei Kauf erst am Ex-Tag e.

        bar_open, high, low, close = values[k]
        if bar_open <= stop:                       # Open ist zeitlich zuerst
            exit_reference, outcome, reason = bar_open, -1, GAP_STOP
            exit_bar = k
            break
        if bar_open >= target:
            exit_reference, outcome, reason = bar_open, 1, GAP_TARGET
            exit_bar = k
            break

        hit_stop, hit_target = low <= stop, high >= target
        ambiguous = hit_stop and hit_target
        stop_wins = hit_stop and (tie == STOP_FIRST or not hit_target)
        if stop_wins:
            exit_reference, outcome, reason = stop, -1, STOP
            exit_bar = k
            break
        if hit_target:
            exit_reference, outcome, reason = target, 1, TARGET
            exit_bar = k
            break
        if k == last:
            exit_reference, outcome, reason = close, 0, TIME
            exit_bar = k
            break

    if exit_reference is None:
        return _failure(CENSORED, event)

    exit_price = exit_reference * (1.0 - b)
    exit_notional = shares * exit_price
    net = exit_notional - entry + dividends - f * entry - f * exit_notional
    return {
        'barrier_class': outcome,
        'meta_label': 1 if net > 0 else 0,
        'net_R': net / risk,
        'entry': entry,
        'exit': exit_price,
        'exit_bar': exit_bar,
        'reason': reason,
        'ambiguous': bool(ambiguous),
        'nonpositive_stop': bool(nonpositive_stop),
        'label_available_after': raw_bars.index[exit_bar],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.12 -m unittest factor_lab.tests.test_s4_labels -v`
Expected: PASS, 19 Tests.

Falls ein handgebautes Fixture den erwarteten Ausgang nicht trifft, **die Erwartung nachrechnen, nicht den Kernel anpassen**. Die Fixtures sind so gewaehlt, dass Entry, Stop und Target exakt bestimmbar sind.

- [ ] **Step 5: Gesamte S4-Suite und Regression des Bestands**

Zwei Konventionen liegen im Testverzeichnis nebeneinander, und das muss der Befehl beruecksichtigen. **`test_stats.py`, `test_costs.py` und `test_portfolio.py` sind assert-basierte Skripte mit `check_*`-Funktionen und ohne `unittest.TestCase`** — unter `python -m unittest` sammeln sie stillschweigend null Tests ein und melden trotzdem Erfolg. Sie gehoeren nicht in die unittest-Liste, sondern einzeln als Skript ausgefuehrt.

Zuerst die unittest-Module (neun, die tatsaechlich Tests beitragen):

```bash
py -3.12 -m unittest factor_lab.tests.test_s4_indicators factor_lab.tests.test_s4_events factor_lab.tests.test_s4_labels factor_lab.tests.test_daily_comparison factor_lab.tests.test_daily_models factor_lab.tests.test_features_2x2 factor_lab.tests.test_models_2x2 factor_lab.tests.test_evaluate_2x2 factor_lab.tests.test_run_feature_model_2x2 -v
```

Dann die skriptartigen Pruefdateien einzeln, mit `PYTHONPATH` auf die Repo-Wurzel, weil ein Skriptpfad-Aufruf sonst das Paket nicht importieren kann:

```bash
PYTHONPATH=. bash -c 'for f in factor_lab/tests/test_stats.py factor_lab/tests/test_costs.py factor_lab/tests/test_portfolio.py; do echo "== $f"; py -3.12 "$f" >/dev/null || echo "FAILED $f"; done'
```

Expected: PASS beim ersten Befehl, keine `FAILED`-Zeile beim zweiten. Ein Fehlschlag im Bestand bedeutet, dass dieser Branch etwas bewegt hat, was er nicht anfassen darf — **nicht** den Bestandstest anpassen.

**Im Bericht ist die Zahl der beitragenden Module zu nennen, nicht die Zahl der aufgerufenen Dateien.** Eine Suite, die null Tests einsammelt, hat nichts abgesichert.

- [ ] **Step 6: Commit**

```bash
git add factor_lab/s4_labels.py factor_lab/tests/test_s4_labels.py
git commit -m "feat(factor_lab): S4 label kernel with gaps, ties, splits and dividends"
```

---

### Task 4: Mechanikbericht

**Files:**
- Create: `docs/superpowers/s4-pullback-mechanics-2026-09-13-results.md`

Kein Code, kein Lauf auf echten Daten. Der Bericht haelt fest, was geprueft wurde und was offen bleibt.

- [ ] **Step 1: Bericht schreiben**

Gliederung, gefuellt mit den tatsaechlichen Testergebnissen:

1. **Auftrag und Abgrenzung** — Mechanik geprueft, kein Census, kein Modell, keine Optimierung.
2. **Datenbefund** — der versiegelte Snapshot erfuellt den OHLC- und Corporate-Action-Vertrag **nicht**: zwei Spalten (`price`, `rate_pa_pct`), keine O/H/L, kein Volumen, keine Dividenden, keine Splits. Zusaetzlich ist die Preisreihe rueckadjustiert (`auto_adjust=True`; SPY 2007-01-03 bei 98,87 gegen real rund 141, TLT 48,23 gegen rund 88), was Abschnitt 8.1 als handelbare Kursbasis ausdruecklich verbietet. Der einzige Volumenfund im Projekt, `market_control_system/data_layer/alpaca_client.py:187-188`, erfindet `bid_volume = volume/2`.
3. **Was die Mechanik jetzt kann** — Modulliste, Testzahl je Datei, welche Spec-Abschnitte abgedeckt sind.
4. **Geprueftes Verhalten** — namentlich: Kausalitaet der Indikatoren, Laufsegmentierung und EMA-Neustart nach Datenbruch, Warmup-Grenze, exakte Kanalreihenfolge, Fristen und Cooldown, keine Emission auf der Beruehrungsbar, Gaps ueber und unter den Barrieren, Open-vor-High/Low-Aufloesung, mehrdeutige Barrieren in beiden Varianten, Gebuehren auf beiden Seiten, Kosten die einen Barrier-Gewinn in ein negatives Meta-Label drehen, Splits vor und nach dem Entry, Dividenden inklusive des entfallenden Anspruchs am Ex-Tag, Zensierung statt erfundener Nullrendite.
5. **Gefundene Mechanikfehler** — ehrlich auflisten, auch wenn keine gefunden wurden.
6. **Der Census bleibt offen** — und warum das kein Nebenbefund ist: die Renditeherleitung aus Spec Abschnitt 11 verlangt bei 0,25% Risiko je Trade einen Ø-Nettoertrag von `48/Trades`, und bei q=1,5 gilt vor Kosten `Ø R ~ 2,5w - 1`. 100 Trades pro Jahr verlangen damit rund 59% Trefferquote, 50 Trades rund 78%, 200 Trades rund 50%; Break-even liegt bei 40%. Der Census beantwortet also direkt, welche Trefferquote dieses Setup liefern muesste, damit 12% ueberhaupt im Bereich liegen.
7. **Was den Census freischalten wuerde** — ein separat versionierter Roh-OHLC-Snapshot mit `auto_adjust=False` und `actions=True`, getrennt vom versiegelten Snapshot, nicht damit vergleichbar, kostenlos ueber die bereits installierte Abhaengigkeit. Ausdruecklich **nicht** in diesem Plan enthalten.
8. **Grenzen** — die Liste aus Spec Abschnitt 13, plus: alle Tests laufen auf synthetischen Fixtures; eine korrekte Mechanik ist kein Hinweis auf einen Edge; der Forschungszaehler steht weiter bei 145 und wird von dieser Stufe nicht erhoeht, weil kein Vergleich gerechnet wurde.

- [ ] **Step 2: Dateiliste pruefen**

Run: `git status --porcelain`
Expected: nur die geplanten Quell-, Test- und Dokumentdateien. Nichts aus `research_archive/`, keine `.pkl`, keine Laufartefakte.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/s4-pullback-mechanics-2026-09-13-results.md
git commit -m "docs(factor_lab): S4 mechanics verification report and open census gate"
```
