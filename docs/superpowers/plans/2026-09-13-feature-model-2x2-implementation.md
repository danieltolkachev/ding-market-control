# 2×2 Features × Modell — Implementierungsplan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Das in der Spec festgelegte 2×2-Raster (bestehende 5 Kanaele vs. erweiterte 10 Kanaele, gekreuzt mit Ridge vs. einer festen LightGBM-Rezeptur) auf dem eingefrorenen Preissnapshot ausfuehren und mit einer vorab festgelegten Primaermetrik gegen die immer-investierte Kontrolle berichten.

**Architecture:** Drei neue, additive Module unter `factor_lab/` plus ein Runner. Kein bestehendes Modul wird veraendert — insbesondere nicht `daily_models.py` oder `daily_comparison.py`, weil `run_horizon_comparison.py` seine Ergebnisse per `assert_reproduction` gegen versiegelte Artefakte prueft und dabei `fit_predict_models` aufruft. Die neue Modellschicht ist kanalzahl-generisch (`lambda = 1.0 * p`) und wird in Task 2 per Aequivalenztest gegen den bestehenden Ridge-Pfad verifiziert. Die fuenf expandierenden Fenster sind ein **Refit-Fahrplan**: jede Auswertungszeile bekommt ihre Prognose von dem Fenster, das sie besitzt; simuliert wird danach **eine** durchgehende Kursbahn ueber 2015-01-02 bis 2026-08-31, damit die Zelle exakt gegen dieselbe Kontrolle laeuft wie im Horizon-Vergleich.

**Tech Stack:** Python 3.12 (`py -3.12`), numpy 2.5.2, pandas 3.0.5, lightgbm 4.7.0. Kein torch in diesem Raster (kein LSTM), kein scikit-learn.

**Spec:** `docs/superpowers/specs/2026-09-11-feature-model-2x2-design.md`

## Global Constraints

- Alles mit `py -3.12` ausfuehren. Niemals bare `python`/`pip` — die loesen auf dieser Maschine inkonsistent auf.
- Tests im Stil der Module, die hier erweitert werden: `unittest`, Ausfuehrung mit `py -3.12 -m unittest factor_lab.tests.<modul> -v`.
- Deutsche Kommentare/Docstrings in ASCII (ue/oe/ae/ss), passend zum Bestand.
- **Nichts veraendern** an: `daily_models.py`, `daily_comparison.py`, `horizon_data.py`, `stats.py`, `costs.py`, `portfolio.py`, `signals.py`, `data_snapshot.py`, `registration*.py`, `run_trend_*`, allem unter `factor_lab/audit/`, `factor_lab/audit_data/`, `market_control_system/` und `research_archive/`.
- **Snapshot ausschliesslich lesend:** `C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl`, Inhalts-SHA256 `36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89`. Der Runner prueft den Hash und bricht bei Abweichung ab.
- **Eingefrorene Bestandteile, in keiner Zelle veraendert:** Label `returns.shift(-2)/vol`, geclippt `[-10, 10]`, `label_delay = 2`; Allokation `make_targets` (inverse Vola, 10-%-Vola-Deckel, Long/Flat-Gate `predictions > 0`); Ausfuehrung `simulate` (Entscheidung t, Fill t+1); Halte-Kadenz **21** Handelstage; Bruttoexposure maximal **1,0x**.
- **Kosten:** 3 bp primaer, 15 bp Sensitivitaet. Keine weiteren Kostenstufen.
- **Regularisierung:** Ridge `lambda = 1.0 * p` mit `p = 20 * Kanalzahl`. Ergibt 100 bei 5 Kanaelen (heutiger Wert) und 200 bei 10.
- **GBM-Rezeptur ist vor dem ersten Lauf eingefroren** (Task 3) und wird danach nicht mehr angefasst: `num_boost_round=300`, `learning_rate=0.05`, `num_leaves=15`, `min_data_in_leaf=200`, `feature_fraction=0.7`, `bagging_fraction=0.7`, `bagging_freq=1`, `seed=7`, `deterministic=True`, `force_row_wise=True`, `num_threads=2`, `verbosity=-1`, `objective='regression'`.
- **Keine Hyperparametersuche, keine fuenfte Zelle, kein Nachjustieren nach Sicht der Ergebnisse.** Ein Nullbefund wird als Nullbefund berichtet.
- **Mehrfachtest-Buchfuehrung:** dieser Lauf fuegt 4 Vergleiche zu den bestehenden 141 unkorrigierten hinzu. Keine Korrektur wird behauptet.
- Nach jedem Task committen. Commit-Nachrichten enden mit einer Leerzeile und dann `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Nicht pushen, keinen PR oeffnen.

## Zwei Umsetzungsentscheidungen, die die Spec offen laesst

Beide sind hier festgelegt, **bevor** Ergebnisse sichtbar sind, und gehoeren in den Ergebnisbericht:

1. **Zeilenausrichtung.** Die erweiterten Kanaele brauchen laengeren Vorlauf (E1 `shift(147)`, E4 `rolling(252)`) als die bestehenden fuenf. Wuerde jede Zelle ihren eigenen maximalen Zeilensatz nutzen, saehen A/B und C/D unterschiedliche Trainingsmengen und die Feature-Achse waere mit einer Stichprobenaenderung vermischt. **Festlegung:** alle vier Zellen laufen auf dem Zeilensatz der erweiterten Kanaele. Der Auswertungszeitraum ab 2015-01-02 ist davon nicht betroffen (2.932 Zeilen in beiden Faellen); es aendern sich nur rund 125 Trainingszeilen ganz am Anfang der Historie. „Zelle A bleibt bitgleich" wird daher als **Arithmetik**-Aussage verifiziert (Task 2, Aequivalenztest gegen `daily_models.fit_predict_models`), nicht als Aussage ueber die Trainingsstichprobe.

2. **Aggregation ueber die fuenf Fenster.** Die Fenster sind zusammenhaengend und ueberdecken 2015-01-02 bis 2026-08-31 disjunkt. Die Primaermetrik wird deshalb auf der **zusammengefuegten** Monatsdifferenz-Reihe berechnet (Stationary-Block-Bootstrap, erwartete Blocklaenge 6 Monate, 10.000 Ziehungen, Seed 7) — das ist ein monatsgleichgewichteter Durchschnitt ueber die Fenster statt eines fenstergleichgewichteten. Bei Blockgroessen von 586/586/586/586/588 Zeilen ist der Unterschied vernachlaessigbar. Die fensterweise Aufschluesselung wird zusaetzlich deskriptiv berichtet.

## Dateistruktur

| Datei | Verantwortung |
|---|---|
| `factor_lab/features_2x2.py` | Baut aus Preisen den ausgerichteten Datensatz: `X_base` [D,A,20,5], `X_ext` [D,A,20,10], Label, Vola, Renditen. Sonst nichts. |
| `factor_lab/models_2x2.py` | Standardisierung (kanalzahl-generisch), Ridge mit `lambda = 1.0*p`, GBM mit eingefrorener Rezeptur. Kennt weder Fenster noch Portfolios. |
| `factor_lab/evaluate_2x2.py` | Fensteraufteilung mit Purge und Zusammensetzen der Prognosen ueber alle Fenster. Kennt keine Dateien und kein CLI. |
| `factor_lab/run_feature_model_2x2.py` | CLI, Snapshot-Pruefung, Kontrolle, Simulation, Statistik, Artefakte, Bericht. |
| `factor_lab/tests/test_features_2x2.py` | Kausalitaet, E1–E5-Definitionen, Ausrichtung, SPY-Sonderfall. |
| `factor_lab/tests/test_models_2x2.py` | Ridge-Aequivalenz gegen Bestand, Lambda-Skalierung, GBM-Determinismus. |
| `factor_lab/tests/test_evaluate_2x2.py` | Fenstergrenzen, Purge, keine Zukunftsnutzung, vollstaendige Ueberdeckung. |

---

### Task 1: Erweiterte Kanaele und ausgerichteter Datensatz

**Files:**
- Create: `factor_lab/features_2x2.py`
- Create: `factor_lab/tests/test_features_2x2.py`

**Interfaces:**
- Consumes: nichts aus frueheren Tasks.
- Produces: `BASE_HORIZONS = (1, 5, 21, 63, 126)`; `EXTENDED_NAMES = ('skip_momentum', 'cross_rank', 'vol_ratio', 'drawdown', 'spy_corr')`; `extended_channels(prices, market='SPY') -> dict[str, pd.DataFrame]`; `build_2x2_dataset(prices, market='SPY') -> dict` mit den Schluesseln `dates` (DatetimeIndex, Laenge D), `X_base` (float32 [D,A,20,5]), `X_ext` (float32 [D,A,20,10]), `y` (float32 [D,A]), `vol` (DataFrame D×A), `returns` (DataFrame ueber den vollen Preisindex), `columns` (Liste der Assetnamen).

- [ ] **Step 1: Write the failing test**

Create `factor_lab/tests/test_features_2x2.py`:

```python
"""Kanaeldefinitionen, Kausalitaet und Zeilenausrichtung des 2x2-Datensatzes."""
import unittest

import numpy as np
import pandas as pd

from factor_lab.daily_comparison import build_dataset
from factor_lab.features_2x2 import (
    BASE_HORIZONS,
    EXTENDED_NAMES,
    build_2x2_dataset,
    extended_channels,
)


def synthetic_prices(periods=900, assets=('SPY', 'AAA', 'BBB'), seed=3):
    """Deterministische, positive Preisreihen auf einem Boersenkalender."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2010-01-04', periods=periods)
    data = {}
    for offset, name in enumerate(assets):
        steps = 0.0004 * (offset + 1) + 0.01 * rng.standard_normal(periods)
        data[name] = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=index)


class ExtendedChannelTests(unittest.TestCase):
    def test_channel_names_and_shapes(self):
        prices = synthetic_prices()
        channels = extended_channels(prices)
        self.assertEqual(tuple(channels), EXTENDED_NAMES)
        for frame in channels.values():
            self.assertTrue(frame.index.equals(prices.index))
            self.assertTrue(frame.columns.equals(prices.columns))

    def test_skip_momentum_matches_definition(self):
        prices = synthetic_prices()
        returns = prices.pct_change(fill_method=None)
        vol = returns.rolling(63, min_periods=63).std().clip(lower=1e-6)
        expected = (prices.shift(21) / prices.shift(147) - 1) / vol / np.sqrt(126)
        got = extended_channels(prices)['skip_momentum']
        pd.testing.assert_frame_equal(got, expected, check_names=False)

    def test_cross_rank_is_bounded_and_purely_cross_sectional(self):
        prices = synthetic_prices()
        channels = extended_channels(prices)
        rank = channels['cross_rank'].dropna(how='all')
        self.assertTrue(((rank >= -1 - 1e-12) & (rank <= 1 + 1e-12)).all().all())
        # Drei Assets: exakt ein Minimum bei -1, ein Maximum bei +1 je Tag.
        row = rank.dropna().iloc[-1]
        np.testing.assert_allclose(sorted(row.to_numpy()), [-1.0, 0.0, 1.0])

    def test_spy_correlation_channel_is_one_for_the_market_itself(self):
        prices = synthetic_prices()
        spy = extended_channels(prices)['spy_corr']['SPY'].dropna()
        self.assertGreater(len(spy), 0)
        np.testing.assert_allclose(spy.to_numpy(), 1.0, atol=1e-10)

    def test_channels_are_causal(self):
        """Aenderung eines zukuenftigen Preises darf Kanaele bei t nicht bewegen."""
        prices = synthetic_prices()
        cut = prices.index[-30]
        tampered = prices.copy()
        tampered.loc[prices.index[-10]:] *= 1.5
        before = extended_channels(prices).items()
        after = dict(extended_channels(tampered))
        for name, frame in before:
            pd.testing.assert_frame_equal(
                frame.loc[:cut], after[name].loc[:cut], check_names=False
            )


class DatasetAlignmentTests(unittest.TestCase):
    def test_base_block_equals_existing_builder_on_shared_rows(self):
        prices = synthetic_prices()
        data = build_2x2_dataset(prices)
        reference = build_dataset(prices)
        shared = data['dates']
        self.assertTrue(shared.isin(reference['dates']).all())
        take = reference['dates'].get_indexer(shared)
        np.testing.assert_allclose(data['X_base'], reference['X'][take], atol=0, rtol=0)
        np.testing.assert_array_equal(
            np.isfinite(data['y']), np.isfinite(reference['y'][take])
        )
        np.testing.assert_allclose(
            np.nan_to_num(data['y']), np.nan_to_num(reference['y'][take]), atol=0, rtol=0
        )

    def test_extended_block_carries_base_channels_first(self):
        data = build_2x2_dataset(synthetic_prices())
        np.testing.assert_allclose(data['X_ext'][..., :5], data['X_base'], atol=0, rtol=0)
        self.assertEqual(data['X_ext'].shape[-1], len(BASE_HORIZONS) + len(EXTENDED_NAMES))

    def test_all_extended_features_are_finite_on_kept_rows(self):
        data = build_2x2_dataset(synthetic_prices())
        self.assertTrue(np.isfinite(data['X_ext']).all())
        self.assertGreater(len(data['dates']), 0)

    def test_alignment_starts_later_than_the_base_only_dataset(self):
        prices = synthetic_prices()
        self.assertGreater(
            build_2x2_dataset(prices)['dates'][0], build_dataset(prices)['dates'][0]
        )


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 -m unittest factor_lab.tests.test_features_2x2 -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'factor_lab.features_2x2'`

- [ ] **Step 3: Write minimal implementation**

Create `factor_lab/features_2x2.py`:

```python
"""Kausale Kanaele fuer das 2x2-Raster: bestehende fuenf plus fuenf erweiterte.

Alle Kanaele nutzen ausschliesslich Daten bis einschliesslich t und sind rein
preisabgeleitet — der Snapshot enthaelt weder Volumen noch OHLC noch Makro.
Beide Merkmalsbloecke werden auf **demselben** Zeilensatz zurueckgegeben (dem
der erweiterten Kanaele), damit die Feature-Achse nicht mit einer Aenderung der
Trainingsstichprobe vermischt wird.
"""
import numpy as np
import pandas as pd

BASE_HORIZONS = (1, 5, 21, 63, 126)
EXTENDED_NAMES = ('skip_momentum', 'cross_rank', 'vol_ratio', 'drawdown', 'spy_corr')
SEQUENCE = 20


def _validated(prices):
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError('prices must be a nonempty DataFrame')
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('Prices need unique chronological dates')
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Prices must be finite and positive')
    return prices


def extended_channels(prices, market='SPY'):
    """Die fuenf zusaetzlichen Kanaele E1-E5 der Spec, Abschnitt 3."""
    prices = _validated(prices)
    if market not in prices.columns:
        raise ValueError(f'market column {market!r} missing')
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(63, min_periods=63).std().clip(lower=1e-6)

    # E1: 126-Tage-Rendite endend vor 21 Tagen; meidet kurzfristige Umkehr.
    skip = (prices.shift(21) / prices.shift(147) - 1) / vol / np.sqrt(126)

    # E2: tagesweiser Rang von E1 ueber die Instrumente, linear auf [-1, +1].
    counts = skip.notna().sum(axis=1)
    ranks = skip.rank(axis=1, method='average')
    cross_rank = (2 * (ranks - 1).div(counts - 1, axis=0) - 1).where(counts > 1)

    # E3: Vola-Regimewechsel ohne separates Regimemodell.
    vol_short = returns.rolling(21, min_periods=21).std().clip(lower=1e-6)
    vol_long = returns.rolling(126, min_periods=126).std().clip(lower=1e-6)
    vol_ratio = np.log(vol_short / vol_long)

    # E4: Abstand zum 252-Tage-Hoch als Marktzustand.
    peak = prices.rolling(252, min_periods=252).max()
    drawdown = (prices / peak - 1) / vol / np.sqrt(252)

    # E5: Marktkopplung. Fuer das Marktinstrument selbst konstant 1 — die
    # Rolling-Korrelation kann numerisch minimal ueber 1 laufen, daher Clip.
    spy_corr = returns.rolling(63, min_periods=63).corr(returns[market]).clip(-1.0, 1.0)

    channels = {'skip_momentum': skip, 'cross_rank': cross_rank, 'vol_ratio': vol_ratio,
                'drawdown': drawdown, 'spy_corr': spy_corr}
    return {name: channels[name].astype(float) for name in EXTENDED_NAMES}


def build_2x2_dataset(prices, market='SPY'):
    """Ausgerichteter Datensatz mit Basis- und erweitertem Merkmalsblock.

    Label und Basiskanaele sind identisch zu ``daily_comparison.build_dataset``;
    nur der behaltene Zeilensatz ist enger, weil E1/E4 laengeren Vorlauf haben.
    """
    prices = _validated(prices)
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(63, min_periods=63).std().clip(lower=1e-6)
    base = [(prices.pct_change(h, fill_method=None) / vol / np.sqrt(h)).to_numpy()
            for h in BASE_HORIZONS]
    extra = [frame.to_numpy() for frame in extended_channels(prices, market).values()]
    features = np.stack(base + extra, axis=-1)
    last = SEQUENCE - 1
    positions = [d for d in range(last, len(prices))
                 if np.isfinite(features[d - last:d + 1]).all()]
    if not positions:
        raise ValueError('No complete 20-day extended feature sequences')
    window = np.stack([features[d - last:d + 1].transpose(1, 0, 2) for d in positions])
    window = window.astype('float32')
    y = (returns.shift(-2) / vol).clip(-10, 10).iloc[positions].to_numpy(dtype='float32')
    dates = prices.index[positions]
    return {'dates': dates, 'X_base': window[..., :len(BASE_HORIZONS)].copy(),
            'X_ext': window, 'y': y, 'vol': vol.loc[dates], 'returns': returns,
            'columns': list(prices.columns)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 -m unittest factor_lab.tests.test_features_2x2 -v`
Expected: PASS, 9 Tests.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/features_2x2.py factor_lab/tests/test_features_2x2.py
git commit -m "feat(factor_lab): causal extended channels and aligned 2x2 dataset"
```

---

### Task 2: Kanalzahl-generische Standardisierung und Ridge

**Files:**
- Create: `factor_lab/models_2x2.py`
- Create: `factor_lab/tests/test_models_2x2.py`

**Interfaces:**
- Consumes: `factor_lab.features_2x2.build_2x2_dataset` (nur im Test).
- Produces: `training_pairs(y, first_test, label_delay=2) -> np.ndarray [n,2]`; `fit_scaler(X, pairs) -> tuple[np.ndarray, np.ndarray]`; `standardize(X, pairs, mean, scale) -> np.ndarray [n,20,C]`; `RIDGE_LAMBDA_PER_FEATURE = 1.0`; `fit_predict_ridge(X, y, first_test, last_test, label_delay=2) -> np.ndarray [last_test-first_test, A]`.

**Warum ein Aequivalenztest:** `daily_models.fit_predict_models` ist durch `run_horizon_comparison.assert_reproduction` an versiegelte Artefakte gebunden und darf nicht angefasst werden. Die neue, generische Implementierung muss deshalb beweisen, dass sie bei 5 Kanaelen dieselbe Arithmetik ausfuehrt.

- [ ] **Step 1: Write the failing test**

Create `factor_lab/tests/test_models_2x2.py`:

```python
"""Aequivalenz zum bestehenden Ridge-Pfad, Lambda-Skalierung, GBM-Determinismus."""
import unittest

import numpy as np
import pandas as pd

from factor_lab.daily_comparison import build_dataset
from factor_lab.daily_models import fit_predict_models
from factor_lab.features_2x2 import build_2x2_dataset
from factor_lab.models_2x2 import (
    RIDGE_LAMBDA_PER_FEATURE,
    fit_predict_ridge,
    fit_scaler,
    standardize,
    training_pairs,
)


def synthetic_prices(periods=700, assets=('SPY', 'AAA', 'BBB'), seed=11):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2011-01-03', periods=periods)
    data = {}
    for offset, name in enumerate(assets):
        steps = 0.0003 * (offset + 1) + 0.012 * rng.standard_normal(periods)
        data[name] = 50.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=index)


class RidgeEquivalenceTests(unittest.TestCase):
    def test_matches_existing_daily_models_ridge_exactly(self):
        """Bei 5 Kanaelen ist lambda = 1.0 * 100 der heutige Wert 100."""
        prices = synthetic_prices()
        reference = build_dataset(prices)
        first = len(reference['dates']) - 60
        expected = fit_predict_models(
            reference['X'], reference['y'], first_test=first, label_delay=2, seed=7
        )['ridge']
        got = fit_predict_ridge(
            reference['X'], reference['y'], first_test=first,
            last_test=len(reference['dates']), label_delay=2
        )
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-12)

    def test_lambda_scales_with_feature_count(self):
        self.assertEqual(RIDGE_LAMBDA_PER_FEATURE, 1.0)
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 40
        for block in ('X_base', 'X_ext'):
            out = fit_predict_ridge(data[block], data['y'], first, len(data['dates']))
            self.assertEqual(out.shape, (40, data['y'].shape[1]))
            self.assertTrue(np.isfinite(out).all())

    def test_stronger_shrinkage_pulls_predictions_toward_the_intercept(self):
        """Sanity: mehr Merkmale bei gleichem lambda pro Merkmal bleibt stabil."""
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 40
        base = fit_predict_ridge(data['X_base'], data['y'], first, len(data['dates']))
        ext = fit_predict_ridge(data['X_ext'], data['y'], first, len(data['dates']))
        self.assertFalse(np.allclose(base, ext))
        self.assertLess(np.abs(ext).max(), 50.0)


class ScalerTests(unittest.TestCase):
    def test_training_pairs_respect_the_label_delay(self):
        y = np.zeros((10, 2), dtype='float32')
        y[9] = np.nan
        pairs = training_pairs(y, first_test=8, label_delay=2)
        self.assertEqual(int(pairs[:, 0].max()), 6)
        self.assertEqual(len(pairs), 14)

    def test_scaler_pools_timesteps_and_guards_zero_scale(self):
        X = np.ones((6, 2, 20, 3), dtype='float32')
        X[..., 1] = np.arange(20, dtype='float32')
        y = np.zeros((6, 2), dtype='float32')
        pairs = training_pairs(y, first_test=6, label_delay=2)
        mean, scale = fit_scaler(X, pairs)
        self.assertEqual(mean.shape, (3,))
        np.testing.assert_allclose(mean[0], 1.0)
        np.testing.assert_allclose(scale[0], 1.0)  # konstante Spalte -> Guard
        np.testing.assert_allclose(mean[1], 9.5)
        standardized = standardize(X, pairs, mean, scale)
        self.assertEqual(standardized.shape, (len(pairs), 20, 3))
        self.assertLessEqual(np.abs(standardized).max(), 10.0)

    def test_standardize_clips_at_ten(self):
        X = np.zeros((4, 1, 20, 1), dtype='float32')
        X[2, 0, :, 0] = 1000.0
        y = np.zeros((4, 1), dtype='float32')
        pairs = training_pairs(y, first_test=4, label_delay=2)
        mean, scale = fit_scaler(X, pairs)
        self.assertLessEqual(np.abs(standardize(X, pairs, mean, scale)).max(), 10.0)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 -m unittest factor_lab.tests.test_models_2x2 -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'factor_lab.models_2x2'`

- [ ] **Step 3: Write minimal implementation**

Create `factor_lab/models_2x2.py`:

```python
"""Prognosemodelle des 2x2-Rasters: Ridge und eine feste LightGBM-Rezeptur.

Kanalzahl-generisch. Ridge nutzt ``lambda = 1.0 * p`` mit ``p = 20 * Kanaele``,
damit die Schrumpfung pro Koeffizient beim Wechsel von 5 auf 10 Kanaele
konstant bleibt und der Feature-Effekt nicht mit einem Regularisierungseffekt
vermischt wird. Bei 5 Kanaelen ergibt das exakt die 100 des Bestands; die
Aequivalenz ist in ``tests/test_models_2x2.py`` gegen
``daily_models.fit_predict_models`` verifiziert.

Der Aufrufer liefert bereits vola-normierte, geclippte Label; dieses Modul
verschiebt, normiert und clippt y nicht. Label i ist ab Schluss
i + label_delay beobachtbar.
"""
import numpy as np

RIDGE_LAMBDA_PER_FEATURE = 1.0
SEQUENCE = 20
BATCH = 256


def _checked(X, y, first_test, last_test, label_delay):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 4 or X.shape[2] != SEQUENCE or y.shape != X.shape[:2]:
        raise ValueError('expected X[D,A,20,C] and y[D,A]')
    days = y.shape[0]
    if (isinstance(label_delay, (bool, np.bool_))
            or not isinstance(label_delay, (int, np.integer)) or label_delay < 2):
        raise ValueError('label_delay must be an integer >=2')
    if (not isinstance(first_test, (int, np.integer))
            or not label_delay <= first_test < last_test <= days):
        raise ValueError('need label_delay <= first_test < last_test <= D')
    if y.shape[1] == 0:
        raise ValueError('no assets')
    return X, y


def training_pairs(y, first_test, label_delay=2):
    """(Tag, Asset)-Paare mit endlichem, bei first_test bereits reifem Label."""
    valid = np.isfinite(np.asarray(y))
    pairs = np.argwhere(valid[:first_test - label_delay + 1])
    if not len(pairs):
        raise ValueError('no finite training labels before the test block')
    return pairs


def _raw(X, pairs):
    batch = np.asarray(X[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
    if not np.isfinite(batch).all():
        raise ValueError('nonfinite features in eligible or prediction examples')
    return batch


def fit_scaler(X, pairs):
    """Kanalweise Mittelwerte/Populationsstreuungen, ueber Zeitschritte gepoolt.

    Zwei Durchlaeufe vermeiden Ausloeschung in der Varianz, ohne eine
    vollstaendige Kopie der Trainingsdaten zu materialisieren.
    """
    channels = X.shape[3]
    total = np.zeros(channels)
    for start in range(0, len(pairs), BATCH):
        total += _raw(X, pairs[start:start + BATCH]).sum(axis=(0, 1))
    mean = total / (len(pairs) * SEQUENCE)
    squares = np.zeros(channels)
    for start in range(0, len(pairs), BATCH):
        squares += ((_raw(X, pairs[start:start + BATCH]) - mean) ** 2).sum(axis=(0, 1))
    scale = np.sqrt(squares / (len(pairs) * SEQUENCE))
    scale[scale == 0] = 1
    return mean, scale


def standardize(X, pairs, mean, scale):
    return np.clip((_raw(X, pairs) - mean) / scale, -10, 10)


def _prediction_pairs(assets, day):
    return np.column_stack((np.full(assets, day), np.arange(assets)))


def fit_predict_ridge(X, y, first_test, last_test, label_delay=2):
    """Ridge-Prognosen fuer die Tage [first_test, last_test).

    Einmalige Anpassung auf allen bei first_test reifen Labeln, ungestrafter
    Achsenabschnitt. Nach dem Clipping zentriert: geclippte Trainings-
    koordinaten haben nicht zwingend Mittelwert 0.
    """
    X, y = _checked(X, y, first_test, last_test, label_delay)
    pairs = training_pairs(y, first_test, label_delay)
    mean, scale = fit_scaler(X, pairs)
    features = SEQUENCE * X.shape[3]
    gram, rhs = np.zeros((features, features)), np.zeros(features)
    z_sum, y_sum = np.zeros(features), 0.0
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start:start + BATCH]
        z = standardize(X, chunk, mean, scale).reshape(len(chunk), features)
        gram += z.T @ z
        targets = np.asarray(y[chunk[:, 0], chunk[:, 1]], dtype=np.float64)
        rhs += z.T @ targets
        z_sum += z.sum(axis=0)
        y_sum += targets.sum()
    z_mean, y_mean = z_sum / len(pairs), y_sum / len(pairs)
    penalty = RIDGE_LAMBDA_PER_FEATURE * features
    weights = np.linalg.solve(
        gram - np.outer(z_sum, z_mean) + penalty * np.eye(features),
        rhs - z_sum * y_mean,
    )
    intercept = y_mean - z_mean @ weights
    assets = y.shape[1]
    out = np.empty((last_test - first_test, assets), dtype=np.float64)
    for offset, day in enumerate(range(first_test, last_test)):
        z = standardize(X, _prediction_pairs(assets, day), mean, scale)
        out[offset] = z.reshape(assets, features) @ weights + intercept
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 -m unittest factor_lab.tests.test_models_2x2 -v`
Expected: PASS, 6 Tests. Der erste Test ist der entscheidende: er beweist die Ridge-Aequivalenz zum eingefrorenen Bestand.

- [ ] **Step 5: Regression der beruehrten Nachbarn**

Run: `py -3.12 -m unittest factor_lab.tests.test_daily_models factor_lab.tests.test_daily_comparison factor_lab.tests.test_horizon_models -v`
Expected: PASS. Nichts am Bestand darf sich bewegt haben.

- [ ] **Step 6: Commit**

```bash
git add factor_lab/models_2x2.py factor_lab/tests/test_models_2x2.py
git commit -m "feat(factor_lab): channel-generic ridge with lambda scaled to feature count"
```

---

### Task 3: Eingefrorene LightGBM-Rezeptur

**Files:**
- Modify: `factor_lab/models_2x2.py` (anhaengen)
- Modify: `factor_lab/tests/test_models_2x2.py` (Testklasse anhaengen)

**Interfaces:**
- Consumes: `training_pairs`, `fit_scaler`, `standardize`, `_checked`, `_prediction_pairs` aus Task 2.
- Produces: `GBM_PARAMS: dict`, `GBM_ROUNDS = 300`; `fit_predict_gbm(X, y, first_test, last_test, label_delay=2) -> np.ndarray [last_test-first_test, A]`.

**Festlegung vor dem ersten Lauf:** die Rezeptur unten ist die der Spec, Abschnitt 4. Sie wird nach dem ersten Lauf nicht mehr veraendert. Beide Modelle sehen dieselbe flache, standardisierte, geclippte Matrix (20 × Kanaele) — die Variante „Baum sieht nur den letzten Zeitschritt" ist ausdruecklich **kein** Teil dieses Rasters.

- [ ] **Step 1: Write the failing test**

An `factor_lab/tests/test_models_2x2.py` anhaengen, **vor** dem `if __name__ == '__main__':`-Block, und den Import oben um `GBM_PARAMS, GBM_ROUNDS, fit_predict_gbm` erweitern:

```python
class GbmTests(unittest.TestCase):
    def test_recipe_is_frozen(self):
        self.assertEqual(GBM_ROUNDS, 300)
        self.assertEqual(GBM_PARAMS['learning_rate'], 0.05)
        self.assertEqual(GBM_PARAMS['num_leaves'], 15)
        self.assertEqual(GBM_PARAMS['min_data_in_leaf'], 200)
        self.assertEqual(GBM_PARAMS['feature_fraction'], 0.7)
        self.assertEqual(GBM_PARAMS['bagging_fraction'], 0.7)
        self.assertEqual(GBM_PARAMS['bagging_freq'], 1)
        self.assertEqual(GBM_PARAMS['seed'], 7)
        self.assertTrue(GBM_PARAMS['deterministic'])

    def test_predictions_have_the_expected_shape_and_are_finite(self):
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 30
        out = fit_predict_gbm(data['X_ext'], data['y'], first, len(data['dates']))
        self.assertEqual(out.shape, (30, data['y'].shape[1]))
        self.assertTrue(np.isfinite(out).all())

    def test_repeated_runs_are_bit_identical(self):
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 30
        a = fit_predict_gbm(data['X_base'], data['y'], first, len(data['dates']))
        b = fit_predict_gbm(data['X_base'], data['y'], first, len(data['dates']))
        np.testing.assert_array_equal(a, b)

    def test_uses_no_labels_from_the_test_block(self):
        """Label im Testblock veraendern darf die Prognosen nicht bewegen."""
        data = build_2x2_dataset(synthetic_prices())
        first = len(data['dates']) - 30
        clean = fit_predict_gbm(data['X_base'], data['y'], first, len(data['dates']))
        tampered = data['y'].copy()
        tampered[first - 1:] = 9.0
        dirty = fit_predict_gbm(data['X_base'], tampered, first, len(data['dates']))
        np.testing.assert_array_equal(clean, dirty)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 -m unittest factor_lab.tests.test_models_2x2 -v`
Expected: FAIL mit `ImportError: cannot import name 'GBM_PARAMS'`

- [ ] **Step 3: Write minimal implementation**

An `factor_lab/models_2x2.py` anhaengen (und `import lightgbm as lgb` oben ergaenzen):

```python
# Eingefrorene Rezeptur, Spec Abschnitt 4. Bewusst klein fuer die kleine
# Stichprobe. Keine Hyperparametersuche; nach dem ersten Lauf unveraendert.
GBM_ROUNDS = 300
GBM_PARAMS = {
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 15,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'seed': 7,
    'deterministic': True,
    'force_row_wise': True,
    'num_threads': 2,
    'verbosity': -1,
}


def fit_predict_gbm(X, y, first_test, last_test, label_delay=2):
    """LightGBM-Prognosen fuer die Tage [first_test, last_test).

    Sieht exakt dieselbe flache, standardisierte, geclippte Matrix wie Ridge.
    Baeume brauchen die 20 stark korrelierten Verzoegerungen nicht; sie
    bekommen sie trotzdem, weil die Modellachse sonst mit einer
    Repraesentationsaenderung vermischt waere.
    """
    X, y = _checked(X, y, first_test, last_test, label_delay)
    pairs = training_pairs(y, first_test, label_delay)
    mean, scale = fit_scaler(X, pairs)
    features = SEQUENCE * X.shape[3]
    design = np.empty((len(pairs), features))
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start:start + BATCH]
        design[start:start + len(chunk)] = standardize(
            X, chunk, mean, scale).reshape(len(chunk), features)
    targets = np.asarray(y[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
    booster = lgb.train(GBM_PARAMS, lgb.Dataset(design, label=targets),
                        num_boost_round=GBM_ROUNDS)
    assets = y.shape[1]
    out = np.empty((last_test - first_test, assets), dtype=np.float64)
    for offset, day in enumerate(range(first_test, last_test)):
        z = standardize(X, _prediction_pairs(assets, day), mean, scale)
        out[offset] = booster.predict(z.reshape(assets, features))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 -m unittest factor_lab.tests.test_models_2x2 -v`
Expected: PASS, 10 Tests.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/models_2x2.py factor_lab/tests/test_models_2x2.py
git commit -m "feat(factor_lab): frozen lightgbm recipe on identical flattened inputs"
```

---

### Task 4: Expandierende Fenster mit Purge und zusammengesetzte Prognosen

**Files:**
- Create: `factor_lab/evaluate_2x2.py`
- Create: `factor_lab/tests/test_evaluate_2x2.py`

**Interfaces:**
- Consumes: `fit_predict_ridge`, `fit_predict_gbm` aus Tasks 2/3.
- Produces: `N_WINDOWS = 5`; `expanding_windows(first_test, last_day, n_windows=N_WINDOWS) -> list[tuple[int, int]]`; `CELLS` (Tupel von `(name, feature_block, model)`); `predict_over_windows(X, y, windows, model, label_delay=2, progress=None) -> np.ndarray [last-first, A]`.

`CELLS` ist genau:

```python
CELLS = (('A_base_ridge', 'X_base', 'ridge'),
         ('B_base_gbm', 'X_base', 'gbm'),
         ('C_ext_ridge', 'X_ext', 'ridge'),
         ('D_ext_gbm', 'X_ext', 'gbm'))
```

**Purge:** es wird kein zusaetzlicher Purge-Parameter eingefuehrt. `training_pairs` schneidet bereits bei `first_test - label_delay` ab; das **ist** der Purge von `label_delay` Tagen an jeder Fenstergrenze und es gibt genau eine Stelle, an der er passiert.

- [ ] **Step 1: Write the failing test**

Create `factor_lab/tests/test_evaluate_2x2.py`:

```python
"""Fenstergrenzen, Purge und Zusammensetzen der Prognosen."""
import unittest

import numpy as np
import pandas as pd

from factor_lab.evaluate_2x2 import (
    CELLS,
    N_WINDOWS,
    expanding_windows,
    predict_over_windows,
)
from factor_lab.features_2x2 import build_2x2_dataset


def synthetic_prices(periods=700, assets=('SPY', 'AAA', 'BBB'), seed=23):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2011-01-03', periods=periods)
    data = {}
    for offset, name in enumerate(assets):
        steps = 0.0002 * (offset + 1) + 0.011 * rng.standard_normal(periods)
        data[name] = 80.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=index)


class WindowTests(unittest.TestCase):
    def test_windows_partition_the_evaluation_range_exactly(self):
        windows = expanding_windows(1501, 4433)
        self.assertEqual(len(windows), N_WINDOWS)
        self.assertEqual(windows[0][0], 1501)
        self.assertEqual(windows[-1][1], 4433)
        for (_, end), (start, _) in zip(windows, windows[1:]):
            self.assertEqual(end, start)
        sizes = [end - start for start, end in windows]
        self.assertEqual(sum(sizes), 2932)
        # Der Rest der Ganzzahldivision faellt komplett in den letzten Block,
        # die uebrigen sind gleich gross. 2932 = 5*586 + 2, also 588 am Ende.
        self.assertEqual(len(set(sizes[:-1])), 1)
        self.assertEqual(sizes[-1] - sizes[0], 2932 % N_WINDOWS)

    def test_block_sizes_for_the_real_evaluation_range(self):
        sizes = [end - start for start, end in expanding_windows(1501, 4433)]
        self.assertEqual(sizes, [586, 586, 586, 586, 588])

    def test_rejects_degenerate_ranges(self):
        with self.assertRaises(ValueError):
            expanding_windows(100, 100)
        with self.assertRaises(ValueError):
            expanding_windows(100, 102, n_windows=5)

    def test_cells_are_the_four_preregistered_ones(self):
        self.assertEqual([name for name, _, _ in CELLS],
                         ['A_base_ridge', 'B_base_gbm', 'C_ext_ridge', 'D_ext_gbm'])
        self.assertEqual({block for _, block, _ in CELLS}, {'X_base', 'X_ext'})
        self.assertEqual({model for _, _, model in CELLS}, {'ridge', 'gbm'})


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.data = build_2x2_dataset(synthetic_prices())
        self.days = len(self.data['dates'])
        self.first = self.days - 100
        self.windows = expanding_windows(self.first, self.days)

    def test_composed_predictions_cover_every_evaluation_row(self):
        out = predict_over_windows(
            self.data['X_base'], self.data['y'], self.windows, 'ridge')
        self.assertEqual(out.shape, (100, self.data['y'].shape[1]))
        self.assertTrue(np.isfinite(out).all())

    def test_each_block_equals_its_own_single_window_fit(self):
        out = predict_over_windows(
            self.data['X_base'], self.data['y'], self.windows, 'ridge')
        from factor_lab.models_2x2 import fit_predict_ridge
        start, end = self.windows[2]
        direct = fit_predict_ridge(self.data['X_base'], self.data['y'], start, end)
        np.testing.assert_allclose(out[start - self.first:end - self.first], direct)

    def test_no_window_sees_labels_from_its_own_or_later_blocks(self):
        clean = predict_over_windows(
            self.data['X_base'], self.data['y'], self.windows, 'ridge')
        tampered = self.data['y'].copy()
        last_start = self.windows[-1][0]
        tampered[last_start - 1:] = 7.0
        dirty = predict_over_windows(
            self.data['X_base'], tampered, self.windows, 'ridge')
        np.testing.assert_array_equal(clean, dirty)

    def test_unknown_model_name_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_over_windows(
                self.data['X_base'], self.data['y'], self.windows, 'lstm')


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 -m unittest factor_lab.tests.test_evaluate_2x2 -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'factor_lab.evaluate_2x2'`

- [ ] **Step 3: Write minimal implementation**

Create `factor_lab/evaluate_2x2.py`:

```python
"""Refit-Fahrplan des 2x2-Rasters: fuenf expandierende Fenster mit Purge.

Die Fenster sind kein Portfolio-Neustart. Jede Auswertungszeile bekommt ihre
Prognose von dem Fenster, das sie besitzt; simuliert wird spaeter **eine**
durchgehende Kursbahn, damit die Zelle gegen exakt dieselbe immer-investierte
Kontrolle laeuft wie im Horizon-Vergleich.

Der Purge von ``label_delay`` Tagen an jeder Fenstergrenze passiert an genau
einer Stelle: ``models_2x2.training_pairs`` schneidet bei
``first_test - label_delay`` ab.
"""
import numpy as np

from factor_lab.models_2x2 import fit_predict_gbm, fit_predict_ridge

N_WINDOWS = 5
CELLS = (('A_base_ridge', 'X_base', 'ridge'),
         ('B_base_gbm', 'X_base', 'gbm'),
         ('C_ext_ridge', 'X_ext', 'ridge'),
         ('D_ext_gbm', 'X_ext', 'gbm'))
_FITTERS = {'ridge': fit_predict_ridge, 'gbm': fit_predict_gbm}


def expanding_windows(first_test, last_day, n_windows=N_WINDOWS):
    """Zusammenhaengende, disjunkte Testbloecke ueber [first_test, last_day).

    Alle Bloecke bis auf den letzten sind gleich gross; die Restzeilen der
    Ganzzahldivision gehen vollstaendig an den letzten Block. Bei 2932 Zeilen
    und fuenf Fenstern sind das 586/586/586/586/588 — der Unterschied ist
    gegenueber der Blockgroesse vernachlaessigbar und die Regel bleibt eine
    Zeile Code statt einer Verteilungslogik.
    """
    if not isinstance(n_windows, (int, np.integer)) or n_windows < 1:
        raise ValueError('n_windows must be a positive integer')
    span = int(last_day) - int(first_test)
    if span < n_windows:
        raise ValueError('evaluation range too short for the requested windows')
    size = span // n_windows
    bounds = [first_test + i * size for i in range(n_windows)] + [last_day]
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_windows)]


def predict_over_windows(X, y, windows, model, label_delay=2, progress=None):
    """Prognosen ueber alle Fenster, aneinandergesetzt zu [last-first, A]."""
    if model not in _FITTERS:
        raise ValueError(f'unknown model {model!r}')
    fitter = _FITTERS[model]
    first, last = windows[0][0], windows[-1][1]
    out = np.empty((last - first, y.shape[1]), dtype=np.float64)
    for number, (start, end) in enumerate(windows, start=1):
        if progress is not None:
            progress(f'window {number}/{len(windows)}: train < {start}, test [{start},{end})')
        out[start - first:end - first] = fitter(X, y, start, end, label_delay)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 -m unittest factor_lab.tests.test_evaluate_2x2 -v`
Expected: PASS, 8 Tests.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/evaluate_2x2.py factor_lab/tests/test_evaluate_2x2.py
git commit -m "feat(factor_lab): expanding refit windows with purged training cutoff"
```

---

### Task 5: Runner mit Pilotmodus

**Files:**
- Create: `factor_lab/run_feature_model_2x2.py`

**Interfaces:**
- Consumes: `build_2x2_dataset`, `CELLS`, `expanding_windows`, `predict_over_windows`; aus dem Bestand `load_trend_snapshot`, `snapshot_content_sha256`, `make_targets`, `simulate`, `annualized_stats`, `monthly_log_returns`, `stationary_block_bootstrap`, `write_json`.
- Produces: ein CLI `py -3.12 -u factor_lab/run_feature_model_2x2.py --snapshot <pfad> --output-root <pfad> [--pilot] [--force]` und ein Ausgabeverzeichnis `feature_2x2_<UTC-Zeitstempel>/`.

**Pilotmodus (`--pilot`):** genau eine Zelle (`A_base_ridge`) ueber genau ein Fenster (das erste), keine Statistik, kein `COMPLETE`. Er dient ausschliesslich der Laufzeitmessung vor dem vollen Lauf, wie in Spec Abschnitt 9 verlangt.

- [ ] **Step 1: Write the runner**

Create `factor_lab/run_feature_model_2x2.py`:

```python
"""Vorab festgelegtes 2x2-Raster Features x Modell; vier Zellen, keine Suche.

Bekannte Historie, kein frisches Holdout. Vier Zellen kommen zu den bestehenden
141 unkorrigierten Vergleichen hinzu; keine Mehrfachtestkorrektur wird behauptet.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shutil

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

# Der dokumentierte Aufruf ist ein Skriptpfad, nicht -m; dann liegt factor_lab/
# auf sys.path statt des Repo-Wurzelverzeichnisses. Sechs der acht run_*.py
# tragen dieselbe Zeile aus demselben Grund.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_lab.daily_comparison import make_targets, simulate
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.evaluate_2x2 import CELLS, N_WINDOWS, expanding_windows, predict_over_windows
from factor_lab.features_2x2 import BASE_HORIZONS, EXTENDED_NAMES, build_2x2_dataset
from factor_lab.models_2x2 import GBM_PARAMS, GBM_ROUNDS, RIDGE_LAMBDA_PER_FEATURE
from factor_lab.run_daily_comparison import write_json
from factor_lab.stats import annualized_stats, monthly_log_returns, stationary_block_bootstrap

EXPECTED_SNAPSHOT_SHA256 = '36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89'
TEST_START = '2015-01-01'
HOLDING_INTERVAL = 21
COSTS = (3., 15.)
CONTROL = 'always_long'
LIMITATIONS = [
    'Previously observed history and universe; new window cuts are not a fresh holdout',
    'Four preregistered cells added to 141 existing uncorrected comparisons; no multiplicity adjustment',
    'Descriptive bootstrap intervals on overlapping monthly data; not a significance test',
    'The selection pipeline falsification audit covers trend-etf-v2 screening only; it does not transfer to this grid',
    'No whole-share rounding, no broker minimum fees, no FX, no taxes, no financing',
    'Five extended channels are a narrow selection; a null result refutes these five in this pipeline, not cross-sectional information in general',
    'The SPY correlation channel is constant 1 for SPY itself; market coupling is uninformative for the market instrument',
    'Volatility ceiling and historical drawdown do not guarantee future risk',
    'The 2026 year is partial; gross exposure capped at 1.0x, below the permitted 1.25x',
]


def build_config(content_hash, symbols, dates, windows, pilot):
    return {
        'classification': 'preregistered_2x2_on_previously_observed_history',
        'pilot': pilot,
        'cells': [list(cell) for cell in CELLS],
        'n_windows': N_WINDOWS,
        'windows': [list(w) for w in windows],
        'base_channels': list(BASE_HORIZONS),
        'extended_channels': list(EXTENDED_NAMES),
        'ridge_lambda_per_feature': RIDGE_LAMBDA_PER_FEATURE,
        'gbm_params': GBM_PARAMS,
        'gbm_rounds': GBM_ROUNDS,
        'label': 'returns.shift(-2)/vol, clipped to [-10,10]',
        'label_delay': 2,
        'holding_interval': HOLDING_INTERVAL,
        'execution': 'next close; drift between scheduled fills',
        'allocation': 'make_targets: inverse volatility, long/flat gate, 10% volatility cap',
        'cost_bp': list(COSTS),
        'primary_cost_bp': 3.0,
        'control': CONTROL,
        'primary_metric': 'annualized relative geometric growth vs control, pooled monthly deltas',
        'gross_ceiling': 1.,
        'user_leverage_ceiling': 1.25,
        'volatility_cap': .10,
        'return_target': .12,
        'drawdown_evaluation_limit': .15,
        'symbols': symbols,
        'test_start': str(dates[0]),
        'test_end': str(dates[-1]),
        'snapshot_content_sha256': content_hash,
        'bootstrap': {'block_months': 6, 'n_boot': 10000, 'seed': 7},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'lightgbm': lgb.__version__},
        'limitations': LIMITATIONS,
    }


def copy_sources(out):
    root = Path(__file__).resolve().parents[1]
    for relative in ['factor_lab/features_2x2.py', 'factor_lab/models_2x2.py',
                     'factor_lab/evaluate_2x2.py', 'factor_lab/run_feature_model_2x2.py',
                     'factor_lab/daily_comparison.py', 'factor_lab/stats.py',
                     'docs/superpowers/specs/2026-09-11-feature-model-2x2-design.md',
                     'docs/superpowers/plans/2026-09-13-feature-model-2x2-implementation.md']:
        source = root / relative
        dest = out / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--pilot', action='store_true',
                        help='one cell, one window, timing only')
    parser.add_argument('--force', action='store_true',
                        help='allow a snapshot whose content hash differs')
    args = parser.parse_args()

    dfs = load_trend_snapshot(args.snapshot)
    content_hash = snapshot_content_sha256(dfs)
    if content_hash != EXPECTED_SNAPSHOT_SHA256 and not args.force:
        raise ValueError(f'Snapshot content hash {content_hash} is not the sealed one')
    symbols = sorted(set(dfs) - {'IRX'})
    prices = pd.concat({s: dfs[s]['price'] for s in symbols}, axis=1, sort=True).dropna()
    data = build_2x2_dataset(prices)
    days = len(data['dates'])
    first = int(data['dates'].searchsorted(pd.Timestamp(TEST_START)))
    if first < 252 or first >= days - 252:
        raise ValueError('Insufficient training/evaluation data')
    windows = expanding_windows(first, days)
    if args.pilot:
        windows = windows[:1]
    dates = data['dates'][windows[0][0]:windows[-1][1]]
    cash = dfs['IRX']['rate_pa_pct'].reindex(prices.index).ffill().shift(1).loc[dates] / 100 / 252
    if cash.isna().any():
        raise ValueError('Missing causally available cash rate')

    stamp = datetime.now(timezone.utc).strftime('feature_2x2_%Y%m%d_%H%M%S_%f')
    out = Path(args.output_root) / (stamp + ('_pilot' if args.pilot else ''))
    out.mkdir(parents=True, exist_ok=False)
    config = build_config(content_hash, symbols, dates, windows, args.pilot)
    write_json(out / 'configuration.json', config)
    copy_sources(out)
    print(f'OUTPUT: {out}', flush=True)
    print(f'Rows {days}, evaluation {len(dates)} from {dates[0].date()} to {dates[-1].date()}',
          flush=True)

    cells = CELLS[:1] if args.pilot else CELLS
    predictions = {}
    for name, block, model in cells:
        started = datetime.now(timezone.utc)
        print(f'{name}: {model} on {data[block].shape[-1]} channels', flush=True)
        values = predict_over_windows(
            data[block], data['y'], windows, model,
            progress=lambda text, cell=name: print(f'  {cell}: {text}', flush=True))
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
        print(f'{name}: fitted in {seconds:.1f}s', flush=True)
        if values.shape != (len(dates), len(symbols)) or not np.isfinite(values).all():
            raise ValueError(f'Invalid forecasts for {name}')
        frame = pd.DataFrame(values, index=dates, columns=symbols)
        frame.to_csv(out / f'{name}_predictions.csv')
        predictions[name] = frame

    if args.pilot:
        print('Pilot complete: timing only, no statistics, no COMPLETE marker.', flush=True)
        return

    signals = dict(predictions)
    signals[CONTROL] = pd.DataFrame(1., index=dates, columns=symbols)
    mask = pd.Series(np.arange(len(dates)) % HOLDING_INTERVAL == 0, index=dates)
    mask.to_csv(out / 'decisions.csv')
    summary = {'configuration': config, 'cells': {}, 'primary': {}, 'per_window': {},
               'prediction_metrics': {}}
    labels = data['y'][windows[0][0]:windows[-1][1]]
    valid = np.isfinite(labels)
    paths = {}
    for name, values in signals.items():
        targets = make_targets(values, data['vol'].loc[dates], data['returns'])
        targets.to_csv(out / f'{name}_targets.csv')
        summary['cells'][name] = {}
        for cost in COSTS:
            net, detail = simulate(data['returns'].loc[dates], cash, targets, cost,
                                   decision_mask=mask)
            if not np.isfinite(net).all() or detail['gross'].max() > 1 + 1e-10:
                raise ValueError('Invalid equity or leverage path')
            paths[f'{name}/{cost}'] = net
            detail.to_csv(out / f'{name}_{cost:g}bp_daily.csv')
            stats = annualized_stats(net, cash)
            stats.update({
                'turnover_sum': float(detail['turnover'].sum()),
                'cost_sum': float(detail['cost'].sum()),
                'max_gross': float(detail['gross'].max()),
                'ending_10000': float(10000 * (1 + net).prod()),
                'historical_target_met': bool(stats['cagr'] >= .12
                                              and stats['max_drawdown'] >= -.15),
                'annual_returns': {str(year): float((1 + g).prod() - 1)
                                   for year, g in net.groupby(net.index.year)}})
            summary['cells'][name][str(cost)] = stats
        if name != CONTROL:
            forecasts = values.to_numpy()
            summary['prediction_metrics'][name] = {
                'n_labels': int(valid.sum()),
                'direction_accuracy': float(((forecasts > 0) == (labels > 0))[valid].mean()),
                'always_up_accuracy': float((labels[valid] > 0).mean()),
                'normalized_mse': float(((forecasts[valid] - labels[valid]) ** 2).mean()),
                'long_share': float((forecasts > 0).mean())}
        print(f'{name}: portfolio calculations complete', flush=True)

    summary['cash_only'] = annualized_stats(cash, cash)
    paths['cash_only'] = cash
    pd.DataFrame(paths).to_csv(out / 'net_returns.csv')
    monthly = {key: monthly_log_returns(net) for key, net in paths.items()}
    for name, _, _ in CELLS:
        summary['primary'][name] = {}
        summary['per_window'][name] = {}
        for cost in COSTS:
            delta = monthly[f'{name}/{cost}'] - monthly[f'{CONTROL}/{cost}']
            summary['primary'][name][str(cost)] = stationary_block_bootstrap(
                delta.to_numpy(), 6, 10000, 7)
            per_window = {}
            for number, (start, end) in enumerate(windows, start=1):
                block = dates[start - windows[0][0]:end - windows[0][0]]
                piece = delta.loc[(delta.index >= block[0].to_period('M').to_timestamp())
                                  & (delta.index <= block[-1].to_period('M').to_timestamp())]
                per_window[str(number)] = {
                    'start': str(block[0].date()), 'end': str(block[-1].date()),
                    'n_months': int(len(piece)),
                    'ann_geom': float(np.exp(12 * piece.mean()) - 1) if len(piece) else float('nan')}
            summary['per_window'][name][str(cost)] = per_window
    print('Paired comparisons complete', flush=True)

    excluding = [name for name, _, _ in CELLS
                 if summary['primary'][name]['3.0']['ci_low_95'] > 0
                 or summary['primary'][name]['3.0']['ci_high_95'] < 0]
    summary['verdict'] = {
        'cells_with_interval_excluding_zero': excluding,
        'reading': ('null result' if not excluding
                    else 'hint, not evidence' if len(excluding) == 1
                    else 'check axis consistency: C-A vs D-B')}
    write_json(out / 'summary.json', summary)
    write_report(out, summary, config, dates, windows)
    write_json(out / 'sha256.json',
               {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob('*') if p.is_file()})
    (out / 'COMPLETE').write_text(
        'Completed preregistered 2x2; all artifacts covered by sha256.json.\n', encoding='utf-8')


def write_report(out, summary, config, dates, windows):
    names = [name for name, _, _ in CELLS] + [CONTROL]
    lines = ['# 2x2 Features x Modell — vorab festgelegtes Raster', '',
             f'Zeitraum: {dates[0].date()} bis {dates[-1].date()}, {len(dates)} Handelstage, '
             f'{N_WINDOWS} expandierende Fenster. Alle Zahlen nach modellierten Kosten, vor Steuern.',
             '', '| Zelle | CAGR 3bp | Max DD 3bp | CAGR 15bp | Max DD 15bp | Turnover | Endwert 10.000 EUR |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        low, high = summary['cells'][name]['3.0'], summary['cells'][name]['15.0']
        lines.append(f"| {name} | {low['cagr']:.2%} | {low['max_drawdown']:.2%} | "
                     f"{high['cagr']:.2%} | {high['max_drawdown']:.2%} | "
                     f"{low['turnover_sum']:.2f} | {low['ending_10000']:,.0f} |")
    lines += ['', f"Nur Cash: {summary['cash_only']['cagr']:.2%} CAGR.", '',
              '## Primaermetrik — annualisiertes relatives geometrisches Wachstum gegen '
              f'`{CONTROL}` (deskriptives 95%-Intervall)', '']
    for name, _, _ in CELLS:
        for cost in COSTS:
            s = summary['primary'][name][str(cost)]
            lines.append(f"- {name} @ {cost:g}bp: {s['ann_geom']:.2%} "
                         f"[{s['ci_low_95']:.2%}, {s['ci_high_95']:.2%}]")
    lines += ['', '## Achsenbeitraege bei 3 bp (Differenz der Punktschaetzer)', '']
    point = {name: summary['primary'][name]['3.0']['ann_geom'] for name, _, _ in CELLS}
    lines += [f"- Feature-Achse: C-A = {point['C_ext_ridge'] - point['A_base_ridge']:.2%}, "
              f"D-B = {point['D_ext_gbm'] - point['B_base_gbm']:.2%}",
              f"- Modell-Achse: B-A = {point['B_base_gbm'] - point['A_base_ridge']:.2%}, "
              f"D-C = {point['D_ext_gbm'] - point['C_ext_ridge']:.2%}"]
    lines += ['', '## Fensterweise Aufschluesselung bei 3 bp (deskriptiv)', '',
              '| Zelle | ' + ' | '.join(f'F{i}' for i in range(1, N_WINDOWS + 1)) + ' |',
              '|---|' + '---:|' * N_WINDOWS]
    for name, _, _ in CELLS:
        per = summary['per_window'][name]['3.0']
        lines.append(f'| {name} | ' + ' | '.join(
            f"{per[str(i)]['ann_geom']:.2%}" for i in range(1, N_WINDOWS + 1)) + ' |')
    lines += ['', '## Prognosemetriken', '',
              '| Zelle | Trefferquote Richtung | Immer-aufwaerts | Norm. MSE | Long-Anteil |',
              '|---|---:|---:|---:|---:|']
    for name, _, _ in CELLS:
        m = summary['prediction_metrics'][name]
        lines.append(f"| {name} | {m['direction_accuracy']:.2%} | {m['always_up_accuracy']:.2%} | "
                     f"{m['normalized_mse']:.4f} | {m['long_share']:.2%} |")
    lines += ['', '## Urteil nach der vorab festgelegten Entscheidungsregel', '',
              f"Zellen mit Intervall ohne Null (3 bp): "
              f"{summary['verdict']['cells_with_interval_excluding_zero'] or 'keine'}.",
              f"Lesart: **{summary['verdict']['reading']}**.", '',
              '## Stueckelung und Gebuehren bei 10.000 EUR', '',
              'Ganze Stuecke, Broker-Mindestgebuehren, FX und Steuern sind **nicht modelliert**. '
              'Die Kosten sind ein reiner Basispunkt-Aufschlag auf den Umsatz (3 bp primaer, '
              '15 bp Sensitivitaet). Bei 10.000 EUR auf 19 Instrumente liegen einzelne Positionen '
              'im niedrigen dreistelligen Bereich, wo Mindestgebuehren und Stueckelung real '
              'spuerbar waeren; die Endwerte oben sind insoweit optimistisch.', '',
              '## Grenzen', ''] + [f'- {item}' for item in config['limitations']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Pilot ausfuehren und Laufzeit messen**

Run:

```bash
py -3.12 -u factor_lab/run_feature_model_2x2.py --snapshot "C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl" --output-root factor_lab/runs_2x2 --pilot
```

Expected, exakt so (die Zahlen sind am 2026-09-13 auf dem versiegelten Snapshot nachgerechnet, nicht geschaetzt):

```
Rows 4433, evaluation 586 from 2015-01-02 to 2017-05-01
```

danach `A_base_ridge: fitted in <N>s` und `Pilot complete`. Kein `COMPLETE`, kein `summary.json`.

**Weicht `Rows` von 4433 ab, ist das ein Fehler in Task 1, kein Kalibrierungsbedarf im Test.** Der ausgerichtete Zeilensatz ist verifiziert: D_ext = 4433, erste Zeile 2009-01-15, `first_test` = 1501, Auswertungsfenster 2.932 Zeilen von 2015-01-02 bis 2026-08-31. In dem Fall die Kanaldefinitionen gegen Spec Abschnitt 3 pruefen, statt die Erwartung anzupassen.

Die fuenf Fenstergrenzen, ebenfalls nachgerechnet:

| Fenster | Von | Bis | Zeilen |
|---:|---|---|---:|
| 1 | 2015-01-02 | 2017-05-01 | 586 |
| 2 | 2017-05-02 | 2019-08-28 | 586 |
| 3 | 2019-08-29 | 2021-12-23 | 586 |
| 4 | 2021-12-27 | 2024-04-25 | 586 |
| 5 | 2024-04-26 | 2026-08-31 | 588 |

**Entscheidungspunkt:** aus der Pilotzeit die Gesamtlaufzeit schaetzen. Ridge skaliert linear in den Fenstern, GBM ist teurer. Grobe Hochrechnung: `2 × 5 × t_ridge + 2 × 5 × t_gbm`. Liegt die Schaetzung unter 30 Minuten, den vollen Lauf im Vordergrund starten. Darueber: detacht ueber ein `.cmd`-Skript plus `Start-Process` mit PID-Datei, und **nach dem Start die erste Ausgabezeile pruefen** — eine direkt verkettete Kommandozeile ist in der letzten Sitzung an einem SyntaxError gescheitert, ohne dass es sofort auffiel.

- [ ] **Step 3: GBM-Laufzeit einzeln messen**

Run:

```bash
py -3.12 -u -c "import time,pandas as pd; from factor_lab.data_snapshot import load_trend_snapshot; from factor_lab.features_2x2 import build_2x2_dataset; from factor_lab.models_2x2 import fit_predict_gbm; dfs=load_trend_snapshot(r'C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl'); s=sorted(set(dfs)-{'IRX'}); p=pd.concat({k:dfs[k]['price'] for k in s},axis=1,sort=True).dropna(); d=build_2x2_dataset(p); f=int(d['dates'].searchsorted(pd.Timestamp('2015-01-01'))); t=time.time(); fit_predict_gbm(d['X_ext'],d['y'],f,f+586); print('gbm window seconds', round(time.time()-t,1))"
```

Expected: eine Zeile `gbm window seconds <N>`. Das ist die teuerste Einzelanpassung (10 Kanaele, erstes Fenster) und die Basis fuer das Budget.

- [ ] **Step 4: Ausgabeverzeichnis aus der Versionierung halten**

Run:

```bash
printf 'runs_2x2/\n' >> factor_lab/.gitignore
```

Falls `factor_lab/.gitignore` nicht existiert, legt der Befehl sie an. Pruefen mit `git status --porcelain` — es darf nichts aus `factor_lab/runs_2x2/` als unversioniert auftauchen.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/run_feature_model_2x2.py factor_lab/.gitignore
git commit -m "feat(factor_lab): 2x2 runner with pilot mode and sealed-snapshot guard"
```

---

### Task 6: Vollstaendiger Lauf, Bericht und Forschungsprotokoll

**Files:**
- Create: `docs/superpowers/feature-model-2x2-2026-09-13-results.md`
- Modify: `docs/superpowers/specs/2026-09-11-feature-model-2x2-design.md` (nur Statuszeile)

- [ ] **Step 1: Gesamte Testsuite gruen**

**Nicht** `unittest discover` verwenden. Die vier Audit-Testdateien (`test_null_params`, `test_null_worlds`, `test_audit_runner`, `test_audit_evaluation`) sind einfache Skripte mit `check_*`-Funktionen und ohne `unittest.TestCase`; `discover` importiert sie, findet null Tests und meldet trotzdem Erfolg. Das waere falsche Sicherheit.

Zuerst die unittest-Module:

```bash
py -3.12 -m unittest factor_lab.tests.test_features_2x2 factor_lab.tests.test_models_2x2 factor_lab.tests.test_evaluate_2x2 factor_lab.tests.test_daily_comparison factor_lab.tests.test_daily_models factor_lab.tests.test_horizon_models factor_lab.tests.test_horizon_execution factor_lab.tests.test_horizon_reproduction factor_lab.tests.test_stats factor_lab.tests.test_costs factor_lab.tests.test_portfolio -v
```

Dann die uebrigen Testdateien einzeln, jede als Skript, weil beide Konventionen im Verzeichnis nebeneinander existieren:

```bash
for f in factor_lab/tests/test_*.py; do echo "== $f"; py -3.12 "$f" >/dev/null || echo "FAILED $f"; done
```

Expected: keine `FAILED`-Zeile. Schlaegt ein Bestandstest fehl, **nicht** den Bestandstest anpassen — der neue Code hat dann etwas beruehrt, was er nicht anfassen darf.

- [ ] **Step 2: Vollen Lauf starten**

Run:

```bash
py -3.12 -u factor_lab/run_feature_model_2x2.py --snapshot "C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl" --output-root factor_lab/runs_2x2
```

Bei geschaetzt ueber 30 Minuten stattdessen detacht starten (siehe Task 5, Step 2) und die erste Ausgabezeile pruefen.

Expected: vier Zellen mit Fenster-Fortschritt, dann `Paired comparisons complete`, dann der vollstaendige Bericht auf stdout, und im Ausgabeverzeichnis `COMPLETE`, `summary.json`, `sha256.json`, `report.md`.

- [ ] **Step 3: Ergebnisse gegen die Entscheidungsregel pruefen**

Das Urteil steht in `summary.json` unter `verdict`. Die Regel der Spec, Abschnitt 7, gilt wie geschrieben:

- Kein Intervall schliesst Null aus → **Nullbefund**, so berichten, keine Nachjustierung, keine fuenfte Zelle.
- Genau eines schliesst Null aus → **Hinweis, kein Nachweis**; Kandidat fuer eine eigene praeregistrierte Familie mit frischem Holdout, nicht fuer eine Verlaengerung dieses Rasters.
- Mehrere → Achsenbeitraege auf Konsistenz pruefen (C−A gegen D−B, B−A gegen D−C); widerspruechliche Vorzeichen deuten auf Rauschen.

**Keine** Reaktion auf die Zahlen, die die Rezeptur, die Kanalauswahl, die Fensterzahl oder die Kadenz veraendert.

- [ ] **Step 4: Ergebnisbericht schreiben**

`docs/superpowers/feature-model-2x2-2026-09-13-results.md` mit folgender Gliederung anlegen, gefuellt mit den tatsaechlichen Zahlen aus `summary.json`:

1. **Was getestet wurde** — die vier Zellen, die eine Primaermetrik, die Entscheidungsregel, alles vor dem Lauf festgelegt.
2. **Ergebnistabelle** — CAGR, MaxDD, Turnover, Endwert je Zelle bei 3 bp und 15 bp, plus Kontrolle und Cash.
3. **Primaermetrik** — Punktschaetzer und 95%-Intervall je Zelle gegen die Kontrolle, bei beiden Kostenstufen.
4. **Achsenbeitraege** — C−A, D−B, B−A, D−C, mit dem Hinweis, dass Differenzen von Punktschaetzern keine getesteten Groessen sind.
   Ebenfalls zu nennen: die fensterweise Aufschluesselung ist **deskriptiv und an den Raendern ueberlappend** — ein Kalendermonat, in dem eine Fenstergrenze liegt, erscheint in beiden angrenzenden Fenstern. Die Primaermetrik ist davon nicht betroffen, sie laeuft auf der zusammengefuegten Monatsreihe.
5. **Urteil** nach der Entscheidungsregel, wortwoertlich aus Abschnitt 7 der Spec zitiert.
6. **Zwingende Einschraenkungen** — die Liste aus `configuration.json['limitations']`, plus die beiden Umsetzungsentscheidungen aus diesem Plan (Zeilenausrichtung, monatsgleichgewichtete Aggregation), plus: der Falsifikations-Audit deckt die v2-Screening-Pipeline ab und **nicht** dieses Raster, ein positives Ergebnis erbt hier also weiterhin eine ungemessene Fehlalarmrate.
7. **Stueckelung und Gebuehren** — ausdruecklich: nicht modelliert, nur Basispunkt-Aufschlag auf den Umsatz; bei 10.000 EUR auf 19 Instrumente waeren Mindestgebuehren und ganze Stuecke real spuerbar.
8. **Mehrfachtest-Buchfuehrung** — 141 + 4 = 145 unkorrigierte Vergleiche. Jede spaetere Variante (Baum nur auf dem letzten Zeitschritt, elfter Kanal, andere Rezeptur) zaehlt neu.
9. **Ausgabeverzeichnis und `sha256.json`-Hash** der `summary.json`, damit der Lauf nachvollziehbar bleibt.
10. **Falls Nullbefund:** der Abschnitt, den der Uebergabe-Prompt verlangt — bei 15 % Drawdown-Deckel und 1,25x Hebel ist 12,7 % mit dem bisher Getesteten nicht erreichbar, und die ehrliche Konsequenz ist eine Zielrevision, nicht die naechste Variante.

- [ ] **Step 5: Spec-Status aktualisieren**

In `docs/superpowers/specs/2026-09-11-feature-model-2x2-design.md` **nur** die Statuszeile aendern:

```
**Status:** Ausgefuehrt am 2026-09-13 — siehe `docs/superpowers/feature-model-2x2-2026-09-13-results.md`
```

Sonst nichts an der Spec anfassen; sie ist das praeregistrierte Dokument.

- [ ] **Step 6: Dateiliste vor dem Staging pruefen**

Run: `git status --porcelain`
Expected: nur die geplanten Quell-, Test- und Dokumentdateien. **Kein** Eintrag aus `factor_lab/runs_2x2/`, keine `.pkl`, keine `.csv` aus Laufverzeichnissen, nichts aus `research_archive/`, nichts aus `market_control_system/`.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/feature-model-2x2-2026-09-13-results.md docs/superpowers/specs/2026-09-11-feature-model-2x2-design.md
git commit -m "docs(factor_lab): 2x2 feature/model grid results and spec status"
```

- [ ] **Step 8: Commit-ID und Status melden**

Run: `git log --oneline -8` und `git status --porcelain`
Die Commit-ID, das Ausgabeverzeichnis des Laufs und den verbleibenden Status berichten. **Nicht pushen, keinen PR oeffnen** — dafuer braucht es einen eigenen Auftrag.
