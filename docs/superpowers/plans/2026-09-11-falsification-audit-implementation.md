# Falsifikations-Audit — Implementierungsplan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Die Fehlalarmrate der bestehenden trend-etf-v2-Screening-Pipeline auf drei mittelwertfreien Nullwelten messen, ohne Originaldaten, versiegelte Kandidaten oder abgeschlossene Ergebnisse zu berühren.

**Architecture:** Neues, isoliertes Paket `factor_lab/audit/` mit vier reinen Modulen (Parameterschätzung, Nullwelt-Generatoren, Audit-Runner, Auswertung). Der Runner ruft die **echte** `run_trend_baseline_v2.run_screening()` auf — keine Nachbildung der Pipeline. Kein bestehendes Modul wird verändert.

**Tech Stack:** Python 3.12 (`py -3.12`), numpy, pandas. Kein scipy, kein sklearn, kein lightgbm nötig (die gehören zum 2×2-Raster, nicht hierher).

**Spec:** `docs/superpowers/specs/2026-09-11-falsification-audit-design.md`

## Global Constraints

- Alles mit `py -3.12` ausführen (niemals bare `python`/`pip` — lösen auf dieser Maschine inkonsistent auf).
- Testkonvention: einfache Skripte, `check_*`-Funktionen, `run_consistency_check()`, `__main__`. Kein pytest.
- Deutsche Docstrings/Kommentare, ASCII (ue/oe/ae).
- **Prüfobjekt ist die v2-Pipeline:** `factor_lab.run_trend_baseline_v2.run_screening`. Eine v3-Familie existiert nicht.
- **Nichts verändern** an: `run_trend_baseline_v2.py`, `run_trend_holdout_v2.py`, `registration*.py`, `stats.py`, `portfolio.py`, `signals.py`, `costs.py`, `data_snapshot.py`, allem unter `market_control_system/`, und allem unter `research_archive/`.
- **Archiv-Snapshot ausschliesslich lesend:** `C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl`, SHA256 `36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89`. Der Hash wird bei jedem Laden geprueft.
- **Der Audit versiegelt nichts:** kein `candidate.json`, kein Tombstone, kein Schreiben nach `factor_lab/logs/`.
- **Synthetische Daten eindeutig kennzeichnen:** Beispiel-Panels nur unter `factor_lab/audit_data/` mit Praefix `SYNTHETIC_NULL_` und `"synthetic": true` im Manifest.
- **Abweichung von der Spec, bewusst und hier festgehalten:** Panels werden NICHT alle persistiert (600 Panels waeren ~450 MB). Sie sind aus `(welt, seed, eingefrorene Parameter)` deterministisch reproduzierbar; persistiert werden nur die eingefrorenen Parameter, je Welt ein Beispiel-Panel zur Inspektion, und die kleinen Ergebnis-Records.
- Nach jedem Task committen; Commit-Nachrichten enden mit einer Leerzeile und dann `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

### Task 1: Nullwelt-Parameter schaetzen und einfrieren

**Files:**
- Create: `factor_lab/audit/__init__.py` (leer)
- Create: `factor_lab/audit/null_params.py`
- Create: `factor_lab/tests/test_null_params.py`

**Interfaces:**
- Produces: `ARCHIVE_SNAPSHOT_PATH`, `ARCHIVE_SNAPSHOT_SHA256`; `load_archive_snapshot() -> dict` (lesend, mit Hash-Pruefung); `fit_garch_grid(returns: np.ndarray) -> dict` (Varianz-Targeting + Gitter-MLE, kein Optimizer); `estimate_null_params(dfs: dict) -> dict`; `freeze_params(params: dict, path: str) -> str` (schreibt JSON, gibt SHA256 zurueck).

- [ ] **Step 1: Write the failing test**

```python
"""
test_null_params.py — Schaetzung der Nullwelt-Parameter auf dem
Archiv-Snapshot (ausschliesslich lesend) und deren Einfrieren.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile

import numpy as np

from factor_lab.audit.null_params import (
    fit_garch_grid, estimate_null_params, freeze_params, load_archive_snapshot,
)


def check_garch_grid_recovers_persistence() -> None:
    """Auf simulierten GARCH-Daten mit bekannter Persistenz muss die
    geschaetzte Persistenz alpha+beta in der richtigen Groessenordnung
    liegen (kein Punktschaetzer-Anspruch, nur Plausibilitaet)."""
    rng = np.random.default_rng(0)
    n = 4000
    omega_true, alpha_true, beta_true = 1e-6, 0.08, 0.90
    r = np.empty(n)
    var = omega_true / (1 - alpha_true - beta_true)
    for i in range(n):
        r[i] = np.sqrt(var) * rng.standard_normal()
        var = omega_true + alpha_true * r[i] ** 2 + beta_true * var
    fit = fit_garch_grid(r)
    persistence = fit["alpha"] + fit["beta"]
    assert 0.80 <= persistence < 0.999, f"Persistenz {persistence:.3f} unplausibel"
    assert fit["omega"] > 0 and fit["alpha"] > 0 and fit["beta"] > 0
    print("fit_garch_grid: Persistenz plausibel rekonstruiert: OK")


def check_garch_grid_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    r = rng.standard_normal(2000) * 0.01
    assert fit_garch_grid(r) == fit_garch_grid(r), "Gitter-Fit muss deterministisch sein"
    print("fit_garch_grid: deterministisch: OK")


def check_estimate_null_params_structure() -> None:
    dfs = load_archive_snapshot()
    params = estimate_null_params(dfs)
    assert set(params) >= {"symbols", "calendar", "n1", "n2", "n3"}
    assert len(params["symbols"]) == 19, f"erwartet 19 Instrumente, bekam {len(params['symbols'])}"
    # N1: zwei Vola-Zustaende plus Uebergangsmatrix
    assert set(params["n1"]) >= {"transition", "state_vol"}
    t = np.asarray(params["n1"]["transition"], dtype=float)
    assert t.shape == (2, 2) and np.allclose(t.sum(axis=1), 1.0), "Uebergangszeilen muessen auf 1 summieren"
    # N2: je Instrument omega/alpha/beta mit Stationaritaet
    assert set(params["n2"]) == set(params["symbols"])
    for sym, p in params["n2"].items():
        assert p["alpha"] + p["beta"] < 1.0, f"{sym}: nichtstationaer"
    # N3: zwei Faktoren, Ladungen je Instrument, idiosynkratische Vola
    assert np.asarray(params["n3"]["loadings"]).shape == (19, 2)
    assert len(params["n3"]["idio_vol"]) == 19
    assert len(params["n3"]["factor_garch"]) == 2
    print("estimate_null_params: Struktur aller drei Welten: OK")


def check_freeze_params_roundtrip_and_hash() -> None:
    params = {"symbols": ["A"], "n1": {"transition": [[0.9, 0.1], [0.2, 0.8]]}}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.json")
        digest = freeze_params(params, path)
        assert len(digest) == 64
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == params
        assert freeze_params(params, os.path.join(tmp, "p2.json")) == digest, (
            "Gleicher Inhalt muss denselben Hash ergeben"
        )
    print("freeze_params: Roundtrip + stabiler Hash: OK")


def run_consistency_check() -> None:
    check_garch_grid_recovers_persistence()
    check_garch_grid_is_deterministic()
    check_estimate_null_params_structure()
    check_freeze_params_roundtrip_and_hash()
    print("\nAlle null_params-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_null_params.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'factor_lab.audit'`

- [ ] **Step 3: Implement `factor_lab/audit/__init__.py` (leer) und `factor_lab/audit/null_params.py`**

```python
"""
null_params.py — Schaetzung der Parameter fuer die drei Nullwelten des
Falsifikations-Audits (Spec Abschnitt 1), ausschliesslich LESEND auf dem
archivierten Preissnapshot.

Bewusst ohne scipy/sklearn: GARCH(1,1) wird per Varianz-Targeting plus
deterministischem Gitter-MLE geschaetzt. Das ist gruber als ein echter
Optimizer, aber reproduzierbar, testbar und ohne zusaetzliche Abhaengigkeit
-- fuer eine Nullwelt genuegt eine plausible, nicht eine optimale
Parametrisierung.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd

ARCHIVE_SNAPSHOT_PATH = (
    r"C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907"
    r"/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl"
)
ARCHIVE_SNAPSHOT_SHA256 = "36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89"

_ALPHA_GRID = np.round(np.arange(0.02, 0.205, 0.01), 3)
_BETA_GRID = np.round(np.arange(0.70, 0.975, 0.01), 3)


def load_archive_snapshot() -> dict:
    """Laedt den archivierten Snapshot read-only. Der Datei-Hash wird NICHT
    geprueft (das Pickle-Format ist nicht bytegleich reproduzierbar); geprueft
    wird der Inhalts-Hash ueber snapshot_content_sha256, der auch im
    Uebergabe-Prompt steht."""
    from factor_lab.data_snapshot import snapshot_content_sha256
    with open(ARCHIVE_SNAPSHOT_PATH, "rb") as f:
        dfs = pickle.load(f)
    digest = snapshot_content_sha256(dfs)
    if digest != ARCHIVE_SNAPSHOT_SHA256:
        raise ValueError(
            f"Archiv-Snapshot hat Inhalts-Hash {digest}, erwartet {ARCHIVE_SNAPSHOT_SHA256}"
        )
    return dfs


def _returns_frame(dfs: dict) -> pd.DataFrame:
    symbols = sorted(s for s in dfs if s != "IRX")
    common = dfs[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(dfs[s].index)
    common = common.sort_values()
    prices = pd.DataFrame({s: dfs[s].loc[common, "price"] for s in symbols})
    return prices.pct_change().dropna()


def fit_garch_grid(returns: np.ndarray) -> dict:
    """GARCH(1,1) mit Varianz-Targeting: omega folgt aus der Stichprobenvarianz
    und (alpha, beta); (alpha, beta) per deterministischem Gitter-MLE unter
    Gauss-Annahme. Mittelwert wird als null angenommen (Nullwelt-Zweck)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    target_var = float(np.mean(r ** 2))
    best = None
    for alpha in _ALPHA_GRID:
        for beta in _BETA_GRID:
            persistence = alpha + beta
            if persistence >= 0.999:
                continue
            omega = target_var * (1.0 - persistence)
            var = target_var
            loglik = 0.0
            for value in r:
                loglik -= 0.5 * (np.log(var) + value * value / var)
                var = omega + alpha * value * value + beta * var
            if best is None or loglik > best[0]:
                best = (loglik, float(omega), float(alpha), float(beta))
    return {"omega": best[1], "alpha": best[2], "beta": best[3], "loglik": best[0]}


def estimate_null_params(dfs: dict) -> dict:
    returns = _returns_frame(dfs)
    symbols = list(returns.columns)

    # --- N1: zwei Vola-Zustaende ueber die Panel-Durchschnittsvola ---
    panel_vol = returns.mean(axis=1).rolling(21, min_periods=21).std().dropna()
    high = (panel_vol > panel_vol.median()).to_numpy()
    transition = np.zeros((2, 2))
    for a, b in zip(high[:-1], high[1:]):
        transition[int(a), int(b)] += 1.0
    transition = transition / transition.sum(axis=1, keepdims=True)
    aligned = returns.loc[panel_vol.index]
    state_vol = {
        s: [float(aligned.loc[~high, s].std()), float(aligned.loc[high, s].std())]
        for s in symbols
    }

    # --- N2: GARCH(1,1) je Instrument ---
    n2 = {s: fit_garch_grid(returns[s].to_numpy()) for s in symbols}

    # --- N3: zwei Hauptkomponenten aus der Korrelationsmatrix ---
    standardized = (returns - returns.mean()) / returns.std()
    eigvals, eigvecs = np.linalg.eigh(np.corrcoef(standardized.to_numpy(), rowvar=False))
    order = np.argsort(eigvals)[::-1][:2]
    loadings = eigvecs[:, order] * np.sqrt(eigvals[order])
    factors = standardized.to_numpy() @ eigvecs[:, order]
    residual = standardized.to_numpy() - factors @ loadings.T
    n3 = {
        "loadings": (loadings * returns.std().to_numpy()[:, None]).tolist(),
        "idio_vol": (residual.std(axis=0) * returns.std().to_numpy()).tolist(),
        "factor_garch": [fit_garch_grid(factors[:, k]) for k in range(2)],
    }

    return {
        "symbols": symbols,
        "calendar": [d.strftime("%Y-%m-%d") for d in returns.index],
        "n1": {"transition": transition.tolist(), "state_vol": state_vol},
        "n2": n2,
        "n3": n3,
    }


def freeze_params(params: dict, path: str) -> str:
    payload = json.dumps(params, indent=2, sort_keys=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_null_params.py`
Expected: PASS. Der `estimate_null_params`-Check laedt den echten Archiv-Snapshot und laeuft dadurch einige Minuten (19 Gitter-MLEs ueber ~4900 Tage). Ist das zu langsam, in Step 3 die inneren Schleifen von `fit_garch_grid` vektorisieren — **nicht** das Gitter verkleinern, ohne es zu dokumentieren.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/audit/__init__.py factor_lab/audit/null_params.py factor_lab/tests/test_null_params.py
git commit -m "feat(audit): estimate and freeze null-world parameters from the archived snapshot"
```

---

### Task 2: Nullwelt-Generatoren

**Files:**
- Create: `factor_lab/audit/null_worlds.py`
- Create: `factor_lab/tests/test_null_worlds.py`

**Interfaces:**
- Consumes: die von Task 1 eingefrorenen Parameter (als dict).
- Produces: `WORLDS = ("n1", "n2", "n3")`; `generate_null_panel(world: str, params: dict, seed: int) -> dict` — liefert exakt die Struktur, die `run_screening` erwartet: `{symbol: DataFrame(columns=['price']), ..., 'IRX': DataFrame(columns=['rate_pa_pct'])}`.

- [ ] **Step 1: Write the failing test**

```python
"""
test_null_worlds.py — Generatoren fuer die drei Nullwelten. Zentrale
Eigenschaft: konstruktionsbedingt kein prognostizierbarer Mittelwert.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.audit.null_worlds import WORLDS, generate_null_panel


def _params(n_days: int = 600, n_symbols: int = 4) -> dict:
    symbols = [f"S{i}" for i in range(n_symbols)]
    calendar = [d.strftime("%Y-%m-%d")
                for d in pd.date_range("2010-01-01", periods=n_days, freq="B")]
    garch = {"omega": 1e-6, "alpha": 0.08, "beta": 0.90, "loglik": 0.0}
    return {
        "symbols": symbols,
        "calendar": calendar,
        "n1": {"transition": [[0.95, 0.05], [0.10, 0.90]],
               "state_vol": {s: [0.006, 0.018] for s in symbols}},
        "n2": {s: dict(garch) for s in symbols},
        "n3": {"loadings": [[0.01, 0.004]] * n_symbols,
               "idio_vol": [0.008] * n_symbols,
               "factor_garch": [dict(garch), dict(garch)]},
    }


def check_panel_shape_matches_pipeline_contract() -> None:
    params = _params()
    for world in WORLDS:
        dfs = generate_null_panel(world, params, seed=0)
        assert set(dfs) == set(params["symbols"]) | {"IRX"}, f"{world}: falsche Schluessel"
        for s in params["symbols"]:
            assert list(dfs[s].columns) == ["price"], f"{world}/{s}: Spalte muss 'price' heissen"
            assert len(dfs[s]) == len(params["calendar"])
            values = dfs[s]["price"].to_numpy()
            assert np.isfinite(values).all() and (values > 0).all(), (
                f"{world}/{s}: Preise muessen endlich und positiv sein"
            )
        assert list(dfs["IRX"].columns) == ["rate_pa_pct"]
    print("generate_null_panel: Struktur/Kalender/Positivitaet fuer alle Welten: OK")


def check_deterministic_per_seed() -> None:
    params = _params()
    for world in WORLDS:
        a = generate_null_panel(world, params, seed=7)["S0"]["price"].to_numpy()
        b = generate_null_panel(world, params, seed=7)["S0"]["price"].to_numpy()
        c = generate_null_panel(world, params, seed=8)["S0"]["price"].to_numpy()
        assert np.array_equal(a, b), f"{world}: gleicher Seed muss gleiches Panel geben"
        assert not np.array_equal(a, c), f"{world}: anderer Seed muss anderes Panel geben"
    print("generate_null_panel: deterministisch je Seed, verschieden ueber Seeds: OK")


def check_mean_is_statistically_zero() -> None:
    """Ueber viele Seeds gemittelt darf die Tagesrendite nicht signifikant
    von null abweichen -- das ist die definierende Eigenschaft der Nullwelt."""
    params = _params(n_days=1200)
    for world in WORLDS:
        means = []
        for seed in range(20):
            dfs = generate_null_panel(world, params, seed=seed)
            r = dfs["S0"]["price"].pct_change().dropna().to_numpy()
            means.append(r.mean())
        means = np.asarray(means)
        t_stat = means.mean() / (means.std(ddof=1) / np.sqrt(len(means)))
        assert abs(t_stat) < 3.0, f"{world}: Mittelwert weicht von null ab (t={t_stat:.2f})"
    print("generate_null_panel: Mittelwert statistisch null in allen Welten: OK")


def check_n3_induces_cross_sectional_correlation() -> None:
    """N3 muss echte Querschnittskorrelation erzeugen, sonst testet sie nicht,
    wozu sie da ist."""
    params = _params(n_days=1500)
    dfs = generate_null_panel("n3", params, seed=3)
    returns = pd.DataFrame({s: dfs[s]["price"].pct_change() for s in params["symbols"]}).dropna()
    corr = returns.corr().to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    assert off_diagonal.mean() > 0.2, (
        f"N3 erzeugt zu wenig Querschnittskorrelation (Mittel {off_diagonal.mean():.2f})"
    )
    print("generate_null_panel: N3 erzeugt Querschnittskorrelation: OK")


def run_consistency_check() -> None:
    check_panel_shape_matches_pipeline_contract()
    check_deterministic_per_seed()
    check_mean_is_statistically_zero()
    check_n3_induces_cross_sectional_correlation()
    print("\nAlle null_worlds-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_null_worlds.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'factor_lab.audit.null_worlds'`

- [ ] **Step 3: Implement `factor_lab/audit/null_worlds.py`**

```python
"""
null_worlds.py — Generatoren der drei Nullwelten (Spec Abschnitt 1).

Gemeinsame, definierende Eigenschaft aller drei: der bedingte Erwartungswert
der Renditen ist konstruktionsbedingt exakt null. Nur die Struktur ZWEITER
Ordnung (Volatilitaetsregime, Clustering, Querschnittskorrelation) wird
nachgebildet -- genau die Strukturen, die scheinbare Prognostizierbarkeit
erzeugen koennen, ohne dass ein echter Edge existiert.

Der Cash-Satz (IRX) wird konstant gehalten: er ist kein Gegenstand der
Falsifikation, und ein zufaelliger Cash-Pfad wuerde nur Rauschen hinzufuegen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WORLDS = ("n1", "n2", "n3")
_CASH_RATE_PA_PCT = 2.0


def _garch_path(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    omega, alpha, beta = spec["omega"], spec["alpha"], spec["beta"]
    var = omega / max(1.0 - alpha - beta, 1e-8)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * out[i] ** 2 + beta * var
    return out


def _regime_states(transition: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    states = np.empty(n, dtype=int)
    state = 0
    for i in range(n):
        states[i] = state
        state = int(rng.random() >= transition[state, 0])
    return states


def _to_panel(returns: np.ndarray, symbols: list[str], calendar: pd.DatetimeIndex) -> dict:
    """Renditen -> Preise (Start 100). Die erste Zeile ist der Startpreis, daher
    werden n-1 Renditen kumuliert."""
    dfs = {}
    for j, symbol in enumerate(symbols):
        prices = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns[:, j]]))
        dfs[symbol] = pd.DataFrame({"price": prices}, index=calendar)
    dfs["IRX"] = pd.DataFrame({"rate_pa_pct": np.full(len(calendar), _CASH_RATE_PA_PCT)},
                              index=calendar)
    return dfs


def generate_null_panel(world: str, params: dict, seed: int) -> dict:
    if world not in WORLDS:
        raise ValueError(f"Unbekannte Nullwelt: {world}")
    symbols = list(params["symbols"])
    calendar = pd.DatetimeIndex(pd.to_datetime(params["calendar"]))
    n = len(calendar) - 1  # eine Rendite weniger als Preise
    rng = np.random.default_rng(seed)

    if world == "n1":
        transition = np.asarray(params["n1"]["transition"], dtype=float)
        states = _regime_states(transition, n, rng)
        returns = np.empty((n, len(symbols)))
        for j, symbol in enumerate(symbols):
            low, high = params["n1"]["state_vol"][symbol]
            sigma = np.where(states == 0, low, high)
            returns[:, j] = sigma * rng.standard_normal(n)
    elif world == "n2":
        returns = np.column_stack([_garch_path(params["n2"][s], n, rng) for s in symbols])
    else:
        loadings = np.asarray(params["n3"]["loadings"], dtype=float)
        idio = np.asarray(params["n3"]["idio_vol"], dtype=float)
        factors = np.column_stack([_garch_path(spec, n, rng)
                                   for spec in params["n3"]["factor_garch"]])
        # Faktoren auf Einheitsvarianz normieren, damit die Ladungen die
        # Skala bestimmen und nicht die GARCH-Parametrisierung.
        factors = factors / factors.std(axis=0, keepdims=True)
        returns = factors @ loadings.T + idio * rng.standard_normal((n, len(symbols)))

    return _to_panel(returns, symbols, calendar)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_null_worlds.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/audit/null_worlds.py factor_lab/tests/test_null_worlds.py
git commit -m "feat(audit): add mean-zero null-world generators (vol regime, GARCH, common factors)"
```

---

### Task 3: Audit-Runner fuer eine Replikation

**Files:**
- Create: `factor_lab/audit/audit_runner.py`
- Create: `factor_lab/tests/test_audit_runner.py`

**Interfaces:**
- Consumes: `generate_null_panel` (Task 2); `factor_lab.run_trend_baseline_v2.run_screening`, `prepare_inputs`, `VARIANT_NAMES` (UNVERAENDERT).
- Produces: `run_one_replication(world: str, params: dict, seed: int) -> dict` mit den Feldern `world`, `seed`, `candidate` (str oder None), `gate_a` (dict Variante -> bool), `passed_all` (dict Variante -> bool), `excess_lower95` (dict Variante -> float), `dev_end` (str), `elapsed_s` (float).

- [ ] **Step 1: Write the failing test**

```python
"""
test_audit_runner.py — eine Audit-Replikation gegen die ECHTE
v2-Screening-Pipeline. Wichtigste Zusicherung neben der Record-Struktur:
der Audit versiegelt nichts und schreibt nicht nach factor_lab/logs/.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from factor_lab.audit.audit_runner import run_one_replication
from factor_lab.run_trend_baseline_v2 import VARIANT_NAMES

_LOGS = os.path.join(os.path.dirname(__file__), "..", "logs")


def _small_params(n_days: int = 900) -> dict:
    """Kleines Panel: genug fuer warmup(315)+Monatsenden, aber schnell."""
    symbols = ["SPY", "TLT", "GLD", "EFA", "EEM", "UUP", "USO"]
    calendar = [d.strftime("%Y-%m-%d")
                for d in pd.date_range("2010-01-01", periods=n_days, freq="B")]
    garch = {"omega": 1e-6, "alpha": 0.08, "beta": 0.90, "loglik": 0.0}
    return {
        "symbols": symbols,
        "calendar": calendar,
        "n1": {"transition": [[0.95, 0.05], [0.10, 0.90]],
               "state_vol": {s: [0.006, 0.018] for s in symbols}},
        "n2": {s: dict(garch) for s in symbols},
        "n3": {"loadings": [[0.01, 0.004]] * len(symbols),
               "idio_vol": [0.008] * len(symbols),
               "factor_garch": [dict(garch), dict(garch)]},
    }


def check_record_structure() -> None:
    record = run_one_replication("n2", _small_params(), seed=0)
    assert record["world"] == "n2" and record["seed"] == 0
    assert record["candidate"] is None or record["candidate"] in VARIANT_NAMES
    assert sorted(record["gate_a"]) == VARIANT_NAMES
    assert sorted(record["passed_all"]) == VARIANT_NAMES
    assert sorted(record["excess_lower95"]) == VARIANT_NAMES
    assert all(isinstance(v, bool) for v in record["gate_a"].values())
    assert record["elapsed_s"] > 0
    print("run_one_replication: Record-Struktur: OK")


def check_writes_nothing_to_logs() -> None:
    before = set(os.listdir(_LOGS)) if os.path.isdir(_LOGS) else set()
    run_one_replication("n1", _small_params(), seed=1)
    after = set(os.listdir(_LOGS)) if os.path.isdir(_LOGS) else set()
    assert before == after, f"Audit hat nach logs/ geschrieben: {after - before}"
    print("run_one_replication: schreibt nichts nach factor_lab/logs/: OK")


def check_candidate_consistent_with_gates() -> None:
    """Eine Kandidatin darf nur gemeldet werden, wenn sie auch alle Gates
    besteht -- sonst misst FA_pipeline etwas anderes als gemeint."""
    for seed in range(3):
        record = run_one_replication("n3", _small_params(), seed=seed)
        if record["candidate"] is not None:
            assert record["passed_all"][record["candidate"]], (
                f"Kandidatin {record['candidate']} gemeldet, besteht aber nicht alle Gates"
            )
    print("run_one_replication: Kandidatin konsistent mit Gates: OK")


def run_consistency_check() -> None:
    check_record_structure()
    check_writes_nothing_to_logs()
    check_candidate_consistent_with_gates()
    print("\nAlle audit_runner-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_audit_runner.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'factor_lab.audit.audit_runner'`

- [ ] **Step 3: Implement `factor_lab/audit/audit_runner.py`**

```python
"""
audit_runner.py — eine Replikation des Falsifikations-Audits.

Ruft die ECHTE Screening-Pipeline auf (run_trend_baseline_v2.run_screening),
keine Nachbildung. Repliziert dabei die Trimmung auf dev_end genau so, wie
run_trend_baseline_v2.main() es tut -- sonst wuerde der Audit auf einer
laengeren Stichprobe messen als der echte Lauf und damit eine andere
CI-Breite und Fehlalarmrate.

Der Audit versiegelt bewusst NICHTS: kein candidate.json, kein Tombstone,
kein Schreiben nach factor_lab/logs/.
"""
from __future__ import annotations

import time

from factor_lab.audit.null_worlds import generate_null_panel
from factor_lab.run_trend_baseline_v2 import prepare_inputs, run_screening


def run_one_replication(world: str, params: dict, seed: int) -> dict:
    started = time.perf_counter()
    dfs = generate_null_panel(world, params, seed)
    dev_end = prepare_inputs(dfs)["dev_end"]
    dev_dfs = {name: df.loc[df.index <= dev_end] for name, df in dfs.items()}
    result, _ = run_screening(dev_dfs, dev_end=dev_end)
    summary = result["summary"]
    return {
        "world": world,
        "seed": seed,
        "candidate": result["candidate"],
        "gate_a": {v: bool(s["gates"]["gate_a_excess_ci"]) for v, s in summary.items()},
        "passed_all": {v: bool(s["gates"]["passed_all"]) for v, s in summary.items()},
        "excess_lower95": {v: float(s["excess_bootstrap"]["ann_geom_lower_1s95"])
                           for v, s in summary.items()},
        "dev_end": str(dev_end),
        "elapsed_s": time.perf_counter() - started,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_audit_runner.py`
Expected: PASS. Laeuft je Replikation spuerbar lange (Bootstrap dominiert) — das ist erwartet und wird in Task 4 gemessen.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/audit/audit_runner.py factor_lab/tests/test_audit_runner.py
git commit -m "feat(audit): run one replication against the real v2 screening pipeline"
```

---

### Task 4: Auswertung und versiegelte Kontroll-Seeds

**Files:**
- Create: `factor_lab/audit/evaluation.py`
- Create: `factor_lab/tests/test_audit_evaluation.py`

**Interfaces:**
- Produces: `false_alarm_rates(records: list[dict]) -> dict` mit `fa_variant`, `fa_variant_ci95`, `fa_pipeline`, `fa_pipeline_ci95`, `n_replications`, `n_variant_tests`; `verdict(rates: dict) -> dict` mit `calibrated` (bool), `selection_inflated` (bool), `text` (str); `WORKING_SEEDS`, `CONTROL_SEEDS`; `seal_control_seeds(path: str) -> str`.

- [ ] **Step 1: Write the failing test**

```python
"""
test_audit_evaluation.py — Fehlalarmraten, vorab festgelegte Kriterien
und Versiegelung der Kontroll-Seeds.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile

from factor_lab.audit.evaluation import (
    CONTROL_SEEDS, WORKING_SEEDS, false_alarm_rates, seal_control_seeds, verdict,
)


def _record(gate_a_hits: int, candidate: bool) -> dict:
    variants = [f"v{i}" for i in range(8)]
    return {
        "world": "n2", "seed": 0,
        "candidate": "v0" if candidate else None,
        "gate_a": {v: (i < gate_a_hits) for i, v in enumerate(variants)},
        "passed_all": {v: (candidate and v == "v0") for v in variants},
        "excess_lower95": {v: 0.0 for v in variants},
    }


def check_rates_counted_over_the_right_denominators() -> None:
    records = [_record(1, True)] + [_record(0, False)] * 9
    rates = false_alarm_rates(records)
    assert rates["n_replications"] == 10
    assert rates["n_variant_tests"] == 80, "8 Varianten x 10 Laeufe"
    assert abs(rates["fa_variant"] - 1 / 80) < 1e-12
    assert abs(rates["fa_pipeline"] - 1 / 10) < 1e-12
    low, high = rates["fa_variant_ci95"]
    assert low <= rates["fa_variant"] <= high
    print("false_alarm_rates: korrekte Nenner und CIs: OK")


def check_verdict_thresholds() -> None:
    calibrated = verdict({"fa_variant": 0.05, "fa_pipeline": 0.08})
    assert calibrated["calibrated"] and not calibrated["selection_inflated"]

    inflated = verdict({"fa_variant": 0.05, "fa_pipeline": 0.22})
    assert inflated["calibrated"] and inflated["selection_inflated"]

    broken = verdict({"fa_variant": 0.18, "fa_pipeline": 0.30})
    assert not broken["calibrated"], "FA_variant > 10% muss als fehlkalibriert gelten"

    conservative = verdict({"fa_variant": 0.01, "fa_pipeline": 0.02})
    assert not conservative["calibrated"], "FA_variant < 2.5% liegt ebenfalls ausserhalb"
    print("verdict: vorab festgelegte Schwellen: OK")


def check_seeds_are_disjoint_and_sealed() -> None:
    assert set(WORKING_SEEDS).isdisjoint(CONTROL_SEEDS), "Seed-Mengen muessen disjunkt sein"
    assert len(WORKING_SEEDS) == 200 and len(CONTROL_SEEDS) == 50
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "seeds.json")
        digest = seal_control_seeds(path)
        assert len(digest) == 64
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["control_seeds"] == list(CONTROL_SEEDS)
        assert seal_control_seeds(os.path.join(tmp, "s2.json")) == digest
    print("Kontroll-Seeds: disjunkt, versiegelbar, stabiler Hash: OK")


def run_consistency_check() -> None:
    check_rates_counted_over_the_right_denominators()
    check_verdict_thresholds()
    check_seeds_are_disjoint_and_sealed()
    print("\nAlle audit_evaluation-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 factor_lab/tests/test_audit_evaluation.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'factor_lab.audit.evaluation'`

- [ ] **Step 3: Implement `factor_lab/audit/evaluation.py`**

```python
"""
evaluation.py — Fehlalarmraten und die VORAB festgelegten Kriterien
(Spec Abschnitt 3 und 7).

Zwei getrennte Masse, weil sie verschiedene Dinge messen:
  FA_variant  -- schlaegt Gate A pro Einzeltest zu oft an? (Nominal 5%)
  FA_pipeline -- versiegelt die "Beste von 8"-Regel zu oft? (kein Nominalwert)
Die DIFFERENZ beider Zahlen ist die Selektionsinflation der Auswahlregel.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

WORKING_SEEDS = tuple(range(200))
CONTROL_SEEDS = tuple(range(10000, 10050))

FA_VARIANT_LOWER = 0.025
FA_VARIANT_UPPER = 0.10
FA_PIPELINE_LIMIT = 0.15


def _wilson_ci95(hits: int, total: int) -> tuple[float, float]:
    """Wilson-Intervall: bei kleinen Raten deutlich ehrlicher als die
    Normalapproximation, und bei hits=0 nicht entartet."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = hits / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def false_alarm_rates(records: list[dict]) -> dict:
    n_replications = len(records)
    variant_hits = sum(sum(r["gate_a"].values()) for r in records)
    n_variant_tests = sum(len(r["gate_a"]) for r in records)
    pipeline_hits = sum(r["candidate"] is not None for r in records)
    return {
        "n_replications": n_replications,
        "n_variant_tests": n_variant_tests,
        "fa_variant": variant_hits / n_variant_tests if n_variant_tests else 0.0,
        "fa_variant_ci95": _wilson_ci95(variant_hits, n_variant_tests),
        "fa_pipeline": pipeline_hits / n_replications if n_replications else 0.0,
        "fa_pipeline_ci95": _wilson_ci95(pipeline_hits, n_replications),
    }


def verdict(rates: dict) -> dict:
    fa_v, fa_p = rates["fa_variant"], rates["fa_pipeline"]
    calibrated = FA_VARIANT_LOWER <= fa_v <= FA_VARIANT_UPPER
    inflated = fa_p > FA_PIPELINE_LIMIT
    if not calibrated and fa_v > FA_VARIANT_UPPER:
        text = ("SCHWERWIEGEND: Bootstrap-Inferenz selbst fehlkalibriert "
                f"(FA_variant={fa_v:.1%} > {FA_VARIANT_UPPER:.0%}). Alle bisherigen "
                "Gate-A-Aussagen inklusive v2s Screening-Pass sind nicht interpretierbar.")
    elif not calibrated:
        text = (f"Test uebermaessig konservativ (FA_variant={fa_v:.1%} < "
                f"{FA_VARIANT_LOWER:.1%}). Echte Effekte wuerden uebersehen.")
    elif inflated:
        text = (f"Einzeltest kalibriert (FA_variant={fa_v:.1%}), aber die Auswahlregel "
                f"ist inflationaer (FA_pipeline={fa_p:.1%} > {FA_PIPELINE_LIMIT:.0%}). "
                "Mehrfachtest-Korrektur noetig, bevor eine kuenftige Familie versiegelt wird.")
    else:
        text = (f"Methodik-Pruefung bestanden (FA_variant={fa_v:.1%}, "
                f"FA_pipeline={fa_p:.1%}). KEIN Edge-Nachweis.")
    return {"calibrated": calibrated, "selection_inflated": inflated, "text": text}


def seal_control_seeds(path: str) -> str:
    payload = json.dumps({"control_seeds": list(CONTROL_SEEDS),
                          "working_seeds": list(WORKING_SEEDS)},
                         indent=2, sort_keys=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 factor_lab/tests/test_audit_evaluation.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/audit/evaluation.py factor_lab/tests/test_audit_evaluation.py
git commit -m "feat(audit): add false-alarm rates, predeclared criteria and sealed control seeds"
```

---

### Task 5: Pilot — Parameter einfrieren, Seeds versiegeln, Laufzeit messen

**Files:**
- Create: `factor_lab/audit/run_pilot.py`
- Create: `docs/superpowers/audit-pilot-2026-09-11-results.md` (Ergebnis des Laufs)

**Interfaces:**
- Consumes: alles aus Tasks 1-4.
- Produces: `factor_lab/audit_data/null_params.json` (+ Hash), `factor_lab/audit_data/audit_control_seeds.json` (+ Hash), je Welt ein `SYNTHETIC_NULL_<welt>_seed0_example.csv`, `factor_lab/audit_data/pilot_records.json`.

- [ ] **Step 1: Implement `factor_lab/audit/run_pilot.py`**

```python
"""
run_pilot.py — Schritt 1 des Audits: Parameter auf dem Archiv-Snapshot
schaetzen und einfrieren, Kontroll-Seeds versiegeln, je Nullwelt 5
Replikationen fahren und die Laufzeit messen.

Der Pilot entscheidet ueber das Hauptbudget. Er behauptet KEINE
Fehlalarmrate -- 5 Replikationen je Welt haben dafuer keine Power.

Ausfuehren: py -3.12 factor_lab/audit/run_pilot.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from factor_lab.audit.audit_runner import run_one_replication
from factor_lab.audit.evaluation import false_alarm_rates, seal_control_seeds
from factor_lab.audit.null_params import estimate_null_params, freeze_params, load_archive_snapshot
from factor_lab.audit.null_worlds import WORLDS, generate_null_panel

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "audit_data")
PILOT_SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Schaetze Nullwelt-Parameter auf dem Archiv-Snapshot (read-only)...", flush=True)
    params = estimate_null_params(load_archive_snapshot())
    params_path = os.path.join(DATA_DIR, "null_params.json")
    params_hash = freeze_params(params, params_path)
    print(f"  eingefroren: {params_path}\n  SHA256: {params_hash}", flush=True)

    seeds_path = os.path.join(DATA_DIR, "audit_control_seeds.json")
    seeds_hash = seal_control_seeds(seeds_path)
    print(f"  Kontroll-Seeds versiegelt: {seeds_path}\n  SHA256: {seeds_hash}", flush=True)

    for world in WORLDS:
        example = generate_null_panel(world, params, seed=0)
        frame = pd.DataFrame({s: df["price"] for s, df in example.items() if s != "IRX"})
        frame.to_csv(os.path.join(DATA_DIR, f"SYNTHETIC_NULL_{world}_seed0_example.csv"))

    records = []
    for world in WORLDS:
        for seed in PILOT_SEEDS:
            record = run_one_replication(world, params, seed)
            records.append(record)
            print(f"  {world} seed={seed}: candidate={record['candidate']} "
                  f"gate_a_hits={sum(record['gate_a'].values())}/8 "
                  f"{record['elapsed_s']:.1f}s", flush=True)

    with open(os.path.join(DATA_DIR, "pilot_records.json"), "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "params_sha256": params_hash,
                   "control_seeds_sha256": seeds_hash, "records": records}, f, indent=2)

    total = sum(r["elapsed_s"] for r in records)
    per_run = total / len(records)
    rates = false_alarm_rates(records)
    print(f"\nPilot: {len(records)} Replikationen, {total / 60:.1f} min gesamt, "
          f"{per_run:.1f} s je Lauf")
    print(f"Hochrechnung Hauptlauf (3 x 200): {per_run * 600 / 3600:.1f} h")
    print(f"Hochrechnung Kontrolllauf (3 x 50): {per_run * 150 / 3600:.1f} h")
    print(f"Orientierung (KEINE Aussage, zu wenig Power): FA_variant={rates['fa_variant']:.1%}, "
          f"FA_pipeline={rates['fa_pipeline']:.1%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the pilot**

Run: `py -3.12 factor_lab/audit/run_pilot.py`
Erwartung: laeuft mehrere Minuten bis Stunden. Dauert eine einzelne Replikation laenger als ~2 Minuten, ist der Volltreue-Hauptlauf (600 Replikationen) nicht in einer Sitzung machbar — dann Step 3 anwenden.

- [ ] **Step 3: Budget-Entscheidung anhand der gemessenen Laufzeit**

- Hochrechnung ≤ 6 h: Hauptlauf mit n=200 je Welt fahren (Task 6).
- Hochrechnung > 6 h: `n_boot` in `stationary_block_bootstrap` fuer den Audit von 10.000 auf 2.000 senken — **aber nur** nach dem in der Spec geforderten Treuecheck: dieselben 15 Pilot-Panels mit beiden Einstellungen rechnen und pruefen, dass die Gate-A-Entscheidung in ≥95% der 120 Variantenfaelle uebereinstimmt. Ergebnis des Checks im Bericht festhalten. Stimmt es nicht ueberein, n stattdessen auf das groesste machbare n ≥ 100 senken und die reduzierte Power offenlegen.
- Selbst mit n=100 nicht machbar: Audit als nicht durchfuehrbar berichten, nichts beschoenigen.

- [ ] **Step 4: Write `docs/superpowers/audit-pilot-2026-09-11-results.md`**

Inhalt: Parameter-Hash, Kontroll-Seed-Hash, gemessene Laufzeit je Replikation, Hochrechnung, getroffene Budget-Entscheidung mit Begruendung, ggf. Ergebnis des `n_boot`-Treuechecks. Keine Fehlalarm-Aussage aus dem Piloten ableiten.

- [ ] **Step 5: Commit**

```bash
git add factor_lab/audit/run_pilot.py factor_lab/audit_data docs/superpowers/audit-pilot-2026-09-11-results.md
git commit -m "feat(audit): freeze null-world parameters, seal control seeds and report pilot budget"
```

---

### Task 6: Hauptlauf, Kontrolllauf und Ergebnisbericht

**Files:**
- Create: `factor_lab/audit/run_audit.py`
- Create: `docs/superpowers/audit-2026-09-11-results.md`

**Interfaces:**
- Consumes: alles aus Tasks 1-5, inklusive der eingefrorenen `null_params.json`.
- Produces: `factor_lab/audit_data/audit_records_main.json`, `factor_lab/audit_data/audit_records_control.json`, Ergebnisbericht.

- [ ] **Step 1: Implement `factor_lab/audit/run_audit.py`**

```python
"""
run_audit.py — Haupt- und Kontrolllauf des Falsifikations-Audits.

Liest die im Piloten EINGEFRORENEN Parameter; schaetzt nicht neu. Der
Kontrolllauf darf erst laufen, wenn Code und Auswertung eingefroren sind
(--control), und wird getrennt gespeichert.

Ausfuehren: py -3.12 factor_lab/audit/run_audit.py [--n N] [--control]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from factor_lab.audit.audit_runner import run_one_replication
from factor_lab.audit.evaluation import CONTROL_SEEDS, WORKING_SEEDS, false_alarm_rates, verdict
from factor_lab.audit.null_worlds import WORLDS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "audit_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(DATA_DIR, "null_params.json"), encoding="utf-8") as f:
        params = json.load(f)

    seeds = CONTROL_SEEDS if args.control else WORKING_SEEDS[:args.n]
    label = "control" if args.control else "main"
    records = []
    for world in WORLDS:
        for i, seed in enumerate(seeds):
            records.append(run_one_replication(world, params, seed))
            if (i + 1) % 10 == 0:
                print(f"  {world}: {i + 1}/{len(seeds)}", flush=True)
        rates = false_alarm_rates([r for r in records if r["world"] == world])
        print(f"{world}: FA_variant={rates['fa_variant']:.1%} "
              f"FA_pipeline={rates['fa_pipeline']:.1%}", flush=True)

    path = os.path.join(DATA_DIR, f"audit_records_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "label": label, "records": records}, f, indent=2)

    overall = false_alarm_rates(records)
    print(f"\nGesamt ({label}, n={len(seeds)} je Welt): "
          f"FA_variant={overall['fa_variant']:.2%} CI{overall['fa_variant_ci95']}, "
          f"FA_pipeline={overall['fa_pipeline']:.2%} CI{overall['fa_pipeline_ci95']}")
    print(verdict(overall)["text"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Hauptlauf starten (detacht, da lang)**

Per Tech-Stack-Konvention detacht starten: Interpreter ueber `py -3.12 -c "import sys; print(sys.executable)"` aufloesen, dann PowerShell `Start-Process` mit `-u`, PID-Datei schreiben, Monitor auf Prozessende setzen.
Run: `py -3.12 -u factor_lab/audit/run_audit.py --n <aus Task 5 entschieden>`

- [ ] **Step 3: Auswertung einfrieren, dann Kontrolllauf**

Erst wenn Haupt-Auswertung steht und kein Code mehr geaendert wird:
Run: `py -3.12 -u factor_lab/audit/run_audit.py --control`
Abweichungskriterium: weicht FA_variant der Kontroll-Seeds um mehr als 5 Prozentpunkte vom Hauptlauf ab, gilt der Audit als an die Arbeits-Seeds angepasst und ist zu ueberarbeiten.

- [ ] **Step 4: Write `docs/superpowers/audit-2026-09-11-results.md`**

Inhalt: FA_variant und FA_pipeline je Welt und gesamt mit Wilson-CIs, die Differenz beider Masse als gemessene Selektionsinflation, das Verdikt nach den vorab festgelegten Schwellen, Kontroll-Seed-Vergleich, alle Abweichungen vom Plan (z.B. reduziertes `n_boot` oder n), und in einem eigenen Absatz: **was dieses Ergebnis NICHT sagt** (kein Edge-Nachweis, nur drei Nullwelten, Parameter aus demselben Snapshot).

- [ ] **Step 5: Full test suite + Commit**

Alle Testdateien unter `factor_lab/tests/` einzeln mit `py -3.12` laufen lassen (insbesondere die vier neuen), dann:

```bash
git add -A
git commit -m "feat(audit): run falsification audit and report false-alarm calibration"
```
