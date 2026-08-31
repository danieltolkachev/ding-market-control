# Cross-Sectional Signal Diagnostics Design

## Kontext & Motivation

Eine externe Code-Review des gemergten Cross-Sectional-Portfolio-PRs (`ef00950`, siehe [PR #1](https://github.com/danieltolkachev/ding-market-control/pull/1)) hat mehrere methodische Probleme aufgedeckt, die vor dem PR nicht erkannt wurden:

1. **Zeitachse/Ausführung:** `ControlLoop`/`SymbolForecaster` entscheiden UND fuellen am selben Bar (`execute()` bekommt denselben `raw_event`, aus dem auch die Prognose kam). Im Quote-basierten Pfad (`fetch_historical_market_data`) kommt dazu, dass Alpaca-Minutenbars linksseitig gestempelt sind (Bar `14:52` enthaelt Trades bis `14:53`), `merge_asof(..., direction="backward")` aber eine Quote `<=14:52` anhaengt -- eine Quote, die bereits vor dem Bar-Ende, teils sogar vor Bar-Beginn, veraltet sein kann. Betrifft `run_live.py`, `run_replay.py` und den fruehesten Session-Spike zu "echte Quotes vs. approximiert" direkt; betrifft den approximierten Pfad (`fetch_historical_bars_approximate`, von der Cross-Sectional-PR genutzt) nur in der schwaecheren Form (kein separates Quote-Timing-Problem, aber weiterhin Entscheidung=Fill-Bar).
2. **Portfolio-Invariante gebrochen:** der in der PR nachgeruestete Rebalance-Deadband wird PRO SYMBOL geprueft, nicht portfolio-weit reprojiziert -- das ausgefuehrte Buch kann von den (eigentlich dollar-neutralen) Zielgewichten abdriften (verifiziert: bis zu ~12,5% Netto-Exposure bei Standardkonfiguration).
3. **Hysterese-Vergleich konfundiert:** die Multi-Seed-Varianten `3/3/Zone5` vs. `5/5/Zone6` aendern Leg-Groesse UND Zone gleichzeitig. Nachgerechnet: der Haltepuffer (`hysteresis_zone - n_long`) schrumpft von 2 auf 1 -- die "breitere" Variante hat tatsaechlich einen KLEINEREN relativen Puffer. Das erklaert den beobachteten Turnover-Anstieg wahrscheinlich vollstaendig als Artefakt. **Die bisherige Schlussfolgerung "breitere Hysterese erhoeht Turnover" gilt als zurueckgezogen, nicht als Befund.**
4. **Keine Score-Normalisierung:** 12 unabhaengig trainierte Modelle werden ueber rohes `mu/sigma^2` direkt gerankt (`cross_sectional_portfolio.py::compute_edges`) -- nichts stellt eine gemeinsame Kalibrierungsskala sicher. Modell-Kalibrierungsunterschiede koennen das Ranking dominieren, nicht echtes Signal.
5. **Kein eingefrorener Datensatz:** jeder Lauf holt Daten relativ zu `datetime.now()` -- Varianten und Seeds laufen nicht garantiert auf identischen Bars.
6. **Falsche Return-/Drawdown-Mathematik:** `backtest_stats.py::summarize_period_returns` summiert Perioden-Returns additiv (`period_returns.sum()`, `cumsum()` fuer Drawdown) statt zu compounden. Die Baseline nutzt Log-Returns (die sich korrekt zu echtem Compounding aufsummieren), die Strategie einfache Returns -- unterschiedliche Renditebegriffe, nicht direkt vergleichbar.

**Kernaussage der Review, der dieses Design folgt:** Diversifikation kann eine vorhandene kleine Edge stabilisieren, aber aus zwoelf Signalen mit Erwartungswert null keine Edge erzeugen. Ohne positiven Rank-IC ist der Erwartungswert vor Kosten null, danach negativ. Deshalb: **kein weiteres LSTM-/Deadband-/Hysterese-Tuning, bevor nicht belegt ist, dass ueberhaupt ein Brutto-Signal existiert.**

## Ziel

Eine von Controller/Ausfuehrung/Kosten komplett entkoppelte Diagnose-Pipeline, die misst, ob die 12 Symbol-Modelle ueberhaupt eine cross-sectional auswertbare Edge liefern -- bevor irgendeine weitere Arbeit an Positionsgroessen, Deadbands oder Hysterese investiert wird.

## Nicht-Ziel (explizit ausserhalb des Scopes)

- Keine Aenderung an `ControlLoop`, `PaperExecutionEngine`, `CrossSectionalPortfolio` oder dem bestehenden Deadband -- dieses Design testet Signalqualitaet OHNE diese Komponenten zu benutzen, nicht als Ersatz fuer sie.
- Keine Kosten-/Deadband-/Controller-Optimierung. Explizites Gate: nur bei positivem, ueber mehrere Walk-Forward-Folds robustem Brutto-Rank-IC wird das ueberhaupt zum naechsten Schritt.
- Kein Fix des Quote-Merge-Zeitachsenfehlers in `fetch_historical_market_data()` -- dieses Design nutzt ausschliesslich den approximierten Bar-Pfad und braucht daher keine separate Quote-Zeitreihe. Der Fix fuer `run_live.py`/`run_replay.py` ist ein eigenes, spaeteres Thema.
- Keine Behebung der Portfolio-Invariante im bestehenden Deadband-Code -- irrelevant, da dieses Design keine Positionsfuehrung simuliert (jede Bewertung nutzt bei jedem Zeitschritt frisch berechnete, exakt neutrale Gewichte ohne Gedaechtnis an die Vorperiode).

## Architektur-Ueberblick

```
1x Datensatz-Snapshot (12 Symbole, 1 Jahr, approximierte Bars)
  -> lokal eingefroren (Pickle/Parquet + Content-Hash), NIE erneut per datetime.now() geladen

Cross-Sectional-Walk-Forward (verallgemeinert training/walk_forward.py):
  Pro Fold (rollierendes Fenster, wie bisher):
    - 12 FRISCHE Modelle trainiert auf dem Fold-Trainingsfenster
    - KEIN Online-Training waehrend der Fold-Evaluierung (verhindert Drift)
    - Score-Varianten pro Symbol berechnet: mu, mu/sigma, kalibriertes Kelly-Edge, p_up,
      plus kausale Score-Kalibrierung (rollierendes Perzentil der eigenen Score-Historie)
    - Pro gemeinsamem Zeitschritt im Testfenster:
        Rank-IC (Spearman: Score_t ueber 12 Symbole vs. tatsaechlicher Return t->t+1)
        Brutto-Top3-minus-Bottom3-Spread (kostenlos, kein Deadband, korrekt compoundend)
    - Baselines im selben Fold/Fenster: Random-Ranking, Momentum, Reversal

Holdout-Fenster (letzter Abschnitt des Snapshots): NIE angerührt, bis ueber alle
Trainings-Folds hinweg ein robuster, positiver Rank-IC gezeigt wurde.
```

## Komponenten im Detail

**Daten-Snapshot (`data_layer/frozen_snapshot.py`, neu).** Laedt alle 12 Symbole (`fetch_historical_bars_approximate`) einmal, richtet sie per Inner-Join aus (wiederverwendet dieselbe Logik wie `run_cross_sectional_backtest.py::align_and_split`), serialisiert das Ergebnis in eine Datei unter `market_control_system/data_snapshots/` (gitignored, wie `logs/`) mit einem Content-Hash im Dateinamen. Alle folgenden Skripte laden ausschliesslich diese Datei, nie erneut per Netzwerk -- ausser beim allerersten Erzeugen. Loest Punkt 5.

**Cross-Sectional-Fold-Generator (`training/cross_sectional_walk_forward.py`, neu, wiederverwendet `generate_fold_slices`/`evaluate_fold`-Muster aus `training/walk_forward.py`).** Nimmt den eingefrorenen Snapshot, erzeugt rollierende (Trainingsfenster, Testfenster)-Paare ueber den gemeinsamen Zeitindex (nicht ueberlappend, analog zu `WalkForwardConfig`). Pro Fold: 12 frische `LSTMForecaster`-Instanzen, `train_epoch` auf dem Fold-Trainingsfenster (identische Hyperparameter wie ueberall im Projekt: hidden_size=32, num_layers=2, 10 Epochen, lr=1e-3, batch_size=64), **kein** `OnlineTrainer`/`SymbolForecaster` -- reines Batch-Predict auf dem Testfenster.

**Score-Berechnung (`controller/cross_sectional_signal_metrics.py`, neu).** Pro Symbol, pro Zeitschritt im Testfenster: `mu`, `sigma`, `p_up` aus `model.predict()`. Vier Score-Varianten separat ausgewertet:
- `mu` (roh)
- `mu/sigma` (Sharpe-artig, ohne Quadrat)
- kalibriertes Kelly-Edge: `calibrate_k()` (bestehend, `controller/exposure_controller.py`) pro Symbol auf dem TRAININGSFENSTER angewandt, liefert `k_i`; Score = `k_i * mu_i/sigma_i^2`
- `p_up` (roh, bereits in [0,1] und damit implizit cross-sectional vergleichbar in seiner eigenen Grössenordnung)
- kausale Score-Kalibrierung: rollierendes Perzentil (Fenstergroesse konfigurierbar, Default 500 Bars) des jeweiligen rohen Scores relativ zur EIGENEN Historie des Symbols, NICHT ein Querschnitts-Z-Score ueber die 12 aktuellen Werte (aendert die Rangfolge nicht, siehe Review-Punkt 4).

**Rank-IC & Brutto-Spread (`controller/cross_sectional_signal_metrics.py`).**
- `compute_rank_ic(scores: dict[str, float], forward_returns: dict[str, float]) -> float`: Spearman-Korrelation ueber die (bis zu) 12 Symbole zu einem Zeitpunkt. `forward_returns` = tatsaechlicher Return von Bar t zu Bar t+1 (nutzt den bestehenden, bereits look-ahead-freien Shift-Mechanismus aus `feature_pipeline.py` mit `horizon=1` -- absichtlich 1 Bar, nicht 5, um die Interpretation "Score bei t sagt Return t->t+1 vorher" eindeutig zu halten und jede Ambiguitaet zwischen Entscheidungs- und Fuellzeitpunkt zu vermeiden).
- `compute_gross_spread(scores, forward_returns, n_long=3, n_short=3) -> float`: Top-n_long gleichgewichtet long, Bottom-n_short gleichgewichtet short, KEIN Deadband, KEINE `PaperExecutionEngine` -- direkte Rueckgabe des ungekuerzten Spread-Returns fuer diesen Zeitschritt.
- `compute_breakeven_cost(gross_spread_series, turnover_series) -> float`: welcher Pro-Trade-Kostensatz den mittleren Brutto-Spread auf 0 druecken wuerde.
- Compounding: alle kumulierten Kennzahlen ueber `prod(1 + r) - 1`, Drawdown ueber eine echte Equity-Kurve (`equity_t = equity_{t-1} * (1 + r_t)`), NICHT ueber additive Summen/`cumsum`. Loest Punkt 6 fuer dieses neue Modul (bestehendes `backtest_stats.py` bleibt unveraendert, da es an anderer Stelle im Projekt bereits verwendet wird und eine Aenderung dort ausserhalb des Scopes dieses Designs liegt).

**Baselines (dieselbe Datei).** Fuer denselben Fold/dasselbe Testfenster:
- `random_ranking_scores(symbols, seed) -> dict`: zufaellige Scores pro Zeitschritt, ueber mehrere Seeds gemittelt, Turnover empirisch auf denselben mittleren Turnover wie das echte Ranking normiert (durch Wiederholung mit mehreren Zufalls-Seeds und Mittelung, nicht durch exakte Turnover-Erzwingung -- YAGNI, die Naeherung reicht fuer einen Plausibilitaets-Vergleich).
- `momentum_scores(returns_window, lookback_bars) -> dict`: Score = Return der letzten `lookback_bars` Bars.
- `reversal_scores(...)`: Score = negativer Momentum-Score.

**Orchestrierung (`orchestration/run_cross_sectional_signal_diagnostics.py`, neu).** Bindet alles zusammen: Snapshot laden/erzeugen, Folds generieren, pro Fold pro Score-Variante Rank-IC/Spread/Breakeven berechnen, Baselines im selben Fold mitlaufen lassen, Ergebnisse als JSON + lesbarer Report speichern (Muster wie alle bisherigen `orchestration/run_*`-Skripte: Fortschritts-Prints, JSON-Summary, `py -3.12` Ausfuehrung).

**Holdout.** Der Snapshot wird nach dem Laden in zwei Teile geschnitten: `WALKFORWARD_FRACTION` (Default 0.8) fuer alle Fold-Iterationen oben, der Rest als `HOLDOUT`-Fenster, das von keinem Code-Pfad in diesem Design vor der finalen Auswertung gelesen wird. Erst wenn die Walk-Forward-Folds einen robusten (im Mittel positiven, in der Mehrheit der Folds positiven) Rank-IC zeigen, wird ein einziger, expliziter Auswertungslauf auf dem Holdout durchgefuehrt -- das Skript daf¸r ist ein separater, manuell auszuloesender Schritt, nicht Teil der automatischen Fold-Schleife (verhindert versehentliches Holdout-Peeking durch wiederholtes Ausfuehren).

## Konfigurierbare Parameter (Startwerte)

| Parameter | Startwert | Begruendung |
|---|---|---|
| Universum | dieselben 12 Symbole wie die PR | Konsistenz mit bisherigen Ergebnissen |
| Walk-Forward Trainingsfenster | 2000 Sequenzen (wie `WalkForwardConfig`-Default) | bewaehrter Wert aus `training/walk_forward.py` |
| Walk-Forward Testfenster | 400 Sequenzen | bewaehrter Wert aus `training/walk_forward.py` |
| `n_long`/`n_short` fuer Brutto-Spread | 3/3 | kleinstes sinnvolles Long/Short-Buch bei 12 Symbolen |
| Rolling-Perzentil-Fenster | 500 Bars | grob 1-2 Handelstage bei 1-Min-Bars, genug Historie fuer ein stabiles Perzentil |
| Horizon fuer Rank-IC/Spread | 1 Bar | eliminiert die Entscheidungs-/Fuellzeitpunkt-Ambiguitaet aus Review-Punkt 1 durch Konstruktion |
| `WALKFORWARD_FRACTION` (Holdout-Split) | 0.8 | 80% fuer iterative Folds, 20% unangetasteter Holdout |

## Testen/Validierung

1. **Unit-Ebene:** `compute_rank_ic`/`compute_gross_spread`/`compute_breakeven_cost` mit synthetischen Scores/Returns testen (z.B. perfekt korrelierte Scores -> Rank-IC nahe 1.0; Scores unabhaengig vom Return -> Rank-IC nahe 0 im Erwartungswert). Plain-Assert-Skript unter `tests/`, wie im gesamten Projekt ueblich (kein pytest).
2. **Compounding-Korrektheit:** ein Testfall mit bekannten Returns, wo `prod(1+r)-1` nachweislich von `sum(r)` abweicht (grosse Returns), um zu verifizieren, dass die neue Compounding-Logik tatsaechlich verwendet wird.
3. **Smoke-Test:** kleine Fold-Konfiguration (wenige Symbole, kurzer Snapshot) vor dem echten Lauf, wie bei jedem bisherigen Orchestrierungsskript in diesem Projekt.
4. **Kein Multi-Seed-Vergleich in dieser Phase** -- Walk-Forward-Folds ueber mehrere Zeitfenster ersetzen hier die Notwendigkeit von Seed-Wiederholungen fuer die Kernfrage "gibt es ueberhaupt Signal"; Seeds sind sekundaer (siehe Review-Punkt 5) und werden erst relevant, falls/wenn das Design das Gate zu Kosten-/Controller-Arbeit passiert.

## Offene Punkte fuer die Implementierungsplanung

- Exakte Serialisierungsform des Snapshots (Pickle vs. Parquet) -- wird in der Planung entschieden, keine funktionale Auswirkung auf das Design.
- Ob `random_ranking_scores`' Turnover-Normalisierung ueber eine feste Anzahl Wiederholungen (z.B. 20 Seeds) oder eine adaptive Toleranz laeuft -- Implementierungsdetail, kein Design-Entscheid.
