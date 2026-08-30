# Cross-Sectional Portfolio Design

## Kontext & Motivation

Drei unabhängige Spikes (2026-08-27, siehe Projekt-Memory zum Overnight-Backtest) haben übereinstimmend gezeigt, dass es in Einzelaktien-1-Minuten-Preis-/Volumen-/Quote-Daten mit dem bestehenden 7-Feature-Set kein nachweisbares Richtungssignal gibt — weder über verschiedene Horizonte (1-30 Min), noch mit echter statt approximierter Quote-Mikrostruktur, noch mit einem relativen (statt absoluten) Zielmaß gegen SPY. Die Regelschleife selbst (Messen → LSTM-Prognose → Controller → Ausführung → Feedback → Online-Korrektur) ist dabei nachweislich korrekt implementiert; das fehlende Element ist ausschließlich das Eingangssignal.

**Die Wette dieses Designs:** nicht, dass ein einzelnes Symbol plötzlich mehr Signal zeigt, sondern dass viele, für sich genommen statistisch nicht nachweisbare Pro-Symbol-Edges, wenn sie über ausreichend unkorrelierte Symbole gebündelt werden, einen im Portfolio nachweisbaren Effekt ergeben können (Portfolio-Sharpe wächst für unkorrelierte Einzel-Bets ungefähr mit der Wurzel der Anzahl Bets — klassische Stat-Arb-Diversifikationslogik). Das ist ein Power-Argument, keine neue Signal-Behauptung.

## Ziel

Ein markt-neutrales Long/Short-Portfolio über ein festes Universum liquider Large Caps, das die bestehenden Pro-Symbol-LSTM-Prognosen cross-sectional rankt und daraus Positionen ableitet, statt jedes Symbol unabhängig long-only zu handeln.

## Nicht-Ziel (explizit außerhalb des Scopes)

- Keine Änderung an Feature-Set, Modellarchitektur oder Trainingslogik pro Symbol — alle drei wurden bereits als nicht der limitierende Faktor identifiziert.
- Kein dynamisches Symbol-Universum (Auswahl/Rotation von Symbolen) — fixe Liste für die erste Iteration.
- Keine Risikofaktor-Modelle (Sektor-/Beta-Neutralität über einfache Long/Short-Gleichgewichtung hinaus).
- Keine Positionsgrößen-Optimierung über Gleichgewichtung innerhalb eines Legs hinaus (kein Mean-Variance-Optimizer o.ä.).
- Universum-Größe > 12 Symbole ist eine spätere Erweiterung, kein Ziel dieser Iteration.

## Architektur-Überblick

```
Pro Symbol (x12, unveraendert):
  Historische Bars -> FeaturePipeline (7 Features) -> Scaling -> LSTM
  -> (mu, sigma) pro Zeitschritt -> edge_i = mu_i / (sigma_i^2 + eps)

NEU -- Synchronisierter Cross-Sectional-Loop:
  Fuer jeden GEMEINSAMEN Zeitschritt (Bar-Timestamp, den ALLE 12 Symbole haben):
    1. Sammle edge_i fuer alle 12 Symbole
    2. Cross-Sectional-Ranking + Hysterese -> Ziel-Leg pro Symbol (long/short/flat)
    3. Positionsgroessen: gleichgewichtet innerhalb jedes Legs, dollar-neutral
    4. Ausfuehrung + Slippage pro Symbol (bestehende Logik, pro Leg)
    5. Portfolio-Return dieses Zeitschritts = Long-Leg-Return - Short-Leg-Return
    6. Feedback/Online-Training WEITERHIN pro Symbol unabhaengig (unveraendert)
```

## Komponenten im Detail

**Symbol-Universum.** Fixe Liste von 12 liquiden Large Caps, bewusst sektorübergreifend gestreut (nicht 12x Tech), damit die sqrt(N)-Diversifikationsannahme nicht durch hohe Kreuzkorrelation unterlaufen wird: AAPL, MSFT, GOOGL, NVDA (Tech), JPM (Finanzen), JNJ (Pharma), XOM (Energie), PG (Konsumgüter), HD (Einzelhandel), DIS (Medien), KO (Getränke), CAT (Industrie).

**Per-Symbol-Modelle.** Unverändert aus `run_backtest.py::run_symbol_backtest()` — gleiches Offline-Pretraining, gleiche Feature-Skalierung, gleicher Online-Trainer pro Symbol. Wiederverwendet, nicht neu gebaut.

**Bar-Ausrichtung über Symbole.** Nur Zeitstempel, an denen ALLE 12 Symbole einen Bar haben, gehen in den Cross-Sectional-Loop ein (Inner-Join über die Zeitindizes). Ein Symbol mit Datenlücke an einem Zeitpunkt fällt für diesen einen Zeitschritt aus dem Ranking, eine bestehende Position wird mit dem letzten bekannten Preis fortgeführt (Mark-to-Market), nicht zwangsliquidiert. Bei den gewählten 12 liquiden Large Caps werden Lücken selten erwartet; falls sie in der Praxis zu häufig sind, ist das ein Befund für die spätere Auswertung, kein Blocker für die erste Iteration.

**Cross-Sectional-Ranking + Hysterese (der eigentlich neue Baustein, neues Modul `controller/cross_sectional_portfolio.py`).** Pro Zeitschritt: alle verfügbaren `edge_i` absteigend sortieren. Ein Symbol, das aktuell NICHT gehalten wird, geht neu long, wenn es in den Top 3 liegt (bzw. short bei Bottom 3). Ein Symbol, das BEREITS long gehalten wird, bleibt long, solange es in den Top 5 bleibt (analog Bottom 5 für short) — erst beim Verlassen dieser breiteren Zone wird die Position geschlossen. Das ist die Cross-Sectional-Entsprechung zu `RiskOverlayConfig.min_rebalance_threshold` (gleicher Zweck: Rauschen an der Entscheidungsgrenze nicht in Turnover übersetzen). Konfigurierbar: `n_long`/`n_short` (Start: 3), `hysteresis_zone` (Start: 5).

**Positionsgrößen.** Gleichgewichtet innerhalb jedes Legs (z.B. 3 Long-Positionen à 1/3 Long-Gesamtexposure). Long-Gesamtexposure = Short-Gesamtexposure (dollar-neutral). Kein Kelly/Edge-proportionales Sizing in dieser ersten Iteration (bewusst einfach gehalten — YAGNI, bis eine einfachere Version als unzureichend nachgewiesen ist).

**Ausführung/Kosten.** Bestehendes Slippage-Modell (`ExecutionConfig`) pro Symbol/Leg wiederverwendet. Explizit erwartet: ein Long/Short-Buch über 12 Namen handelt strukturell MEHR als ein Long-Only-Einzelsymbol (zwei Legs, mehr Symbole) — Turnover/Slippage sind hier eher wichtiger als beim bisherigen System, nicht unwichtiger. Deshalb Hysterese von Beginn an Teil des Designs, nicht nachträgliche Reparatur.

**Auswertung.** Portfolio-Return pro Zeitschritt = mittlerer Return des Long-Legs minus mittlerer Return des Short-Legs. Diese Zeitreihe geht in die bestehende `backtest_stats.py`-Periodenstatistik (unverändert wiederverwendet). Vergleich verschiedener Parametrisierungen (`n_long`/`n_short`, `hysteresis_zone`) ausschließlich über die bestehende Multi-Seed-Infrastruktur (`run_multi_seed_comparison.py`-Muster) — kein Einzellauf wird als Ergebnis interpretiert, nach der Lehre aus der gesamten bisherigen Session.

## Konfigurierbare Parameter (Startwerte)

| Parameter | Startwert | Begründung |
|---|---|---|
| Symbol-Universum | 12 Symbole (Liste oben) | Klein genug für vertretbare Rechenzeit, sektordivers genug für Diversifikationsannahme |
| `n_long` / `n_short` | 3 / 3 | Genug Diversifikation innerhalb von 12, nicht zu granular |
| `hysteresis_zone` | 5 | Cross-Sectional-Analogon zum bereits bewährten Deadband-Prinzip |
| Rebalance-Kadenz | jeder Bar (wie bisher) | Ein einziger Turnover-Kontrollmechanismus (Hysterese) statt zwei überlagerten (Hysterese + Zeit-Throttle) — einfacher zu verstehen und zu debuggen; ein Zeit-Throttle kann später ergänzt werden, falls die Hysterese allein nicht reicht |

## Testen/Validierung

1. **Unit-Ebene:** Ranking+Hysterese-Logik in `cross_sectional_portfolio.py` mit synthetischen Edge-Werten testen (z.B. ein Symbol pendelt knapp um Rang 3/4 — muss dank Hysterese NICHT bei jedem Schritt rotieren).
2. **Smoke-Test:** 2-3 Tage, 4-5 Symbole (statt 12), vor jedem echten Lauf — Muster aus der gesamten Session beibehalten.
3. **Multi-Seed-Vergleich:** verschiedene `n_long`/`hysteresis_zone`-Kombinationen über mehrere Seeds, bevor irgendeine Parametrisierung als "besser" gilt.
4. **Baseline-Vergleich:** das resultierende Long/Short-Portfolio gegen ein triviales Long-Only-Gleichgewicht aller 12 Symbole (kein Ranking, keine Prognose) stellen — nur so lässt sich zeigen, dass das Ranking selbst etwas beiträgt und nicht nur Marktexposure kaschiert.

## Offene Punkte für die Implementierungsplanung

- Exakte Reihenfolge/Struktur des neuen Orchestrierungsskripts (vermutlich `orchestration/run_cross_sectional_backtest.py`, analog zu `run_backtest.py`).
- Ob der synchronisierte Multi-Symbol-Loop die 12 `ControlLoop`-Instanzen direkt wiederverwendet (ein `ControlLoop.step()`-Aufruf pro Symbol pro Zeitschritt, Ergebnisse eingesammelt) oder eine leichtgewichtigere Variante ohne den Single-Symbol-Positions-/Risk-Overlay-Anteil von `ControlLoop` braucht (da Positionsgröße jetzt vom Cross-Sectional-Layer kommt, nicht vom Single-Symbol-Controller) — wird in der Implementierungsplanung entschieden.
