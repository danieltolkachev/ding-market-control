# Daily-Factor-Lab, Bein 1: Time-Series-Trend auf Cross-Asset-ETFs

**Datum:** 2026-09-01 (v2 — Revision nach externem Review, gleicher Tag)
**Status:** v2 in Review; Implementierung pausiert bis zur Freigabe
**Experiment-Familie:** `trend-etf-v1` (siehe Abschnitt 12, Registrierung)

**Änderungshistorie:**
- v1 (2026-09-01, gemergt in `5b5424b`): Erstfassung.
- v2 (2026-09-01): Revision nach externem Review. P0-Korrekturen: ausführbare
  Execution-Konvention mit vollem Handelstag Lag; gemeinsames Evaluationsfenster
  für alle Varianten und Benchmarks; Matched-No-Signal-Benchmark und Gates auf
  gepaartem Mehrertrag; versiegeltes One-Shot-Holdout; Gross-Cap-Semantik
  präzisiert; Stationary-Block-Bootstrap statt unabhängiger Monats-Blöcke;
  Screening-/Bestätigungs-Trennung; Kostenleiter; verzinstes Cash; fixes
  Snapshot-Ende und versiegelte Datenartefakte; operationale Präregistrierung.
  Der zugehörige Bugfix (`max_drawdown_from_returns` ignorierte die Start-Equity)
  liegt im selben Korrektur-PR.

**Vorgeschichte:** Die 1-Minuten-LSTM-Linie wurde nach dem 188-Fold-Rank-IC-
Nullergebnis (PR #2) beendet. Dieses Lab ist ein NEUES Forschungsprojekt mit
Tageshorizont, einfachen Regeln und ohne LSTM.

## 1. Forschungsfrage

Liefert einfacher Time-Series-Trend auf einer festen Liste liquider Cross-Asset-
ETFs NACH Kosten einen belastbaren MEHRWERT GEGENÜBER PASSIVEM HALTEN DERSELBEN
INSTRUMENTE MIT IDENTISCHEM SIZING — bei historischem Max-Drawdown <= 15 % und
ohne Leverage?

Ehrlich formuliert: Trend allein liefert die 12.7 %-Zielrendite nicht — es ist
EIN Baustein (Beta + kleines aktives Alpha − Kosten). Ein positives absolutes
Ergebnis allein beweist keinen Trend-Effekt (das kann Beta sein); deshalb messen
die Gates den gepaarten Mehrertrag gegenüber einem Matched-Benchmark (Abschnitt 9).
Das Drawdown-Gate ist ein HISTORISCHES Kriterium des Backtests, keine Garantie
für zukünftige Drawdowns.

## 2. Nicht-Ziele

- KEIN Cross-Sectional-Momentum auf Einzelaktien (Bein 2, wartet auf die
  Datenquellen-/Budget-Entscheidung).
- KEIN Training/ML, keine Parameteroptimierung über die 8 präregistrierten
  Varianten hinaus.
- KEINE Live-/Paper-Execution. Erst Signalnachweis, dann (separates Projekt)
  Execution mit echten Brokerorders, Ledger, Reconciliation.
- KEIN Leverage in den Zielgewichten (Abschnitt 6), keine Intraday-Daten.

## 3. Entscheidungen (User-fixiert)

| Entscheidung | Wert |
|---|---|
| Startscope | Trend-Leg zuerst |
| Drawdown-Cap (historisches Gate) | 15 % auf der Netto-Equity-Kurve inkl. Start-Equity 1.0 |
| Ort | gleiches Repo, Top-Level-Package `factor_lab/` |
| Ansatz | A — minimales präregistriertes Baseline-Lab |

## 4. Daten

- **Universum (12 ETFs, Auflage <= 2006):** SPY, QQQ, IWM (US-Aktien), EFA, EEM
  (Intl/EM), TLT, IEF, LQD (Anleihen), GLD, SLV, DBC (Gold/Silber/Rohstoffe),
  VNQ (REITs). Sleeves für Gate D: US-Equity {SPY, QQQ, IWM}, Intl-Equity
  {EFA, EEM}, Bonds {TLT, IEF, LQD}, Real Assets {GLD, SLV, DBC, VNQ}.
- **Cash-Verzinsung:** 13-Wochen-T-Bill-Rendite als 13. Serie (yfinance `^IRX`,
  Prozent p.a.); Cash verdient täglich `IRX/100/252`. Näherung (Discount-Yield,
  kein Reinvestitions-Detail), dokumentiert; gilt identisch für Strategie und
  Benchmarks. Excess-Kennzahlen (Sharpe über Cash) werden mit ausgewiesen.
- **Quelle:** yfinance, Tagesdaten, `auto_adjust=True` (Splits + Dividenden in
  den Preisen; Returns sind Total-Return-Näherungen; TER steckt im Preis).
- **Fixes Zeitfenster:** `SNAPSHOT_START = 2007-01-01`,
  `SNAPSHOT_END = 2026-09-01` (exklusiv). Beide gehen in den Parameter-Hash des
  Snapshot-Dateinamens ein.
- **Versiegelter Snapshot (eigener Schritt VOR jeder Auswertung):** separates
  Build-Skript fetcht einmalig, friert ein (Pickle), berechnet Content-SHA256,
  schreibt ein Manifest (Zeilen + Zeitraum je Serie) und führt fail-closed
  Sanity-Checks aus: keine Duplikate im Index, keine NaN/nichtpositiven Preise,
  keine Serie endet früher als `SNAPSHOT_END` minus 5 Handelstage, gemeinsamer
  Kalender = Schnittmenge mit mindestens 4800 Tagen. Das Manifest wird
  COMMITTET, BEVOR irgendein Baseline-Lauf startet. Das Pickle bleibt als
  unveränderliches lokales Artefakt erhalten (gitignored, Umzugs-Regel wie beim
  PR-#2-Snapshot).
- **Auswertungen fetchen NIE:** `run_trend_baseline.py`/`run_trend_holdout.py`
  laden nur, verifizieren den Content-Hash gegen das committete Manifest und
  brechen bei Abweichung ab.
- **Split:** `DEV_END` = letzter Monatsultimo des gemeinsamen Kalenders, der
  <= dem 80%-Quantil-Datum liegt (deterministische Regel, beim Snapshot-Seal
  einmal ausgerechnet und in die Registrierung geschrieben). Entwicklungsfenster
  = Tage <= `DEV_END`; **Holdout** = Tage > `DEV_END`, nur über den versiegelten
  Mechanismus aus Abschnitt 10 lesbar.
- **Versionsprotokoll:** jede Ergebnis-Provenance enthält Python-, pandas-,
  numpy-, yfinance-Version, Git-SHA und den Config-Hash (Abschnitt 12).

## 5. Signale (präregistriert, KEINE weiteren Varianten)

Je Instrument i und Handelstag t:

- `mom_L(i, t) = sign( AdjClose(i, t) / AdjClose(i, t - L) - 1 )` für
  L ∈ {63, 126, 252} Handelstage.
- `combo` = Gleichgewichts-Mittel der drei Vorzeichen (in {-1, -1/3, +1/3, +1}).

Zwei Portfolio-Modi je Signalvariante: **long/short** (klassisches TSMOM) und
**long/flat** (negativ → Cash). long/short ist wegen unkalibrierter
Borrow-/Margin-Annahmen ausdrücklich als FORSCHUNGSVARIANTE gelabelt; die
operational glaubwürdige Familie für ein kleines Konto ist long/flat.

Ergibt **8 präregistrierte Varianten**. Alle 8 werden immer gerechnet und
veröffentlicht. Änderungen nur über die Amendment-Regel (Abschnitt 12).

## 6. Execution-Konvention und Portfolio-Konstruktion

- **Ausführbare Close-only-Konvention mit vollem Handelstag Lag:** Entscheidung
  am Monatsultimo t auf Basis von Close(t) (Signal, Vol, Zielgewichte); Fill zum
  Close(t+1) — Handelskosten fallen an t+1 an; die neue Position wirkt erstmals
  auf den Return von t+1 nach t+2. Kein Wert, der in die Zielgewichte eingeht,
  ist zum Fill-Zeitpunkt unbekannt.
- **Inverse-Vol-Sizing:** Rohgewicht `w_i ∝ signal_i / sigma_i`; `sigma_i` =
  annualisierte EWMA-Vol der Tagesreturns (Span 63, min_periods 63, kausal bis t).
- **Vol-OBERGRENZE (nicht "Ziel"):** Normierung auf Gross 1.0, dann Skalierung
  `min(1, 0.10 / realisierte 63-Tage-Vol des Kandidatenportfolios)` — EINE
  Formel: einfache Standardabweichung der mit den Kandidatengewichten
  gewichteten letzten 63 Tagesreturns, annualisiert mit sqrt(252). (Kein
  EWMA-Kovarianz-Wording mehr; v1 war hier doppeldeutig.) Wegen Gross <= 1 ist
  0.10 eine Obergrenze; unerreichte Vol wird nicht hochgehebelt.
- **Gross-Cap-Semantik (präzisiert):** Der Cap 1.0 gilt für die REBALANCE-
  ZIELGEWICHTE. Zwischen Rebalances driften die Gewichte mit den Preisen; im
  long/short-Modus kann das tägliche Gross dadurch vorübergehend über 1 liegen
  (long/flat beweisbar nie). Die Baseline erzwingt KEIN tägliches Deleveraging
  (keine synthetischen Zwangstrades); stattdessen wird das maximale tägliche
  Gross je Variante berechnet und im Report ausgewiesen.
- **Rebalancing:** monatlich am letzten gemeinsamen Handelstag (Entscheidung),
  Ausführung am Folgetag (siehe oben), dazwischen Drift.
- **Cash:** verzinst mit der T-Bill-Näherung (Abschnitt 4), identisch in
  Strategie und Benchmarks.

## 7. Gemeinsames Evaluationsfenster

Alle 8 Varianten und ALLE Benchmarks werden auf einem IDENTISCHEN PnL-Index
ausgewertet:

- Warmup = 252 Handelstage (längster Lookback) + 63 (Vol) ab Beginn des
  gemeinsamen Kalenders; frühere Tage dienen ausschließlich als Warmup.
- `EVAL_DECISION_0` = erster Monatsultimo nach dem Warmup; erste Ausführung am
  Folgetag; **`EVAL_START` = erster PnL-Tag danach** — für alle Varianten und
  Benchmarks derselbe.
- Ein Pflichttest erzwingt: identischer Return-Index über alle 8 Varianten und
  alle Benchmarks (kein variantenabhängiger Start wie in v1, der mom63 andere
  Marktregime gegeben hätte als combo).

## 8. Kostenmodell

- **One-way-Turnover-Definition:** `Turnover = Σ_i |Δw_i|` je Rebalance; ein
  Flip von +1 nach −1 ist Turnover 2 (Pflichttest).
- Pro Trade: Half-Spread + Slippage-Puffer in bp vom gehandelten Nominal,
  Kommission 0: 1.5 bp (SPY, QQQ, IWM, TLT, IEF, GLD), 3.0 bp (EFA, EEM, LQD,
  SLV, DBC, VNQ). **Ehrliche Einordnung:** das sind konservativ gemeinte, aber
  UNKALIBRIERTE Schätzwerte (angesetzt 2026-09-01, ohne Quoted-Spread-Studie);
  genau deshalb läuft die **Kostenleiter 1× / 2× / 5×** als fester Bestandteil
  jedes Laufs, plus **Break-even-bp je Variante** (Kostensatz, der den
  Mehrertrag auf 0 drückt).
- long/short zusätzlich: 50 bp p.a. Borrow auf Short-Nominal (täglich
  anteilig), mit ausgewiesener Sensitivität (25/50/100 bp p.a.); Verfügbarkeit,
  Margin und Forced-Cover sind NICHT modelliert — Teil des
  Forschungsvarianten-Labels aus Abschnitt 5.
- ETF-TERs stecken in den adjustierten Preisen.

## 9. Benchmarks und Kennzahlen

**Benchmarks (identische Kostenlogik, identisches Evaluationsfenster, identisches
Entry-Timing):**

1. **`matched_long` (primärer Vergleich):** Signal konstant +1 auf allen 12
   ETFs, ansonsten IDENTISCHE Pipeline (Inverse-Vol, Vol-Obergrenze, Gross-Cap,
   Monats-Rebalance, Lag, Kosten, Cash-Verzinsung). Misst: was liefert dasselbe
   Portfolio OHNE Timing? Der gepaarte Mehrertrag `Trend − matched_long` ist die
   primäre Messgröße.
2. SPY Buy-and-Hold, 3. 60/40 SPY/TLT (monatlich): Kontext, nicht Gate-relevant.

**Kennzahlen** (alles multiplikativ, Equity-Kurve inkl. Startwert 1.0): Netto-
CAGR, Vol, Sharpe und Excess-Sharpe (über Cash), Max-Drawdown, max. tägliches
Gross, Jahres-Turnover, Kostenanteil, Returns je Kalenderjahr, Instrument-
Attribution (inkl. anteiliger Kosten), Break-even-bp, Kostenleiter-Zeilen.

**Inferenz:** primäre Beobachtungseinheit = MONATLICHE Log-Mehrerträge
(`log(1+r)`-Summen je Kalendermonat, Trend minus matched_long).
**Stationary-Block-Bootstrap** (Politis/Romano) mit erwarteter Blocklänge
**6 Monate** (präregistriert primär; 3 und 12 Monate nur als Sensitivität) —
unabhängiges Resampling einzelner Monate (v1) hätte die Persistenz über
Monatsgrenzen zerstört, die 63–252-Tage-Signale gerade ausmacht. B = 10.000,
p-Werte als `(extreme + 1) / (B + 1)`, DIESELBEN Bootstrap-Ziehungen (gleiche
Blockstruktur-Zufallsfolge) für alle 8 Varianten. Zusätzlich ein Sign-Flip-
Permutationstest auf den monatlichen Mehrerträgen mit 10.000 Permutationen.

## 10. Gates und Holdout (präregistriert)

**Das Entwicklungsfenster ist ausdrücklich SCREENING, keine Bestätigung** — acht
unkorrigierte 95%-Tests haben keine gemeinsame 95%-Fehlerrate. Bestätigen kann
ausschließlich der eine versiegelte Holdout-Test.

Eine Variante besteht das Screening, wenn:

- **Gate A (Signalmehrwert):** die einseitige 95%-Bootstrap-Untergrenze des
  annualisierten geometrischen NETTO-MEHRERTRAGS gegenüber `matched_long` > 0.
- **Gate B (historisches Risiko):** Max-Drawdown <= 15 % unter Basis- UND
  2×-Kosten (Equity inkl. Startwert 1.0).
- **Gate C (ökonomische Relevanz):** Netto-CAGR unter 2×-Kosten >= **+2.0 %
  p.a.** (präregistrierter Mindestwert, nicht bloß > 0).
- **Gate D (Breite):** echte Leave-one-instrument-out-Reruns (12 komplette
  Pipeline-Läufe) und Leave-one-sleeve-out-Reruns (4 Sleeves, Abschnitt 4);
  der gepaarte Mehrertrag (Punktschätzer, jeweils gegen den OHNE dasselbe
  Instrument/Sleeve neu gerechneten matched_long) bleibt in JEDEM Rerun > 0.
  Der additive Brutto-Beitragsabzug aus v1 entfällt (er enthielt keine Kosten
  und war kein Gegenfaktual). „Bestes Jahr entfernen": compoundierter Ertrag
  der übrigen VOLLSTÄNDIGEN Kalenderjahre bleibt > 0 (erstes/letztes
  unvollständiges Jahr ausgeschlossen).

**Versiegeltes One-Shot-Holdout:**

1. Der Screening-Lauf erzeugt bei >= 1 bestandener Variante ein unveränderliches
   `candidate.json`: Familie, GENAU EINE Kandidatin (höchster Netto-Sharpe des
   Mehrertrags unter den Bestandenen; Tie-Break: alphabetisch erster
   Variantenname), Snapshot-Content-Hash, Git-SHA, Config-Hash, `DEV_END`,
   SHA256 des Screening-Ergebnis-JSONs.
2. `run_trend_holdout.py` akzeptiert KEINEN Variantennamen — es liest NUR
   `candidate.json`, verifiziert alle Hashes und verweigert die Ausführung,
   wenn bereits ein Holdout-Ergebnis der Familie existiert (Tombstone-Datei).
3. Der Holdout-Lauf startet in CASH und eröffnet die Position am ersten
   ausführbaren Termin nach `DEV_END` inklusive voller Einstiegskosten; Signale
   dürfen Preise vor `DEV_END` sehen (kausal), gewertet und attribuiert werden
   AUSSCHLIESSLICH Holdout-Tage (alle Zeitreihen werden erst vollständig
   gespeichert, dann geschnitten — kein Dev-Anteil in irgendeinem Holdout-Gate).
4. Holdout-Passregel (final): Gate A, B, C analog auf dem Holdout-Fenster
   (Gate D entfällt dort — zu kurz für Jahres-/Sleeve-Zerlegung, dafür wird die
   Instrument-Attribution berichtet).
5. Besteht die Kandidatin nicht, endet die Experiment-Familie `trend-etf-v1`.
   KEINE Runner-up-Variante, kein zweiter Zugriff.

## 11. Struktur

```
factor_lab/
  build_trend_snapshot.py  # EIGENER Schritt: Fetch, Sanity-Checks, Freeze, Hash, Manifest
  data_snapshot.py         # Laden + Hash-Verifikation gegen committetes Manifest (fetcht nie)
  signals.py               # mom63/126/252, combo — reine Funktionen
  costs.py                 # bp-Modell, Borrow, Kostenleiter
  portfolio.py             # Sizing, Lag-Konvention, Drift, Backtest-Loop, matched_long, Benchmarks
  stats.py                 # Monats-Aggregation, Stationary-Block-Bootstrap, Permutation, Gates, Verdikt
  registration.py          # Config-Hash, candidate.json-Siegel, Tombstone-Logik
  run_trend_baseline.py    # Screening: 8 Varianten + Benchmarks + Leiter + Gates + candidate.json
  run_trend_holdout.py     # One-Shot: liest NUR candidate.json, schreibt Tombstone
  tests/                   # check_*-Konvention, TDD
  logs/, data_snapshots/   # gitignored; Manifest wird force-committet
```

Neue Dependency: `yfinance` (nur Build-Skript). Pflichttests u.a.: Future-Poison/
Fill-Lag, identischer Return-Index aller Varianten+Benchmarks, `[-0.10]` →
−10 % Drawdown, Flip=Turnover 2, Brutto/Kosten/Borrow/Netto-Reconciliation,
Gross-Drift-Reporting, Snapshot-Manipulation → Hashfehler, Holdout nimmt nur
versiegelte Kandidatin und nur einmal, Holdout-Attribution nur aus Holdout-
Tagen, Splits/Rebalances auf Monatsgrenzen.

## 12. Registrierung und Amendment-Regel

- **Familie:** `trend-etf-v1`. **Config-Hash:** SHA256 über den normalisierten
  Registrierungsblock (Universum, Zeitfenster, DEV_END-Regel, 8 Varianten,
  Sizing-, Kosten-, Inferenz- und Gate-Parameter dieser Spec); wird in jede
  Provenance geschrieben.
- Alle 8 Varianten-Ergebnisse werden IMMER veröffentlicht (auch Fehlschläge).
- Kostenleiter- und Leave-one-out-Diagnosen dienen NIE der Auswahl neuer
  Varianten — sie sind Bestandteil der fixierten Gates bzw. reine Reports.
- **Amendment-Regel:** Jede inhaltliche Änderung NACH Sichtung eines
  Screening-Ergebnisses (Parameter, Gates, Varianten, Daten) erzeugt eine neue
  Experiment-Familie (`trend-etf-v2`, …) mit neuem, unberührtem Holdout. Die
  alte Familie wird als beendet dokumentiert.

## 13. Bekannte Grenzen (bewusst akzeptiert)

- Milder Survivorship-Bias durch Auswahl heutiger Überlebender-ETFs
  (Mega-AUM, ~20 Jahre Historie — dokumentiert, vertretbar).
- yfinance ist eine inoffizielle API — nur für den einmaligen Build relevant.
- Adjusted-Close-Returns und `^IRX`-Cash sind Näherungen.
- Close(t+1)-Fills sind eine Konvention, kein Mikrostruktur-Modell; die
  Kostenleiter deckt Unsicherheit teilweise ab, ersetzt aber keine
  Spread-Kalibrierung.
- Ein Datenanbieter, ETFs statt Futures: beantwortet bewusst die engere Frage
  „trägt es auf MEINEN handelbaren Instrumenten".
