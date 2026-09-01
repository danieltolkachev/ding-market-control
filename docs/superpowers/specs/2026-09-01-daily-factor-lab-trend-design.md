# Daily-Factor-Lab, Bein 1: Time-Series-Trend auf Cross-Asset-ETFs

**Datum:** 2026-09-01
**Status:** Design freigegeben (Chat), Implementierung noch nicht begonnen
**Vorgeschichte:** Die 1-Minuten-LSTM-Linie wurde nach dem 188-Fold-Rank-IC-Nullergebnis
(PR #2, `logs/signal_diagnostics_20260831_210031/`) beendet. Dieses Lab ist ein NEUES
Forschungsprojekt mit Tages-/Wochenhorizont, einfachen Regeln und ohne LSTM. Ein
neuronales Netz kommt erst wieder ins Spiel, wenn es bereits funktionierende Signale
kombinieren soll — nicht vorher.

## 1. Forschungsfrage

Liefert einfacher Time-Series-Trend auf einer festen Liste liquider Cross-Asset-ETFs
NACH Kosten einen belastbaren Baustein Richtung des Renditeziels (>= 12.7 % Netto-CAGR
ueber Jahre, siehe Memory `project_target_return`), bei einem harten Drawdown-Cap von
15 % auf der Netto-Equity-Kurve und ohne Leverage?

Erwartung ehrlich formuliert: Trend allein liefert die 12.7 % nicht — es ist EIN
Baustein (Beta + kleines aktives Alpha − Kosten). Das Gate prueft deshalb "traegt
netto positiv und robust bei", nicht "erreicht allein 12.7 %".

## 2. Nicht-Ziele

- KEIN Cross-Sectional-Momentum auf Einzelaktien (Bein 2, wartet auf die
  Datenquellen-/Budget-Entscheidung: survivorship-bias-freier Datensatz noetig).
- KEIN Training/ML, keine Parameteroptimierung ueber die praeregistrierten
  Varianten hinaus.
- KEINE Live-/Paper-Execution. Erst Signalnachweis, dann (separates Projekt)
  Execution mit echten Brokerorders, Ledger, Reconciliation.
- KEIN Leverage (Gross-Cap 1.0), keine Intraday-Daten.

## 3. Entscheidungen (vom User fixiert, 2026-09-01)

| Entscheidung | Wert |
|---|---|
| Startscope | Trend-Leg zuerst (Momentum-Leg spaeter, separat) |
| Drawdown-Cap (hartes Gate) | 15 % auf der Netto-Equity-Kurve |
| Ort | gleiches Repo, neues Top-Level-Package `factor_lab/` |
| Ansatz | A — minimales praeregistriertes Baseline-Lab |

## 4. Daten

- **Universum (12 ETFs, alle Auflage <= 2006):** SPY, QQQ, IWM (US-Aktien),
  EFA, EEM (Intl/EM-Aktien), TLT, IEF, LQD (Anleihen), GLD, SLV, DBC
  (Gold/Silber/Rohstoffe), VNQ (REITs).
- **Quelle:** yfinance, Tagesdaten ab 2007-01-01 (~19.7 Jahre), auto_adjust=True
  (Splits + Dividenden in den Preisen; Returns aus Adjusted Close sind damit
  Total-Return-Naeherungen; ETF-TER steckt ebenfalls im Preis).
- **Frozen Snapshot:** einmalig fetchen, auf Platte einfrieren, Content-SHA256
  berechnen, menschenlesbares Manifest (Zeilen + Zeitraum je Symbol) COMMITTEN —
  exakt das Muster aus `market_control_system/data_layer/frozen_snapshot.py`
  (dortige generische Funktionen werden importiert oder minimal adaptiert, da der
  Fetch-Pfad ein anderer ist). Jeder Ergebnis-Output traegt den Content-Hash in
  der Provenance.
- **Split:** Entwicklungsfenster = erste 80 % der gemeinsamen Handelstage;
  **Holdout = letzte 20 %**, wird vom Baseline-Skript NIE gelesen — nur von
  einem separaten, manuell auszufuehrenden `run_trend_holdout.py` (Muster aus
  `run_cross_sectional_holdout_eval.py`).
- Gemeinsamer Kalender: Schnittmenge der Handelstage aller 12 ETFs.

## 5. Signal (praeregistriert, KEINE weiteren Varianten)

Je Instrument i und Tag t:

- `mom_L(i, t) = sign( AdjClose(i, t) / AdjClose(i, t - L) - 1 )` fuer
  L ∈ {63, 126, 252} Handelstage.
- Vierte Signalvariante: `combo` = Gleichgewichts-Mittel der drei Vorzeichen
  (Wert in {-1, -1/3, +1/3, +1}).

Zwei Portfolio-Modi je Signalvariante:

- **long/short:** Position folgt dem Vorzeichen (klassisches TSMOM).
- **long/flat:** negatives Signal → Position 0 (realistischer fuer ein kleines
  Cash-Konto; keine Borrow-Kosten, kein Margin-Bedarf).

Ergibt **8 praeregistrierte Varianten** (4 Signale × 2 Modi). Alle 8 werden
gerechnet und geloggt. Es gibt keinen Mechanismus, weitere hinzuzufuegen, ohne
diese Spec zu aendern.

## 6. Portfolio-Konstruktion

- **Inverse-Vol-Sizing:** Rohgewicht `w_i ∝ signal_i / sigma_i` mit `sigma_i` =
  annualisierte EWMA-Volatilitaet der Tagesreturns (Span 63 Tage, kausal).
- **Vol-Targeting:** Skalierung des Gesamtportfolios auf 10 % p.a. Ziel-Vol,
  geschaetzt aus der kausalen EWMA-Kovarianz-Naeherung (vereinfachend: Skalierung
  ueber die realisierte Portfolio-Vol der letzten 63 Tage).
- **Gross-Cap 1.0:** wenn Σ|w_i| > 1 nach Skalierung, wird proportional auf
  Gross 1.0 herunterskaliert. Es wird NIE hochgehebelt — Vol-Targeting kann
  Exposure nur senken, nie ueber Gross 1 heben.
- **Rebalancing:** monatlich am letzten gemeinsamen Handelstag; zwischen
  Rebalances driften die Positionen mit den Preisen (kein taegliches Nachziehen).
- **Cash:** unverzinst (konservative Vereinfachung, im Report ausgewiesen).

## 7. Kostenmodell

Pro Trade, als Basispunkte vom gehandelten Nominal (Half-Spread + Slippage-Puffer,
Kommission 0):

- 1.5 bp: SPY, QQQ, IWM, TLT, IEF, GLD
- 3.0 bp: EFA, EEM, LQD, SLV, DBC, VNQ

Im long/short-Modus zusaetzlich pauschal **50 bp p.a. Borrow** auf das
Short-Nominal (taeglich anteilig). **Stresstest:** kompletter Lauf zusaetzlich mit
allen Kosten × 2 (Gate c unten prueft dagegen).

## 8. Evaluation & Inferenz

Alle Equity-Kennzahlen MULTIPLIKATIV (prod(1+r)−1, echte Equity-Kurve — Lehre aus
Review-Korrektur 3 der alten Linie):

- Netto-CAGR, annualisierte Vol, Sharpe (rf=0), Max-Drawdown, mittlerer
  Jahres-Turnover, Kostenanteil am Bruttoertrag, Returns je Kalenderjahr,
  Ertragsbeitrag je Instrument.
- **Inferenz:** Monats-Block-Bootstrap (~190 Monatsbloecke im Entwicklungsfenster)
  fuer das 95%-CI des mittleren Netto-Tagesreturns, plus Monats-Sign-Flip-
  Permutationstest — beides via Import der getesteten Funktionen
  `day_block_bootstrap` / `day_sign_flip_pvalue` aus
  `market_control_system/controller/cross_sectional_signal_metrics.py`
  (Bloecke = Kalendermonate statt Tage; die Funktionen gruppieren nach
  `index.date` — `factor_lab/stats.py` normalisiert dafuer den Timestamp-Index
  vor dem Aufruf auf den jeweiligen Monatsanfang, sodass jeder Kalendermonat
  ein Block ist; die Bestandsfunktionen bleiben unveraendert).
- **Benchmarks** im selben Fenster, mit derselben Kostenlogik: SPY Buy-and-Hold
  und 60/40 SPY/TLT (monatlich rebalanced).
- Output je Lauf: JSON (Summary + Provenance inkl. Snapshot-Hash, Kostensatz,
  Varianten) + CSV der Tagesreturns je Variante, in `factor_lab/logs/<run_id>/`.

## 9. Gates (praeregistriert, als Code im Verdikt)

Eine Variante BESTEHT das Entwicklungsfenster, wenn:

- (a) das 95%-Monats-Block-Bootstrap-CI des mittleren Netto-Tagesreturns
  vollstaendig > 0 liegt,
- (b) Max-Drawdown <= 15 %,
- (c) der Netto-CAGR-Punktschaetzer auch bei Kosten × 2 positiv bleibt,
- (d) kein Einzeljahr-Artefakt: der compoundierte Ertrag der uebrigen
  Kalenderjahre bleibt positiv, wenn das beste Jahr entfernt wird; kein
  Einzelinstrument-Artefakt: die Summe der additiven Instrument-P&L-Beitraege
  bleibt positiv, wenn der Beitrag des besten Instruments abgezogen wird
  (Beitragsabzug als definierte Naeherung — KEINE Neuberechnung des Portfolios
  ohne das Instrument, das waere ein anderes Portfolio).

**Holdout-Regel (vorab fixiert):** Bestehen eine oder mehrere Varianten alle vier
Gates, wird GENAU EINE auf dem Holdout evaluiert — die mit dem hoechsten
Netto-Sharpe unter den Bestehenden. Einmalig, manuell, via `run_trend_holdout.py`.
Besteht keine Variante, wird das Holdout NICHT angefasst und das Ergebnis als
Nullbefund dokumentiert.

## 10. Struktur

```
factor_lab/
  data_snapshot.py      # yfinance-Fetch, Freeze, Content-Hash, Manifest
  signals.py            # mom_63/126/252, combo — reine Funktionen
  portfolio.py          # inverse-Vol, Vol-Targeting, Gross-Cap, Monats-Rebalance
  costs.py              # bp-Kostenmodell + Borrow, x2-Stress
  stats.py              # Re-Use der Metriken/Inferenz aus market_control_system
  run_trend_baseline.py # Entwicklungsfenster: 8 Varianten + Benchmarks + Gates
  run_trend_holdout.py  # manuell, genau eine Variante, einmalig
  tests/                # check_*-Konvention wie im Bestand, TDD
  logs/                 # gitignored; Manifeste/Summaries wie gehabt behandeln
```

Neue Dependency: `yfinance` (nur fuer den einmaligen Snapshot-Build noetig;
alle Auswertungen laufen gegen den eingefrorenen Snapshot).

## 11. Bekannte Grenzen (bewusst akzeptiert)

- Auswahl heutiger Ueberlebender-ETFs ist milder Survivorship-Bias; bei
  Mega-AUM-Instrumenten mit ~20 Jahren Historie vertretbar und dokumentiert.
- yfinance ist eine inoffizielle API — durch den Frozen Snapshot nur fuer den
  einmaligen Build relevant.
- Adjusted-Close-Returns sind Total-Return-Naeherungen (Reinvestitions-Timing
  von Dividenden vereinfacht).
- Cash unverzinst → Ergebnisse sind konservativ verzerrt (real gaebe es
  Geldmarktzins auf Cash, besonders im long/flat-Modus).
- Ein Datenanbieter, eine Assetklasse ETFs: kein Ersatz fuer die
  Futures-Universen der Literatur — beantwortet bewusst die engere Frage
  "traegt es auf MEINEN handelbaren Instrumenten".
