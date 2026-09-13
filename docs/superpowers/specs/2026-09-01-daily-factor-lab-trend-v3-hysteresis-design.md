# Daily-Factor-Lab, Familie trend-etf-v3: Signal-Persistenz gegen Whipsaw

## Verbindliche Revision vom 2026-09-07

Diese Revision ersetzt die widersprechenden Aussagen des historischen Entwurfs
unten. Der Entwurf bleibt als Forschungsprotokoll erhalten, nicht als unveraendert
auszufuehrende Spezifikation.

### Aussagekraft und Vergleich

Die gesamte Historie bis 2026-08-31 gilt als bereits verwendet: 2020-2022 waren
Teil des v2-Screenings, Dezember 2022 bis August 2026 wurde als Holdout ausgewertet
und anschliessend diagnostiziert. Weder die neue Familie noch das 60%-Quantil
erzeugen frische Daten. Alle historischen v3-Ergebnisse sind explorativ und
duerfen weder als Blindtest noch als unabhaengige Bestaetigung bezeichnet werden.
Das 60%-Quantil bleibt ausschliesslich ein deskriptiver Periodenschnitt.

Whipsaw ist eine Hypothese. Absolute negative Instrumentenbeitraege belegen
keinen relativen Schaden gegen matched_long. Die bisherige Monatszaehlung und
Ausreisserbeschreibung sind vor einer erneuten Verwendung aus den gespeicherten
Zeitreihen nachzurechnen; sie sind keine Voraussetzung des Experiments.

Alle acht bisherigen Varianten werden jeweils unveraendert und mit Persistenz
auf exakt denselben Daten, Entscheidungsdaten, Kosten und Startbedingungen
verglichen. matched_long bleibt eine weitere Referenz. Kein nachtraeglicher
Sieger wird zum bestaetigten Kandidaten erklaert. Auszugeben sind CAGR, Vola,
Drawdown, Turnover, Kosten und gepaarte monatliche Log-Renditedifferenzen
Persistenz minus Rohregel sowie Strategie minus matched_long. Bootstrapintervalle
sind deskriptiv, nicht familienuebergreifend korrigierte Signifikanznachweise.

### Monatlicher Zustandsautomat

apply_signal_persistence erhaelt ausschliesslich vollstaendige Monatsultimo-
Entscheidungen. Zwei aufeinanderfolgende Monate bestaetigen einen Wechsel;
zwei taegliche Beobachtungen tun dies nicht. combo entsteht vorher aus rohen
Einzelsignalen. Der erste gueltige Monatswert wird unmittelbar uebernommen;
Magnitude innerhalb derselben Richtung wird sofort aktualisiert. Null ist eine
eigene Richtung. NaN setzt den Zustand zurueck. Fehlende Kalendermonate setzen
eine laufende Bestaetigungsserie zurueck. Unsortierte oder doppelte Monatsindizes
sowie unendliche Werte werden abgelehnt. Die Monatswerte werden ausschliesslich
vorwaerts auf den Tagesindex uebertragen; keine Rueckwaertsfuellung. Ausfuehrung
bleibt am folgenden Handelsschluss. Ein unvollstaendiger Schlussmonat ist kein
Entscheidungsmonat. Vor dem Periodenschnitt beobachtete Signalzustaende duerfen
kausal weitergefuehrt werden; beide Portfolios starten je Vergleichsperiode cash.

### Daten und Grenzen

Der vorhandene verifizierte v2-Snapshot wird fuer den gepaarten Vergleich
lesend wiederverwendet. Kein erneuter Yahoo-Abruf: veraenderte Daten wuerden den
Methodenvergleich unnoetig erschweren. Vor Ausfuehrung werden Snapshot, Manifest,
Konfiguration und spaeter Ergebnisse ausserhalb eines Wegwerf-Worktrees gesichert
und ihre SHA256-Werte verifiziert. Alte Siegel und Tombstones bleiben erhalten.

Die historischen Zielgewichte bleiben auf Gross 1.0 begrenzt, identisch zur
Referenz. 15% Drawdown ist eine Bewertungsgrenze, keine Verlustgarantie; 1.25x
ist die beschlossene Projektobergrenze, kein einzustellender Experimentalhebel.
Es entsteht kein neuer one-shot-Holdout-Runner fuer bereits verwendete Daten.
Eine spaetere Vorwaertsvalidierung benoetigt einen vorab festgelegten Start,
eingefrorene Regeln und neue Beobachtungen.

Implementierung: Der alte Plan vom 2026-09-01 ist zurueckgezogen; seine
Codebloecke duerfen insbesondere wegen der Tages-/Monatsverwechslung nicht
unveraendert uebernommen werden.

## Historischer Entwurf vom 2026-09-01 (durch Revision eingeschraenkt)

**Datum:** 2026-09-01
**Status:** Design zur Freigabe
**Experiment-Familie:** `trend-etf-v3` (neu, unabhaengig von `trend-etf-v1`/`trend-etf-v2`)
**Vorgeschichte:** `trend-etf-v2` (19-Instrumente-Universum, Spec
`2026-09-01-daily-factor-lab-trend-v2-universe-design.md`) bestand das Screening
(`combo_long_flat` versiegelt, alle vier Gates bestanden) aber **scheiterte am
einmaligen Holdout** (Dez 2022 - Aug 2026): Gate A schlug fehl, die 95%-Bootstrap-CI
des Mehrertrags gegenueber `matched_long` lag komplett im Negativen
(`ann_geom_lower_1s95=-5.71%`, `p(<=0)=0.9865`). Per Amendment-Regel (Spec v1 §12) ist
die Familie `trend-etf-v2` damit **beendet** — keine Runner-up-Variante, kein Nachjustieren
am bestehenden Holdout. Dieses Dokument definiert `trend-etf-v3` als **neue, unabhaengige
Familie** mit einer literaturgestuetzten Signal-Aenderung und einem eigenen, teilweise
frischen Holdout-Fenster.

## 1. Diagnose: warum v2 scheiterte (Motivation, nicht Rechtfertigung)

Eine Post-hoc-Analyse des gescheiterten v2-Holdouts (zulaessig: Ergebnis verstehen ist
keine Praeregistrierungs-Verletzung, solange daraus keine Anpassung AN DENSELBEN
Holdout folgt) zeigt:

- **Kein einzelner Crash, sondern chronisches Verbluten:** der monatliche Mehrertrag war
  in 30 von 45 Holdout-Monaten negativ, ueberwiegend klein, mit einem Ausreisser
  (November 2023: -4.3%, mehr als dreimal so gross wie jeder andere Monat).
- **Verlustkonzentration in Anleihen (TLT, IEF, LQD), Small Caps (IWM) und Energie
  (USO, UNG)** — Instrumentenklassen mit haeufigen, aber kurzlebigen Trendwechseln im
  Holdout-Zeitraum (mehrfache Zinserwartungs-Umschwuenge, Small-Cap-Underperformance,
  Energie-Boom-Bust-Zyklen).
- **Interpretation:** das monatlich neu entschiedene, binaere long/flat-Signal geht bei
  jedem kurzen Ruecksetzer defensiv und verpasst dann die anschliessende schnelle
  Erholung — der klassische Schwachpunkt von Trendfolge in einem Markt mit haeufigen
  kurzen Schrecken statt eines anhaltenden Abwaertstrends.

**Nicht-Ziel:** dies ist KEIN Versuch, November 2023 oder die Bond/Small-Cap/Energie-
Verluste spezifisch "wegzuoptimieren". Der hier eingefuehrte Mechanismus (Abschnitt 3)
ist ein allgemeiner, in der Trendfolge-Literatur etablierter Kniff gegen Whipsaw
(Signal-Traegheit/Hysterese), der VOR erneuter Sichtung irgendeines Holdout-Ergebnisses
fest verdrahtet wird — keine nachtraeglich an die beobachteten Verlierer angepasste Regel.

## 2. Universum: identisch zu trend-etf-v2 (bewusst unveraendert)

Alle 19 Instrumente aus v2 (SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, GLD, SLV, DBC, VNQ,
UUP, FXE, FXY, USO, UNG, DBA, EMB), dieselben 7 Sleeves, dieselbe `min_common_days`-
Logik. **Warum unveraendert:** v3 testet ausschliesslich, ob der Signal-Mechanismus
(Abschnitt 3) das v2-Problem behebt. Wuerde gleichzeitig auch das Universum geaendert,
liesse sich ein etwaiger Erfolg oder Misserfolg nicht mehr eindeutig zuordnen.

## 3. Mechanismus: Signal-Persistenz (Hysterese)

Neue, reine Funktion `apply_signal_persistence(signal_frame: pd.DataFrame, min_confirm:
int = 2) -> pd.DataFrame`, angewendet auf JEDES der vier Basissignale (mom63, mom126,
mom252, combo) VOR der bestehenden, unveraenderten `portfolio.py`-Maschinerie
(`rebalance_weights`/`trend_weight_provider`). **Reihenfolge, um Mehrdeutigkeit
auszuschliessen:** `combo` wird zuerst wie bisher unveraendert aus den drei RAW-Signalen
(mom63/126/252, jeweils vor Persistenz) ueber das bestehende, unveraenderte
`combo_signal()` gebildet; ERST DANACH wird `apply_signal_persistence` einheitlich auf
alle vier resultierenden Serien (mom63, mom126, mom252, combo) je einzeln angewendet.
Persistenz wird also nicht in die Kombination hineingerechnet, sondern als einheitlicher
letzter Schritt auf alle vier fertigen Signale — vermeidet, dass "combo aus bereits
persistenten Einzelsignalen" und "persistentes combo aus rohen Einzelsignalen" zwei
unterschiedliche, verwechselbare Konstruktionen waeren.

Pro Instrumentenspalte ein Zustandsautomat, unabhaengig ueber alle anderen Spalten:

- Zustand: `effective` (zuletzt bestaetigter Signalwert, initial `NaN`), `pending_sign`,
  `pending_streak` (beide initial 0/`NaN`).
- Bei `NaN` im Rohsignal (Warmup): Zustand wird zurueckgesetzt, Ausgabe `NaN` — identisches
  Warmup-Verhalten wie die unveraenderten Rohsignale.
- Bei gueltigem Rohwert `raw[t]`, `sign_t = sign(raw[t])`:
  - Erster gueltiger Wert ueberhaupt: `effective = raw[t]`, sofort uebernommen.
  - `sign_t == sign(effective)`: **sofortige Uebernahme** (`effective = raw[t]`,
    `pending_streak` zurueckgesetzt) — Hysterese bremst NUR Vorzeichenwechsel, nicht
    Magnitudenaenderungen innerhalb derselben Richtung.
  - `sign_t != sign(effective)` (Kandidat fuer Richtungswechsel):
    - `pending_sign == sign_t`: `pending_streak += 1`.
    - sonst: `pending_sign = sign_t`, `pending_streak = 1`.
    - Ist `pending_streak >= min_confirm`: **jetzt erst** `effective = raw[t]` (der
      tatsaechliche Rohwert, nicht nur das Vorzeichen), `pending_streak` zurueckgesetzt.
    - Sonst bleibt `effective` unveraendert (alte Position wird gehalten) — ein
      einzelner Ausreisser-Monat in eine neue Richtung wird ignoriert, wenn ihm nicht
      im Folgemonat derselbe neue Richtungswert folgt.
- `min_confirm=2` ist der praeregistrierte Standardwert (ein Richtungswechsel muss sich
  einmal wiederholen, bevor er wirkt) — kein Suchraum ueber mehrere Werte, kein
  Abstimmen nach Sichtung von Ergebnissen.

Beispiel (eine Instrumentenspalte, Rohsignal ueber die Zeit: -1,-1,-1,+1,-1,+1,+1,+1):
effektives Signal: -1,-1,-1,-1 (Wechsel zu +1 noch nicht bestaetigt),-1 (sofort, da
gleiches Vorzeichen wie aktuelles `effective`, Pending zurueckgesetzt),-1 (erster +1 seit
Reset, Pending=1),+1 (zweiter +1 in Folge, bestaetigt),+1.

## 4. Dev/Holdout-Split: 60%-Quantil statt 80% (prinzipienbasiert, kein Datum von Hand)

v1/v2 nutzten `dev_end_rule`: "letzter Monatsultimo <= 80%-Quantil-Datum des
gemeinsamen Kalenders". v3 aendert **nur den Quantil-Parameter** auf **60%** — weiterhin
dieselbe Formel, kein hart codiertes Datum. **Warum die Aenderung:** der Grossteil des
bisherigen v2-Holdout-Fensters (Dez 2022 - Aug 2026) wurde in Abschnitt 1 bereits im
Detail durchleuchtet; ein v3-Holdout, das sich damit vollstaendig ueberschneidet, waere
kein blinder Test mehr. Das 60%-Quantil verschiebt das Dev-Ende auf voraussichtlich
Ende 2019 (exakter Wert erst nach dem echten Snapshot-Build bekannt, wie schon bei v1/v2)
und verlaengert den Holdout entsprechend auf ca. 6-7 Jahre (statt v2s 45 Monate) — davon
sind die Jahre 2020-2022 (COVID-Crash, Erholung, 2022er Baermarkt) fuer diese Analyse
bisher ungesehen, nur der letzte Teil (2023-2026) ist der bereits inspizierte Abschnitt.

**Bekannte Einschraenkung, offen gelegt statt verschwiegen:** ein "Bestehen" von v3s
Holdout ist dadurch **nicht vollstaendig blind** — der letzte Teil des Fensters wurde
schon einmal (im Rahmen der v2-Diagnose) inhaltlich betrachtet. Ein Bestehen sollte als
"die Diagnose-getriebene Reparatur haelt auch auf einem laengeren, ueberwiegend aber
nicht vollstaendig neuen Fenster" gelesen werden, nicht als vollstaendig unabhaengige
Bestaetigung. Echte, unbelastete Bestaetigung kaeme erst aus zukuenftigem Live-/
Paper-Trading nach diesem Datum.

## 5. Architektur: getrenntes Modul-Set, gemeinsame Bausteine (wie v1 -> v2)

Neue Dateien `factor_lab/registration_v3.py`, `factor_lab/build_trend_snapshot_v3.py`,
`factor_lab/run_trend_baseline_v3.py`, `factor_lab/run_trend_holdout_v3.py` sowie die
neue, eigenstaendige `factor_lab/signal_persistence.py` (Abschnitt 3). Importieren
UNVERAENDERT aus `signals.py`, `portfolio.py`, `stats.py`, `costs.py` (keine der 19
Instrumenten-Kosten aendert sich). **Warum getrennte statt parametrisierter Runner:**
identische Begruendung wie beim Uebergang v1 -> v2 — der bereits gemergte, gepruefte
v2-Code (inkl. seines bereits verbrauchten, tombstone-versiegelten Holdouts) bleibt
komplett unangetastet. `registration_v3.py` dupliziert die Sealing/Tombstone-Funktionen
erneut (wie `registration_v2.py` es gegenueber `registration.py` tat) statt sie zu
parametrisieren.

## 6. Alles andere identisch zu trend-etf-v2

`SNAPSHOT_START="2007-01-01"`, `SNAPSHOT_END_EXCLUSIVE="2026-09-01"`, `min_common_days
=4200`, 19-Instrumente-Kostenmodell, Signale (mom63/126/252 + combo, je long_short/
long_flat, 8 Varianten — jetzt jeweils MIT vorgeschalteter Signal-Persistenz), Vol-Cap
0.10, Gross-Cap 1.0, monatliches Rebalancing, `matched_long`-Benchmark, Stationary-
Block-Bootstrap (6-Monats-Bloecke primaer), die vier Gates (A-D unveraendert), das
versiegelte One-Shot-Holdout (candidate.json + eigener Tombstone), die Kostenleiter
1x/2x/5x + Breakeven-bp + Borrow-Sensitivitaet 25/50/100bp fuer long_short — alles
uebernommen aus Spec v2.

## 7. Bekannte Grenzen

- Die in Abschnitt 4 offen gelegte partielle Ueberschneidung mit dem bereits
  inspizierten v2-Holdout-Fenster ist die wichtigste Einschraenkung dieser Familie —
  siehe dort.
- Signal-Persistenz erhoeht die effektive Reaktionszeit des Signals um mindestens einen
  Monat bei jedem echten Trendwechsel — das ist der bewusste Kompromiss (weniger
  Whipsaw-Kosten gegen spaeteres Einsteigen bei echten neuen Trends), keine kostenlose
  Verbesserung.
- `min_confirm=2` ist ein einzelner, vorregistrierter Wert, kein optimierter — sollte er
  sich als falsch erweisen, ist das ein Nullbefund fuer DIESE spezifische Hysterese-
  Staerke, keine Aussage ueber Hysterese-Mechanismen generell.
