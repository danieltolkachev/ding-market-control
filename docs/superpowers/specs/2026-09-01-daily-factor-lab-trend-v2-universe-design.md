# Daily-Factor-Lab, Familie trend-etf-v2: erweitertes Cross-Asset-Universum

**Datum:** 2026-09-01
**Status:** Design zur Freigabe
**Experiment-Familie:** `trend-etf-v2` (neu, unabhaengig von `trend-etf-v1`)
**Vorgeschichte:** `trend-etf-v1` (Spec `2026-09-01-daily-factor-lab-trend-design.md`, v2) wurde
vollstaendig implementiert, gegen echte Daten gelaufen und lieferte einen sauberen
Nullbefund: keine der 8 praeregistrierten Varianten zeigte eine statistisch robuste
Ueberrendite gegenueber `matched_long` (Gate A schlug fuer alle 8 fehl); der Holdout
wurde korrekt nie beruehrt. Dieses Dokument definiert `trend-etf-v2` als **neue,
unabhaengige Familie** mit einem breiteren Instrumenten-Universum — per Amendment-Regel
aus Spec v1 §12 (jede inhaltliche Aenderung nach Sichtung eines Ergebnisses erzeugt eine
neue Familie mit frischem, unberuehrtem Holdout).

## 1. Warum ueberhaupt eine neue Familie

Die Trend-Following-Literatur (Hurst/Ooi/Pedersen, bereits in Spec v1 zitiert) belegt
die Ueberrendite von Trendfolge ueberwiegend durch **viele unkorrelierte Trend-Wetten
gleichzeitig** — typischerweise Dutzende Futures ueber Aktienindizes, Anleihen,
**Waehrungen** und **einzelne Rohstoffe** hinweg. `trend-etf-v1`s 12-ETF-Universum hatte
**keine Waehrungsposition** und nur **eine einzige gebuendelte Rohstoffposition** (DBC)
statt disaggregierter Rohstofftrends. Das ist eine schwaechere Diversifikationsbasis als
das, was die zitierten Studien tatsaechlich testen — ein plausibler, VOR Sichtung
jeglicher neuer Daten identifizierter Grund fuer den schwachen v1-Befund.

**Nicht-Ziel:** dies ist KEIN Versuch, das v1-Nullergebnis nachtraeglich zu "reparieren".
Alle 12 v1-Instrumente bleiben unveraendert erhalten — es werden ausschliesslich neue
Instrumente HINZUGEFUEGT, niemals bestehende aufgrund ihrer beobachteten Performance
entfernt. Ein selektives Entfernen nach Sichtung der Einzelbeitraege waere genau das
nachtraegliche Rosinenpicken, das die gesamte Praeregistrierungs-Methodik verhindern soll.

## 2. Universum (19 Instrumente = 12 v1-Instrumente + 7 neue)

**Unveraendert aus v1 (12):** SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, GLD, SLV, DBC, VNQ.

**Neu (7), gezielt gegen die in Abschnitt 1 benannte Luecke:**

| Symbol | Assetklasse | Warum |
|---|---|---|
| UUP | Waehrung (USD) | Waehrungstrend fehlte komplett in v1 |
| FXE | Waehrung (EUR) | s.o. |
| FXY | Waehrung (JPY) | s.o. |
| USO | Rohstoff (Oel) | disaggregierter Energie-Trend statt nur gebuendeltes DBC |
| UNG | Rohstoff (Erdgas) | disaggregierter Energie-Trend, historisch niedrig korreliert zu Oel |
| DBA | Rohstoff (Agrar) | disaggregierter Agrar-Trend, historisch niedrig korreliert zu Energie/Aktien |
| EMB | Anleihe (EM-USD-Staatsanleihen) | andere Durations-/Kredit-/Waehrungsrisiken als TLT/IEF/LQD |

**Bewusst NICHT Kupfer (CPER):** CPER wurde erwogen, hat aber erst 2011 Handelsstart —
das haette die gemeinsame Historie um ca. 5 Jahre verkuerzt. DBA (Agrar, Handelsstart
Anfang 2007) deckt eine aehnlich unkorrelierte Rohstoff-Nische ab, ohne die Historie zu
verkuerzen — und ist fuer Trendfolge-Diversifikation eher noch geeigneter, da
Agrar-Rohstoffe historisch niedriger mit Industrie-/Aktienzyklen korrelieren als Kupfer.

**Sleeves fuer Gate D (Leave-one-sleeve-out), 7 statt bisher 4:**

| Sleeve | Instrumente |
|---|---|
| us_equity | SPY, QQQ, IWM |
| intl_equity | EFA, EEM |
| bonds | TLT, IEF, LQD |
| em_bonds | EMB |
| real_assets | GLD, SLV, DBC, VNQ |
| currencies | UUP, FXE, FXY |
| granular_commodities | USO, UNG, DBA |

## 3. Architektur: getrenntes Modul-Set, gemeinsame Bausteine

Neue Dateien `factor_lab/registration_v2.py`, `factor_lab/build_trend_snapshot_v2.py`,
`factor_lab/run_trend_baseline_v2.py`, `factor_lab/run_trend_holdout_v2.py` — mit eigener
`FAMILY`/`REGISTRATION`, eigenem Snapshot, eigenen Log-Verzeichnissen. Importieren
UNVERAENDERT aus den bestehenden, bereits gemergten und geprueften Bausteinen:
`signals.py`, `portfolio.py`, `stats.py` (keine Aenderung noetig). **Eine additive
Aenderung an einer geteilten Datei:** `costs.py`s `COST_BP`-Dict bekommt die 7 neuen
Symbole hinzugefuegt (alle bei 3.0bp — dieselbe "weniger liquide als SPY-Tier"-Stufe wie
EFA/EEM/LQD/etc., siehe Abschnitt 4). Das ist rein additiv und kann v1s Verhalten nicht
beeinflussen: v1s Code fragt `COST_BP` ausschliesslich fuer sein eigenes 12-Symbol-
Universum ab, die 7 neuen Keys werden dort nie gelesen. **Warum getrennte statt
parametrisierter Runner:** der bereits gemergte, hochgradig geprüfte v1-Code (inkl. des
irreversiblen Holdout-Runners) bleibt dadurch komplett unangetastet — null Risiko einer
Regression durch die Erweiterung.

Die Tombstone-Isolation zwischen Familien ist bereits im bestehenden Code
familennamen-basiert (`holdout_tombstone_{FAMILY}.json`, `registration.py`,
unveraendert) — `trend-etf-v2` bekommt automatisch einen eigenen Tombstone, unabhaengig
von `trend-etf-v1`s (bereits nicht existentem, da nie ausgeloestem) Tombstone.

**Verbesserung gegenueber v1, offen gelegt:** `REGISTRATION_V2` enthaelt zusaetzlich das
Feld `"sleeves"` (die Tabelle aus Abschnitt 2) — in v1 war `SLEEVES` eine reine
Modul-Konstante in `run_trend_baseline.py`, NICHT Teil des gehashten `REGISTRATION`-Dicts.
Das war eine kleine Luecke in v1s Siegel-Vollstaendigkeit (welche Sleeve-Einteilung fuer
Gate D verwendet wurde, war nicht Teil des manipulationssicheren Hashes). v2 schliesst
das: `config_hash()` deckt jetzt auch die Sleeve-Definition ab.

## 4. Kostenmodell fuer die neuen Symbole

Alle 7 neuen Symbole: 3.0bp (Half-Spread + Slippage-Puffer), identische Kommission 0,
identische unkalibrierte-Schaetzung-Einordnung wie in Spec v1 §8 — die Kostenleiter
1x/2x/5x deckt die Unsicherheit weiterhin ab. Keine der neuen Positionen wird jemals
geshortet abweichend von den bestehenden Regeln (long_short/long_flat gelten unveraendert
pro Signalvariante ueber das gesamte 19-Symbol-Universum, nicht pro Instrument).

## 5. Zeitfenster und Datenverfuegbarkeit

`SNAPSHOT_START="2007-01-01"`, `SNAPSHOT_END_EXCLUSIVE="2026-09-01"` — identisch zu v1,
fuer eine methodisch saubere Vergleichsbasis (dieselbe Endzeit, kein nachtraeglich
verlaengertes Fenster). Die neuen Instrumente haben teils spaetere Handelsstarts als
2007-01-01 (die genauen Daten werden empirisch beim Snapshot-Build ermittelt, nicht
vorab geschaetzt) — der gemeinsame Handelskalender (Schnittmenge aller 19 Symbole) wird
dadurch mechanisch kuerzer als v1s. Das ist eine unvermeidbare Konsequenz der
Universums-Erweiterung, VOR Sichtung jeglicher Daten festgelegt, keine
nachtraegliche Anpassung. **Konkrete Folge:** `min_common_days` in `REGISTRATION_V2`
wird von v1s 4800 auf **4200** (~16.7 Jahre) gesenkt — rein mechanisch begruendet durch
die spaeteren Handelsstarts, nicht durch ein gewuenschtes Ergebnis. `build_trend_snapshot_v2.py`s
bestehender fail-closed Sanity-Check bricht weiterhin hart ab, falls die tatsaechliche
gemeinsame Historie selbst darunter liegen sollte.

## 6. Alles andere identisch zu trend-etf-v1

Signale (mom63/126/252 + combo, je long_short/long_flat, 8 Varianten), Execution-
Konvention (Close(t)-Entscheidung, Fill Close(t+1)), Vol-Cap 0.10, Gross-Cap 1.0,
monatliches Rebalancing, `matched_long`-Benchmark, Stationary-Block-Bootstrap
(6-Monats-Bloecke primaer), die vier Gates (A: Bootstrap-Untergrenze > 0; B: DD-Cap 15%;
C: Stress-CAGR-Floor 2%; D: kein Einzeljahr-/Einzelsleeve-Treiber), das versiegelte
One-Shot-Holdout (candidate.json + Tombstone), die Kostenleiter 1x/2x/5x +
Breakeven-bp + Borrow-Sensitivitaet 25/50/100bp fuer long_short (siehe die
Holdout-Nachbesserung vom 2026-09-01) — unveraendert aus Spec v1 uebernommen.

## 7. Bekannte Grenzen (zusaetzlich zu Spec v1 §13)

- USO/UNG sind bekannt fuer Roll-/Contango-Kosten, die ueber das reine Bid-Ask-Spread-
  Modell hinausgehen koennen — die Kostenleiter (bis 5x) deckt einen Teil dieser
  Unsicherheit ab, ersetzt aber keine spezifische Futures-Roll-Kostenkalibrierung.
- 19 statt 12 Instrumente bedeuten mehr Leave-one-instrument-out-Reruns in Gate D
  (19 statt 12) plus 7 statt 4 Sleeve-Reruns — laengere Laufzeit fuer A-C-Passer, keine
  methodische Aenderung.
- Waehrungs-ETFs (UUP/FXE/FXY) bilden Termingeschaefte auf Waehrungspaare nach, nicht
  Spot-Positionen direkt — kleine Tracking-Differenz zu einer echten FX-Forward-Position,
  als Naeherung akzeptiert (analog zur bestehenden ETF-statt-Futures-Einschraenkung aus
  Spec v1 §13).

## 8. Bekannte Einschraenkung: v1s Holdout-Glob ist nicht v2-bewusst

`run_trend_holdout.py` (trend-etf-v1, nicht veraenderbar) sucht Kandidaten ueber das
blosse Muster `trend_screening_*/candidate.json`. Da `trend-etf-v2`s Screening-
Ausgabeverzeichnis `trend_screening_v2_<runid>` heisst und `'v'` lexikografisch nach
Ziffern sortiert, wuerde dieses Glob HEUTE tatsaechlich das v2-Verzeichnis treffen, wenn
v1s Holdout jemals NACH einem v2-Screening-Lauf ausgefuehrt wuerde (v1s eigenes Screening
hat nie eine Kandidatin versiegelt, daher ist das bisher nie eingetreten). Das faellt
nicht offen aus: `read_and_verify_candidate()` prueft `family == "trend-etf-v1"` und
wirft bei einem v2-candidate.json einen klaren `ValueError` (Familien-Mismatch) --
keine stille Korruption, kein falsches Siegel. Ein zukuenftiger Operator sollte diesen
Fehler aber nicht mit Manipulation verwechseln. **Workaround, falls dieser Fall jemals
relevant wird:** das `trend_screening_v2_*`-Verzeichnis vor dem Ausfuehren von v1s
Holdout temporaer beiseite verschieben (z.B. umbenennen oder in ein anderes
Verzeichnis verschieben) und danach zurueckverschieben. `run_trend_holdout.py` selbst
bleibt davon unangetastet (nicht veraenderbar).
