# Begrenzter 2×2-Vergleich: Features × Modell

**Datum:** 2026-09-11
**Status:** Spezifikation zur Freigabe — **noch keine Implementierung**
**Grundlage:** `docs/superpowers/ml4t-repository-findings-2026-09-11.md`, Punkt 1
**Auftrag:** Schritt 4 des Übergabe-Prompts vom 2026-09-11

## 0. Was getestet wird — und was nicht

**Getestet wird** genau eine Frage in zwei getrennten Achsen: Trägt **erweiterte Querschnitts-/Marktzustandsinformation** (Feature-Achse) oder **ein nichtlineares Baummodell** (Modell-Achse) etwas bei — und zwar so, dass sich beide Beiträge nicht vermischen.

**Nicht getestet wird:** Allokationsregel, Handelsschwellen, Prognosekalibrierung, Halte-Kadenz. Die bleiben auf den bestehenden, eingefrorenen Werten. Grundsatz aus dem Findings-Bericht: *Modell, Auswahlregel und Ausführung nicht gleichzeitig ändern.*

**Erwartungshaltung, vorab notiert:** Alle bisherigen Linien sind null. Die ehrliche Prior ist, dass auch dieser Vergleich null ausgeht. Die Spec legt unten fest, wie ein Nullbefund aussieht und dass er als solcher berichtet wird.

## 1. Abhängigkeit vom Falsifikations-Audit

Der Audit (`2026-09-11-falsification-audit-design.md`) ist **nicht** implementiert. Solange die Fehlalarmrate der Auswahlpipeline unbekannt ist, gilt:

> Ein positives Ergebnis dieses 2×2-Vergleichs ist **nicht interpretierbar** — es erbt die unkalibrierte Fehlalarmrate. Ein negatives Ergebnis ist dagegen weiterhin aussagekräftig (ein Verfahren, das auf echten Daten nichts findet, findet auch mit unbekannter Fehlalarmrate nichts).

Empfohlene Reihenfolge bleibt daher: Audit zuerst. Wird der 2×2 vorgezogen, ist diese Einschränkung im Ergebnisbericht zu nennen.

## 2. Das Raster

|  | **Ridge** (bestehend) | **GBM** (eine feste Rezeptur) |
|---|---|---|
| **Bestehende Features** (5 Kanäle) | Zelle A — Referenz, exakt der heutige Code | Zelle B — isoliert den **Modell**beitrag |
| **Erweiterte Features** (10 Kanäle) | Zelle C — isoliert den **Feature**beitrag | Zelle D — Interaktion |

Vier Zellen, **vorab festgelegt**, keine Suche. Beitrag der Features = (C−A) und (D−B); Beitrag des Modells = (B−A) und (D−C).

## 3. Feature-Achse

### Bestehend (5 Kanäle, unverändert aus `daily_comparison.build_dataset`)

Volatilitätsnormierte Renditen über 1, 5, 21, 63, 126 Tage: `pct_change(h) / vol / sqrt(h)`, mit `vol` = 63-Tage-Rollstd. Sequenzlänge 20, also X = [D, A, 20, 5].

### Erweitert (+5 Kanäle, ausschließlich preisabgeleitet)

Keine Volumen-, Makro- oder Orderbuchdaten — die enthält der Snapshot nicht und sie werden nicht erfunden.

| # | Kanal | Definition | Warum |
|---|---|---|---|
| E1 | Skip-Recent-Momentum | `(prices.shift(21)/prices.shift(147) − 1) / vol / sqrt(126)` | 126-Tage-Rendite endend vor 21 Tagen; meidet kurzfristige Umkehr |
| E2 | Querschnittsrang | Tagesweiser Rang von E1 über alle 19 Instrumente, linear auf [−1, +1] abgebildet | **Relative Stärke** — die im Findings-Bericht benannte Lücke; rein querschnittlich |
| E3 | Volatilitätsverhältnis | `log(vol_21 / vol_126)` | Vola-Regimewechsel ohne separates Regimemodell |
| E4 | Drawdown | `(prices / prices.rolling(252).max() − 1) / vol / sqrt(252)` | Abstand zum 252-Tage-Hoch als Marktzustand |
| E5 | SPY-Korrelation | `returns.rolling(63).corr(returns['SPY'])` | Marktkopplung. **Für SPY selbst konstant 1** — degenerierte Spalte, wird vom bestehenden `scale == 0 → 1`-Guard in `daily_models` sauber behandelt; im Bericht zu erwähnen |

Alle Kanäle sind kausal (nur Daten bis einschließlich t) und durchlaufen dieselbe Standardisierungs- und Clipping-Kette (±10) wie die bestehenden fünf.

### Ein Detail, das sonst die Feature-Achse verfälscht

Ridge nutzt heute `λ = 100` bei `p = 100` Merkmalen (20 × 5, flach). Bei 10 Kanälen sind es `p = 200`. Bliebe λ konstant, bekäme die erweiterte Zelle **weniger Schrumpfung pro Koeffizient** — der Feature-Effekt wäre mit einem Regularisierungseffekt vermischt.

**Festlegung:** `λ = 1.0 · p`. Das ergibt bei p = 100 exakt die heutige 100 (Zelle A bleibt bitgleich zum bestehenden Code) und bei p = 200 eben 200.

## 4. Modell-Achse

**Ridge:** unverändert, geschlossene Lösung wie in `daily_models.fit_predict_models`, ungestrafter Achsenabschnitt.

**GBM:** **eine** feste Rezeptur, keine Hyperparametersuche. Vorschlag (an die kleine Stichprobe angepasst, bewusst klein gehalten): 300 Bäume, `learning_rate` 0,05, `num_leaves` 15, `min_data_in_leaf` 200, `feature_fraction` 0,7, `bagging_fraction` 0,7 mit `bagging_freq` 1, fester Seed. Diese Zahlen werden **vor** dem ersten Lauf festgeschrieben und danach nicht mehr angefasst.

### Offene Abhängigkeit — Entscheidung liegt bei dir

`lightgbm` und `scikit-learn` sind in dieser Umgebung **nicht installiert** (vorhanden: torch 2.13, numpy 2.5.2, pandas 3.0.5).

| Option | Bewertung |
|---|---|
| **`pip install lightgbm`** (empfohlen) | Kostenlos und quelloffen, verletzt die "keine kostenpflichtigen Abos"-Grenze nicht. Standardimplementierung, die auch die ML4T-Quelle nutzt. Eine neue Abhängigkeit im Environment. |
| `pip install scikit-learn` → `HistGradientBoostingRegressor` | Ebenfalls frei, sehr ähnliches Verfahren, etwas weniger direkt vergleichbar zur Quelle. |
| GBM selbst in numpy implementieren | **Nicht empfohlen.** Fügt mehr Fehlerrisiko hinzu als es vermeidet; ein selbstgebautes Boosting wäre eine zusätzliche unvalidierte Komponente in einer Kette, deren Fehlalarmrate ohnehin ungeprüft ist. |

Ohne diese Entscheidung ist die Modell-Achse nicht lauffähig.

## 5. Kontrollregeln des Vergleichs

- **Beide Modelle sehen identische Eingaben** — dieselbe flache Matrix (20 × Kanäle). Bäume brauchen die 20 stark korrelierten Verzögerungen eigentlich nicht; sie bekommen sie trotzdem, weil sonst die Modell-Achse mit einer Repräsentationsänderung vermischt wäre. Die Variante "Baum sieht nur den letzten Zeitschritt" ist ein **separater späterer Test**, nicht Teil dieses Rasters.
- Identisch in allen vier Zellen: Label (`returns.shift(−2)/vol`, geclippt ±10, `label_delay = 2`), Allokation (`make_targets`, inverse Vola, 10%-Vola-Deckel), Ausführung (`simulate`, Entscheidung t → Fill t+1), Halte-Kadenz **21 Tage**, Kosten, Risikobudget, Kalender.
- Bruttoexposure bleibt bei maximal 1,0x. Die erlaubten 1,25x werden hier **nicht** ausgeschöpft.

## 6. Auswertungsprotokoll

**Wiederholte chronologische Fenster** statt eines einzelnen Splits (Findings-Bericht Punkt 4): 5 expandierende Trainingsfenster mit je ~2 Jahren Testblock über den Zeitraum 2015-01-02 bis 2026-08-31 (2.932 Handelstage). Labels vollständig reif, Purge von `label_delay` Tagen an jeder Fenstergrenze.

**Benchmark:** dieselbe immer-investierte inverse-Volatilitäts-Kontrolle mit 21-Tage-Kadenz wie im Horizon-Vergleich (4,01% CAGR / −15,72% MaxDD bei 3 bp). Zellenvergleich immer **gegen die Kontrolle**, nicht gegeneinander in absoluten Zahlen.

**Kosten:** 3 bp primär, 15 bp als Sensitivität — beides, weil im Horizon-Vergleich die Kontrolle bei 15 bp bereits vorn lag.

## 7. Vorab festgelegte Metrik und Entscheidungsregel

**Primärmetrik (eine):** annualisiertes relatives geometrisches Wachstum der Zelle gegen die passende Kontrolle, gemittelt über die 5 Fenster, mit deskriptivem 95%-CI.

**Sekundär (berichtet, nicht entscheidend):** CAGR, MaxDD, Turnover, Kostenempfindlichkeit 3 bp ↔ 15 bp, Benchmarkdifferenz.

| Ergebnis | Interpretation |
|---|---|
| Kein CI einer Zelle schließt null aus | **Nullbefund.** So berichten, keine Nachjustierung, keine fünfte Zelle. |
| Genau ein CI schließt null aus | Hinweis, **kein Nachweis** — 4 Zellen, unkorrigiert, Audit ausstehend. Wäre Kandidat für eine eigene, sauber präregistrierte Familie mit frischem Holdout. |
| Mehrere CIs schließen null aus | Auf Konsistenz der Achsenbeiträge prüfen (C−A vs. D−B). Widersprüchliche Vorzeichen deuten auf Rauschen, nicht auf Effekt. |

Bei 10.000 EUR Depot zusätzlich offenzulegen: tatsächlich implementierte Stückelung und Mindestgebühren. Werden ganze Stücke nicht modelliert, ist das explizit zu sagen statt zu implizieren.

## 8. Mehrfachtest-Buchführung

Dieser Vergleich fügt **4 vorab festgelegte Zellen × 1 Primärmetrik = 4 Vergleiche** hinzu, zu den bestehenden 141 nicht korrigierten. Kein Nachreichen zusätzlicher Zellen; jede später gewünschte Variante (Baum auf letztem Zeitschritt, TLT-Korrelation als 11. Kanal, andere GBM-Rezeptur) zählt als **neuer** Vergleich und wird im Forschungsprotokoll geführt.

## 9. Budget

Ridge ist geschlossen lösbar (Sekunden). GBM mit fester Rezeptur auf ~50k Zeilen × 200 Merkmalen liegt im Sekunden- bis Minutenbereich. 4 Zellen × 5 Fenster = **20 Modellanpassungen**, kein LSTM in diesem Raster. Das ist deutlich billiger als der Audit und sollte in einer Sitzung durchlaufen — die tatsächliche Laufzeit ist aber wie beim Audit erst durch einen Pilot (eine Zelle, ein Fenster) zu messen, bevor der volle Lauf startet.

## 10. Bekannte Grenzen

- Universum und Historie sind bekannt und bereits mehrfach betrachtet; neue Fensterschnitte machen daraus **kein frisches Holdout**.
- Fünf zusätzliche Kanäle sind eine enge Auswahl. Ein Nullbefund widerlegt nicht "Querschnittsinformation hilft nie", sondern nur diese fünf Kanäle in dieser Pipeline.
- Der Snapshot enthält nur Preisreihen. OHLC-, Volumen- und Makro-Features der ML4T-Quelle sind hier grundsätzlich nicht prüfbar.
- E5 ist für SPY konstant; der Informationsgehalt der Marktkopplung fehlt damit ausgerechnet für das Marktinstrument selbst.
- Die ML4T-Zahlen (16,5% Validierung, 2,2% Holdout) sind Autorenangaben, nicht reproduziert, und erfüllen unser Ziel ohnehin nicht.

## 11. Nächster Schritt

Diese Spec ist der Entwurf. Für die Umsetzung wäre ein task-weiser Implementierungsplan zu schreiben — **erst auf ausdrücklichen Implementierungsauftrag**, und sinnvollerweise erst nach dem Audit. Vorher ist die Entscheidung aus Abschnitt 4 (GBM-Abhängigkeit) nötig.
