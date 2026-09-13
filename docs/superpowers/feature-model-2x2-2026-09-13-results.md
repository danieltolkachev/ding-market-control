# 2x2 Features x Modell -- Ergebnis des praeregistrierten Rasters

**Ergebnis in einem Satz:** Nullbefund. Keine der vier Zellen schlaegt die immer investierte Inverse-Volatilitaets-Kontrolle; kein 95%-Intervall der Primaermetrik schliesst die Null aus, bei keiner der beiden Kostenstufen. Alle vier Punktschaetzer liegen unter null, also schlechter als die Kontrolle.

Lauf am 2026-09-13. Zeitraum 2015-01-02 bis 2026-08-31, 2.932 Handelstage, 19 ETFs aus dem versiegelten Preis-Snapshot `trend_snapshot_a654e3a4d7368cf2` (Inhalts-Hash `36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89`), fuenf expandierende Fenster. Alle Zahlen nach modellierten Kosten, vor Steuern.

## 1. Was getestet wurde

Vier Zellen, vollstaendig vor dem Lauf festgelegt in `docs/superpowers/specs/2026-09-11-feature-model-2x2-design.md`:

| Zelle | Kanaele | Modell |
|---|---|---|
| A_base_ridge | 5 Basiskanaele (Renditefenster 1, 5, 21, 63, 126) | Ridge |
| B_base_gbm | dieselben 5 Basiskanaele | eingefrorene LightGBM-Rezeptur |
| C_ext_ridge | 10 Kanaele (Basis + skip_momentum, cross_rank, vol_ratio, drawdown, spy_corr) | Ridge |
| D_ext_gbm | dieselben 10 Kanaele | dieselbe LightGBM-Rezeptur |

Label: `returns.shift(-2)/vol`, auf [-10, 10] geklippt, Label-Verzoegerung 2 Tage. Haltedauer 21 Handelstage, Ausfuehrung zum naechsten Schluss, dazwischen Drift ohne Handel. Allokation: inverse Volatilitaet, Long/Flat-Gate, 10%-Volatilitaetsdeckel, Bruttogrenze 1,0x. Kontrolle: `always_long`, dieselbe Allokation ohne Prognosegate, dieselbe Kadenz.

**Eine** Primaermetrik: annualisiertes relatives geometrisches Wachstum der Zelle gegen die Kontrolle, aus den zusammengefuegten Monatsdifferenzen, mit deskriptivem 95%-Intervall aus einem stationaeren Block-Bootstrap (erwartete Blocklaenge 6 Monate, 10.000 Ziehungen, Seed 7, 140 Monatsbeobachtungen). Primaere Kostenstufe 3 bp, Sensitivitaet 15 bp. Entscheidungsregel: Abschnitt 7 der Spec, unten woertlich zitiert.

Zur Aggregation gibt es eine Abweichung vom Wortlaut der Spec, die der Plan vorab festgelegt hat: die Spec schreibt "gemittelt ueber die 5 Fenster", umgesetzt ist die zusammengefuegte Monatsreihe, also ein monats- statt fenstergewichteter Durchschnitt. Abschnitt 6 (b) legt das offen und begruendet, warum der Unterschied hier unerheblich ist.

Alles daran -- Zellen, Kanaele, Rezeptur, Fensterzahl, Kadenz, Metrik, Regel -- stand vor dem ersten Lauf fest und wurde danach nicht angefasst; die beiden Umsetzungsentscheidungen in Abschnitt 6 ebenfalls.

## 2. Ergebnistabelle

| Zelle | CAGR 3bp | MaxDD 3bp | Endwert 3bp | CAGR 15bp | MaxDD 15bp | Endwert 15bp | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_base_ridge | 3,09% | -10,21% | 14.253 | 2,36% | -10,28% | 13.121 | 68,97 |
| B_base_gbm | 2,87% | -8,88% | 13.898 | 2,12% | -8,97% | 12.770 | 70,49 |
| C_ext_ridge | 2,96% | -9,83% | 14.045 | 2,20% | -9,90% | 12.888 | 71,65 |
| D_ext_gbm | 3,69% | -10,88% | 15.239 | 2,97% | -11,33% | 14.057 | 67,30 |
| always_long (Kontrolle) | 4,01% | -15,72% | 15.799 | 3,87% | -15,80% | 15.554 | 13,03 |
| Nur Cash | 2,08% | -0,0015% | -- | 2,08% | -0,0015% | -- | 0 |

Endwerte auf 10.000 EUR Startkapital. Turnover ist die Summe ueber den gesamten Zeitraum bei 3 bp. Die Cash-Zeile ist bei beiden Kostenstufen identisch, weil sie keinen Umsatz hat.

Zwei Beobachtungen dazu, beide deskriptiv: die Kontrolle hat den hoechsten CAGR und zugleich mit Abstand den tiefsten Drawdown (-15,7%, knapp ueber der 15%-Auswertungsgrenze); die vier Modellzellen kaufen ihre geringere Drawdown-Tiefe mit rund fuenffachem Turnover und niedrigerer Rendite. Keine Zelle und auch nicht die Kontrolle erreichen das 12%-Ziel der Laufkonfiguration (`configuration.return_target = 0.12`); `historical_target_met` ist in allen zehn Kosten-Zellen `false`. Das davon zu unterscheidende 12,7%-Ziel des Uebergabe-Prompts, das Abschnitt 10 behandelt, liegt noch hoeher und ist damit erst recht nicht erreicht.

Die Prognosemetriken sind mit diesem Bild konsistent: die Richtungstrefferquote liegt zwischen 50,12% und 51,02% und damit in allen vier Zellen **unter** der trivialen Immer-aufwaerts-Quote von 51,65%; der normierte MSE liegt bei 1,15 bis 1,16, also ueber 1. Die Modelle sagen in diesem Aufbau nichts Verwertbares vorher.

## 3. Primaermetrik

Annualisiertes relatives geometrisches Wachstum gegen `always_long`, Punktschaetzer und deskriptives 95%-Intervall:

| Zelle | 3 bp | 15 bp |
|---|---|---|
| A_base_ridge | -0,88% [-2,87%, +1,31%] | -1,45% [-3,44%, +0,74%] |
| B_base_gbm | -1,09% [-2,69%, +0,67%] | -1,68% [-3,25%, +0,07%] |
| C_ext_ridge | -1,00% [-3,01%, +1,22%] | -1,60% [-3,61%, +0,62%] |
| D_ext_gbm | -0,31% [-1,75%, +1,24%] | -0,86% [-2,30%, +0,69%] |

Acht Intervalle, acht Mal die Null eingeschlossen. Alle acht Punktschaetzer sind negativ. Am naechsten daran, die Null auszuschliessen, kommt B_base_gbm bei 15 bp mit einer Obergrenze von +0,07% -- es schliesst die Null knapp ein, und selbst wenn es sie ausschloesse, zeigte es in die falsche Richtung: die Zelle waere dann nachweislich **schlechter** als die Kontrolle, kein Fund.

## 4. Achsenbeitraege

Differenzen der Punktschaetzer bei 3 bp (15 bp in Klammern):

- Feature-Achse: C-A = -0,12% (-0,15%), D-B = +0,78% (+0,81%)
- Modell-Achse: B-A = -0,21% (-0,23%), D-C = +0,69% (+0,74%)

Diese Differenzen sind **keine getesteten Groessen**. Sie haben kein Intervall, keine Fehlerrechnung und keine Entscheidungsregel hinter sich; sie stehen hier nur zur Beschreibung. Beide Achsen haben widerspruechliche Vorzeichen: zusaetzliche Kanaele helfen dem Baum (+0,78%) und schaden dem Ridge (-0,12%); der Baum schlaegt den Ridge auf zehn Kanaelen (+0,69%) und verliert gegen ihn auf fuenf (-0,21%). Das ist genau das Muster, das die Spec vorab als Rauschen und nicht als Effekt gelesen hat. Da ohnehin kein Intervall die Null ausschliesst, gibt es hier nichts zu interpretieren.

**Fensterweise Aufschluesselung bei 3 bp -- deskriptiv und an den Raendern ueberlappend.** Ein Kalendermonat, in dem eine Fenstergrenze liegt, erscheint in **beiden** angrenzenden Fenstern; die Monatssumme der Fenster (29+28+29+29+29 = 144) uebersteigt deshalb die 140 Monate der zusammengefuegten Reihe. Die Primaermetrik ist davon nicht betroffen: sie laeuft auf der zusammengefuegten Monatsreihe, in der jeder Monat genau einmal vorkommt.

| Zelle | F1 (2015-01 bis 2017-05) | F2 (bis 2019-08) | F3 (bis 2021-12) | F4 (bis 2024-04) | F5 (bis 2026-08) |
|---|---:|---:|---:|---:|---:|
| A_base_ridge | -0,17% | -0,55% | -3,24% | +3,69% | -3,98% |
| B_base_gbm | -1,75% | -1,18% | -2,11% | +2,67% | -3,03% |
| C_ext_ridge | +0,31% | -1,65% | -3,36% | +3,21% | -4,12% |
| D_ext_gbm | -0,74% | -0,21% | -1,27% | +2,94% | -2,30% |

Alle vier Zellen zeigen im Kern dasselbe Muster: ueberwiegend negativ -- drei von fuenf Fenstern bei C_ext_ridge, vier von fuenf bei den uebrigen -- und positiv im vierten Fenster (2022 bis Anfang 2024) bei allen vier Zellen, dem Fenster mit dem grossen Marktrueckgang, in dem ein Long/Flat-Gate mechanisch hilft. C_ext_ridge ist zusaetzlich im ersten Fenster leicht positiv (+0,31%), die einzige Ausnahme im Raster. Der Beitrag des vierten Fensters ist ein Drawdown-Schutz, kein Prognosevorteil, und im Gesamtergebnis wiegt er die negativen Fenster in keiner der vier Zellen auf.

## 5. Urteil nach der vorab festgelegten Entscheidungsregel

`summary.json` unter `verdict`:

```json
{
  "cells_with_interval_excluding_zero": [],
  "reading": "null result"
}
```

Abschnitt 7 der Spec, erste Zeile der Entscheidungstabelle, woertlich (Umlaute hier nach ASCII umgesetzt):

> Kein CI einer Zelle schliesst null aus -> **Nullbefund.** So berichten, keine Nachjustierung, keine fuenfte Zelle.

Das ist der eingetretene Fall. Es gab keine Nachjustierung, keine fuenfte Zelle, kein erneutes Schneiden der Fenster, keine zweite Kostenstufe nach dem Ergebnis und keine Veraenderung der Rezeptur, der Kanalauswahl oder der Kadenz.

## 6. Zwingende Einschraenkungen

Aus `configuration.json['limitations']`, vollstaendig:

- Bereits gesehene Historie und bereits gesehenes Universum; neue Fensterschnitte sind kein frisches Holdout.
- Vier praeregistrierte Zellen zusaetzlich zu 141 bestehenden unkorrigierten Vergleichen; keine Multiplizitaetskorrektur.
- Deskriptive Bootstrap-Intervalle auf ueberlappenden Monatsdaten; kein Signifikanztest.
- Der Falsifikations-Audit der Selektionspipeline deckt nur das trend-etf-v2-Screening ab; er uebertraegt sich nicht auf dieses Raster.
- Keine ganzen Stuecke, keine Broker-Mindestgebuehren, kein FX, keine Steuern, keine Finanzierung.
- Die fuenf erweiterten Kanaele sind eine enge Auswahl; ein Nullbefund widerlegt diese fuenf in dieser Pipeline, nicht Querschnittsinformation im Allgemeinen.
- Der SPY-Korrelationskanal ist fuer SPY selbst konstant 1; die Marktkopplung ist fuer das Marktinstrument uninformativ.
- Volatilitaetsdeckel und historischer Drawdown garantieren kein zukuenftiges Risiko.
- Das Jahr 2026 ist unvollstaendig; die Bruttoquote ist auf 1,0x gedeckelt und damit unter den erlaubten 1,25x.

Dazu die zwei Umsetzungsentscheidungen, die der Plan vorab festgelegt hat:

**(a) Zeilenausrichtung.** Alle vier Zellen laufen auf der Zeilenmenge der erweiterten Kanaele (4.433 Zeilen, Auswertung ab 2015-01-02). Die Feature-Achse ist damit nicht mit einem Stichprobenwechsel verwechselbar -- waeren die Basiszellen auf ihrer eigenen, laengeren Zeilenmenge gelaufen, waere jeder Unterschied zwischen A und C zugleich ein Unterschied der Trainingsdaten gewesen. Der Preis dieser Entscheidung: die Aussage "Zelle A ist bit-identisch zum bestehenden Code" ist eine **arithmetische** Aussage und keine Aussage ueber die Trainingsstichprobe. Sie wird durch einen Aequivalenztest gegen den eingefrorenen `daily_models`-Ridge geprueft (`factor_lab/tests/test_models_2x2.py::test_matches_existing_daily_models_ridge_exactly`), nicht durch den Lauf selbst.

**(b) Monatsgleichgewichtete Aggregation.** Die Primaermetrik fuegt die Monatsdifferenzen ueber die fuenf zusammenhaengenden Fenster zusammen und mittelt sie als Monatsreihe, nicht als Mittel der fuenf Fensterwerte. Das ist ein monats- statt fenstergewichteter Durchschnitt. Bei Blockgroessen von 586/586/586/586/588 Zeilen ist der Unterschied unerheblich.

Und ausdruecklich, weil es leicht untergeht: der bereits abgeschlossene Falsifikations-Audit deckt die **Screening-Pipeline von trend-etf-v2** ab und **nicht dieses Raster**. Ein positives Ergebnis haette hier also weiterhin eine ungemessene Fehlalarmrate geerbt. Dass das Ergebnis negativ ist, aendert daran nichts -- es macht die Frage nur gegenstandslos.

## 7. Stueckelung und Gebuehren

Ganze Stuecke, Broker-Mindestgebuehren, FX und Steuern sind **nicht modelliert**. Die Kosten bestehen ausschliesslich aus einem Basispunkt-Aufschlag auf den Umsatz, 3 bp primaer und 15 bp als Sensitivitaet. Bei 10.000 EUR auf 19 Instrumente liegt eine Durchschnittsposition bei rund 526 EUR, und die inverse Volatilitaetsgewichtung drueckt einen Teil der Positionen deutlich darunter; dort waeren Mindestgebuehren und die Rundung auf ganze Stuecke real deutlich spuerbar. Die Endwerte in Abschnitt 2 sind insoweit optimistisch -- fuer die Zellen staerker als fuer die Kontrolle, weil die Zellen den rund fuenffachen Umsatz haben. Eine realistischere Gebuehrenmodellierung wuerde den Abstand zur Kontrolle also vergroessern, nicht verkleinern.

## 8. Mehrfachtest-Buchfuehrung

141 bestehende unkorrigierte Vergleiche aus der bisherigen Projektgeschichte, plus die 4 Zellen dieses Rasters: **145 unkorrigierte Vergleiche**. Keine Multiplizitaetskorrektur ist angewandt.

Jede spaetere Variante zaehlt neu und erhoeht diesen Zaehler weiter: ein Baum nur auf dem letzten Zeitschritt, ein elfter Kanal, eine andere Rezeptur, eine andere Haltedauer. Das ist der Grund, warum die Entscheidungsregel bei genau einem auffaelligen Intervall nur "Hinweis, kein Nachweis" zulaesst und eine eigene praeregistrierte Familie mit frischem Holdout verlangt.

## 9. Ausgabeverzeichnis und Hash

Ausgabeverzeichnis: `factor_lab/runs_2x2/feature_2x2_20260913_120450_206376` (git-ignoriert, nicht eingecheckt).

SHA-256 von `summary.json` laut `sha256.json`:

```
bc3c4c3028e3180a8a166a2e6768c35a472e4de53fdf52def291678de13999ab
```

`sha256.json` deckt alle 32 Artefakte des Laufs ab: 24 Ausgabedateien plus 8 Eintraege unter `source/`, naemlich die sechs beteiligten Quelldateien, den Plan und die Spec. `COMPLETE` ist geschrieben. Reproduktionsbefehl:

```powershell
py -3.12 -u factor_lab/run_feature_model_2x2.py --snapshot "C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl" --output-root factor_lab/runs_2x2
```

Umgebung: Python 3.12.10, numpy 2.5.2, pandas 3.0.5, lightgbm 4.7.0. Laufzeit rund 42 Sekunden.

## 10. Konsequenz des Nullbefunds fuer das Renditeziel

Das Ergebnis ist ein Nullbefund, damit gilt dieser Abschnitt.

Der Stand nach diesem Lauf: die Trendbeine allein kappen bei rund 7,3% CAGR -- diese Zahl stammt nicht aus diesem Lauf, sondern aus dem frueheren Projektbefund zum Trendbein (Zielrendite-Notiz, `project_target_return`), auf den sich der Uebergabe-Prompt stuetzt. Das zweite Bein -- Querschnittsinformation, in genau der Form, die dieses Raster geprueft hat -- traegt nichts bei. Vier Zellen, zwei Kostenstufen, acht Intervalle, alle acht mit negativem Punktschaetzer und alle acht die Null einschliessend. Die immer investierte Kontrolle ist besser als jede Modellzelle und erreicht selbst nur 4,01% CAGR bei -15,7% Drawdown.

Daraus folgt ohne Beschoenigung: **bei einem Drawdown-Deckel von 15% und einer Hebelgrenze von 1,25x ist ein CAGR-Ziel von 12,7% mit dem bisher Getesteten nicht erreichbar.** Kein getestetes Modell, keine getestete Kadenz, keine getestete Kanalmenge kommt in die Naehe; die beste Zahl im gesamten Raster ist 3,69% und liegt unter der trivialen Kontrolle.

Die ehrliche Konsequenz ist eine **Revision des Ziels**, nicht die naechste Variante. Jede weitere Variante auf derselben Historie erhoeht den Mehrfachtest-Zaehler, erbt dasselbe fehlende frische Holdout und hat nach allem hier Gemessenen eine schlechte Ausgangswahrscheinlichkeit. Wer das Ziel halten will, muesste entweder die Randbedingungen aendern (hoehere Drawdown-Toleranz, hoeherer Hebel) oder eine Informationsquelle erschliessen, die ausserhalb dieser Pipeline liegt -- und dann mit frischem, vorab abgetrenntem Holdout, nicht auf dieser Historie.
