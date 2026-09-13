# Monats-Persistenz: explorative Ergebnisse vom 2026-09-07

Die zweimonatige Bestaetigung senkt den Handelsumsatz, verbessert aber nicht
systematisch Rendite und Drawdown. Keine unabhaengige Bestaetigung; alle
historischen Daten waren bereits Gegenstand frueherer Forschung.

Snapshot-Content-SHA256:
`36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89`.
Gesamte Auswertung 2009-04-01 bis 2026-08-31 (4381 Handelstage).
60%-Kalenderschnitt tatsaechlich 2019-02-28, spaete Periode ab 2019-03-01.
Je Teilperiode cash-Start mit erster Ausfuehrung nach der vorausgehenden
Monatsentscheidung; kausaler Signalzustand wird aus der Vorgeschichte bestimmt.

## Gesamthistorie bei einfachen Kosten

| Variante | CAGR roh | CAGR persistent | MaxDD roh | MaxDD persistent |
|---|---:|---:|---:|---:|
| combo_long_flat | 5.36% | 4.69% | -13.71% | -15.52% |
| combo_long_short | 2.73% | 2.71% | -16.71% | -17.28% |
| mom126_long_flat | 4.75% | 4.63% | -14.83% | -16.01% |
| mom126_long_short | 2.08% | 1.70% | -14.54% | -15.10% |
| mom252_long_flat | 4.44% | 4.55% | -14.92% | -16.83% |
| mom252_long_short | 2.14% | 2.03% | -14.01% | -12.78% |
| mom63_long_flat | 4.16% | 5.08% | -20.42% | -15.20% |
| mom63_long_short | 1.13% | 1.81% | -17.02% | -20.88% |

Handelsumsatz sinkt je Variante um 15.15% bis 33.72%. CAGR steigt bei drei
von acht Varianten; maximaler Drawdown verbessert sich bei zwei von acht.
matched_long erreicht 4.60% CAGR bei -15.66% MaxDD.

## combo_long_flat ueber Zeit

Fruehe Periode: CAGR 5.09% roh gegen 3.62% persistent.
Spaete Periode: 5.78% gegen 6.19%, Drawdown jedoch -13.71% gegen -15.52%.
Bei doppelten Kosten gesamt: CAGR 5.20% gegen 4.56%.
Deskriptives 95%-Bootstrapintervall des annualisierten geometrischen
relativen Wachstums persistent/roh: gesamt [-1.72%, +0.44%], spaet
[-1.06%, +1.91%]. Das sind relative Wachstumsraten, keine einfachen
CAGR-Prozentpunktdifferenzen. Kein korrigierter Mehrfachtest und kein Gate-Pass.

## Artefakte und Grenzen

Vollstaendiger Lauf unter
`C:/Users/Daniel/Desktop/Ding/research_archive/persistence_comparisons/persistence_20260907_171943_384682`:
Konfiguration, Summary, Tagesreihen, monatliche Log-Differenzen, Tabelle,
Quellcodekopie und geprueftes SHA256-Inventar. Die Kostenleiter 1/2/5 ist
vollstaendig in der Summary enthalten. Originale separat in
`research_archive/trend_v2_preserved_20260907` gesichert und gegen Quelle gehasht.

Vorlaeufiger Lauf `persistence_20260907_171600_340875` wurde nach Review wegen
Periodenstart-Luecke abgebrochen und explizit INCOMPLETE markiert. Keine Auswahl
zwischen diesen Laeufen: nur der vollstaendige korrigierte Lauf ist massgeblich.

Das Experiment verwendet unveraenderte v2-Ausfuehrungs-/Kostenannahmen und
Gross-Ziel 1.0 mit Drift. 15% Drawdown bleibt eine Bewertungsgrenze, keine
Garantie. Keine neuen Daten abgerufen, kein neuer Holdout oder Live-Handel.
