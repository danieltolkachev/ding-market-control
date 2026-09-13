# Explorative Monats-Persistenz Umsetzung

Ersetzt den zurueckgezogenen Plan vom 2026-09-01. Massgeblich ist die Revision
vom 2026-09-07 in der v3-Spec. Keine neue Holdout-Familie, keine Sieger-Selektion.

## Umsetzung und Verifikation

- [x] Originale aus dem v2-Holdout-Worktree ausserhalb aller Worktrees kopieren;
  11 Dateien inklusive Snapshot, Kandidat, Summary und Tombstone per SHA256
  gegen die Originale pruefen. Archiv: `research_archive/trend_v2_preserved_20260907`.
- [x] `factor_lab/signal_persistence.py`: Monatswerte filtern, zwei konsekutive
  Monate bestaetigen, NaN und Monatsluecken behandeln, taeglich nur vorwaerts
  uebertragen. Ein terminaler Teilmonat wird konservativ ausgeschlossen.
- [x] `factor_lab/tests/test_monthly_persistence.py`: handberechnete Wechsel,
  Magnitude, Null, NaN, Luecke, Indexvalidierung, Zukunft und Teilmonat pruefen.
  Erst fehlendes Modul beobachtet, danach erfolgreicher Test.
- [x] `factor_lab/run_persistence_comparison.py`: vorhandene prepare_inputs/
  run_variant verwenden, nur Signalframes austauschen. Alle acht Varianten,
  raw/persistent, Kosten 1/2/5 und matched_long. Gesamthistorie und 60%-Teilung,
  jede Periode cash-Start; Signalvorgeschichte bleibt kausal verfuegbar.
- [x] `factor_lab/tests/test_persistence_comparison.py`: konstantes Signal muss
  exakt gleiche Pfade liefern; echte Kosten, identische Indizes, Inputschutz.
  Erst fehlendes Modul beobachtet, danach erfolgreicher Test.
- [x] Bestehende `test_portfolio.py` prueft Fill-Lag und Future-Poison erneut.
- [x] Historischen Lauf abschliessen und Ergebnisse pruefen. Vollstaendiger Lauf:
  `research_archive/persistence_comparisons/persistence_20260907_171943_384682`.
  SHA256-Inventar umfasst 85 Ergebnis- und Quelldateien.
- [x] Unabhaengiges Review abgeschlossen. Ein fehlender Monat am Periodenschnitt
  wurde durch einen zuvor fehlschlagenden Grenztest reproduziert und behoben;
  Nachreview ohne weitere wichtige Findings. Erster Lauf als unvollstaendig
  markiert und angehalten, korrigierter Lauf komplett neu ausgefuehrt.

## Ausfuehrung

Im v3-Worktree, Python 3.12:

```powershell
py -3.12 factor_lab/tests/test_monthly_persistence.py
py -3.12 factor_lab/tests/test_persistence_comparison.py
py -3.12 factor_lab/tests/test_portfolio.py
py -3.12 -u factor_lab/run_persistence_comparison.py --snapshot C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl --output-root C:/Users/Daniel/Desktop/Ding/research_archive/persistence_comparisons
```

Ausgaben: Konfiguration und Quellcode vor Berechnung; pro Periode Tagesrenditen,
monatliche Log-Renditen, gepaarte Differenzen, Summary; abschliessend Markdown-
Tabelle und SHA256-Inventar. Bootstrap: 10000 Wiederholungen, Seed 0,
erwartete Blocklaenge sechs Monate, rein deskriptiv. Nicht auf Gates abstimmen.

## Bewusste Grenzen

V2-Ausfuehrungsmodell unveraendert; Zielgross 1.0 mit Drift, kein harter
15%-Verlustschutz. Volatilitaetsskalierung identisch. Keine neuen Daten,
keine Live-Orders, keine bestaetigte Renditeaussage. Das Archiv ist gegen
Worktree-Loeschung geschuetzt, aber keine Sicherung gegen Festplattenausfall.
