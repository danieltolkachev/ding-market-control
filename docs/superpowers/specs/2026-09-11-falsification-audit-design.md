# End-to-End-Falsifikations-Audit der Auswahlpipeline

**Datum:** 2026-09-11
**Status:** Plan zur Freigabe — **noch keine Implementierung**
**Grundlage:** Nikolopoulos, *Spurious Predictability in Financial Machine Learning*, arXiv 2604.15531 (v1)
**Auftrag:** Schritt 2 des Übergabe-Prompts vom 2026-09-11 (`docs/superpowers/continuation-prompt-2026-09-11.md`)

## 0. Was hier geprüft wird — und was nicht

**Geprüft wird:** die Fehlalarmrate der *gesamten adaptiven Auswahlpipeline* auf Daten, die nachweislich keinen prognostizierbaren Mittelwert haben. Nicht ein einzelner Test, sondern die vollständige Kette: 8 Varianten rechnen → 4 Gates anwenden → beste Passerin nach Excess-Sharpe auswählen → versiegeln.

**Nicht geprüft wird:** ob ein Markt-Edge existiert. Ein bestandener Audit sagt ausschließlich: *"Die Pipeline meldet auf diesen drei Nullwelten ungefähr so selten einen Treffer, wie sie es laut ihrem eigenen 95%-Kriterium tun sollte."* Er sagt nichts über echte Daten und nichts über Nullwelten, die hier nicht getestet werden.

**Warum jetzt:** Die 141 bisherigen Vergleiche sind nicht für Mehrfachtests korrigiert, und die Fehlalarmrate der Pipeline ist unbekannt. Jede künftige Erweiterung (Feature-Sets, Modellklassen, Schwellen) vervielfacht den Suchraum. Ohne kalibrierte Fehlalarmrate ist ein künftiges "Gate A bestanden" nicht interpretierbar.

## 1. Die drei Nullwelten

Alle drei erzeugen ein Panel derselben Form wie der echte Snapshot: 19 Preisreihen plus eine Cash-Reihe, Tagesfrequenz, gleiche Länge und gleicher Handelskalender wie `trend_snapshot_a654e3a4d7368cf2.pkl`. Alle drei haben **konstruktionsbedingt einen bedingten Erwartungswert von exakt null** — kein Modell kann dort einen echten Prognose-Edge finden.

Entscheidend: die Nullwelten müssen die *marginalen* Eigenschaften echter Märkte plausibel nachbilden (Volatilitätsniveau, Clustering, Querschnittskorrelation). Eine Nullwelt, die nicht wie ein Markt aussieht, macht einen bestandenen Audit wertlos.

| ID | Nullwelt | Konstruktion | Was sie isoliert |
|---|---|---|---|
| **N1** | Mittelwertfreie Renditen mit wechselnder Volatilität | r_t = σ_t · ε_t, ε_t ~ N(0,1) i.i.d.; σ_t folgt einem zweizuständigen Markov-Schalter (ruhig/stressig), Übergangswahrscheinlichkeiten aus dem echten Snapshot geschätzt | Erzeugt allein **Volatilitätsstruktur** scheinbare Prognostizierbarkeit? |
| **N2** | GARCH(1,1), mittelwertfrei | r_t = σ_t · ε_t mit σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}; (ω, α, β) je Instrument auf dem echten Snapshot geschätzt, Mittelwert hart auf null gesetzt | Erzeugt allein **Volatilitäts-Clustering** scheinbare Prognostizierbarkeit? Realistischste der drei. |
| **N3** | Gemeinsame Faktoren ohne prognostizierbaren Mittelwert | r_t = B·f_t + e_t; f_t zwei latente Faktoren, beide mittelwertfrei mit GARCH-Vola; Ladungen B aus der echten Korrelationsmatrix (PCA, zwei Hauptkomponenten); e_t idiosynkratisch mittelwertfrei | Erzeugt allein **Querschnittskorrelation** scheinbare Prognostizierbarkeit? Der für Trendfolge über ein korreliertes ETF-Universum relevanteste Fall. |

**Schätzung der Nullwelt-Parameter** erfolgt einmalig auf dem echten Snapshot, wird als JSON eingefroren und gehasht. Danach werden die Parameter nicht mehr angefasst. Der echte Snapshot wird dabei nur gelesen, nie verändert.

**Kennzeichnung:** Alle erzeugten Panels landen ausschließlich unter `factor_lab/audit_data/` mit Dateinamen-Präfix `SYNTHETIC_NULL_` und einem `"synthetic": true`-Feld im Manifest. Sie dürfen niemals in ein `trend_snapshot_*`-Verzeichnis geraten.

## 2. Die auditierte Pipeline

Der Audit fährt **dieselbe Codepfad-Kette wie ein echter Screening-Lauf**, nicht eine Nachbildung:

```
Null-Panel  →  prepare_inputs()  →  8 Varianten × 3 Kostenmultiplikatoren
            →  Bootstrap-CI + Permutation je Variante
            →  Gates A/B/C  →  (bei A-C-Passerin) Gate D mit LOO über 19 Instrumente + 7 Sleeves
            →  screening_verdict()  →  Kandidatin versiegelt ja/nein
```

Die Auswahlregel ("höchster Excess-Sharpe unter Bestehenden, Tie-Break alphabetisch") ist **Teil des Auditierten**, nicht Beiwerk. Genau diese "nimm die Beste von 8"-Regel ist der Mechanismus, dessen Inflation gemessen werden soll.

## 3. Zwei Messgrößen, nicht eine

| Messgröße | Definition | Nominalwert | Kriterium |
|---|---|---|---|
| **FA_variant** | Anteil der Läufe × Varianten, in denen **Gate A** (Bootstrap-Untergrenze > 0) anschlägt | 5% (einseitiges 95%-Kriterium) | Pipeline gilt als **fehlkalibriert**, wenn FA_variant außerhalb [2,5%, 10%] liegt |
| **FA_pipeline** | Anteil der Läufe, in denen **irgendeine** Variante alle Gates besteht und eine Kandidatin versiegelt wird | kein etablierter Nominalwert | **>15% ⇒ die "Beste von 8"-Regel braucht eine Mehrfachtest-Korrektur**, bevor einer künftigen Familie geglaubt wird |

**Begründung der 15%-Schwelle (vorab festgelegt, Ermessensentscheidung, keine hergeleitete Konstante):** Wären die 8 Varianten unabhängig und Gate A exakt bei 5% kalibriert, läge FA_pipeline bei 1−0,95⁸ ≈ 34%. Die Varianten sind stark korreliert (gleiches Universum, überlappende Lookbacks), und die Gates B/C/D filtern zusätzlich. Ein Wert unter ~10% würde bedeuten, dass die Zusatz-Gates faktisch Mehrfachtest-Kontrolle leisten. Über 15% leisten sie das erkennbar nicht.

**Die Differenz zwischen beiden Zahlen ist das eigentliche Ergebnis des Audits** — sie quantifiziert, wie viel Selektionsinflation die Pipeline erzeugt. Diese Zahl existiert für dieses Projekt bisher nicht.

## 4. Laufbudget

**Statistische Power.** Um FA_variant = 5% von 10% zu unterscheiden (einseitig, α = 5%):

| Wiederholungen je Nullwelt | Standardfehler bei p=0,05 | Abstand zu 10% | Power |
|---:|---:|---:|---:|
| 50 | 3,08% | 1,6 SE | ≈55% — **unbrauchbar** |
| 100 | 2,18% | 2,3 SE | ≈80% — Untergrenze |
| **200** | **1,54%** | **3,2 SE** | **≈96% — Zielwert** |

**Kostentreiber.** Der Bootstrap dominiert: `stationary_block_bootstrap` läuft eine Python-Schleife über `n_boot`, innen `stationary_bootstrap_indices` über ~160 Monate. Pro Variante fallen 3 Bootstrap-Aufrufe an (primär + 2 Sensitivitäten), bei 8 Varianten also 24 × 10.000 Ziehungen je Lauf. Gate Ds LOO-Zweig (26 Reruns) feuert nur bei A-C-Passerinnen — auf Nullwelten also selten, was den Audit relativ günstig macht.

**Gestuftes Budget:**

1. **Pilot: 5 Läufe je Nullwelt (15 gesamt).** Zweck: Plumbing validieren und **die tatsächliche Laufzeit je Lauf messen**. Erst danach wird über das Hauptbudget entschieden. Ohne diese Messung wäre jede Budgetzahl geraten.
2. **Hauptlauf: 200 je Nullwelt (600 gesamt)**, falls der Pilot das zulässt. Andernfalls das größte n ≥ 100, das ins Zeitbudget passt — mit **offengelegter reduzierter Power**. Unter n = 100 wird der Audit nicht gefahren, sondern als nicht durchführbar berichtet.
3. **Kontrolllauf: 50 versiegelte Seeds je Nullwelt (150 gesamt)**, genau einmal, nach Einfrieren des Audit-Codes.

**Falls das Volltreue-Budget nicht reicht:** `n_boot` darf von 10.000 auf 2.000 gesenkt werden (5× günstiger), **aber nur** wenn im Pilot geprüft wird, dass die Gate-A-Entscheidung bei 2.000 und 10.000 Ziehungen in ≥95% der Fälle übereinstimmt. Sonst auditiert man eine andere Pipeline als die echte. Eine solche Reduktion wird im Ergebnisbericht als Einschränkung ausgewiesen, nicht stillschweigend übernommen.

## 5. Unangetastete Kontroll-Seeds

Der Audit kann selbst überangepasst werden — bewusst oder unbewusst, indem Nullwelt-Parameter oder Auswertungsdetails so lange justiert werden, bis das Ergebnis "gut aussieht". Dagegen:

- **Arbeits-Seeds:** 0–199 je Nullwelt. Ausschließlich diese werden während der Audit-Entwicklung verwendet.
- **Kontroll-Seeds:** 10000–10049 je Nullwelt. Werden **vor** dem Hauptlauf in `audit_control_seeds.json` festgeschrieben, die Datei wird committet, ihr SHA256 im Ergebnisbericht genannt. Sie werden erst ausgeführt, wenn Audit-Code und Auswertung eingefroren sind.
- **Abweichungskriterium:** Weicht FA_variant auf den Kontroll-Seeds um mehr als 5 Prozentpunkte vom Arbeits-Seed-Ergebnis ab, gilt der Audit als an die Arbeits-Seeds angepasst und muss überarbeitet werden.

## 6. Isolation gegenüber bestehender Forschung

- Neues, getrenntes Modulset unter `factor_lab/audit/`; keine Änderung an `registration*.py`, `run_trend_*`, `stats.py`, `portfolio.py`, `signals.py`, `costs.py`.
- Der echte Snapshot wird **read-only** für die Parameterschätzung gelesen. Versiegelte Kandidaten, Tombstones und abgeschlossene Ergebnisberichte werden nicht berührt.
- Synthetische Daten ausschließlich unter `factor_lab/audit_data/`, Präfix `SYNTHETIC_NULL_`, `"synthetic": true` im Manifest.
- Der Audit versiegelt **keine** Kandidatin und schreibt **keinen** Tombstone in `factor_lab/logs/`.

## 7. Vorab festgelegte Ergebnisinterpretation

Vor dem Lauf festgeschrieben, damit das Ergebnis nicht nachträglich umgedeutet wird:

| Ergebnis | Interpretation | Konsequenz |
|---|---|---|
| FA_variant in [2,5%, 10%] **und** FA_pipeline ≤ 15% | Pipeline ist auf diesen Nullwelten ungefähr wie beworben kalibriert | Methodik-Prüfung bestanden. **Kein Edge-Nachweis.** Erweiterungen (2×2-Vergleich) können folgen. |
| FA_variant in [2,5%, 10%], FA_pipeline > 15% | Einzeltest kalibriert, Auswahlregel inflationär | Mehrfachtest-Korrektur nötig, bevor eine künftige Familie versiegelt wird. Bisherige "bestandene" Screenings sind entsprechend zu diskontieren. |
| FA_variant > 10% | Bootstrap-Inferenz selbst fehlkalibriert | **Schwerwiegend.** Alle bisherigen Gate-A-Aussagen — inklusive v2s Screening-Pass — wären nicht interpretierbar. Methodik reparieren, bevor irgendetwas anderes läuft. |
| FA_variant < 2,5% | Test übermäßig konservativ | Echte Effekte würden übersehen; Konservativität dokumentieren, keine Lockerung ohne eigene Begründung. |

Ein Fehlschlag des Audits ist ein **verwertbares Ergebnis über die Methodik**, kein gescheitertes Experiment. Er wird als solcher berichtet, nicht wegoptimiert.

## 8. Bekannte Grenzen dieses Audits

- Drei Nullwelten decken nicht alle Formen scheinbarer Prognostizierbarkeit ab. Mikrostruktur-Effekte sind mangels Orderbuch-/Volumendaten gar nicht prüfbar.
- Die Nullwelt-Parameter stammen aus demselben Snapshot, auf dem die bisherige Forschung lief — die Nullwelten sind dem echten Datensatz also ähnlich, aber nicht unabhängig von ihm.
- Der Audit prüft die *Trend-Screening*-Pipeline. Für die späteren LSTM-/2×2-Erweiterungen wäre er in angepasster Form zu wiederholen; ein Pass hier überträgt sich nicht automatisch.
- Kennzahlen aus dem Paper (BIF, effektive Suchbreite) werden **nicht** als universelle Korrekturformeln übernommen; die Autorenresultate wurden nicht unabhängig reproduziert.

## 9. Was als Nächstes nötig wäre

Dieser Plan ist der Entwurf, nicht die Umsetzung. Für die Implementierung wäre ein eigener, task-weiser Plan zu schreiben (Nullwelt-Generatoren mit Tests, Audit-Runner, Auswertung, Kontrolllauf) — **erst auf ausdrücklichen Implementierungsauftrag**, wie im Übergabe-Prompt festgelegt.
