# Falsifikations-Audit — Pilot-Ergebnisse (2026-09-11)

Dieser Bericht dokumentiert Schritt 1-3 des Falsifikations-Audits: Parameter
auf dem archivierten Snapshot schaetzen und einfrieren, Kontroll-Seeds
versiegeln, 15 Pilot-Replikationen (5 je Nullwelt) fahren, die reale
Laufzeit messen und daraus das Hauptlauf-Budget ableiten.

**Der Pilot liefert KEINE Fehlalarmrate.** 5 Replikationen je Welt haben
dafuer keine statistische Power. Die im Konsolen-Output ausgegebenen
`FA_variant`/`FA_pipeline`-Werte sind reine Orientierungsgroessen und werden
unten entsprechend gekennzeichnet, nicht als Ergebnis interpretiert.

## Eingefrorene Artefakte

| Artefakt | Pfad | SHA256 |
|---|---|---|
| Nullwelt-Parameter | `factor_lab/audit_data/null_params.json` | `ae7c512cf50e7c4a34f44a73706f80443a6c295a4e56d8df9cc8d13fb728c9d0` |
| Kontroll-Seeds | `factor_lab/audit_data/audit_control_seeds.json` | `9ffb0f014183680af41d87d644989b379cdac4ad574d81e7864ccaac953f5a77` |

Beispiel-Panels (je Welt, seed=0, zur Sichtpruefung der Nullwelt-Struktur):
`factor_lab/audit_data/SYNTHETIC_NULL_{n1,n2,n3}_seed0_example.csv`.

Der vollstaendige Pilot-Datensatz (15 Records inkl. `elapsed_s`, `gate_a`,
`passed_all`, `excess_lower95`, `dev_end`) liegt in
`factor_lab/audit_data/pilot_records.json`.

## Gemessene Laufzeit

15 Replikationen, 30,2 min gesamt, **120,8 s im Mittel je Lauf**.

Die Kostenstruktur ist **bimodal**, nicht gleichverteilt um den Mittelwert:

| Fall | Anzahl Laeufe | Laufzeit |
|---|---|---|
| Keine Variante besteht Gate A-C | 13 / 15 | **88,6 s** |
| Mindestens eine Variante besteht Gate A-C | 2 / 15 | **330,7 s** |

Grund: `run_screening()` durchlaeuft fuer jede Variante, die Gate A-C besteht,
zusaetzlich den Gate-D-LOO-Zweig (Leave-one-symbol-out) mit 26 Backtest-Reruns
je Treffer. Dieser Zweig ist backtest-dominiert, nicht bootstrap-dominiert —
die ~242 s Differenz zwischen den beiden Faellen kommt aus den Reruns, nicht
aus `stationary_block_bootstrap`.

Pro Welt:

| Welt | Pipeline-Kandidat (Treffer/5) | Gate-A-Variantentreffer (von 40) |
|---|---|---|
| n1 | 1 (`mom126_long_flat`, seed=4) | 3 |
| n2 | 1 (`mom126_long_flat`, seed=4) | 2 |
| n3 | 0 | 0 |

## Hochrechnung

- Hauptlauf (3 Welten x 200 Replikationen = 600): **20,1 h**
- Kontrolllauf (3 Welten x 50 Replikationen = 150): **5,0 h**

Die Hochrechnung fuer den Hauptlauf liegt weit ueber der 6-Stunden-Schwelle
aus dem Brief.

## Ruling 1 (Controller): volle Treue, n=100 je Welt statt n_boot-Reduktion

Der Plan sah bei einer Hochrechnung > 6 h die Option vor, `n_boot` von
10.000 auf 2.000 zu senken (nach Treuecheck). Die gemessene, bimodale
Kostenstruktur widerlegt jedoch die Annahme, auf der diese Option beruhte:
eine `n_boot`-Reduktion trifft ausschliesslich den Bootstrap-Anteil des
88,6-s-Grundlaufs, **nicht** den teuren Gate-D-LOO-Zweig (330,7 s), der
backtest- und nicht bootstrap-dominiert ist.

Selbst in der optimistischsten Rechnung — der 88,6-s-Grundlauf bestuende zu
100% aus Bootstrap-Zeit und liesse sich durch die `n_boot`-Reduktion auf
nahe null druecken — kaeme man bei n=200 auf ca. 8,8 h Hauptlaufzeit.
Dagegen steht n=100 bei **unveraendertem** `n_boot=10.000`: ca. 10,1 h. Der
Zeitgewinn der Reduktion ist marginal (8,8 h vs. 10,1 h), der Preis waere,
dass eine gegenueber der Registrierung modifizierte statt der echten
Pipeline auditiert wird — mit einem Treuecheck, dessen zulaessige
Abweichungsquote (bis zu 5% divergierende Gate-Entscheidungen) in derselben
Groessenordnung liegt wie die Rate, die das Audit ueberhaupt messen soll.

**Entscheidung: n = 100 Replikationen je Welt (300 gesamt, ca. 10,1 h
Laufzeit). `n_boot` bleibt unveraendert bei 10.000, wie in
`REGISTRATION_V2` registriert.** Ein Treuecheck ist nicht noetig, weil
nichts an der Pipeline reduziert wird — der Hauptlauf auditiert exakt die
registrierte Konfiguration.

**Offengelegte statistische Power:** Bei n=100 je Welt liegt die Power, eine
Fehlalarmrate von 10% (an der oberen Grenze des vorregistrierten
Toleranzbands `FA_VARIANT_UPPER=0.10`) von der Nominalrate 5% zu
unterscheiden, bei ca. 80% — das ist die im Design-Dokument genannte
Untergrenze fuer ein noch aussagekraeftiges Ergebnis, nicht die urspruenglich
mit n=200 angestrebte hoehere Power. Dies wird hier offengelegt statt
beschoenigt.

## Ruling 2 (Controller): Seeds ueber Welten hinweg nicht unabhaengig — offengelegt

Alle drei Nullwelten (`n1`, `n2`, `n3`) nutzen fuer eine gegebene Replikation
denselben Seed-Wert (z.B. `seed=4` in allen drei Welten). Da jede Welt den
Seed als Ausgangspunkt eines eigenen `np.random.default_rng(seed)`-Stroms
verwendet, sind die resultierenden Panels je Welt zwar intern unabhaengig
voneinander, aber der **Seed-Wert selbst** korreliert ueber Welten hinweg:
"schwierige" oder "guenstige" Seeds koennen system­atisch in mehreren Welten
gleichzeitig auftreten. Die drei Welten liefern damit **keine drei
unabhaengigen Ziehungen** im strengen Sinne.

Im Piloten fielen beide Pipeline-Treffer auf `seed=4` (in n1 und n2, beide
`mom126_long_flat`). Bei 2 Treffern aus 15 Replikationen ist das mit reinem
Zufall gut vereinbar (keine belastbare Evidenz fuer einen systematischen
Effekt), aber der Mechanismus, der eine solche Haeufung erzeugen koennte,
ist real und muss offengelegt werden.

**Entscheidung: Die Seeds bleiben wie versiegelt (kein Offset zwischen
Welten) — die versiegelte Kontroll-Seed-Datei soll literal gelten.** Als
Konsequenz gilt: **die Fehlalarmraten je Welt sind das primaere Ergebnis**
des Hauptlaufs. Eine ueber alle drei Welten gepoolte Zahl ist rein
deskriptiv und muss im Hauptlauf-Bericht ausdruecklich als solche
gekennzeichnet werden, weil die Ziehungen ueber Welten hinweg korreliert
sein koennen und eine Poolung die effektive Stichprobengroesse ueberschaetzen
wuerde.

## Orientierungswerte aus dem Piloten (KEINE Fehlalarm-Aussage)

Konsolen-Output des Piloten: `FA_variant=4,2%`, `FA_pipeline=13,3%`
(gepoolt ueber alle 15 Replikationen / 120 Variantenfaelle). Diese Werte
werden hier **ausschliesslich als Orientierung** benannt. Mit n=5 je Welt
(n=15 gepoolt) hat kein Test die Power, eine kalibrierte von einer
fehlkalibrierten Rate zu unterscheiden; jede Interpretation dieser Zahlen
als Ergebnis waere methodisch nicht gedeckt und wird hier bewusst
unterlassen.

## Zusammenfassung der Entscheidung fuer Task 6

- Hauptlauf: **n = 100 Replikationen je Welt (300 gesamt)**, unveraenderte
  Pipeline (`n_boot=10.000`, keine Modifikation an `stats.py` oder
  `run_trend_baseline_v2.py`).
- Erwartete Laufzeit: ca. 10,1 h (auf Basis des gemessenen bimodalen
  Kostenmodells; die tatsaechliche Trefferquote im Hauptlauf bestimmt den
  genauen Wert, da Treffer den teuren LOO-Zweig ausloesen).
- Statistische Power bei n=100: ca. 80% fuer 5% vs. 10% FA_variant —
  offengelegt, nicht beschoenigt.
- Primaeres Ergebnis des Hauptlaufs: **Fehlalarmraten je Welt getrennt**,
  wegen der ueber Welten korrelierten Seeds. Eine gepoolte Zahl ist rein
  deskriptiv zu kennzeichnen.
- Kontrolllauf-Hochrechnung (3 x 50 = 150): ca. 5,0 h, unveraendert relevant
  fuer eine spaetere Planung.

## Hinweis zu den Betriebsdateien des detachten Piloten-Laufs

Der Pilot wurde detacht gestartet (15 Replikationen ueberschreiten das
10-Minuten-Limit interaktiver Shell-Tools). Dabei entstanden zusaetzlich zu
den in diesem Bericht referenzierten Artefakten die Betriebsdateien
`factor_lab/audit_data/pilot_run.log`, `pilot_run.err.log` und
`pilot_run.pid`. Diese sind reine Prozess-Verwaltungsartefakte des
detachten Starts (Log-Mitschnitt, leerer Stderr-Mitschnitt, PID-Datei) ohne
eigenen Erkenntniswert ueber das hinaus, was bereits strukturiert in
`pilot_records.json` steht, und wurden vor dem Commit entfernt, um das
Repository nicht mit Lauf-Metadaten eines einzelnen lokalen Prozesses zu
verschmutzen.
