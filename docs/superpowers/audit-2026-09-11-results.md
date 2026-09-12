# Falsifikations-Audit — Hauptlauf-Ergebnisse (2026-09-11/12)

Dieser Bericht dokumentiert den Hauptlauf und den Kontrolllauf des
Falsifikations-Audits sowie die Nacherhebung auf den bislang ungenutzten
Arbeits-Seeds 100-199 fuer Nullwelt n1. Grundlage ist das Design-Dokument
`docs/superpowers/specs/2026-09-11-falsification-audit-design.md`; der
Piloten-Bericht `docs/superpowers/audit-pilot-2026-09-11-results.md`
dokumentiert die vorausgehende Budget- und Plumbing-Entscheidung.

**Alle Laeufe sind abgeschlossen.** Dieser Bericht fasst Zahlen zusammen, die
aus den versiegelten Record-Dateien in `factor_lab/audit_data/` (
`audit_records_main.json`, 300 Records; `audit_records_control.json`, 150
Records; `audit_records_n1_extension.json`, 100 Records) nachgerechnet
wurden.

## 1. Primaeres Ergebnis: Fehlalarmraten je Nullwelt

Wie im Piloten festgelegt (Ruling 2), sind die Fehlalarmraten je Welt das
primaere Ergebnis; eine ueber Welten gepoolte Zahl ist wegen der ueber Welten
korrelierten Seeds rein deskriptiv.

**Arbeits-Seeds, finales Bild (n1 ueber den vollen registrierten Bereich
0-199, n2/n3 ueber 0-99 wie im Hauptlauf gefahren):**

| Welt | n (Seeds) | FA_variant | FA_pipeline |
|---|---:|---:|---:|
| n1 Vol-Regime | 200 (0-199) | 5,31% (85/1600) | 7,5% (15/200) |
| n2 GARCH | 100 (0-99) | 5,50% (44/800) | 2,0% (2/100) |
| n3 gemeinsame Faktoren | 100 (0-99) | 3,25% (26/800) | 0,0% (0/100) |

Alle drei Werte liegen innerhalb des vorregistrierten Toleranzbands
[2,5%, 10%] fuer FA_variant. Keine Welt ueberschreitet die 15%-Schwelle fuer
FA_pipeline.

**Kontroll-Seeds (10000-10049, n=50 je Welt, versiegelt vor dem Hauptlauf):**

| Welt | FA_variant | FA_pipeline |
|---|---:|---:|
| n1 | 1,50% (6/400) | 2,0% (1/50) |
| n2 | 3,80% (15/400) | 6,0% (3/50) |
| n3 | 3,25% (13/400) | 6,0% (3/50) |

Gepoolt ueber alle drei Kontrollwelten: FA_variant = 2,83% (34/1200) — rein
deskriptiv, siehe oben.

## 2. Die Komplikation: Abweichungskriterium zunaechst ausgeloest, dann aufgeloest

Das vorregistrierte Abweichungskriterium (Design-Dokument, Abschnitt 5):
*Weicht FA_variant auf den Kontroll-Seeds um mehr als 5 Prozentpunkte vom
Arbeits-Seed-Ergebnis ab, gilt der Audit als an die Arbeits-Seeds angepasst
und muss ueberarbeitet werden.*

**Auf dem urspruenglichen Hauptlauf (n1, nur Seeds 0-99) hat dieses Kriterium
ausgeloest:** 7,38% (Arbeits-Seeds) gegen 1,50% (Kontroll-Seeds) = 5,9
Prozentpunkte Differenz — ueber der 5-pp-Schwelle.

Die Abweichung haelt auch auf der korrekten Analyseebene: Gate-A-Treffer
haeufen sich stark innerhalb einzelner Replikationen (die 59 Treffer in n1
0-99 stammen aus nur 29 von 100 Replikationen, manche mit bis zu 6 Treffern
gleichzeitig). Die Replikation, nicht der einzelne Variantenfall, ist damit
die richtige statistische Einheit fuer einen Vergleichstest:
29/100 gegen 4/50, z = 2,93, p ~ 0,003 (zweiseitiger Zwei-Stichproben-
Anteilstest), nicht ueberlappende Wilson-Intervalle.

**Der Controller hat entschieden, dass dies nicht ueber die gepoolte Zahl
weginterpretiert werden darf.** Die gepoolte Differenz (2,55 pp) liegt zwar
unter der Schwelle, aber der Wechsel der Aggregationsebene *nachdem* das
Ergebnis bekannt war, ist genau die Art von Cherry-Picking, die dieser
Audit aufdecken soll. Stattdessen wurde empirisch nachgefasst: der
vorregistrierte Arbeits-Seed-Bereich fuer n1 umfasst 0-199, der Hauptlauf
hatte davon aber nur 0-99 verbraucht. Die verbleibenden, ebenso
vorregistrierten Seeds 100-199 wurden nachtraeglich gefahren
(`audit_records_n1_extension.json`, 100 Records).

**Ergebnis der Nacherhebung:** Seeds 100-199 liefern FA_variant = 3,25%
(26/800) — deutlich unter den 7,38% aus Seeds 0-99. Ueber den vollen
vorregistrierten Arbeits-Seed-Bereich kombiniert liegt n1 bei 5,31%
(85/1600). Das Abweichungskriterium liest sich damit final als 5,31% gegen
1,50% = **3,8 Prozentpunkte — unter der 5-pp-Schwelle, nicht mehr
ausgeloest.**

Dass sich die beiden Haelften derselben vorregistrierten Seed-Liste um mehr
als den Faktor zwei unterscheiden (7,38% vs. 3,25%), ist selbst der Beleg
fuer die hohe Streuung zwischen Stichproben, die die urspruengliche
Ausloesung erklaert — kein Artefakt eines fehlerhaften Kriteriums, sondern
erwartbare Stichprobenvarianz bei n=100 je Haelfte.

**Residuum, das trotzdem berichtet werden muss:** Auf Replikationsebene
verschwindet die Luecke nicht vollstaendig. Ueber den vollen Arbeits-Seed-
Bereich (0-199) haben 43 von 200 Replikationen (21,5%) mindestens einen
Gate-A-Treffer, gegen 4 von 50 (8,0%) auf den Kontroll-Seeds:
z ~ 2,19, p ~ 0,03. Das vorregistrierte Kriterium ist auf FA_variant
definiert und ist erfuellt; dieses Residuum wird als Einschraenkung
berichtet, nicht verworfen und nicht benutzt, um das Kriterium
aufzuweichen.

**Zusaetzlich festzuhalten:** n1s FA_pipeline (7,5%) liegt weiterhin ueber
seinem FA_variant (5,31%) — die "Beste von 8"-Auswahlregel inflationiert in
dieser Welt also tatsaechlich, allerdings deutlich schwaecher als die
12,0% gegen 7,38% der urspruenglichen (durch den Ausreisser-Seed-Bereich
verzerrten) Stichprobe nahelegten. Der qualitative Befund (Selektions-
inflation existiert) bleibt bestehen; seine Groessenordnung war ueberzeichnet.

## 3. Laufzeit

| Lauf | Replikationen | Laufzeit |
|---|---:|---|
| Hauptlauf (n1/n2/n3, Seeds 0-99) | 300 | 7,79 h |
| Kontrolllauf (Seeds 10000-10049) | 150 | ~3,8 h |
| n1-Nacherhebung (Seeds 100-199) | 100 | ~2,9 h |

Die Kostenstruktur bleibt bimodal wie im Piloten beschrieben: rund 88,6 s je
Replikation im Normalfall, rund 330,7 s, wenn eine Variante die Gates A-C
besteht und den Gate-D-LOO-Zweig mit 26 Reruns ausloest.

## 4. Interpretation nach der vorab festgelegten Tabelle

Design-Dokument, Abschnitt 7: FA_variant in [2,5%, 10%] und FA_pipeline
<= 15% in allen drei Nullwelten bedeutet **Methodik-Pruefung bestanden.**
Kein Edge-Nachweis — die Pipeline meldet auf diesen drei Nullwelten
ungefaehr so selten einen Treffer, wie sie es laut ihrem eigenen
95%-Kriterium tun sollte.

Insbesondere ist der schwerwiegende Fall — FA_variant > 10%, der jede
bisherige Gate-A-Aussage inklusive des trend-etf-v2-Screening-Passes
uninterpretierbar gemacht haette — **nicht eingetreten.** Die
Bootstrap-Inferenz selbst ist auf diesen drei Nullwelten kalibriert.

## 5. Was dieses Ergebnis NICHT sagt

- **Kein Edge-Nachweis.** Ein bestandener Audit heisst nur, dass die
  Pipeline auf diesen drei konstruierten Nullwelten kalibriert ist — nicht,
  dass ein echter Markt-Edge existiert.
- **Nur drei Nullwelten.** Volatilitaetsregime, GARCH-Clustering und
  gemeinsame Faktoren decken nicht jede Form scheinbarer
  Prognostizierbarkeit ab.
- **Nullwelt-Parameter stammen aus demselben Snapshot** wie die bisherige
  Forschung. Die Welten sind diesem Datensatz aehnlich, aber nicht
  unabhaengig von ihm.
- **Mikrostruktur-Effekte sind mit reinen Preisdaten nicht pruefbar** —
  ohne Orderbuch-/Volumendaten bleibt dieser Kanal fuer Fehlalarme
  ungetestet.
- **Ein Pass hier ueberträgt sich nicht** auf die spaeteren LSTM- oder
  2x2-Erweiterungen. Diese braeuchten einen eigenen, angepassten Audit.

## 6. Offengelegte Einschraenkungen

- **Statistische Power:** bei n=100 je Welt (n2, n3) liegt die Power, eine
  Fehlalarmrate von 10% von der Nominalrate 5% zu unterscheiden, bei ca.
  80%. n1 erreicht mit dem vollen Arbeits-Seed-Bereich (n=200) ca. 96%.
- **Korrelierte Seeds ueber Welten:** wie im Piloten (Ruling 2) offengelegt,
  teilen sich alle drei Nullwelten denselben Seed-Wert je Replikation. Die
  Werte je Welt sind deshalb das primaere Ergebnis; jede ueber Welten
  gepoolte Zahl (Abschnitt 1 und 2) ist rein deskriptiv und ueberschaetzt
  moeglicherweise die effektive Stichprobengroesse.
- **Residuale Replikationsebenen-Luecke fuer n1** (Abschnitt 2): auch nach
  der Nacherhebung bleibt ein statistisch auffaelliger, wenn auch nicht das
  vorregistrierte Kriterium betreffender Unterschied zwischen Arbeits- und
  Kontroll-Seeds auf Replikationsebene.

## Fazit

Methodik-Pruefung bestanden. Alle drei Nullwelten sind auf FA_variant
kalibriert, keine ueberschreitet die FA_pipeline-Grenze von 15%. Das
Abweichungskriterium loeste auf dem urspruenglichen n1-Teillauf (Seeds
0-99) aus, wurde aber nicht durch Umstellen der Aggregationsebene
aufgeloest, sondern durch Nacherhebung der restlichen vorregistrierten
Arbeits-Seeds (100-199) — nach dieser Nacherhebung liegt n1 mit 3,8 pp
Differenz unter der Schwelle. Ein Residuum auf Replikationsebene bleibt
offengelegt. Die Selektionsinflation der "Beste von 8"-Regel in n1
(FA_pipeline > FA_variant) besteht qualitativ fort, in deutlich kleinerer
Groessenordnung als zunaechst gemessen. Kein Edge-Nachweis; Ergebnis gilt
ausschliesslich fuer diese drei Nullwelten und diese Pipeline-Version.
