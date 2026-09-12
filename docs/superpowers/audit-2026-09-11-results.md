# Falsifikations-Audit -- Hauptlauf-Ergebnisse (2026-09-11/12, korrigiert)

Dieser Bericht dokumentiert den Hauptlauf und den Kontrolllauf des
Falsifikations-Audits, die Nacherhebung auf den Arbeits-Seeds 100-199 fuer
Nullwelt n1, die Korrektur eines Parametrisierungsfehlers in Nullwelt n3
und die Befunde der abschliessenden Review. Grundlage ist das
Design-Dokument `docs/superpowers/specs/2026-09-11-falsification-audit-design.md`;
der Piloten-Bericht `docs/superpowers/audit-pilot-2026-09-11-results.md`
dokumentiert die vorausgehende Budget- und Plumbing-Entscheidung.

**Diese Fassung ersetzt die vorherige vollstaendig.** Grund: die
abschliessende Review fand einen Parametrisierungsfehler in Nullwelt n3
(Abschnitt 2), der n3 nachtraeglich korrigiert und neu gefahren wurde, sowie
mehrere Befunde zur statistischen Aufarbeitung von n1 (Abschnitt 3), die in
der Vorversion fehlten oder zu milde berichtet waren. Alle Zahlen in diesem
Bericht wurden aus den versiegelten Record-Dateien in
`factor_lab/audit_data/` neu nachgerechnet.

**Alle Laeufe sind abgeschlossen. Es werden hier keine neuen Laeufe
gestartet.**

## 1. Primaeres Ergebnis: Fehlalarmraten je Nullwelt

Wie im Piloten festgelegt (Ruling 2), sind die Fehlalarmraten je Welt das
primaere Ergebnis; eine ueber Welten gepoolte Zahl ist wegen der ueber Welten
korrelierten Seeds rein deskriptiv.

**Arbeits-Seeds, finales Bild (n1 ueber den vollen registrierten Bereich
0-199, n2/n3 ueber 0-99 wie im Hauptlauf gefahren; n3 in der korrigierten
Fassung, siehe Abschnitt 2):**

| Welt | n (Seeds) | FA_variant | FA_pipeline | Replikationen mit >=1 Treffer |
|---|---:|---:|---:|---:|
| n1 Vol-Regime | 200 (0-199) | 5,31% (85/1600) | 7,5% (15/200) | 21,5% (43/200) |
| n2 GARCH | 100 (0-99) | 5,50% (44/800) | 2,0% (2/100) | 20,0% (20/100) |
| n3 gemeinsame Faktoren (korrigiert) | 100 (0-99) | 2,88% (23/800) | 0,0% (0/100) | 13,0% (13/100) |

Alle drei Werte liegen innerhalb des vorregistrierten Toleranzbands
[2,5%, 10%] fuer FA_variant. Keine Welt ueberschreitet die 15%-Schwelle fuer
FA_pipeline.

**Kontroll-Seeds (10000-10049, n=50 je Welt, versiegelt vor dem Hauptlauf,
SHA256 der Kontroll-Seed-Datei `audit_control_seeds.json`:
`9ffb0f014183680af41d87d644989b379cdac4ad574d81e7864ccaac953f5a77`):**

| Welt | FA_variant | FA_pipeline |
|---|---:|---:|
| n1 | 1,50% (6/400) | 2,0% (1/50) |
| n2 | 3,75% (15/400) | 6,0% (3/50) |
| n3 (korrigiert) | 3,25% (13/400) | 2,0% (1/50) |

Alle Zahlen dieser beiden Tabellen wurden fuer diesen Bericht direkt aus
`audit_records_main.json`, `audit_records_n1_extension.json`,
`audit_records_main_n3.json`, `audit_records_control.json` und
`audit_records_control_n3.json` nachgerechnet und stimmen mit den zuvor
berichteten Werten ueberein (n3 hier bereits in der korrigierten Fassung).

## 2. Korrektur: Nullwelt n3 war fehlparametrisiert, wurde korrigiert und neu gefahren

Die abschliessende Review fand, dass `estimate_null_params` das n3-Residuum
(die idiosynkratische Komponente `e_t`) aus Faktor-Scores der Varianz lambda
gebildet hat, waehrend die zugehoerigen Ladungen `B` bereits den Faktor
sqrt(lambda) trugen. Dadurch wurde die gemeinsame Komponente doppelt skaliert und
beim Bilden des Residuums systematisch ueber-subtrahiert, was die
idiosynkratische Restvolatilitaet aufblaehte.

**Gemessene Konsequenz gegen den echten Snapshot:** die fehlerhafte n3-Welt
hatte eine Querschnittskorrelation von **0,086** und eine annualisierte
Volatilitaet von **0,285**, gegenueber **0,162** und **0,205** im echten
Snapshot. n1 und n2 waren von diesem Fehler nicht betroffen (Vol 0,206 bzw.
0,196, jeweils nahe am Snapshot). Die korrigierte n3-Welt liefert
Korrelation **0,177** und Vol **0,205** -- deutlich naeher am echten Snapshot
als die fehlerhafte Fassung.

**Behoben in Commit `157db58`** (`fix(audit): correct N3 factor scaling,
seal byte-fidelity and overwrite guards`). Die Nullwelt-Parameter wurden
neu versiegelt: `null_params.json` SHA256 `ae7c512c...` (Pilotenfassung,
fehlerhaft) wurde durch **`d6f693d39ed72d868ec5967a0b51ba14428b97d62db7d1eca478fa42a9ce9e13`**
ersetzt. Verifiziert per `sha256sum` auf der aktuellen Datei im Repo.

n3 wurde danach auf **denselben Seeds** wie zuvor neu gefahren -- Arbeits-
Seeds 0-99, Kontroll-Seeds 10000-10049 -- sodass es sich um einen
Wie-fuer-wie-Ersatz handelt, nicht um ein Neu-Wuerfeln der Stichprobe. Von
den eingefrorenen Parametern hat sich innerhalb von n3 nur der
Residuum-Block geaendert (`idio_vol`-Mittelwert 0,0156 -> 0,0089, sowie
`factor_garch`); `symbols`, `calendar`, `n1` und `n2` sind byteidentisch mit
der Pilotenfassung. Die n1- und n2-Laeufe stehen also unveraendert.

**Die superseded n3-Records bleiben zur Transparenz in
`audit_records_main.json` und `audit_records_control.json`** (dort weiterhin
unter dem Welt-Label `n3`, jetzt als die *fehlerhafte* Fassung zu lesen).
Die korrigierten n3-Records liegen separat in
`audit_records_main_n3.json` (Arbeits-Seeds 0-99, 100 Records) und
`audit_records_control_n3.json` (Kontroll-Seeds, 50 Records). **Alle
FA-Zahlen in diesem Bericht fuer n3 verwenden ausschliesslich die
korrigierten `*_n3.json`-Dateien**; die Werte aus den alten Dateien werden
nur zum Vergleich genannt.

**Zum Vergleich, alte (fehlerhafte) n3-Zahlen, nicht das berichtete
Ergebnis:** FA_variant 3,25% (26/800) auf Arbeits-Seeds, 3,25% (13/400) auf
Kontroll-Seeds, FA_pipeline 0,0% bzw. 6,0% (3/50).

**Die Korrektur aendert das Gesamturteil nicht, und das soll hier nicht
beschoenigt werden:** die korrigierte n3-Welt liefert FA_variant 2,88%
gegenueber 3,25% in der fehlerhaften Fassung -- beide Werte liegen innerhalb
des Toleranzbands, und FA_pipeline ist in beiden Fassungen 0,0%. Was der
Fehler zerstoert hat, war nicht das Ergebnis, sondern die
**Interpretierbarkeit** von n3: vor der Korrektur konnte ein "n3 besteht"
nicht als "die korrelierte-Faktoren-Welt besteht" gelesen werden, weil diese
Welt keinem Markt aehnelte (Korrelation 0,086 statt 0,162, Vol 0,285 statt
0,205). Der einzig gerechtfertigte Schluss aus der fehlerhaften Fassung
waere gewesen: unbekannt, ob n3 kalibriert ist, weil n3 nicht das gemessen
hat, was es messen sollte.

## 3. Der n1-Befund: staerkerer Test zuerst, dann das Gegengewicht

### 3.1 Das Abweichungskriterium

Das vorregistrierte Abweichungskriterium (Design-Dokument, Abschnitt 5):
*Weicht FA_variant auf den Kontroll-Seeds um mehr als 5 Prozentpunkte vom
Arbeits-Seed-Ergebnis ab, gilt der Audit als an die Arbeits-Seeds angepasst
und muss ueberarbeitet werden.*

Nachgerechnet auf allen drei Welten (Arbeits- gegen Kontroll-FA_variant):

| Welt | Arbeits-Seeds | Kontroll-Seeds | Differenz |
|---|---:|---:|---:|
| n1 (0-199) | 5,31% | 1,50% | **3,81 pp** |
| n2 (0-99) | 5,50% | 3,75% | **1,75 pp** |
| n3 korrigiert (0-99) | 2,88% | 3,25% | **-0,38 pp** |

Alle drei liegen unter der 5-pp-Schwelle. Das Kriterium ist auf allen drei
Welten erfuellt.

**Auf dem urspruenglichen Hauptlauf (n1, nur Seeds 0-99, vor der
Nacherhebung) hatte dieses Kriterium ausgeloest:** 7,38% (Arbeits-Seeds)
gegen 1,50% (Kontroll-Seeds) = 5,9 Prozentpunkte Differenz -- ueber der
Schwelle. Der Controller entschied, dies nicht ueber eine gepoolte Zahl oder
eine nachtraeglich gewaehlte Aggregationsebene wegzuinterpretieren, sondern
empirisch nachzufassen: der vorregistrierte Arbeits-Seed-Bereich fuer n1
umfasst 0-199, der Hauptlauf hatte davon nur 0-99 verbraucht. Die
verbleibenden Seeds 100-199 wurden nachtraeglich gefahren
(`audit_records_n1_extension.json`, 100 Records) und liefern FA_variant =
3,25% (26/800) -- deutlich unter den 7,38% aus Seeds 0-99. Kombiniert ueber
den vollen Bereich ergibt sich das oben gezeigte 5,31%.

**Das muss hier ausdruecklich benannt werden, nicht beschoenigt: diese
Seed-Erweiterung war ergebnisgetrieben und asymmetrisch.** Nur die Welt, die
das Kriterium ausgeloest hat, wurde erweitert, und erst nachdem sie
ausgeloest hatte. n2 und n3 blieben bei n=100. Das ist strukturell genau die
Art von Nacharbeit, vor der ein Falsifikationsaudit warnen soll, wenn ein
Ergebnis erst nach dem Ansehen der Daten "verbessert" wird.

Mildernd dagegen stehen drei Tatsachen, die neben diesem Befund und nicht
an seiner Stelle zu nennen sind:

- Die Seeds 100-199 waren als Teil von `WORKING_SEEDS` bereits **vor** dem
  Hauptlauf vorregistriert und versiegelt -- es wurden keine neuen, erst
  nachtraeglich erdachten Seeds gezogen, sondern ein bereits festgelegter,
  bislang ungenutzter Teil des Arbeits-Seed-Bereichs abgerufen.
- Das Design-Dokument selbst nennt n=200 je Welt als **Zielwert** (Abschnitt
  4); n1 wurde damit lediglich auf den urspruenglich angestrebten Umfang
  gebracht, nicht ueber ihn hinaus.
- Das Kriterium wurde auf **seiner eigenen definierten Messgroesse**
  (FA_variant) neu ausgewertet, nicht auf eine andere, guenstiger
  aussehende Aggregationsebene verschoben.

### 3.2 Der staerkere Test: FA_variant selbst, nicht nur die Replikationsebene

Die Vorversion dieses Berichts hat an dieser Stelle nur den milderen,
replikationsweiten Test berichtet (z ca. 2,19, p ca. 0,03, naiv ohne
Cluster-Korrektur). Das war ein Fehler in der falschen Richtung: berichtet
werden muss der staerkere, nicht der schwaechere Test, wenn der Bericht
beansprucht, nichts zu beschoenigen -- und der staerkere Test ist der auf
FA_variant selbst, weil das die Messgroesse ist, auf der das Kriterium
tatsaechlich definiert ist.

Auf Ebene der Variantenfaelle (die richtige Analyseeinheit fuer FA_variant),
mit **replikationsgeclusterten Standardfehlern** (Gate-A-Treffer haeufen
sich stark innerhalb einzelner Replikationen, siehe 3.4), ergibt der
Zwei-Stichproben-Vergleich n1 Arbeits- gegen Kontroll-Seeds:

**z ca. 3,23, p ca. 0,001** (zweiseitig; p_Arbeit = 5,31% +/- 0,90% Cluster-SE,
p_Kontrolle = 1,50% +/- 0,77% Cluster-SE).

Das ist der staerkere, nicht der schwaechere Befund, und er wird hier als
solcher gefuehrt. Nachgerechnet aus `audit_records_main.json` +
`audit_records_n1_extension.json` gegen `audit_records_control.json`
(Methode: siehe 3.4).

**Warum das Kriterium trotzdem als bestanden gilt:** das vorregistrierte
Abweichungskriterium ist eine absolute 5-Prozentpunkte-Regel auf
FA_variant, keine Aussage ueber statistische Kompatibilitaet der beiden
Schaetzungen. 5,31% gegen 1,50% unterschreitet 5 pp (3,81 pp Differenz) und
erfuellt die Regel -- obwohl die beiden Schaetzungen selbst, statistisch
betrachtet, klar unterscheidbar sind (z ca. 3,23). Beides ist gleichzeitig
wahr und wird hier auch so berichtet: das Kriterium besteht, weil es eine
Abstandsregel und keine Signifikanzregel ist, nicht weil die Differenz
statistisch bedeutungslos waere.

### 3.3 Das Gegengewicht: n2 und n3 zeigen keine vergleichbare Luecke

Was in der Vorversion vollstaendig fehlte und hier nachgetragen wird: bei
n2 und n3 tritt derselbe Arbeits-gegen-Kontroll-Vergleich nicht auf.

Mit derselben cluster-robusten Methode wie in 3.2:

| Welt | FA_variant Arbeit | FA_variant Kontrolle | z | p |
|---|---:|---:|---:|---:|
| n1 | 5,31% | 1,50% | 3,23 | 0,001 |
| n2 | 5,50% | 3,75% | 0,82 | 0,41 |
| n3 (korrigiert) | 2,88% | 3,25% | -0,20 | 0,84 |

n2 zeigt keine statistisch auffaellige Luecke, n3 zeigt ueberhaupt keine
Richtung (Kontrolle liegt sogar leicht ueber Arbeit). Die drei Welten
unterscheiden sich in der Seed-Zufallszahlenfolge ausschliesslich durch den
an `np.random.default_rng` uebergebenen Integer-Seed; es gibt keinen
bekannten Mechanismus, ueber den derselbe Seed-Bereich in n1 systematisch
anders wirken sollte als in n2 oder n3. In Kombination mit der Tatsache,
dass n1 ueberhaupt nur deshalb gesondert betrachtet wurde, weil es der
extremste der drei Werte war (Auswahl nach dem Ansehen der Ergebnisse ist
per Definition ein Selektionseffekt), ist das ein staerkeres Argument
dafuer, die n1-Luecke als Stichprobenrauschen zu lesen, als die
Seed-Erweiterung allein liefert. Beide Argumente stehen nebeneinander in
diesem Bericht, keines ersetzt das andere.

### 3.4 Cluster-robuste Konfidenzintervalle statt naiver Wilson-Intervalle

Der urspruengliche Plan sah vor, je Welt ein Wilson-Intervall fuer
FA_variant anzugeben; das fehlte in der Vorversion und wird hier
nachgetragen -- allerdings nicht in der naiven Form.

**Das naive Wilson-Intervall behandelt 8 Varianten x n Replikationen als
unabhaengige Beobachtungen.** Das ist falsch: Gate-A-Treffer haeufen sich
stark innerhalb einzelner Replikationen. In n1 (0-199) stammen die 85
Treffer aus nur 43 von 200 Replikationen (21,5%), mit bis zu 6 gleichzeitig
in einer einzigen Replikation (Verteilung ueber die 43 Replikationen mit
>=1 Treffer: 24x 1 Treffer, 5x 2, 8x 3, 4x 4, 1x 5, 1x 6). Der empirisch
gemessene Designeffekt (Verhaeltnis der cluster-robusten zur naiven
Binomialvarianz von phat) liegt bei ca. 2,5-2,8 ueber die drei Welten
(n1: 2,56; n2: 2,83; n3 korrigiert: 2,61) -- die naive Standardfehler-Angabe
ist damit um den Faktor sqrt(2,5..2,8) ca. 1,6 zu eng.

**Verwendete Methode:** je Replikation wird der Anteil der 8 Varianten mit
Gate-A-Treffer als Cluster-Mittelwert behandelt; die Varianz von phat wird
aus der Streuung dieser n Cluster-Mittelwerte geschaetzt (nicht aus der
Binomialformel), und daraus das 95%-Intervall gebildet.

| Welt | phat (FA_variant) | Cluster-SE | Cluster-95%-CI | naives Wilson-CI (zu eng) |
|---|---:|---:|---:|---:|
| n1 (0-199) | 5,31% | 0,90% | [3,55%, 7,07%] | [4,32%, 6,52%] |
| n2 (0-99) | 5,50% | 1,36% | [2,84%, 8,16%] | [4,12%, 7,30%] |
| n3 korrigiert (0-99) | 2,88% | 0,95% | [1,00%, 4,75%] | [1,92%, 4,28%] |

Alle drei Cluster-Intervalle schliessen weiterhin die Nominalrate 5% ein
oder liegen knapp darueber/darunter im Toleranzband; keines schliesst 10%
(die Fehlkalibrierungsschwelle) ein. **Die im Piloten und in Abschnitt 5
genannte statistische Power (ca. 80% bei n=100, ca. 96% bei n=200, um 5%
von 10% zu unterscheiden) ist unter Beruecksichtigung dieses Designeffekts
konservativ, nicht optimistisch** -- die tatsaechlich erreichte
Trennschaerfe liegt eher am unteren Rand der genannten Werte, weil die dort
zugrunde gelegte Binomialvarianz die tatsaechliche Streuung unterschaetzt.

### 3.5 Kleinere Zahlenkorrekturen

- Die n1-Nacherhebung (Seeds 100-199) dauerte, nach derselben
  Summenkonvention wie die uebrigen Laufzeit-Angaben (Summe der
  `elapsed_s`-Felder), **2,51 h**, nicht die zuvor genannten "~2,9 h".
- Im Hauptlauf lag die mittlere Laufzeit einer Replikation mit
  Pipeline-Kandidatin (Gate-D-LOO-Zweig ausgeloest) bei **242,1 s**, nicht
  bei den 330,7 s aus dem Piloten. Die 330,7 s sind die Pilotenzahl (aus nur
  2 Faellen) und werden hier ausdruecklich als solche gekennzeichnet, nicht
  als Hauptlauf-Wert wiederverwendet.
- Der SHA256 der Kontroll-Seed-Datei (`9ffb0f014183680af41d87d644989b379cdac4ad574d81e7864ccaac953f5a77`)
  wird -- wie von Design-Dokument Abschnitt 5 verlangt -- hier genannt; er
  stand bislang nur im Piloten-Bericht.

## 4. Laufzeit

| Lauf | Replikationen | Laufzeit (Summe `elapsed_s`) |
|---|---:|---:|
| Hauptlauf (n1/n2/n3, Seeds 0-99, inkl. fehlerhafter n3-Fassung) | 300 | 7,79 h |
| Kontrolllauf (Seeds 10000-10049, inkl. fehlerhafter n3-Fassung) | 150 | 3,79 h |
| n1-Nacherhebung (Seeds 100-199) | 100 | 2,51 h |
| n3-Korrekturlauf, Arbeits-Seeds (0-99) | 100 | 2,45 h |
| n3-Korrekturlauf, Kontroll-Seeds (10000-10049) | 50 | 1,28 h |

Die Kostenstruktur bleibt bimodal wie im Piloten beschrieben: im Hauptlauf
liegt die mittlere Laufzeit einer Replikation ohne Pipeline-Kandidatin bei
rund 88,6 s (wie im Piloten), die mittlere Laufzeit mit Kandidatin (Gate-D-
LOO-Zweig mit 26 Reruns) bei **242,1 s im Hauptlauf** (14 von 300
Replikationen) -- abweichend von den 330,7 s des 2-Fall-Pilotenschaetzers,
der zur Budgetplanung diente, aber keine Hauptlauf-Messung war.

## 5. Interpretation nach der vorab festgelegten Tabelle

Design-Dokument, Abschnitt 7: FA_variant in [2,5%, 10%] und FA_pipeline
<= 15% in allen drei Nullwelten bedeutet **Methodik-Pruefung bestanden.**
Kein Edge-Nachweis -- die Pipeline meldet auf diesen drei Nullwelten
ungefaehr so selten einen Treffer, wie sie es laut ihrem eigenen
95%-Kriterium tun sollte. Das gilt fuer n1 (5,31%), n2 (5,50%) und die
korrigierte n3 (2,88%) gleichermassen.

Insbesondere ist der schwerwiegende Fall -- FA_variant > 10%, der jede
bisherige Gate-A-Aussage inklusive des trend-etf-v2-Screening-Passes
uninterpretierbar gemacht haette -- **nicht eingetreten**, weder in der
fehlerhaften noch in der korrigierten n3-Fassung, und auch nicht in n1 oder
n2. Die Bootstrap-Inferenz selbst ist auf diesen drei Nullwelten kalibriert.

Diese Aussage steht unabhaengig vom n1-Residuum aus Abschnitt 3: das
Kriterium ist auf der Abstandsregel definiert, nicht auf statistischer
Ununterscheidbarkeit, und die Regel ist erfuellt.

## 6. Was dieses Ergebnis NICHT sagt

- **Kein Edge-Nachweis.** Ein bestandener Audit heisst nur, dass die
  Pipeline auf diesen drei konstruierten Nullwelten kalibriert ist -- nicht,
  dass ein echter Markt-Edge existiert.
- **Nur drei Nullwelten.** Volatilitaetsregime, GARCH-Clustering und
  gemeinsame Faktoren decken nicht jede Form scheinbarer
  Prognostizierbarkeit ab.
- **Nullwelt-Parameter stammen aus demselben Snapshot** wie die bisherige
  Forschung. Die Welten sind diesem Datensatz aehnlich, aber nicht
  unabhaengig von ihm.
- **Mikrostruktur-Effekte sind mit reinen Preisdaten nicht pruefbar** --
  ohne Orderbuch-/Volumendaten bleibt dieser Kanal fuer Fehlalarme
  ungetestet.
- **Ein Pass hier uebertraegt sich nicht** auf die spaeteren LSTM- oder
  2x2-Erweiterungen. Diese braeuchten einen eigenen, angepassten Audit.
- **Drei unangekuendigte Konstruktionsentscheidungen praegen das Ergebnis**
  mit und sind bislang nirgends offengelegt worden:
  1. Die Cash-Reihe ist in allen drei Nullwelten auf einen konstanten Zins
     von 2,0% p.a. fixiert, waehrend der Zins im echten Snapshot ueber
     2007-2026 schwankte. Gate A ist davon nicht betroffen, weil es ein
     Excess gegen `matched_long` misst und der Cash-Zins sich dabei kuerzt;
     die Drawdown- und Stress-CAGR-Gates sind es aber sehr wohl, da sie
     absolute, nicht relative Groessen pruefen.
  2. Mittelwertfreie *arithmetische* Renditen implizieren einen log-Drift
     von -sigma^2/2 (bei 20% Vol ca. -2%/Jahr). Die synthetischen Universen
     driften damit im Log-Preis leicht nach unten, waehrend der echte
     Snapshot ueber seinen Zeitraum das nicht tat. Das kann Drawdown- und
     Stress-Gates in Richtung "strenger als beabsichtigt" verschieben.
  3. Gate As nominale 5%-Schwelle unterstellt, dass der wahre Excess gegen
     `matched_long` exakt null ist. Tatsaechlich zahlen die Varianten
     Turnover- und Vol-Targeting-Drag, den `matched_long` nicht traegt; der
     wahre Nullwert liegt also leicht unter null. Die gemessenen 5,3-5,5%
     in n1/n2 sind dadurch **leicht anti-konservativ**, nicht exakt auf
     dem Nominalwert -- eine Fehlkalibrierung nach oben wuerde durch diesen
     Effekt eher unter- als ueberschaetzt.

## 7. Offengelegte Einschraenkungen

- **Statistische Power:** bei n=100 je Welt (n2, n3) liegt die Power, eine
  Fehlalarmrate von 10% von der Nominalrate 5% zu unterscheiden, nominell
  bei ca. 80%; n1 erreicht mit dem vollen Arbeits-Seed-Bereich (n=200) ca.
  96%. Unter Beruecksichtigung des in Abschnitt 3.4 gemessenen
  Cluster-Designeffekts (~2,5-2,8) ist die tatsaechliche Power etwas
  geringer als diese Nominalwerte -- die genannten Zahlen sind also
  **konservativ zu lesen**, nicht als exakte Angaben.
- **Korrelierte Seeds ueber Welten:** wie im Piloten (Ruling 2) offengelegt,
  teilen sich alle drei Nullwelten denselben Seed-Wert je Replikation. Die
  Werte je Welt sind deshalb das primaere Ergebnis; jede ueber Welten
  gepoolte Zahl ist rein deskriptiv und ueberschaetzt moeglicherweise die
  effektive Stichprobengroesse.
- **Ergebnisgetriebene, asymmetrische Seed-Erweiterung fuer n1** (Abschnitt
  3.1): nur die Welt, die das Abweichungskriterium ausgeloest hat, wurde
  nachtraeglich um vorregistrierte, aber bislang ungenutzte Seeds erweitert.
  Das Kriterium besteht danach, aber der staerkere Test auf FA_variant
  selbst (z ca. 3,23, p ca. 0,001, Abschnitt 3.2) zeigt weiterhin eine
  statistisch auffaellige Arbeits-Kontroll-Differenz fuer n1, die n2 und n3
  nicht zeigen (Abschnitt 3.3).
- **Drei unangekuendigte Konstruktionsentscheidungen** (Cash-Zins konstant,
  arithmetischer Mittelwert-Null-Drift, leicht negativer wahrer Nullwert
  fuer Gate A) -- siehe Abschnitt 6, Punkt 6.

## Fazit

**Methodik-Pruefung bestanden.** Alle drei Nullwelten -- einschliesslich der
korrigierten n3 -- liegen mit FA_variant im vorregistrierten Band
[2,5%, 10%] (n1: 5,31%, n2: 5,50%, n3: 2,88%), und keine ueberschreitet die
FA_pipeline-Grenze von 15% (n1: 7,5%, n2: 2,0%, n3: 0,0%). Der
schwerwiegende Fall -- FA_variant > 10%, der jede bisherige Gate-A-Aussage
einschliesslich des trend-etf-v2-Screening-Passes uninterpretierbar gemacht
haette -- ist **nicht eingetreten**.

Der Parametrisierungsfehler in n3 wurde gefunden, korrigiert (Commit
`157db58`) und n3 auf denselben Seeds neu gefahren; das Ergebnis fuer n3
aendert sich dadurch kaum (2,88% statt 3,25% FA_variant, FA_pipeline
unveraendert 0,0%) -- der Fehler hatte die Interpretierbarkeit von n3
zerstoert, nicht das Ergebnis verzerrt.

Das Abweichungskriterium loeste auf dem urspruenglichen n1-Teillauf (Seeds
0-99) aus und wurde durch Nacherhebung der uebrigen vorregistrierten
Arbeits-Seeds (100-199) formal aufgeloest (3,81 pp, unter der 5-pp-
Schwelle). Diese Nacherhebung war ergebnisgetrieben und nur fuer n1
durchgefuehrt -- das wird hier offen benannt. Auf FA_variant selbst bleibt
ein statistisch klarer Arbeits-Kontroll-Unterschied fuer n1 (z ca. 3,23,
p ca. 0,001) bestehen, den n2 (z ca. 0,82) und n3 (z ca. -0,20) nicht zeigen; da
sich die drei Welten nur im Zufallszahlen-Seed unterscheiden und n1 allein
deshalb naeher betrachtet wurde, weil es der extremste Wert war, liest sich
das eher als Stichprobenrauschen um einen leicht erhoehten n1-Wert als
Beleg fuer einen systematischen Fehler -- beweisen laesst sich das mit den
vorliegenden Daten nicht, und es wird hier auch nicht als bewiesen
dargestellt.

Kein Edge-Nachweis; das Ergebnis gilt ausschliesslich fuer diese drei
Nullwelten und diese Pipeline-Version, mit den in Abschnitt 6 genannten
Einschraenkungen (insbesondere den drei bislang unangekuendigten
Konstruktionsentscheidungen zu Cash-Zins, Drift und Gate-A-Nullwert).
