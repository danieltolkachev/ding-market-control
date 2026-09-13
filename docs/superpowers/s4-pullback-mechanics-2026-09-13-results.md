# S4 Pullback -- Mechanikbericht und offenes Census-Gate

**Ergebnis in zwei Saetzen:** Die Mechanik des S4-Setups -- Indikatoren, Featurevektor, Eventgenerator und Labelkernel -- ist als vier Module mit 55 Tests umgesetzt und gruen; in dieser Stufe wurde kein Mechanikfehler gefunden, der nicht behoben wurde. **Zwei der fuenf beauftragten Lieferungen fallen negativ aus:** die vorhandenen Daten erfuellen den OHLC- und Corporate-Action-Vertrag **nicht**, und der Event-Census konnte deshalb **nicht** laufen -- ueber die Zahl und Verteilung realer Events ist nach dieser Stufe nichts bekannt.

Stand 2026-09-13. Alle Tests laufen ausschliesslich auf synthetischen, im Code als solche gekennzeichneten Fixtures. Kein Lauf auf echten Daten, kein Modelltraining, keine Parameteroptimierung.

## 1. Auftrag und Abgrenzung

Beauftragt war fuer diese Stufe:

1. pruefen, ob die vorhandenen Daten den OHLC- und Corporate-Action-Vertrag erfuellen;
2. Eventgenerator und Labelmechanik **isoliert** implementieren, mit Tests fuer Timing, Gaps, Gebuehren und mehrdeutige Barrieren;
3. Zahl und Verteilung der Events festhalten -- **falls die Datenqualitaet reicht**;
4. keine Parameteroptimierung, kein LSTM-Training;
5. Datenluecken, Mechanikfehler und die Frage beantworten, ob genug Events fuer die naechste Forschungsstufe existieren.

Erledigt sind (1), (2) und (4). Punkt (1) endet mit einem negativen Befund, und wegen dieses Befundes entfaellt (3) vollstaendig. Punkt (5) ist beantwortet, aber die Antwort auf den letzten Teil lautet: **unbekannt**, nicht "ja" und nicht "nein".

Nicht umgesetzt und ausdruecklich nicht Teil dieses Plans: Census auf echten Daten, Nullaudit, Baseline-Regel, logistische Regression, LSTM, Portfoliointegration.

## 2. Datenbefund: der Vertrag ist nicht erfuellt

Der versiegelte Snapshot `trend_snapshot_a654e3a4d7368cf2.pkl` wurde geoeffnet und geprueft. Er enthaelt **20 Eintraege mit genau zwei Spalten**: `price` fuer 19 ETFs und `rate_pa_pct` fuer IRX. Kein Open, kein High, kein Low, kein Volumen, keine Dividenden, keine Splitfaktoren.

Schwerwiegender als die fehlenden Spalten ist die Natur der vorhandenen: die Preisreihe ist **rueckadjustiert**. `factor_lab/build_trend_snapshot_v2.py:35-36` ruft `yf.download(..., auto_adjust=True, progress=False)`, und die Zahlen bestaetigen es -- SPY steht am 2007-01-03 bei 98,87 gegen real rund 141 tatsaechlich quotierte, TLT bei 48,23 gegen rund 88. Abschnitt 8.1 der Spec verbietet genau das: eine Total-Return-nahe Reihe als handelbare Kursbasis.

Im uebrigen Projekt existiert kein OHLC. Der einzige volumenaehnliche Fund, `market_control_system/data_layer/alpaca_client.py:187-188`, setzt `bid_volume = volume / 2.0` und `ask_volume = volume / 2.0` -- erfundene Mikrostruktur aus der beendeten Minuten-LSTM-Linie, als Quelle unbrauchbar.

**Folge.** S4 prueft die EMA-Beruehrung ueber `L[t]` und `H[t]`; der Labelkernel braucht Roh-Opens fuer den Entry und Kapitalmassnahmen fuer Splits und Dividenden. Beides ist nicht vorhanden. Ein Ersatz durch Close waere nach Abschnitt 2 des Quelldokuments eine **andere Strategie**, keine Datenreparatur. Der Snapshot bleibt unangetastet und weiterhin gueltig fuer das, wofuer er versiegelt wurde -- er ist nur fuer diese Frage die falsche Datenbasis.

## 3. Was die Mechanik jetzt kann

Vier Module, drei Testdateien. Alle neu, kein bestehendes Projektmodul geaendert.

| Modul | Inhalt | Spec-Abschnitt | Tests |
|---|---|---|---|
| `factor_lab/s4_indicators.py` | `compute_indicators(bars)`; `WARMUP=260`, `SEQUENCE=60`; TR, ATR20, sigma5/20, ER20, EMA20/50/200, Uptrend-Flag, Laufsegmentierung | 3, 4 | 17 (mit Features) |
| `factor_lab/s4_features.py` | `build_features(ind, bars, p, t)` -> Sequenz `(60,10)` und Kontext `(4,)` in fester Kanalreihenfolge; `ValueError` statt Imputation | 7 | (in obigen 17) |
| `factor_lab/s4_events.py` | `generate_events(bars, instrument, ind=None)`; Zustandsautomat IDLE/WAIT_CONFIRM; `CONFIRM_MAX=3`, `HOLD_N=10`, `TARGET_Q=1.5`, `COOLDOWN_BARS=10`, `TOUCH_FRACTION=0.25`, `EVENT_VERSION='S4-v1'` | 4, 6 | 13 |
| `factor_lab/s4_labels.py` | `label_event(event, raw_bars, actions, b, f, tie)`; Status `CENSORED`/`DATA_ERROR`/`ACTION_UNSUPPORTED`, Gruende `STOP`/`TARGET`/`GAP_STOP`/`GAP_TARGET`/`TIME` | 5, 8, 9 | 25 |

Testdateien: `factor_lab/tests/test_s4_indicators.py`, `test_s4_events.py`, `test_s4_labels.py`.

**Beobachtete Testzahlen** (Lauf am 2026-09-13, `py -3.12 -m unittest ... -v`):

```
test_s4_indicators   Ran 17 tests   OK
test_s4_events       Ran 13 tests   OK
test_s4_labels       Ran 25 tests   OK
S4 gesamt            Ran 55 tests   OK
```

Zwoelf-Modul-Regression zusammen mit den bestehenden Projektsuiten (`test_daily_comparison`, `test_daily_models`, `test_features_2x2`, `test_models_2x2`, `test_evaluate_2x2`, `test_run_feature_model_2x2`, `test_stats`, `test_costs`, `test_portfolio`):

```
Ran 101 tests in 4.775s   OK
```

Keine Fehler, keine Warnungen. Die bestehenden Suiten sind von den neuen Modulen unberuehrt.

## 4. Geprueftes Verhalten

Namentlich abgedeckt sind:

**Indikatoren und Features.** Kausalitaet aller Indikatorkanaele -- kein Wert bei `t` haengt von Daten nach `t` ab (geprueft ueber `logret`, `tr`, `atr20`, `sigma5`, `sigma20`, `er20`, `ema20`, `ema50`, `ema200`, `up`). EMA-Seed als SMA der ersten n Closes. ATR als einfacher gleitender Mittelwert, ausdruecklich nicht Wilder. Sigma als Stichproben-Standardabweichung der Log-Renditen. Uptrend nur, wenn alle drei Bedingungen erfuellt sind. Laufsegmentierung: `run_length` zaehlt zusammenhaengende gueltige Bars, und der EMA-Zustand **startet nach einem Datenbruch neu**. Nichtpositive und invertierte Bars gelten als ungueltig. Strukturverletzungen -- falsche Spalten, nicht-monotone oder nicht-DatetimeIndex-Indizes -- werden abgewiesen. Exakte Kanalreihenfolge und Formen `(60,10)`/`(4,)` sind gepinnt; fehlende Historie loest `ValueError` aus, statt zu imputieren; die Nullspannen-Konvention einer Bar mit `high == low` ist festgeschrieben.

**Eventgenerator.** Warmup-Grenze: vor `WARMUP=260` wird kein Event emittiert. **Keine Emission auf der Beruehrungsbar selbst** -- der Bestaetigungsbar `t` liegt echt nach dem Setup-Start `p`. Bestaetigungsfenster hoechstens drei Bars. Cooldown von 10 Bars wird eingehalten, ist fest und ergebnisunabhaengig; Labelfenster ueberlappen je Instrument nie. Ein Datenbruch loescht ein anstehendes Setup und sperrt den anschliessenden Cooldown. Events nutzen nur bei `t` verfuegbare Information. Der Generator ist deterministisch. Ein Featurefehler entfernt ein Event **nicht** aus dem Strom: es bleibt mit `sequence=None`, `context=None` und gesetztem `feature_error` erhalten (FEATURE_INVALID), und der Eventstrom ist identisch mit einem Lauf ohne Featurefehler.

**Labelkernel.** Barriereaufloesung fuer Target, Stop und Timeout am letzten gehaltenen Bar. Gaps ueber und unter den Barrieren fuellen **am Open, nicht an der Barriere**. Open wird vor High/Low aufgeloest: eine Bar, die unter dem Stop oeffnet, settelt dort, auch wenn ihre Spanne das Target enthaelt. Der Haircut allein kann den Trade auf seiner eigenen Entry-Bar gap-stoppen -- dieser Fall ist mit durchgerechneter Arithmetik gepinnt. Mehrdeutige Barrieren in beiden Varianten: ein Doppeltreffer wird `ambiguous=True` markiert und ueber `tie` aufgeloest, `STOP_FIRST` (Default) zum Stop, `TARGET_FIRST` zum Target -- eine Modellierungskonvention, keine Behauptung ueber die wahre Intrabar-Reihenfolge. Gebuehren auf beiden Seiten; der Haircut hebt den Entry und senkt den Exit; **Kosten koennen einen Barriere-Gewinn in ein negatives Meta-Label drehen** und tun es im gepinnten Fall. Splits vor dem Entry rechnen `R` auf die Entry-Stueckbasis um, Splits nach dem Entry lassen `net_R` unveraendert. Dividenden nach dem Entry werden gebucht und nicht als Preisgewinn gezaehlt; **am Ex-Tag gekauft entfaellt der Anspruch**. Zensierung statt erfundener Nullrendite: fehlende Entry-Bar, Luecke im Haltefenster, abgeschnittene Historie vor dem Zeitausstieg und ein Haltefenster der Laenge null ergeben alle `CENSORED`. Nichtpositive, nicht-finite oder NaN-Splits und NaN-Dividenden ergeben `ACTION_UNSUPPORTED`; nichtpositives oder NaN-`R` ergibt `DATA_ERROR`.

## 5. Gefundene Mechanikfehler

**Im ausgelieferten Code: keine.** Kein Test deckte einen Rechen-, Timing- oder Kostenfehler in der Mechanik auf, und ich habe keinen gefunden, der offen geblieben waere. Ich formuliere keine kuenstlichen Vorbehalte, um gruendlich zu wirken.

Der Bau war aber nicht reibungslos. Die Reviews der drei Implementierungsschritte fanden zusammen zehn Punkte, alle behoben, alle mit Test abgedeckt. Wert haben sie vor allem als Beleg dafuer, **welche Art von Luecke** hier ueberhaupt auffaellt -- fast durchweg unbelegte Pfade, nicht falsche Formeln:

- *Indikatoren* (3 Befunde, alle Minor): die Kausalitaetspruefung liess `logret` und `ema200` aus; der DatetimeIndex-Vertrag war dokumentiert, aber nicht erzwungen; die Konstante `CONTEXT` war deklariert, ohne etwas zu pruefen. Behoben, ein Test hinzugekommen (16 -> 17).
- *Eventgenerator* (1 Befund, Important): der Test zum FEATURE_INVALID-Pfad durchlief diesen Pfad nie, weil das Warmup-Gate `build_features` im Fixture nie scheitern laesst -- genau das Verhalten, das die Spec als "wenn hier etwas falsch laeuft, dann das" heraushebt, war ohne Abdeckung. Behoben ueber Fehlerinjektion, ein Test hinzugekommen (12 -> 13).
- *Labelkernel* (6 Befunde, alle Minor): ein toter `pandas`-Import; `DATA_ERROR`, die Guard-Clauses, NaN-Split, NaN-Dividende, das Haltefenster der Laenge null und der Gap-Stop auf der Entry-Bar waren saemtlich ungeprueft. Behoben, sechs Tests hinzugekommen (19 -> 25).

Kein Befund erforderte eine Aenderung an einer Indikatorformel, an der Fensterarithmetik, an der Kanalreihenfolge, an den eingefrorenen Konstanten oder an der Tie-Konvention.

## 6. Der Census bleibt offen -- und das ist kein Nebenbefund

Der Census konnte nicht laufen, weil die Daten ihn nicht zulassen (Abschnitt 2). Damit ist **die Zahl und Verteilung realer S4-Events unbekannt**. Das ist keine Formalie und keine blosse Stichprobenfrage, sondern die Machbarkeitsfrage dieses Setups.

Die Renditeherleitung des Nutzers, festgehalten in Abschnitt 11 der Spec, lautet naeherungsweise und ohne Zinseszins:

```
Jahresrendite ~ Trades x Risikoanteil x mittlerer Nettoertrag in R
```

Bei geplant 0,25% Risiko je Trade verlangen 12% jaehrlich also einen mittleren Nettoertrag von `48/Trades` R. Bei `q = 1,5` und Stop bei -1R ist der Mittelwert vor Kosten und Timeouts durch die Trefferquote gedeckelt: `mittleres R ~ 2,5w - 1`. Daraus:

| Trades pro Jahr | noetiger mittlerer Nettoertrag | noetige Trefferquote (vor Kosten) |
|---|---|---|
| 50 | 0,96 R | ~78% |
| 100 | 0,48 R | ~59% |
| 200 | 0,24 R | ~50% |

Break-even liegt bei 40%. Timeouts ziehen den Mittelwert Richtung null und Kosten druecken ihn darunter -- beide verschieben die noetige Quote **nach oben**, nicht nach unten.

Der Census beantwortet also nicht die Frage "haben wir genug Beobachtungen zum Trainieren", sondern die Frage "welche Trefferquote muesste dieses Setup liefern, damit 12% ueberhaupt im Bereich liegen". Liefert die Eventdefinition auf 19 ETFs ueber rund 19 Jahre nur wenige Dutzend Trades pro Jahr, dann liegt die verlangte Quote in einem Bereich, den man vor jedem Modelltraining als unplausibel abhaken kann. Genau deshalb steht der Census im Validierungsvertrag vor dem Modell und nicht daneben.

**Eine Zahl, die man nicht ueberlesen darf:** das synthetische Fixture der Eventgenerator-Tests erzeugte 6 Events auf 600 Bars. Das ist eine Eigenschaft einer handgestimmten Reihe, in der alle 37 Bars ein Pullback erzwungen wird. Es ist **kein** Hinweis auf reale Eventhaeufigkeit und darf nicht als Vorschau auf den Census gelesen werden.

## 7. Was den Census freischalten wuerde

Ein **separat versionierter Roh-OHLC-Snapshot**, gebaut mit `auto_adjust=False` und `actions=True`, damit Open/High/Low/Close unadjustiert und Splits sowie Dividenden als eigene Spalten vorliegen. Er muss vom versiegelten Snapshot **getrennt** gefuehrt werden und ist mit ihm **nicht bar-fuer-bar vergleichbar** -- die beiden Reihen messen Verschiedenes. Er ist ueber die bereits installierte `yfinance`-Abhaengigkeit kostenlos erreichbar.

Diese Arbeit ist **ausdruecklich nicht Teil dieses Plans** und waere getrennt zu beauftragen. Erst danach sind Census, Verteilung und die Machbarkeitsrechnung aus Abschnitt 6 ueberhaupt rechenbar.

## 8. Grenzen

Die Liste aus Abschnitt 13 der Spec gilt unveraendert:

- Der Census ist mit den vorhandenen Daten nicht durchfuehrbar; ohne ihn ist keine Aussage ueber Machbarkeit moeglich.
- Aus 19 hoch korrelierten ETFs entsteht kein gleichwertiger Ersatz fuer 19 unabhaengige Maerkte.
- Rueckblickend ausgewaehltes Universum; historische Universen muessten zeitpunktgetreu sein.
- Diese Stufe hebt weder den gescheiterten Trend-Holdout noch die Momentum- und Minuten-LSTM-Nullbefunde noch den 2x2-Nullbefund auf und beansprucht keinen profitablen Edge.

Dazu kommt aus dieser Stufe:

- **Alle 55 Tests laufen auf synthetischen Fixtures.** Kein Zahlenwert in diesem Bericht stammt aus Marktdaten.
- **Eine korrekte Mechanik ist kein Hinweis auf einen Edge.** Dass Gaps, Ties, Kosten und Kapitalmassnahmen richtig verrechnet werden, sagt exakt nichts darueber aus, ob S4 Geld verdient. Es sagt nur, dass ein spaeterer Befund -- positiv oder negativ -- nicht an einem Rechenfehler in der Mechanik liegen wird. Mehr war hier auch nicht beabsichtigt.
- **Die Konstanten sind Forschungskonventionen, keine optimierten Parameter.** `CONFIRM_MAX=3`, `HOLD_N=10`, `TARGET_Q=1,5`, `COOLDOWN_BARS=10`, `TOUCH_FRACTION=0,25`, `WARMUP=260` wurden vorab festgelegt und in dieser Stufe nicht variiert. Das ist Absicht: eine Variation ohne Census waere Optimierung auf nichts.
- **Der Forschungszaehler steht weiter bei 145** unkorrigierten Vergleichen und wird von dieser Stufe **nicht** erhoeht, weil kein Vergleich gerechnet wurde.
