# S4 Pullback — Eventgenerator und Label-Mechanik

**Datum:** 2026-09-13
**Status:** Spezifikation zur Umsetzung der **Mechanik**. Kein Backtest, keine bestaetigte Strategie, keine Parameteroptimierung.
**Herkunft:** Transkription der normativen Teile des Nutzerdokuments „Chan und Grimes: testbare Trading-Events fuer ein LSTM-Meta-Modell" vom 2026-09-13. Abschnitt 2 und Abschnitt 8 sind hier vollstaendig wiedergegeben, weil die Implementierung sie woertlich braucht; die uebrigen Abschnitte sind als Kontext zusammengefasst.

## 0. Kennzeichnung und Geltungsbereich

Das Quelldokument kennzeichnet durchgaengig **A = Literatur**, **B = eigene quantitative Interpretation**, **C = ungetestete Hypothese**. Alles in dieser Spec unter Abschnitt 2 und 8 ist **B**: eigene Entwuerfe, keine Reproduktion fertiger Buchstrategien. Die Zahlenwerte sind ausdruecklich Forschungskonventionen, keine optimierten Parameter und keine angeblich wortgetreuen Buchregeln.

Quellen (gezielt geprueft, nicht erneut zu lesen): Chan, *Algorithmic Trading*, 2013, Kap. 1, 2, 3, 6, 7. Grimes, *The Art and Science of Technical Analysis*, 2012, Kap. 3–6.

**Zwei Korrekturen am Buchmaterial**, die das Quelldokument festhaelt und die hier gelten: Chan interpretiert auf S. 46 einen p-Wert als Wahrscheinlichkeit der Nullhypothese — statistisch falsch. Auf S. 156 steht `positions.*(op-cl)./op`, was fuer einen Long das falsche Vorzeichen hat; der Preisgewinn ist `close-open`.

## 1. Was diese Spec umsetzt — und was nicht

**Umgesetzt wird:** die Mechanik aus Abschnitt 8 — Indikatoren, Featurevektor, Eventgenerator-Zustandsautomat und Labelkernel — isoliert und gegen synthetische OHLC-Fixtures geprueft.

**Nicht umgesetzt wird:** Census auf echten Daten, Nullaudit, Baseline-Regel, logistische Regression, LSTM-Training, Parameteroptimierung, Portfoliointegration. Das sind spaetere, getrennt zu beauftragende Stufen.

**Erwartungshaltung, vorab notiert:** Diese Spec liefert keinen Nachweis eines Edge. Sie liefert einen pruefbaren Mechanismus, dessen Fehler sich billig zeigen sollen, bevor Daten oder Modelle Geld und Budget kosten.

## 2. Datenbefund (verifiziert am 2026-09-13)

Der versiegelte Snapshot `trend_snapshot_a654e3a4d7368cf2.pkl` enthaelt **20 Eintraege mit genau zwei Spalten**: `price` (19 ETFs) und `rate_pa_pct` (IRX). Kein Open, High, Low, Volumen, keine Dividenden, keine Splitfaktoren.

Schwerwiegender: die Preisreihe ist **rueckadjustiert**. `build_trend_snapshot_v2.py` ruft `yf.download(..., auto_adjust=True)`, und die Zahlen bestaetigen es — SPY steht am 2007-01-03 bei 98,87 gegen real rund 141, TLT bei 48,23 gegen real rund 88. Abschnitt 8.1 verbietet genau das: eine Total-Return-nahe Reihe als handelbaren OHLC-Kurs.

Im uebrigen Projekt existiert kein OHLC. Der einzige Volumenfund, `market_control_system/data_layer/alpaca_client.py:187-188`, setzt `bid_volume = volume/2` und `ask_volume = volume/2` — erfundene Mikrostruktur aus der beendeten Minuten-LSTM-Linie, als Quelle unbrauchbar.

**Folge:** Der Census ist auf vorhandenen Daten nicht durchfuehrbar. S4 prueft die EMA-Beruehrung ueber `L[t]` und `H[t]`, der Labelkernel braucht Roh-Opens und Kapitalmassnahmen. Ein Ersatz durch Close waere nach Abschnitt 2 des Quelldokuments eine andere Strategie, keine Datenreparatur. Die Mechanik wird deshalb ausschliesslich gegen **synthetische, eindeutig gekennzeichnete** OHLC-Fixtures geprueft.

## 3. Zeit, Indikatoren, Datenvertrag (Abschnitt 2 des Quelldokuments)

`t` bezeichnet eine vollstaendig abgeschlossene Bar. Alle Intervalle `[a:b]` sind **einschliesslich beider Endpunkte**. Standardannahme: Tagesbars, Long-Seite, Signal nach Close, Order fruehestens danach. Short-Spiegelungen sind getrennte Experimente mit Leihe, Finanzierung und Ausfuehrbarkeit; sie sind kein kostenlos verdoppeltes Sample.

```text
O[t], H[t], L[t], C[t], V[t] = Open, High, Low, Close, echtes Handelsvolumen
r[t] = log(C[t] / C[t-1])
TR[t] = max(H[t]-L[t], abs(H[t]-C[t-1]), abs(L[t]-C[t-1]))
ATR_n[t] = mean(TR[t-n+1:t])                 # hier SMA, NICHT Wilder-ATR
sigma_n[t] = sample_std(r[t-n+1:t], ddof=1)
EMA_n: alpha=2/(n+1), Initialwert=SMA der ersten n gueltigen Closes
U_N[t] = max(H[t-N:t-1])                    # aktuelle Bar ausgeschlossen
D_N[t] = min(L[t-N:t-1])
A[t] = ATR_20[t-1]                          # Vor-Bar-Normalisierung
ER_20[t] = abs(C[t]-C[t-20]) / sum(abs(C[j]-C[j-1]), j=t-19..t)
UP[t] = EMA_50[t] > EMA_200[t]
        AND EMA_50[t] > EMA_50[t-10]
        AND C[t] > EMA_50[t]
```

Keine Berechnung bei unvollstaendigem Fenster, Null-Nennern oder ungueltigen Preisen. Ein epsilon ersetzt hier keine fehlende Datenqualitaet. Mindestens **260 zusammenhaengende gueltige Bars** vor Eventbeginn. Ein Datenbruch beendet offene Setup-Zustaende; ein fehlender Kurs waehrend eines Labels erzeugt `CENSORED`, keine erfundene Nullrendite.

ATR und Volatilitaet sind unterschiedliche Groessen. Ein Ersatz von High/Low durch Close ist eine neue Strategie, keine zulaessige stille Datenreparatur.

**OHLC-Konsistenz:** keine adjustierten Closes mit unadjustiertem High/Low mischen. Rohdaten unveraendert archivieren, Corporate Actions explizit in Stueckzahlen, Barrieren und Cashflows abbilden. Ein Aktiensplit darf keinen Breakout erzeugen. Dividenden gehoeren in die P&L; sie sind kein tatsaechlich handelbarer Kursanstieg.

## 4. S4 — bestaetigter Pullback im Trend (Abschnitt 3 des Quelldokuments)

**A/Quelle:** Grimes, Kap. 3, S. 65–69; Kap. 6, S. 154–165. Der EMA-/ATR-Automat ist unsere objektive Annaeherung, keine vollstaendige Replikation seiner diskretionaeren Methode.

**B/oekonomische Interpretation:** Ein zeitweiliger Rueckgang innerhalb eines anhaltenden Aufwaertszustands koennte einen spaeteren Einstieg ermoeglichen. Der Wiederanstieg ist ein beobachtbarer Trigger; dass er zusaetzlichen Prognosewert hat, bleibt offen.

**C:** Eine Sequenz koennte die Qualitaet der Gegenbewegung erfassen. Ein einfacher Endzustands-Classifier koennte genauso gut oder besser sein.

**Leakage-Falle:** ZigZag oder zentrierte Swing-Tiefs als Echtzeit-Pivot behandeln. Ein Pivot mit k rechten Bestaetigungsbars ist erst k Bars spaeter bekannt.

## 5. Feste Konventionen und Corporate Actions (Abschnitt 8.1)

Der Datenadapter stellt zwei Sichten bereit: unveraenderte handelbare Roh-OHLC fuer Fills und eine bei jeder Entscheidung ausschliesslich mit bereits wirksamen Splits auf die aktuelle Stueckbasis gebrachte Historie fuer Indikatoren. Historische Preise/EMA-Zustaende werden bei Splitfaktor s durch s geteilt, Volumen mit s multipliziert; aktive Preisanker und ATR-Werte des noch nicht emittierten Setup-Zustands ebenfalls durch s teilen. **Ein emittiertes `event.R` bleibt unveraenderlich auf Signal-Stueckbasis**; ausschliesslich der Labelkernel konvertiert es beim Entry. Bereits emittierte dimensionslose Features bleiben ebenfalls unveraendert. Keine Dividend-Total-Return-Reihe als handelbaren OHLC-Kurs verwenden. Ex-Dividenden-Gaps bleiben echte Kursbewegungen; zugehoerige Ansprueche werden separat verbucht.

Fuer den Labelkernel heisst `s[k]` neue Stueckzahl je altem Stueck, wirksam vor Open k (ohne Split 1); `d[k]` ist der Dividendenanspruch je Stueck auf der nach diesem Split geltenden Stueckbasis am Ex-Tag (sonst 0). Vor dem Entry wirksame Splits konvertieren das Signal-R auf Entry-Stueckbasis. Nach Entry werden Stueckzahl und Barrieren umgerechnet. Dividenden werden als Anspruch in der Trade-P&L erfasst; tatsaechlich auszahlbares Portfolio-Cash entsteht erst am Zahltag. Unbekannte bzw. nicht unterstuetzte Kapitalmassnahmen fuehren zu `ACTION_UNSUPPORTED` und einem offenzulegenden Datenqualitaetsbefund, nicht zu einer stillen Entfernung schlechter Trades.

```text
TIMEFRAME = daily, regulaere abgeschlossene Bars
SIDE = long
WARMUP = 260 gueltige zusammenhaengende Bars
SEQUENCE = 60
EMA = 20, 50, 200; Definition gemaess Abschnitt 3
ATR = SMA(TR, 20)
TOUCH_BAND = 0.25 * ATR20[p-1]
CONFIRM_MAX = 3 Bars nach p, einschliesslich p+3
HOLD_N = 10; Entry-Bar zaehlt als 1
TARGET_Q = 1.5
R = ATR20[t-1], eingefroren bei Event t
EVENT_COOLDOWN = bis einschliesslich t+10
                # unabhaengig vom tatsaechlichen Exit oder Trade/Skip
PRIMARY_OHLC_TIE = stop_first, plus ambiguous=true
TIME_EXIT = vorab geplante Schliessung am Close von t+10
```

Kosten werden als verpflichtendes Manifest uebergeben. Keine stillen Nullkosten. Fuer eine rein technische Referenzsimulation sind explizit zu kennzeichnende, **unkalibrierte** Szenarien `b=3/6/15 bp pro Seite` als kombinierter halber Spread plus Slippage und `f=1 bp Provision pro Seite` moeglich. Das ersetzt keine Brokergebuehrenpruefung. Zusaetzlich Mindestgebuehren als eigene Szenarien, etwa 0/1/3 EUR pro Order, jeweils mit vollstaendiger Stueckzahl- und Waehrungsrechnung; diese Zahlen sind Sensitivitaetsannahmen, keine Anbieterpreise.

Im Ein-Stueck-Labelkernel ist `b` der gewaehlte adverse Fill-Haircut und `f` die proportionale Provision. Mindestgebuehren gehoeren in die getrennte mengenabhaengige Portfolio-P&L. Das Modell darf deshalb Ein-Stueck-Netto-R nicht ungeprueft als wirtschaftlichen 10.000-EUR-Trade bewerten.

## 6. Eventgenerator (Abschnitt 8.2)

`t` ist der fortlaufende Index der planmaessigen Handelssitzungen des Instruments im offiziellen Boersenkalender. Fehlende Sitzungen bleiben als ungueltige Datensaetze erhalten; Wochenenden/Feiertage sind keine Bars. Der Generator laeuft einmal chronologisch ab einem festgehaltenen Datenursprung. An Train-/Kalibrierungs-/Testgrenzen werden weder Setup-Zustand noch Cooldown zurueckgesetzt. Nach einem tatsaechlichen Datenbruch beginnen Indikatorinitialisierung und 260 zusammenhaengende gueltige Sitzungen Warmup neu; die absolute Cooldown-Grenze bleibt erhalten.

```text
state = IDLE
pending_p = NONE
blocked_through = -1

for each completed bar t, in chronological order:
    if a required bar/indicator is invalid or warmup is incomplete:
        state = IDLE
        pending_p = NONE
        continue

    if t <= blocked_through:
        continue

    if state == WAIT_CONFIRM:
        # Invalidierung VOR Bestaetigung; keine neue Vorstufe am selben Bar.
        if t > pending_p+3 or C[t] <= EMA50[t-1]:
            state = IDLE
            pending_p = NONE
            continue

        if UP[t] and C[t] > H[t-1]:
            event = {
                id: (instrument, t, 'S4-v1'),
                t: t,
                signal_time: close_timestamp[t],
                setup_start: pending_p,
                side: +1,
                R: ATR20[t-1],
                q: 1.5,
                N: 10,
                features: BUILD_FEATURES(pending_p, t),
                earliest_entry_bar: t+1
            }
            emit event                         # auch falls Modell spaeter SKIP sagt
            blocked_through = t+10
            state = IDLE
            pending_p = NONE
            continue

        if t == pending_p+3:
            state = IDLE
            pending_p = NONE
        continue

    # IDLE: nur erste passende Beruehrung speichern
    A = ATR20[t-1]
    center = EMA20[t-1]
    touched = (L[t] <= center+0.25*A) and (H[t] >= center-0.25*A)
    if UP[t-1] and C[t-1] > center and C[t] < C[t-1] \
       and touched and C[t] > EMA50[t-1]:
        pending_p = t
        state = WAIT_CONFIRM
        # Kein Event und kein Trade auf dieser ersten Beruehrungsbar.
```

Die Cooldown-Laenge ist unabhaengig von fruehrem Stop-/Target-Erreichen. So koennen Labels und Modellannahmen den zukuenftigen Eventstrom nicht veraendern. Nach Ablauf darf erst eine neue Beruehrung einen Zustand starten; das alte p wird nie wiederverwendet. Die Regel reduziert Ueberschneidung pro Instrument, beseitigt aber keine Cross-Asset-Korrelation.

## 7. Featurevektor (Abschnitt 8.3)

```text
BUILD_FEATURES(p,t):
    for j from t-59 through t:
        a = ATR20[j-1]
        s = sigma20[j-1]
        range = H[j]-L[j]
        append the following 10 values, in this exact order:
          1. r[j] / s
          2. log(C[j]/C[j-5]) / (s*sqrt(5))
          3. log(C[j]/C[j-20]) / (s*sqrt(20))
          4. (C[j]-EMA20[j-1]) / a
          5. (C[j]-EMA50[j-1]) / a
          6. (EMA50[j-1]-EMA50[j-11]) / a
          7. TR[j] / a
          8. sigma5[j-1] / s
          9. ER20[j]
         10. (C[j]-L[j])/range if range>0 else 0.5

    append separate event context, in this exact order:
      1. (t-p)/3
      2. (max(H[p-20:p-1])-min(L[p:t])) / ATR20[p-1]
      3. (C[t]-H[t-1]) / ATR20[t-1]
      4. ATR20[t-1]/C[t]                  # relative Risiko-/Kosten-Skala
    return sequence[60,10], context[4]
```

Ein negativer Wert in Kontext 2 ist mathematisch zulaessig; nicht heimlich auf 0 kappen. Fehlt ein benoetigtes Datum, ist das Event als `FEATURE_INVALID` zu protokollieren und nicht fuer Training/Handel zu verwenden. Keine nachtraegliche Selektion anhand seines spaeteren Ertrags. Hauptversuch **ohne Volumen und ohne Cross-Asset-Features**, um Datenbedarf und Freiheitsgrade klein zu halten.

**Sequenzvertrag:** 60 abgeschlossene Bars bis einschliesslich Event t. Historische Rolling-Features bleiben auf ihrem jeweiligen damaligen Stand. Die erst bei p bekannten Setup-Anker werden als separater Event-Kontext angehaengt, nicht rueckwirkend als angeblich schon damals bekannte Zeitreiheninformation ausgegeben.

## 8. Labelkernel (Abschnitt 8.4)

OHLC sind beobachtete Handelskurse, keine Bid-/Ask-Pfade. Das Fill-Modell ist eine reproduzierbare Naeherung. Die Eroeffnungsorder wird nach Signal t vor der naechsten Sitzung aufgegeben; **keine Auswahl anhand des erst spaeter bekannten `O[t+1]`**. `E` wird erst beim simulierten Fill bekannt. Barrieren werden unmittelbar danach aktiviert. Keine Verwendung spaeterer H/L zur Auswahl des Entry.

```text
LABEL(event, OHLC, b, f):
    e = event.t + 1
    last = e + event.N - 1
    if raw_O[e] unavailable or s[e] invalid:
        return CENSORED

    E = raw_O[e] * (1+b)             # modeled market-on-open fill
    R = event.R / s[e]               # ggf. Split zwischen Signal und Entry
    shares = 1.0                     # Label normiert auf 1 Entry-Stueck
    dividends = 0.0
    S = E-R
    T = E+event.q*R
    if E <= 0 or R <= 0:
        return DATA_ERROR

    # S<=0 ist bei positiven Preisen moeglich, aber kein erreichbarer Stop;
    # nicht rueckblickend den Trade entfernen. Kennzeichnen, weiter verfolgen.
    flag_nonpositive_stop = (S <= 0)

    for k from e through last:
        if any required raw_O,raw_H,raw_L,raw_C or action record is invalid:
            return CENSORED with event metadata

        if k > e:
            shares = shares * s[k]
            S = S / s[k]
            T = T / s[k]
            dividends = dividends + shares*d[k]
            # E und R bleiben auf ORIGINALER Entry-Stueckbasis fuer die P&L.
            # Kein Dividendenanspruch bei Kauf erst am Ex-Tag e.

        O,H,L,C = raw_O[k],raw_H[k],raw_L[k],raw_C[k]

        # Vor High/Low: Open ist zeitlich zuerst.
        if O <= S:
            exit_reference = O
            y = -1
            reason = GAP_STOP
            ambiguous = false
            break
        if O >= T:
            exit_reference = O     # market-on-touch at available open
            y = +1
            reason = GAP_TARGET
            ambiguous = false
            break

        hit_stop = (L <= S)
        hit_target = (H >= T)
        ambiguous = hit_stop and hit_target

        if hit_stop:               # stop-first, including dual touch
            exit_reference = S
            y = -1
            reason = STOP
            break
        if hit_target:
            exit_reference = T
            y = +1
            reason = TARGET
            break
        if k == last:
            exit_reference = C     # expiry was known before close
            y = 0
            reason = TIME
            break

    X = exit_reference * (1-b)
    exit_notional = shares*X
    net_pnl_per_share = exit_notional-E + dividends - f*E - f*exit_notional
    net_R = net_pnl_per_share / R
    meta_label = 1 if net_pnl_per_share > 0 else 0
    return {
        barrier_class:y, meta_label:meta_label, net_R:net_R,
        entry:E, exit:X, exit_bar:k, reason:reason,
        ambiguous:ambiguous, nonpositive_stop:flag_nonpositive_stop,
        label_available_after:close_timestamp[k]
    }
```

Bei der Entry-Bar liegt die Order am Open; daher duerfen spaetere H/L dieser Bar fuer den Ausgang verwendet werden. Ein kostenbedingt bereits hinter dem Stop liegender erster beobachteter Preis fuehrt nach diesem konservativen Modell zu sofortigem Stop, nicht zu Ausschluss. Die gleiche Ausfuehrungsfunktion muss fuer Traininglabels und Backtest verwendet werden. Splitfaktoren muessen positiv sein und Corporate-Action-Daten vollstaendig vorliegen.

Fuer dual-touch-Bars ist eine zweite Auswertung mit **Target-priority** zu erstellen und Ergebnisunterschied sowie Haeufigkeit zu berichten. Wenn daraus ein anderer Forschungsschluss entsteht, sind die Daten zu grob fuer diesen Schluss.

**Label-Maturitaet:** Ein TP/SL-Ausgang ist erst verfuegbar, wenn sein beobachtbarer Exit abgeschlossen und verbucht ist; konservativ nach Close der Exit-Bar. Ein Timeout erst nach tatsaechlichem Time-Exit.

## 9. Zwei verschiedene Labels (Abschnitt 5 des Quelldokuments)

1. **Barrier-Prognose:** `P(Y_barrier=+1|...)`, P(-1), P(0), summieren zu 1.
2. **Wirtschaftliches Meta-Label:** `Y_meta=1` genau dann, wenn der hypothetische Trade unter vollem Kostenvertrag positiven Nettoertrag erzielt.

Diese Labels sind nicht identisch. Ein Timeout kann Gewinn oder Verlust bedeuten. Selbst ein oberes Barrier-Ereignis kann bei sehr kleinem R nach Mindestgebuehren unprofitabel sein.

Break-even-Trefferquoten folgen aus `p*q-(1-p)=0`: q=1 → 50%, q=1,5 → 40%, q=2 → 33,3%. Mit konstanten Roundtrip-Kosten c in R-Einheiten und nur diesen beiden Ausgaengen waere `p>(1+c)/(q+1)` erforderlich. **Mit Timeouts und Gap-Verlusten gilt diese vereinfachte Schwelle nicht.**

## 10. Meta-Entscheidung (Abschnitt 8.5) — spaetere Stufe, hier nicht umgesetzt

```text
DECIDE(event, model_artifact, calibration_artifact):
    require model/calibration training cutoffs < event.signal_time
    require every used training label was mature at its fitting cutoff
    require artifact feature order == BUILD_FEATURES order
    require event and cost contract versions match the artifacts
    p = calibrated_predict_proba(event.features)  # [+1,-1,0]
    require finite p, p>=0, sum(p)=1 within 1e-6
    expected_net_R = dot(p, calibration_artifact.mean_net_R_by_class)
    return REQUEST_LONG_NEXT_OPEN if expected_net_R > 0 else SKIP
```

Der Schwellenwert 0 ist ein wirtschaftlicher Referenzpunkt, kein nachtraeglich gefittetes Konfidenz-Gate. Die Referenz wird unabhaengig davon fuer alle Events gelabelt. Handelsrestriktionen, Stueckzahl und reale Gebuehren koennen einen angefragten Trade zusaetzlich verhindern; das wird als **Portfolio-Ablehnung**, nicht als Modell-SKIP protokolliert.

## 11. Abgrenzung zum 10.000-EUR-Portfolio (Abschnitt 8.6)

Ein korrektes Eventlabel ist noch kein Portfolioergebnis. Als vorsichtige zu pruefende Ausgangskonvention waeren 0,25% geplantes Eigenkapitalrisiko pro Trade und hoechstens 20% geplanter Einzeltitelanteil moeglich; diese Werte sind **nicht als bestehende Projektparameter gesetzt**. Stueckzahlen muessen vor dem Fill kausal bestimmt werden; Gaps koennen geplante Exposure-/Risikowerte ueberschreiten.

Projektgrenzen bleiben **1,25x Hebel-Obergrenze** und **15% historisches Drawdown-Kriterium**. Ein Stop kann wegen Gaps mehr als 1R verlieren; die Summe geplanter Einzelrisiken garantiert keine 15%-Grenze. Mindestens 12% jaehrliche CAGR ist ein Forschungsziel, keine Eingabe zur Signaloptimierung.

**Renditeherleitung des Nutzers (Rechenbeispiel, keine Prognose):** bei geplant 0,25% Risiko pro Trade gilt naeherungsweise und ohne Zinseszins `Jahresrendite ~ Trades × Risikoanteil × Ø-Nettoertrag in R`. 100 Trades × 0,0025 × 0,48R = 12%. Der noetige Ø-Nettoertrag ist damit `48/Trades`. Bei q=1,5 und Stop bei -1R gilt vor Kosten und Timeouts `Ø R ~ 2,5w - 1`, also: 100 Trades verlangen ~59% Trefferquote, 50 Trades ~78%, 200 Trades ~50%. Break-even liegt bei 40%. Timeouts und Kosten verschieben die noetige Quote nach oben. **Der Census entscheidet damit ueber die Machbarkeit, nicht nur ueber die Datenmenge.**

## 12. Validierungsvertrag (Abschnitt 7 des Quelldokuments) — bindend fuer spaetere Stufen

Event-Census vor Modelltraining. Chronologische Trennung mit globalem Kalender-Split ueber alle Instrumente; Random-Splits einzelner Events unzulaessig. Purging aller Ereignisse, deren Informations- oder Labelintervalle die Folgeperiode ueberlappen. Gemeinsame Vergangenheit ist nicht automatisch Leakage, erzeugt aber Abhaengigkeit — Unsicherheit mit gemeinsamen Kalenderblöcken schaetzen, nicht mit iid-Event-Bootstrap. Online strikt getrennt und nur auf ausgereiften Labels. Forschungszaehlung: Eventdefinition, Filter, q, N, Modell, Kostenvariante und Auswahlregel vorab registrieren.

Vergleichsreihenfolge fuer spaetere Stufen: **alle regelbasierten Events handeln → multinomiale logistische Regression → kleines LSTM** (1 Layer, 16 Hidden Units, eingefroren, keine Architektursuche) auf derselben Informationsbasis. Kein Zwang, das LSTM beizubehalten, wenn es Baselines nicht uebertrifft.

## 13. Bekannte Grenzen

- Der Census ist mit den vorhandenen Daten nicht durchfuehrbar; ohne ihn ist keine Aussage ueber Machbarkeit moeglich.
- Aus 19 hoch korrelierten ETFs entsteht kein gleichwertiger Ersatz fuer 19 unabhaengige Maerkte.
- Rueckblickend ausgewaehltes Universum; historische Universen muessten zeitpunktgetreu sein.
- Diese Spec hebt weder den gescheiterten Trend-Holdout noch die Momentum- und Minuten-LSTM-Nullbefunde noch den 2x2-Nullbefund auf und beansprucht keinen profitablen Edge.
- Der Forschungszaehler steht nach dem 2x2 bei **145** unkorrigierten Vergleichen. Jede spaetere S4-Stufe zaehlt neu.
