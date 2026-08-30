# LSTM-zentriertes Market-Control-System

Energie-Regelungssystem-Analogie auf den Aktienmarkt uebertragen:

```
Messen -> Sequenzdaten aufbauen -> LSTM-Prognose -> Controller-Entscheidung
   -> Paper-Ausfuehrung -> Feedback -> Online-Korrektur
   ^_____________________________________________________________________|
```

## Architektur

```
1. MARKET DATA STREAM         data_layer/alpaca_client.py (Alpaca, 1-Minuten-Bars, Symbol konfigurierbar)
   fetch_historical_market_data() fuer Offline-Vortraining,
   AlpacaLiveMarketDataStream fuer Live-/Paper-Betrieb -- beide liefern
   dasselbe Schema wie _generate_synthetic_market_data (kein Downstream-
   Code-Aenderung noetig)
        |
2. FEATURE ENGINEERING LAYER  feature_engineering/feature_pipeline.py
   - log_return, realized_vol (rolling std), orderbook_imbalance,
     spread_norm, vwap_deviation, trade_intensity, mid_price_return
   - kausal (nur Vergangenheit), Batch- (pandas) und Live-Pfad (O(1)/Tick)
        |
3. SEQUENCE BUFFER            feature_engineering/sequence_buffer.py
   X in R^(Timesteps x Features), Ring-Buffer, fail-fast bei fehlenden/NaN-Features
        |
4. LSTM / HYBRID-MODELL       models/lstm_forecaster_torch.py (primaer)
                               models/lstm_forecaster_tf.py (Vergleich, ungetestet)
   Outputs: expected_return (mu), expected_volatility (sigma > 0), probability_up
   Loss: Gaussian NLL(mu, sigma) + BCE(p_up)
        |
4b. WALK-FORWARD-VALIDATION    training/walk_forward.py
   Rollierende Trainings-/Test-Fenster statt statischem 80/20-Split
        |
5. EXPOSURE CONTROLLER         controller/exposure_controller.py + risk_overlay.py
   edge = expected_return / (expected_volatility^2 + eps)
   target_position = clip(k * edge, -max_position, +max_position)
   + Risk Overlay: Rate Limiter, sigma-Regime-Filter, Drawdown-Cooldown
        |
6. PAPER EXECUTION ENGINE      execution/paper_execution.py
   Mid-Preis-Fill + halber Spread als Slippage, mark-to-market der VORHERIGEN Position
        |
7. FEEDBACK BUFFER             feedback/feedback_buffer.py
   Horizon-verzoegerte (X, actual_future_return)-Paare fuer den Online Trainer
        |
8. ONLINE TRAINER              training/online_trainer.py
   Periodisches Fine-Tuning (alle N Bars, kleine LR)

orchestration/control_loop.py verdrahtet Layer 2-8 zu einem lauffaehigen Regelkreis.
```

## Modulstruktur

```
market_control_system/
├── data_layer/
│   └── alpaca_client.py      # historische Bars+Quotes (Offline) + Live-Stream-Adapter (Paper)
├── config/
│   └── settings.py           # laedt Alpaca-Credentials aus .env (nie hartkodiert)
├── feature_engineering/
│   ├── feature_pipeline.py   # FeaturePipeline (Batch), LiveFeatureEngine (Streaming), RollingZScoreScaler
│   └── sequence_buffer.py    # SequenceBuffer (Live), SequenceWindowBuilder (Batch)
├── models/
│   ├── lstm_forecaster_torch.py  # primaerer Pfad
│   └── lstm_forecaster_tf.py     # Framework-Vergleich, in dieser Umgebung nicht ausgefuehrt
├── training/
│   └── walk_forward.py       # rollierende Walk-Forward-Validation (ersetzt statischen 80/20-Split)
├── controller/
│   ├── exposure_controller.py  # edge -> target_position (zustandslose Formel)
│   └── risk_overlay.py         # Rate Limiter, Regime-Filter, Drawdown-Cooldown
├── execution/
│   └── paper_execution.py    # Fill-Simulation (Mid-Preis + halber Spread als Slippage)
├── feedback/
│   └── feedback_buffer.py    # Horizon-verzoegerte (X, y)-Paare fuer Online-Training
├── orchestration/
│   └── control_loop.py       # verdrahtet Layer 2-8, End-to-End-Demo auf synthetischen Daten
├── tests/
│   └── test_consistency.py   # Batch/Live-Paritaet, alle 7 Features
└── requirements.txt
```

## Status (Stand 2026-08-24)

Layer 1-4 stehen und sind end-to-end getestet:

- `test_consistency.py`: Batch- und Live-Pfad liefern fuer alle 7 Features identische Werte (max. Abweichung < 1e-12, reine Gleitkomma-Rundung).
- `sequence_buffer.py`: Alignment zwischen Batch-Fenstern und Live-Ring-Buffer verifiziert, Fail-Fast bei fehlenden Features getestet.
- `lstm_forecaster_torch.py`: End-to-end auf synthetischen Daten trainiert (Walk-Forward-Split 80/20). Modell kollabiert auf Random-Walk-Daten korrekt auf die unbedingte Mittelwertschaetzung (Richtungsgenauigkeit ~0.5) -- erwartetes, gesundes Verhalten, kein Look-Ahead-Bug.

**Fix in dieser Version:** `LiveFeatureEngine` lieferte urspruenglich nur 4 von 7 Features (vwap_deviation, trade_intensity, mid_price_return fehlten). Das haette den Live-Betrieb sofort mit einer `KeyError` in `SequenceBuffer.push()` blockiert. Behoben via `RollingSum`-Hilfsklasse fuer inkrementelle VWAP- und Trade-Intensity-Berechnung.

**Walk-Forward-Validation (`training/walk_forward.py`):** 9 rollierende Folds (train=2000/test=400 Samples, kein expanding window, frisches Modell pro Fold) auf synthetischen Daten. Ergebnis: direction_accuracy pro Fold zwischen 0.398 und 0.535 (Mittel 0.471, std 0.049) -- durchgehend nahe Zufallsniveau ueber alle Folds hinweg, kein Fold mit auffaelligem Ausreisser. Praedizierte sigma liegt in jedem Fold nah an der tatsaechlichen Residual-Streuung (~0.03-0.04 in beiden). Das ist auf Random-Walk-Daten das erwartete, gesunde Ergebnis und zeigt: die Pipeline selbst produziert kein strukturelles Overfitting/Regime-Artefakt -- Voraussetzung, um als Naechstes echte Positionsgroessen an mu/sigma zu haengen.

**Exposure Controller + Risk Overlay:** Formel und Clipping verifiziert (siehe Sanity-Checks in den jeweiligen `__main__`-Bloecken). Rate Limiter glaettet Sprung-Signale ueber mehrere Schritte, Regime-Filter zwingt Position auf 0 bei sigma oberhalb einer Schwelle, Drawdown-Cooldown zwingt nach kumulierten Verlusten ueber ein Lookback-Fenster N Schritte auf Position 0. Cooldown-Mechanismus ist bereits jetzt funktional, auch ohne dass Feedback Buffer/Execution Engine existieren (einfach ungenutzt, bis diese ihn aufrufen).

**End-to-End-Regelkreis (`orchestration/control_loop.py`):** Kompletter Loop laeuft fehlerfrei auf synthetischen Daten -- Offline-Vortraining (10 Epochen), datengetriebene k-Kalibrierung, dann Live-Loop mit periodischem Online-Fine-Tuning (alle 100 Bars). Wichtiger Fund beim ersten Lauf, direkt aus dem End-to-End-Test:

> **Rohe Kelly-Sizing (k=1.0 bzw. selbst k=0.5) saettigt die Position praktisch permanent am `max_position`-Clip.** Bei k=0.5 lag der maximale Drawdown auf reinem Random-Walk-Rauschen (kein echtes Signal!) bei **-84%** des kumulierten Returns. Ursache: sigma ist bei Finanzrenditen naturgemaess klein (~0.03), sigma^2 also winzig (~0.0009) -- dadurch erzeugt schon minimales Rauschen in mu einen zweistelligen edge-Wert, der den Clip sofort saettigt. Das ist **keine Modell-Fehlkalibrierung** (sigma trifft die tatsaechliche Residualstreuung nachweislich gut, siehe Walk-Forward-Ergebnis oben), sondern eine bekannte Eigenschaft roher Kelly-Formeln: sie ignorieren Schaetzunsicherheit in mu selbst.
>
> **Fix:** `controller/exposure_controller.py` bietet jetzt `calibrate_k()` -- k wird aus der tatsaechlichen historischen |edge|-Verteilung des jeweiligen Modells abgeleitet (95. Perzentil -> 50% von max_position), statt per Hand geraten zu werden. Im End-to-End-Lauf kalibriert `control_loop.py` k automatisch aus den Vorhersagen des offline-vortrainierten Modells (Ergebnis: k~0.021, unabhaengig von der urspruenglich per Hand gefundenen 0.02 -- bestaetigt, dass der Wert plausibel war). Ergebnis: max. Drawdown sinkt auf -13% bei nahezu gleichem (nahe-Null) kumuliertem Return -- das erwartete, gesunde Verhalten auf Rauschdaten. Zusaetzlich gibt es `ControllerConfig.max_edge` als optionale harte Sicherheitsbegrenzung des rohen edge-Werts (z.B. gegen sigma-Ausreisser durch einen schlechten Online-Trainings-Schritt).

**Alpaca-Datenanbindung (`data_layer/alpaca_client.py`):** `fetch_historical_market_data()` (historische Bars+Quotes, per `merge_asof` kausal zusammengefuehrt) gegen den echten Paper-Account getestet -- 6 Tage AAPL, 3335 Bars, liefen OHNE JEDE Code-Aenderung durch `FeaturePipeline` -> `SequenceWindowBuilder` -> `LSTMForecaster`-Training. Bestaetigt: das Schema-Versprechen ("gleiche raw_event-Struktur wie synthetische Daten") haelt in der Praxis. `AlpacaLiveMarketDataStream` (Websocket, kombiniert Quotes+Bars zu raw_event pro abgeschlossenem 1-Min-Bar) ist geschrieben; die Konstruktion (Verbindungsaufbau + Subscriptions) wurde erfolgreich getestet, der blockierende `.run()`-Loop selbst NICHT (laesst sich nicht sinnvoll in einem einzelnen Sandbox-Kommando verifizieren -- vor echtem Dauerbetrieb einmal manuell laufen lassen).

**`orchestration/run_live.py`:** echter Einstiegspunkt fuer Live-/Paper-Betrieb (im Gegensatz zu `control_loop.py`'s synthetischer Demo). Kompletter Ablauf gegen den echten Account verifiziert: 5096 echte AAPL-Bars (10 Tage) geladen -> 5023 Trainingssequenzen -> Offline-Vortraining (10 Epochen, Loss faellt sauber monoton von -1.07 auf -3.80) -> `calibrate_k()` auf den ECHTEN mu/sigma-Vorhersagen -> **k=0.00155**. Bemerkenswert: das ist **13x kleiner** als das aus synthetischen Daten kalibrierte k (~0.021) -- ein konkreter Beleg dafuer, dass synthetische Kalibrierung nicht auf echtes Kapital uebertragen werden darf, sondern jedes Mal auf dem tatsaechlichen Symbol neu berechnet werden muss.

## Offene Punkte / naechste Schritte

1. **`run_live.py` einmal manuell live laufen lassen** (waehrend die Boerse offen ist) und beobachten, ob Bars/Quotes wie erwartet eintreffen und der Regelkreis stabil bleibt -- der blockierende Live-Anteil wurde bisher nicht ausgefuehrt.
2. Bekannte Schwaechen: keine Trennung Aleatorik/Epistemik in sigma (kein MC-Dropout/Ensemble) -- `calibrate_k()` mindert die Symptome (weniger Dauer-Saettigung), loest aber nicht das zugrundeliegende Problem, dass sigma keine Modellunsicherheit erfasst; `hidden_size`/`num_layers` ungetuned; `RollingSum` kann bei sehr lange laufenden Streams Gleitkomma-Drift ansammeln (unkritisch bei Tagesrestart); Walk-Forward bisher nur auf synthetischen Random-Walk-Daten validiert; Drawdown-Cooldown in `RiskOverlay` im End-to-End-Lauf noch nicht unter Realbedingungen getriggert getestet (drawdown_limit war im Demo-Lauf grosszuegig genug).

## Setup

```bash
pip install -r requirements.txt
python tests/test_consistency.py
python models/lstm_forecaster_torch.py
```
