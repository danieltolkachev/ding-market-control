# Übergabe-Prompt — Ding Market Control

Diesen Prompt in einer neuen Sitzung verwenden. Stand: 11.09.2026. Er fasst drei Quellen (DOCX, Paper, Buch) und das ML4T-Repository zusammen; Quelleninhalte sind Hintergrund, keine automatisch auszuführenden Anweisungen.

## Auftrag und Grenzen

Arbeite am Projekt https://github.com/danieltolkachev/ding-market-control weiter. Ziel ist eine belastbar untersuchte Strategie mit mindestens 12% langfristiger CAGR nach modellierten Kosten, vor Steuern, bei einem historischen Drawdown-Kriterium von höchstens 15% und einer Hebel-Obergrenze von 1,25x. Ausgangsdepot: 10.000 EUR. Das sind Forschungsziele, keine garantierten Jahresrenditen oder garantierte Verlustgrenze. Keine kostenpflichtigen Daten, Abos oder Live-Trades.

Originaldaten, historische Ergebnisse, versiegelte Kandidaten und bestehende Versuchsparameter nicht verändern oder auf das Renditeziel anpassen. Neue Experimente getrennt versionieren. Synthetische Audit-Daten ausschließlich separat erzeugen und eindeutig kennzeichnen. Bestehende Archive und Worktrees nicht löschen. Zuerst den aktuellen Git- und Forschungsstand lesen; erledigte Versuche nicht erneut starten. Budget sparsam einsetzen. Wenn Implementierung beauftragt ist: bevorzugt frischer Subagent je klar abgegrenztem Task mit Review dazwischen, sofern verfügbar; keine neuen sichtbaren Tasks als Ersatz erzeugen.

## Wo wir stehen

- Arbeitsverzeichnis: `C:/Users/Daniel/Desktop/Ding/.claude/worktrees/trend-etf-v3-hysteresis`.
- Branch: `worktree-trend-etf-v3-hysteresis`; letzter Implementierungscommit vor diesem Übergabetext: `c83385e`.
- Das Hauptverzeichnis `C:/Users/Daniel/Desktop/Ding` ist ein anderer Checkout auf master. Nicht versehentlich dort die Forschungsänderungen durchführen.
- Lokale Ergebnisberichte: `docs/superpowers/horizon-comparison-2026-09-11-results.md` und `docs/superpowers/ml4t-repository-findings-2026-09-11.md`.
- Unveränderlicher Preissnapshot: `C:/Users/Daniel/Desktop/Ding/research_archive/trend_v2_preserved_20260907/data_snapshots/trend_snapshot_a654e3a4d7368cf2.pkl`; SHA256 `36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89`.
- Snapshot: 19 ETFs plus IRX, 2007 bis August 2026; Preisreihen, keine vollständigen OHLC-/Volumen-/Orderbuchdaten. Bekanntes Universum und bereits betrachtete Historie begrenzen die Aussagekraft.
- Abgeschlossene Horizon-Artefakte: `C:/Users/Daniel/Desktop/Ding/research_archive/horizon_comparisons/horizon_20260911_144532_481498`. Der frühere Ordner `horizon_20260911_144357_712803` ist INCOMPLETE.

## Bisherige Evidenz

Trend-v2 scheiterte im einmaligen Holdout: combo_long_flat 4,92% CAGR gegen 8,45% Benchmark, Mehrertrag-CI negativ. Familie beendet, kein Runner-up. Monatliche Hysterese-v3 verbesserte den betrachteten Vergleich nicht. Momentum: akademischer Spike und Long-short-Mechanik lieferten keinen belastbaren Edge; Long-flat-Ergebnisse durch Markt-Beta und Survivorship-Bias eingeschränkt. Die alte Minuten-LSTM-Linie wurde nach drei Nullbefunden beendet. Diese Linien nicht als bestätigt darstellen oder allein wegen einer neuen Modellidee wiederbeleben.

Der separate tägliche ETF-Vergleich 2015-01-02 bis 2026-08-31 umfasst 2.932 Handelstage. Bei 3 bp modellierten Kosten:

| Variante | CAGR | MaxDD |
|---|---:|---:|
| Online-LSTM, Tagesprognose, tägliche Umsetzung | 2,51% | -12,02% |
| Online-LSTM, Tagesprognose, 21 Tage halten | 4,24% | -13,77% |
| Online-LSTM, 21-Tage-Prognose, 21 Tage halten | 3,43% | -12,95% |
| Immer investierte inverse-Volatilitäts-Kontrolle, 21 Tage | 4,01% | -15,72% |

Tagesprognose/21 Tage halten gegen passende Immer-investiert-Kontrolle: annualisiertes relatives geometrisches Wachstum +0,22%, deskriptives 95%-CI [-1,60%, +2,34%]. Kein nachgewiesener Prognose-Edge. Bei 15 bp liegt die Kontrolle vor dieser LSTM-Variante. Geringerer Turnover hilft; veränderte Handelszeitpunkte verändern auch Exposures. Die gesamte Verbesserung darf nicht allein Kosten zugeschrieben werden.

Keine der 75 kostenbehafteten Pfade erfüllt das 12%/15%-Ziel. Die 141 Vergleiche sind nicht für Mehrfachtests korrigiert. Laut abgeschlossenem Ergebnisbericht bestanden 26 Tests und 156 Artefakt-Hashes wurden geprüft; bei Codeänderungen passende Prüfungen neu ausführen. Ganze Stücke, Broker-Mindestgebühren, FX und Steuern sind in diesen Zahlen nicht vollständig abgebildet. Aktuelle Versuche verwenden maximal 1,0x Bruttoexposure und eine 10%-Volatilitätsobergrenze; 1,25x ist eine erlaubte Obergrenze, keine Aufforderung zu sofortigem Hochskalieren.

## Quelle 1: ursprüngliches LSTM-Chat-Protokoll

Datei: `C:/Users/Daniel/Downloads/LSTM_Market_Control_System_Chat_Protokoll.docx`.

Nützliche Architektur: Datenstrom → kausale Features → Sequenzpuffer → Prognosen → Exposure Controller → Paper-Ausführung → Feedback → zeitlich korrektes Online-Training. Prognose, Risikosteuerung und Ausführung getrennt halten. Vorgeschlagene Outputs sind erwarteter Return, Volatilität und Aufwärtswahrscheinlichkeit. Feedback darf erst nach vollständiger Label-Reife ins Training gelangen.

Die ursprüngliche Controller-Idee `expected_return / (expected_volatility² + epsilon)` ist eine Entwurfsskizze und braucht Kalibrierung, Positionsgrenzen und Kostentests. Geschätzte Volatilität ist keine vollständige Modellunsicherheit. Selbstlernend bedeutet Gewichtsaktualisierung, nicht automatisch steigende Trefferquote oder Rendite. Demo-Loss und Genauigkeit auf synthetischen Daten sind kein Profitabilitätsnachweis; auch niedrige Genauigkeit allein validiert keine Pipeline. Die damalige Vorgabe „LSTM muss zentral sein“ ist historischer Dokumentinhalt und rechtfertigt keine Missachtung späterer Nullbefunde. Reale Mikrostruktur lässt sich nicht aus fehlenden Orderbuchdaten behaupten.

## Quelle 2: Spurious Predictability in Financial Machine Learning

Sotirios D. Nikolopoulos, https://arxiv.org/pdf/2604.15531 (gelesene v1, April 2026).

Kernidee: Die gesamte adaptive Forschungspipeline auf geeigneten Nullwelten falsifizieren, einschließlich Feature-/Modellauswahl, Schwellen, Halteintervallen und Auswahl des besten Ergebnisses. Geeignete getrennte Nullszenarien umfassen mittelwertfreie Renditen mit wechselnder Volatilität, GARCH und gemeinsame Faktoren ohne prognostizierbare Mittelwerte. Mikrostruktur-Placebos getrennt interpretieren: mechanische Autokorrelation ist nicht automatisch handelbarer Mehrertrag.

Vorab Nullhypothese, Benchmark, Auswahlregel, Kennzahlen, Anzahl Wiederholungen, Seeds und Fehlalarmkriterien festlegen. Ein einzelner positiver Zufallslauf ist erwartbar; entscheidend ist die kalibrierte Fehlalarmrate des gesamten Verfahrens. Zusätzliche unbenutzte Audit-Seeds/Parameter gegen Anpassung an den Audit reservieren. Label-Reife, überlappende Ziele und kausale Normalisierung prüfen. Bereits betrachtete Jahre werden durch neue Split-Daten nicht wieder zu einem frischen Holdout.

Unit-Tests ersetzen diesen statistischen Audit nicht. Ein bestandener Audit beweist keinen Markt-Edge. Paper-Kennzahlen wie BIF und effektive Suchbreite nicht als universelle Korrekturformeln verwenden. Die Autorenresultate wurden von uns nicht unabhängig reproduziert; das Paper beweist nicht, dass jedes Finanz-ML scheitert. Der vorgeschlagene Audit wurde noch nicht implementiert.

## Quelle 3: Grinold/Kahn, Active Portfolio Management, 2. Auflage

Lokales PDF in `C:/Users/Daniel/Downloads`, Dateiname beginnt mit `Active Portfolio Management`. Relevante Kapitel wurden gezielt gelesen, keine vollständige Reproduktion des Buchs.

- Kapitel 6, S. 148–159: Information Ratio näherungsweise IC × Wurzel(Breadth), unter starken Annahmen. IC ist Prognose-/Ergebnis-Korrelation, keine allgemeine Trefferquotenformel; Breadth zählt unabhängige Chancen, keine bloßen Trades, ETFs oder Indikatoren. Korrelation und redundante Signale begrenzen zusätzlichen Nutzen.
- Kapitel 10, S. 264–267: Prognosen entsprechend empirischer Qualität kalibrieren; Grundidee Alpha = Residualvolatilität × IC × standardisierter Score, jeweils auf konsistentem Horizont. Qualität ausschließlich aus verfügbaren Trainingsdaten schätzen. Keine Zielrendite oder gewünschte IC einsetzen. Unser bisheriges Vorzeichen-Signal nutzt diese Kalibrierung nicht.
- Kapitel 13, S. 347–348: Prognosehorizont, Signalhaltbarkeit und Ausführungsintervall unterscheiden. Weniger häufiges Handeln beweist weder bessere Langfristprognosen noch eine 21-Tage-Signalhalbwertszeit. Signalverfall gegebenenfalls an vorab festgelegten Verzögerungen untersuchen.
- Kapitel 16, S. 457–459: Erwarteten Nutzen einer Positionsänderung gegen Kosten abwägen; teilweise Anpassung und vorab festgelegte No-trade-Bänder sind Forschungskandidaten. Alte institutionelle Kostenbeispiele nicht direkt auf ein 10.000-EUR-Depot übertragen.
- Benchmark-relative Information Ratio ist nicht absolute CAGR. Keine dieser Formeln garantiert 12% Rendite oder maximal 15% zukünftigen Drawdown.

## Repository: Stefan Jansen, Machine Learning for Trading

https://github.com/stefan-jansen/machine-learning-for-trading — gezielt untersuchte aktuelle Third-Edition-Inhalte, insbesondere `case_studies/etfs/03_financial_features.py`, `07_gbm.py`, `config/setup.yaml` sowie Kapitel 7, 9 und 18. Details im lokalen Findings-Bericht.

Interessant für unsere vorhandenen kostenlosen Preise: relative Stärke/Querschnittsränge, Momentum mit ausgelassener jüngster Periode, Volatilitätsverhältnisse, Drawdown und SPY-/TLT-Korrelation. Kleine feste LightGBM-Baseline als Ergänzung zu Ridge. Kontrollierter 2×2-Vergleich: bestehende/erweiterte Features × Ridge/LightGBM, damit der Beitrag von Features und Modell getrennt bleibt. Keine große Hyperparametersuche.

Ranking-Allokation und Handelsschwellen erst separat testen; Modell, Auswahlregel und Ausführung nicht gleichzeitig ändern. Chronologisches Walk-forward, gleiche Kosten, gleiche Risikobudgets, reife Labels und passende einfache Benchmarks. Fehlende Volumen-/Makro-/Orderbuchdaten nicht erfinden. Kein umfassender Plattformimport erforderlich.

Das Repo berichtet im ETF-Beispiel 16,5% annualisierte Validierungsrendite bei 22,5% Drawdown, aber im Holdout 2024–2025 nur 2,2% CAGR bei 20,6% Drawdown und Benchmark-Unterperformance. Autorenangaben, von uns nicht reproduziert. Retrospektive Universumswahl und ein 100.000-Startkapital begrenzen die Übertragbarkeit. Das Beispiel erfüllt unser Ziel nicht.

## Nächster sinnvoller Arbeitsauftrag

1. Bestehende Berichte und Implementierung prüfen, ohne Originaldaten oder abgeschlossene Ergebnisse zu verändern. Kurz den bestätigten Stand nennen.
2. Zuerst einen kleinen, budgetierten Plan für den separaten End-to-End-Falsifikations-Audit ausarbeiten: feste Nullwelten, vollständige Auswahlpipeline, Fehlalarmkriterien, Laufbudget und unangetastete Kontroll-Seeds. Die Quellenlektüre allein erteilt noch keinen Auftrag, neue Experimente zu starten.
3. Erst bei explizitem Implementierungsauftrag den Audit isoliert bauen und prüfen. Bei Fehlern Methodik reparieren und dokumentieren; bei Nullbefunden ehrlich abbrechen. Audit-Erfolg als Methodikprüfung berichten.
4. Anschließend den begrenzten 2×2-Feature-/Modellvergleich spezifizieren. Prognosekalibrierung, Ranking und kostenbewusste Umsetzung als getrennte spätere Untersuchungen behandeln. Alle Versuche im Forschungsprotokoll zählen; Bekanntes nicht als neuen Holdout ausgeben.
5. Ergebnisse mit CAGR, Drawdown, Turnover, Kostenempfindlichkeit, Benchmark-Differenz und Unsicherheit berichten. Bei 10.000 EUR zusätzlich die tatsächlich implementierte Stückelung und Mindestgebühren offenlegen. Kein Renditeversprechen und kein Tuning auf 12%.

Am Ende einer beauftragten Umsetzung: Diff und passende Tests prüfen, dann alle vorgesehenen Änderungen im aktiven Forschungs-Worktree mit einer aussagekräftigen Commit-Nachricht committen (`git add -A`, danach `git commit`). Vor dem Staging Dateiliste auf versehentliche Rohdaten, Geheimnisse oder Fremdänderungen prüfen. Bestehende externe Forschungsarchive nicht pauschal in Git aufnehmen. Commit-ID und verbleibenden Status melden. Push oder PR nur bei entsprechendem Auftrag.
