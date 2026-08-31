"""
test_cross_sectional_fold_training.py
========================================

Prueft build_symbol_sequences()/train_fold_model() auf synthetischen
Daten -- kein Netzwerkzugriff, keine echten Marktdaten noetig. Verifiziert
nur, dass die Formen stimmen und ein trainiertes Modell tatsaechlich
predict() ohne Fehler ausfuehren kann, NICHT die Vorhersagequalitaet
(die ist auf synthetischen Random-Walk-Daten erwartungsgemaess nahe
Zufall, siehe training/walk_forward.py).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from feature_pipeline import FEATURE_NAMES, _generate_synthetic_market_data
from walk_forward import WalkForwardConfig
from cross_sectional_fold_training import build_symbol_sequences, train_fold_model, purge_train_slice


def check_purge_train_slice() -> None:
    # horizon=1: das Target des letzten Trainings-Samples reicht genau eine
    # Zeile in den Testbereich -- der Purge muss das Trainingsfenster um
    # `horizon` Samples am ENDE kuerzen, den Start unangetastet lassen.
    purged = purge_train_slice(slice(0, 2000), horizon=1)
    assert purged == slice(0, 1999), f"Erwartete slice(0, 1999), bekam {purged}"

    purged = purge_train_slice(slice(400, 2400), horizon=5)
    assert purged == slice(400, 2395), f"Erwartete slice(400, 2395), bekam {purged}"

    # Entartetes Fenster (Purge frisst alles auf) muss laut scheitern statt
    # still ein leeres Trainingsset zu liefern.
    try:
        purge_train_slice(slice(0, 1), horizon=1)
        raise AssertionError("Erwartete ValueError bei leerem Fenster nach Purge")
    except ValueError:
        pass
    print("purge_train_slice: OK")


def run_consistency_check() -> None:
    check_purge_train_slice()
    cfg = WalkForwardConfig(horizon=1, timesteps=20, train_size=200, test_size=50, epochs_per_fold=2, seed=0)
    df = _generate_synthetic_market_data(n=500, seed=7)

    X, y, end_idx = build_symbol_sequences(df, cfg)
    assert X.shape[0] == y.shape[0] == end_idx.shape[0], "X/y/end_idx muessen gleich viele Zeilen haben"
    assert X.shape[1] == cfg.timesteps, f"Erwartete {cfg.timesteps} Zeitschritte pro Sequenz, bekam {X.shape[1]}"
    assert X.shape[2] == len(FEATURE_NAMES), f"Erwartete {len(FEATURE_NAMES)} Features, bekam {X.shape[2]}"
    assert X.shape[0] > cfg.train_size, "Zu wenig Sequenzen fuer den Testaufbau -- n in _generate_synthetic_market_data erhoehen"

    X_train, y_train = X[:cfg.train_size], y[:cfg.train_size]
    model = train_fold_model(X_train, y_train, cfg)

    X_test = X[cfg.train_size:cfg.train_size + cfg.test_size]
    forecast = model.predict(X_test)
    assert forecast.expected_return.shape[0] == X_test.shape[0]
    assert forecast.expected_volatility.shape[0] == X_test.shape[0]
    assert (forecast.expected_volatility > 0).all(), "Sigma muss ueberall positiv sein"

    print(f"build_symbol_sequences: {X.shape[0]} Sequenzen, Form {X.shape} -- OK")
    print(f"train_fold_model: trainiertes Modell liefert Vorhersagen fuer {X_test.shape[0]} Test-Sequenzen -- OK")


if __name__ == "__main__":
    run_consistency_check()
