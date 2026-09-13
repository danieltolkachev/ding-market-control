"""Prognosemodelle des 2x2-Rasters: Ridge und eine feste LightGBM-Rezeptur.

Kanalzahl-generisch. Ridge nutzt ``lambda = 1.0 * p`` mit ``p = 20 * Kanaele``,
damit die Schrumpfung pro Koeffizient beim Wechsel von 5 auf 10 Kanaele
konstant bleibt und der Feature-Effekt nicht mit einem Regularisierungseffekt
vermischt wird. Bei 5 Kanaelen ergibt das exakt die 100 des Bestands; die
Aequivalenz ist in ``tests/test_models_2x2.py`` gegen
``daily_models.fit_predict_models`` verifiziert.

Der Aufrufer liefert bereits vola-normierte, geclippte Label; dieses Modul
verschiebt, normiert und clippt y nicht. Label i ist ab Schluss
i + label_delay beobachtbar.
"""
import lightgbm as lgb
import numpy as np

RIDGE_LAMBDA_PER_FEATURE = 1.0
SEQUENCE = 20
BATCH = 256


def _checked(X, y, first_test, last_test, label_delay):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 4 or X.shape[2] != SEQUENCE or y.shape != X.shape[:2]:
        raise ValueError('expected X[D,A,20,C] and y[D,A]')
    days = y.shape[0]
    if (isinstance(label_delay, (bool, np.bool_))
            or not isinstance(label_delay, (int, np.integer)) or label_delay < 2):
        raise ValueError('label_delay must be an integer >=2')
    if (not isinstance(first_test, (int, np.integer))
            or not label_delay <= first_test < last_test <= days):
        raise ValueError('need label_delay <= first_test < last_test <= D')
    if y.shape[1] == 0:
        raise ValueError('no assets')
    return X, y


def training_pairs(y, first_test, label_delay=2):
    """(Tag, Asset)-Paare mit endlichem, bei first_test bereits reifem Label."""
    valid = np.isfinite(np.asarray(y))
    pairs = np.argwhere(valid[:first_test - label_delay + 1])
    if not len(pairs):
        raise ValueError('no finite training labels before the test block')
    return pairs


def _raw(X, pairs):
    batch = np.asarray(X[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
    if not np.isfinite(batch).all():
        raise ValueError('nonfinite features in eligible or prediction examples')
    return batch


def fit_scaler(X, pairs):
    """Kanalweise Mittelwerte/Populationsstreuungen, ueber Zeitschritte gepoolt.

    Zwei Durchlaeufe vermeiden Ausloeschung in der Varianz, ohne eine
    vollstaendige Kopie der Trainingsdaten zu materialisieren.
    """
    channels = X.shape[3]
    total = np.zeros(channels)
    for start in range(0, len(pairs), BATCH):
        total += _raw(X, pairs[start:start + BATCH]).sum(axis=(0, 1))
    mean = total / (len(pairs) * SEQUENCE)
    squares = np.zeros(channels)
    for start in range(0, len(pairs), BATCH):
        squares += ((_raw(X, pairs[start:start + BATCH]) - mean) ** 2).sum(axis=(0, 1))
    scale = np.sqrt(squares / (len(pairs) * SEQUENCE))
    scale[scale == 0] = 1
    return mean, scale


def standardize(X, pairs, mean, scale):
    return np.clip((_raw(X, pairs) - mean) / scale, -10, 10)


def _prediction_pairs(assets, day):
    return np.column_stack((np.full(assets, day), np.arange(assets)))


def fit_predict_ridge(X, y, first_test, last_test, label_delay=2):
    """Ridge-Prognosen fuer die Tage [first_test, last_test).

    Einmalige Anpassung auf allen bei first_test reifen Labeln, ungestrafter
    Achsenabschnitt. Nach dem Clipping zentriert: geclippte Trainings-
    koordinaten haben nicht zwingend Mittelwert 0.
    """
    X, y = _checked(X, y, first_test, last_test, label_delay)
    pairs = training_pairs(y, first_test, label_delay)
    mean, scale = fit_scaler(X, pairs)
    features = SEQUENCE * X.shape[3]
    gram, rhs = np.zeros((features, features)), np.zeros(features)
    z_sum, y_sum = np.zeros(features), 0.0
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start:start + BATCH]
        z = standardize(X, chunk, mean, scale).reshape(len(chunk), features)
        gram += z.T @ z
        targets = np.asarray(y[chunk[:, 0], chunk[:, 1]], dtype=np.float64)
        rhs += z.T @ targets
        z_sum += z.sum(axis=0)
        y_sum += targets.sum()
    z_mean, y_mean = z_sum / len(pairs), y_sum / len(pairs)
    penalty = RIDGE_LAMBDA_PER_FEATURE * features
    weights = np.linalg.solve(
        gram - np.outer(z_sum, z_mean) + penalty * np.eye(features),
        rhs - z_sum * y_mean,
    )
    intercept = y_mean - z_mean @ weights
    assets = y.shape[1]
    out = np.empty((last_test - first_test, assets), dtype=np.float64)
    for offset, day in enumerate(range(first_test, last_test)):
        z = standardize(X, _prediction_pairs(assets, day), mean, scale)
        out[offset] = z.reshape(assets, features) @ weights + intercept
    return out


# Eingefrorene Rezeptur, Spec Abschnitt 4. Bewusst klein fuer die kleine
# Stichprobe. Keine Hyperparametersuche; nach dem ersten Lauf unveraendert.
GBM_ROUNDS = 300
GBM_PARAMS = {
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 15,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'seed': 7,
    'deterministic': True,
    'force_row_wise': True,
    'num_threads': 2,
    'verbosity': -1,
}


def fit_predict_gbm(X, y, first_test, last_test, label_delay=2):
    """LightGBM-Prognosen fuer die Tage [first_test, last_test).

    Sieht exakt dieselbe flache, standardisierte, geclippte Matrix wie Ridge.
    Baeume brauchen die 20 stark korrelierten Verzoegerungen nicht; sie
    bekommen sie trotzdem, weil die Modellachse sonst mit einer
    Repraesentationsaenderung vermischt waere.
    """
    X, y = _checked(X, y, first_test, last_test, label_delay)
    pairs = training_pairs(y, first_test, label_delay)
    mean, scale = fit_scaler(X, pairs)
    features = SEQUENCE * X.shape[3]
    design = np.empty((len(pairs), features))
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start:start + BATCH]
        design[start:start + len(chunk)] = standardize(
            X, chunk, mean, scale).reshape(len(chunk), features)
    targets = np.asarray(y[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
    booster = lgb.train(GBM_PARAMS, lgb.Dataset(design, label=targets),
                        num_boost_round=GBM_ROUNDS)
    assets = y.shape[1]
    out = np.empty((last_test - first_test, assets), dtype=np.float64)
    for offset, day in enumerate(range(first_test, last_test)):
        z = standardize(X, _prediction_pairs(assets, day), mean, scale)
        out[offset] = booster.predict(z.reshape(assets, features))
    return out
