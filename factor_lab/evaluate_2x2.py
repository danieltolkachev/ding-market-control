"""Refit-Fahrplan des 2x2-Rasters: fuenf expandierende Fenster mit Purge.

Die Fenster sind kein Portfolio-Neustart. Jede Auswertungszeile bekommt ihre
Prognose von dem Fenster, das sie besitzt; simuliert wird spaeter **eine**
durchgehende Kursbahn, damit die Zelle gegen exakt dieselbe immer-investierte
Kontrolle laeuft wie im Horizon-Vergleich.

Der Purge von ``label_delay`` Tagen an jeder Fenstergrenze passiert an genau
einer Stelle: ``models_2x2.training_pairs`` schneidet bei
``first_test - label_delay`` ab.
"""
import numpy as np

from factor_lab.models_2x2 import fit_predict_gbm, fit_predict_ridge

N_WINDOWS = 5
CELLS = (('A_base_ridge', 'X_base', 'ridge'),
         ('B_base_gbm', 'X_base', 'gbm'),
         ('C_ext_ridge', 'X_ext', 'ridge'),
         ('D_ext_gbm', 'X_ext', 'gbm'))
_FITTERS = {'ridge': fit_predict_ridge, 'gbm': fit_predict_gbm}


def expanding_windows(first_test, last_day, n_windows=N_WINDOWS):
    """Zusammenhaengende, disjunkte Testbloecke ueber [first_test, last_day).

    Alle Bloecke bis auf den letzten sind gleich gross; die Restzeilen der
    Ganzzahldivision gehen vollstaendig an den letzten Block. Bei 2932 Zeilen
    und fuenf Fenstern sind das 586/586/586/586/588 - der Unterschied ist
    gegenueber der Blockgroesse vernachlaessigbar und die Regel bleibt eine
    Zeile Code statt einer Verteilungslogik.
    """
    if not isinstance(n_windows, (int, np.integer)) or n_windows < 1:
        raise ValueError('n_windows must be a positive integer')
    span = int(last_day) - int(first_test)
    if span < n_windows:
        raise ValueError('evaluation range too short for the requested windows')
    size = span // n_windows
    bounds = [first_test + i * size for i in range(n_windows)] + [last_day]
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_windows)]


def predict_over_windows(X, y, windows, model, label_delay=2, progress=None):
    """Prognosen ueber alle Fenster, aneinandergesetzt zu [last-first, A]."""
    if model not in _FITTERS:
        raise ValueError(f'unknown model {model!r}')
    fitter = _FITTERS[model]
    first, last = windows[0][0], windows[-1][1]
    out = np.empty((last - first, y.shape[1]), dtype=np.float64)
    for number, (start, end) in enumerate(windows, start=1):
        if progress is not None:
            progress(f'window {number}/{len(windows)}: train < {start}, test [{start},{end})')
        out[start - first:end - first] = fitter(X, y, start, end, label_delay)
    return out
