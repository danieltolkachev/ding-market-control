"""S4-Eventgenerator, Spec Abschnitt 6.

Long-Events auf bestaetigten Pullbacks in einem quantitativ definierten
Aufwaertstrend. Der Generator laeuft einmal chronologisch und haengt weder
von spaeteren Labels noch von einer spaeteren Modellentscheidung ab: ein
Event wird auch dann emittiert, wenn das Meta-Modell spaeter SKIP sagt oder
wenn die Features nicht berechenbar sind.

Der Cooldown ist unabhaengig davon, ob ein frueherer Trade Stop oder Target
erreicht hat. Sonst koennten Labels und Modellannahmen den zukuenftigen
Eventstrom veraendern. An Train-/Kalibrierungs-/Testgrenzen wird weder der
Setup-Zustand noch der Cooldown zurueckgesetzt.

Der Generator konsumiert die INDIKATORSICHT der Preise (split-konsistente
Historie). Roh-Bars fuer Fills sind Sache des Labelkernels; die beiden
Sichten ineinander umzurechnen ist Aufgabe des Datenadapters, nicht dieses
Moduls.
"""
import numpy as np

from factor_lab.s4_features import build_features
from factor_lab.s4_indicators import WARMUP, compute_indicators

CONFIRM_MAX = 3
HOLD_N = 10
TARGET_Q = 1.5
COOLDOWN_BARS = 10
TOUCH_FRACTION = 0.25
EVENT_VERSION = 'S4-v1'

_IDLE, _WAIT_CONFIRM = 'IDLE', 'WAIT_CONFIRM'


def _usable(ind, t):
    """Warmup-Gate: mindestens WARMUP zusammenhaengende gueltige Bars bis t."""
    if t < 1 or not ind['valid'][t] or ind['run_length'][t] < WARMUP:
        return False
    return np.isfinite([ind['atr20'][t - 1], ind['ema20'][t - 1], ind['ema50'][t - 1]]).all()


def generate_events(bars, instrument, ind=None):
    """Chronologischer Eventstrom eines Instruments; ein Durchlauf, kein Reset."""
    if ind is None:
        ind = compute_indicators(bars)
    close = bars['close'].to_numpy(dtype=float)
    high = bars['high'].to_numpy(dtype=float)
    low = bars['low'].to_numpy(dtype=float)

    events = []
    state, pending_p, blocked_through = _IDLE, None, -1
    for t in range(len(bars)):
        if not _usable(ind, t):
            state, pending_p = _IDLE, None
            continue
        if t <= blocked_through:
            continue

        if state == _WAIT_CONFIRM:
            # Invalidierung VOR Bestaetigung; keine neue Vorstufe am selben Bar.
            if t > pending_p + CONFIRM_MAX or close[t] <= ind['ema50'][t - 1]:
                state, pending_p = _IDLE, None
                continue
            if ind['up'][t] and close[t] > high[t - 1]:
                events.append(_emit(bars, ind, instrument, pending_p, t))
                blocked_through = t + COOLDOWN_BARS
                state, pending_p = _IDLE, None
                continue
            if t == pending_p + CONFIRM_MAX:
                state, pending_p = _IDLE, None
            continue

        # IDLE: nur die erste passende Beruehrung speichern, kein Event hier.
        a = ind['atr20'][t - 1]
        center = ind['ema20'][t - 1]
        touched = (low[t] <= center + TOUCH_FRACTION * a
                   and high[t] >= center - TOUCH_FRACTION * a)
        if (ind['up'][t - 1] and close[t - 1] > center and close[t] < close[t - 1]
                and touched and close[t] > ind['ema50'][t - 1]):
            pending_p, state = t, _WAIT_CONFIRM
    return events


def _emit(bars, ind, instrument, p, t):
    try:
        sequence, context = build_features(ind, bars, p, t)
        error = None
    except ValueError as problem:      # FEATURE_INVALID, Event bleibt im Strom
        sequence, context, error = None, None, str(problem)
    return {
        'id': (instrument, t, EVENT_VERSION),
        't': t,
        'signal_time': bars.index[t],
        'setup_start': p,
        'side': 1,
        'R': float(ind['atr20'][t - 1]),
        'q': TARGET_Q,
        'N': HOLD_N,
        'sequence': sequence,
        'context': context,
        'earliest_entry_bar': t + 1,
        'feature_error': error,
    }
