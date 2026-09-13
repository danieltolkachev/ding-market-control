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

Die Indikatorsicht muss POINT-IN-TIME sein: nur zum jeweiligen Bar bereits
wirksame Splits duerfen eingerechnet sein. Das ist keine Pedanterie. Jede
Entscheidung des Generators ist skalenfrei -- EMA50 > EMA200, C > EMA50, ein
ATR-skaliertes Touch-Band -- deshalb erzeugt eine voll rueckadjustierte Reihe
(die naheliegende Wahl fuer einen kuenftigen Adapter) denselben Eventstrom,
und nichts sieht falsch aus. `event['R'] = atr20[t-1]` staende dann aber auf
der heutigen Stueckbasis, und `risk = event['R'] / split[e]` im Labelkernel
wuerde eine Korrektur anwenden, die bereits vorgenommen wurde -- `net_R` waere
fuer jeden betroffenen Trade stillschweigend um den Splitfaktor verschoben.
Deshalb stempelt jedes Event seine Basis in `share_basis = 'point_in_time'`;
`label_event` lehnt eine unbekannte Basis mit `ValueError` ab. Ein fehlender
Schluessel gilt als Legacy und wird akzeptiert.

Das Warmup-Gate garantiert, dass build_features bei jedem emittierten Event
alle benoetigten Indizes vorfindet; der feature_error-Pfad wird deshalb auf
gewoehnlichen Daten praktisch nie ausgeloest und nur auf entarteten Daten
erreicht, etwa einer flachen Strecke, die atr20 oder sigma20 auf null zieht.
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
SHARE_BASIS = 'point_in_time'
FEATURE_INVALID = 'feature_error'

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
                # NICHT redundant, trotz des naechsten Schleifendurchlaufs, der
                # t > pending_p + CONFIRM_MAX ohnehin abweisen wuerde. Mit
                # diesem Reset ist der Zustand bei Bar p+4 bereits IDLE, also
                # kann p+4 selbst eine neue Beruehrung (einen neuen
                # pending_p) eroeffnen. Ohne diesen Reset liefe p+4 stattdessen
                # zuerst in den WAIT_CONFIRM-Zweig, faende dort
                # t > pending_p + CONFIRM_MAX und wuerde konsumiert -- p+4
                # koennte dann NIE selbst zur neuen Beruehrungsbar werden.
                # Entfernen wuerde den Eventstrom still veraendern.
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
        FEATURE_INVALID: error,
        'share_basis': SHARE_BASIS,
    }
