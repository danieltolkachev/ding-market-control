"""Labelkernel der S4-Mechanik, Spec Abschnitt 8.

OHLC sind beobachtete Handelskurse, keine Bid-/Ask-Pfade. Das Fill-Modell ist
deshalb eine reproduzierbare NAEHERUNG, kein Nachweis erreichbarer Fills.
Ausgaenge sind ausdruecklich market-on-touch, keine garantierten passiven
Limit-Fills; Spread und Slippage werden am Fill beruecksichtigt.

Die Eroeffnungsorder wird nach Signal t vor der naechsten Sitzung aufgegeben.
E wird erst beim simulierten Fill bekannt; es findet keine Auswahl anhand des
spaeter bekannten O[t+1] statt und keine Verwendung spaeterer H/L fuer den
Entry.

Dieses Modul verbraucht die ROH-Sicht der Preise plus Kapitalmassnahmen. Die
Indikatorsicht ist Sache des Eventgenerators; beide ineinander umzurechnen ist
Aufgabe des Datenadapters.

Ein Ein-Stueck-Netto-R ist KEIN wirtschaftlicher 10.000-EUR-Trade:
Mindestgebuehren gehoeren in die mengenabhaengige Portfolio-P&L.
"""
import numpy as np

CENSORED = 'CENSORED'
DATA_ERROR = 'DATA_ERROR'
ACTION_UNSUPPORTED = 'ACTION_UNSUPPORTED'

KNOWN_SHARE_BASES = ('point_in_time',)

GAP_STOP = 'GAP_STOP'
GAP_TARGET = 'GAP_TARGET'
STOP = 'STOP'
TARGET = 'TARGET'
TIME = 'TIME'

STOP_FIRST = 'stop_first'
TARGET_FIRST = 'target_first'

COLUMNS = ('open', 'high', 'low', 'close')
ACTION_COLUMNS = ('split', 'dividend')


def _failure(status, event, bar=None, field=None):
    """bar/field machen v.a. ACTION_UNSUPPORTED zu einem disclosablen Datenqualitaets-Fund."""
    return {'status': status, 'event_id': event['id'], 'bar': bar, 'field': field}


def _bar_invalid(values):
    """open/high/low/close; finite, strikt positiv, O und C innerhalb [L, H]."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        return True
    bar_open, high, low, close = values
    return low > min(bar_open, close) or max(bar_open, close) > high


def label_event(event, raw_bars, actions, b, f, tie=STOP_FIRST):
    """Barrier-Klasse, Meta-Label und Netto-R eines hypothetischen Long-Trades."""
    if tuple(raw_bars.columns) != COLUMNS or tuple(actions.columns) != ACTION_COLUMNS:
        raise ValueError('unexpected column layout')
    if not raw_bars.index.equals(actions.index):
        raise ValueError('bars and actions must share one calendar')
    if tie not in (STOP_FIRST, TARGET_FIRST):
        raise ValueError('tie must be stop_first or target_first')
    if not (0 <= b < 1) or not (0 <= f < 1):
        raise ValueError('cost parameters out of range')

    signal_time = event.get('signal_time')
    if signal_time is not None:
        if not (0 <= event['t'] < len(raw_bars)) or raw_bars.index[event['t']] != signal_time:
            raise ValueError('event signal_time does not match raw_bars calendar')
    share_basis = event.get('share_basis')
    if share_basis is not None and share_basis not in KNOWN_SHARE_BASES:
        raise ValueError(f'unrecognised share_basis {share_basis!r}')

    e = event['t'] + 1
    last = e + event['N'] - 1
    if e >= len(raw_bars):
        return _failure(CENSORED, event)

    values = raw_bars.to_numpy(dtype=float)
    split = actions['split'].to_numpy(dtype=float)
    dividend = actions['dividend'].to_numpy(dtype=float)
    if _bar_invalid(values[e]):
        return _failure(CENSORED, event)
    # Spec-Pseudocode sagt hier CENSORED; ACTION_UNSUPPORTED ist die bewusst
    # bessere Lesart, weil ein ungueltiger Splitfaktor ein Datenqualitaets-
    # befund ist, keine schlicht fehlende Beobachtung. Nicht zurueckdrehen.
    if not np.isfinite(split[e]) or split[e] <= 0:
        return _failure(ACTION_UNSUPPORTED, event, bar=e, field='split')

    entry = values[e][0] * (1.0 + b)
    risk = event['R'] / split[e]
    if not np.isfinite([entry, risk]).all() or entry <= 0 or risk <= 0:
        return _failure(DATA_ERROR, event)

    shares, dividends = 1.0, 0.0
    stop, target = entry - risk, entry + event['q'] * risk
    # S<=0 ist kein erreichbarer Stop, aber kein Grund, den Trade zu entfernen.
    nonpositive_stop = stop <= 0

    exit_reference, outcome, reason, ambiguous, exit_bar = None, None, None, False, None
    for k in range(e, last + 1):
        ambiguous = False   # pro Iteration frisch, nicht ueber Bars hinweg getragen
        if k >= len(raw_bars) or _bar_invalid(values[k]):
            return _failure(CENSORED, event)
        if not np.isfinite(split[k]) or split[k] <= 0:
            return _failure(ACTION_UNSUPPORTED, event, bar=k, field='split')
        if not np.isfinite(dividend[k]) or dividend[k] < 0:
            return _failure(ACTION_UNSUPPORTED, event, bar=k, field='dividend')

        if k > e:
            shares *= split[k]
            stop /= split[k]
            target /= split[k]
            dividends += shares * dividend[k]
            # Kein Dividendenanspruch bei Kauf erst am Ex-Tag e.

        bar_open, high, low, close = values[k]
        if bar_open <= stop:                       # Open ist zeitlich zuerst
            exit_reference, outcome, reason = bar_open, -1, GAP_STOP
            exit_bar = k
            break
        if bar_open >= target:
            exit_reference, outcome, reason = bar_open, 1, GAP_TARGET
            exit_bar = k
            break

        hit_stop, hit_target = low <= stop, high >= target
        ambiguous = hit_stop and hit_target
        stop_wins = hit_stop and (tie == STOP_FIRST or not hit_target)
        if stop_wins:
            exit_reference, outcome, reason = stop, -1, STOP
            exit_bar = k
            break
        if hit_target:
            exit_reference, outcome, reason = target, 1, TARGET
            exit_bar = k
            break
        if k == last:
            exit_reference, outcome, reason = close, 0, TIME
            exit_bar = k
            break

    if exit_reference is None:
        # N=0 oder ein sonst leeres Haltefenster landet hier: die Schleife
        # laeuft nie, also wird nichts erfunden -- fail closed als CENSORED.
        return _failure(CENSORED, event)

    exit_price = exit_reference * (1.0 - b)
    exit_notional = shares * exit_price
    net = exit_notional - entry + dividends - f * entry - f * exit_notional
    return {
        'barrier_class': outcome,
        'meta_label': 1 if net > 0 else 0,
        'net_R': net / risk,
        'entry': entry,
        'exit': exit_price,
        'exit_bar': exit_bar,
        'reason': reason,
        'ambiguous': bool(ambiguous),
        'nonpositive_stop': bool(nonpositive_stop),
        'label_available_after': raw_bars.index[exit_bar],
    }
