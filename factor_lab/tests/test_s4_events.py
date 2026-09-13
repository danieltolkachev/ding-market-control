"""Zustandsautomat des S4-Eventgenerators, Spec Abschnitt 6.

ALLE Daten in dieser Datei sind SYNTHETISCH und dienen ausschliesslich der
Mechanikpruefung. Sie stammen aus keiner Marktquelle.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from factor_lab.s4_events import (
    COOLDOWN_BARS,
    EVENT_VERSION,
    HOLD_N,
    TARGET_Q,
    generate_events,
)
from factor_lab.s4_indicators import compute_indicators


def trending_bars(n=600, seed=7, drift=0.0012, noise=0.006):
    """Synthetischer Aufwaertstrend mit Ruecksetzern, damit S4 ueberhaupt feuert."""
    rng = np.random.default_rng(seed)
    steps = drift + noise * rng.standard_normal(n)
    steps[::37] -= 4 * noise           # regelmaessige Pullbacks erzwingen
    close = 100.0 * np.exp(np.cumsum(steps))
    width = close * 0.008
    high = close + width * rng.uniform(0.3, 1.0, n)
    low = close - width * rng.uniform(0.3, 1.0, n)
    open_ = low + (high - low) * rng.uniform(0.0, 1.0, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close},
                        index=pd.bdate_range('2008-01-02', periods=n))


class ContractTests(unittest.TestCase):
    def test_constants_are_frozen(self):
        self.assertEqual(COOLDOWN_BARS, 10)
        self.assertEqual(HOLD_N, 10)
        self.assertEqual(TARGET_Q, 1.5)
        self.assertEqual(EVENT_VERSION, 'S4-v1')

    def test_emitted_events_carry_the_frozen_contract(self):
        bars = trending_bars()
        ind = compute_indicators(bars)
        events = generate_events(bars, 'SYNTH', ind)
        self.assertGreater(len(events), 0)
        for event in events:
            self.assertEqual(event['id'], ('SYNTH', event['t'], 'S4-v1'))
            self.assertEqual(event['side'], 1)
            self.assertEqual(event['q'], 1.5)
            self.assertEqual(event['N'], 10)
            self.assertEqual(event['earliest_entry_bar'], event['t'] + 1)
            self.assertAlmostEqual(event['R'], ind['atr20'][event['t'] - 1], places=12)
            self.assertEqual(event['signal_time'], bars.index[event['t']])
            self.assertIn(event['t'] - event['setup_start'], (1, 2, 3))

    def test_no_event_before_the_warmup_is_complete(self):
        bars = trending_bars()
        for event in generate_events(bars, 'SYNTH'):
            self.assertGreaterEqual(event['t'], 259)


class TimingTests(unittest.TestCase):
    def test_events_respect_the_cooldown(self):
        events = generate_events(trending_bars(), 'SYNTH')
        times = [event['t'] for event in events]
        self.assertEqual(times, sorted(times))
        for earlier, later in zip(times, times[1:]):
            self.assertGreater(later, earlier + COOLDOWN_BARS)

    def test_label_windows_never_overlap_per_instrument(self):
        """Entry t+1, letzte gehaltene Bar t+10, naechstes Event fruehestens t+11."""
        events = generate_events(trending_bars(), 'SYNTH')
        for earlier, later in zip(events, events[1:]):
            self.assertGreaterEqual(later['t'] + 1, earlier['t'] + HOLD_N + 1)

    def test_confirmation_window_is_at_most_three_bars(self):
        for event in generate_events(trending_bars(), 'SYNTH'):
            self.assertLessEqual(event['t'], event['setup_start'] + 3)

    def test_generator_is_deterministic(self):
        bars = trending_bars()
        a = [event['t'] for event in generate_events(bars, 'SYNTH')]
        b = [event['t'] for event in generate_events(bars, 'SYNTH')]
        self.assertEqual(a, b)


class StateMachineTests(unittest.TestCase):
    def test_no_event_is_emitted_on_the_touch_bar_itself(self):
        for event in generate_events(trending_bars(), 'SYNTH'):
            self.assertGreater(event['t'], event['setup_start'])

    def test_events_use_only_information_available_at_t(self):
        """Bars nach t veraendern weder Zeitpunkt noch R eines Events."""
        bars = trending_bars()
        events = generate_events(bars, 'SYNTH')
        self.assertGreater(len(events), 2)
        cut = events[1]['t']
        tampered = bars.copy()
        tampered.iloc[cut + 1:] *= 1.7
        after = generate_events(tampered, 'SYNTH')
        early_before = [(e['t'], e['setup_start'], e['R']) for e in events if e['t'] <= cut]
        early_after = [(e['t'], e['setup_start'], e['R']) for e in after if e['t'] <= cut]
        self.assertEqual(early_before, early_after)

    def test_a_data_break_clears_the_pending_setup(self):
        bars = trending_bars()
        events = generate_events(bars, 'SYNTH')
        self.assertGreater(len(events), 0)
        target = events[0]
        broken = bars.copy()
        broken.iloc[target['setup_start'] + 1] = np.nan
        survivors = [e['t'] for e in generate_events(broken, 'SYNTH')]
        self.assertNotIn(target['t'], survivors)

    def test_no_event_inside_the_cooldown_window_after_a_data_break(self):
        """Regressionsschutz, kein isolierter Beweis der Cooldown-Persistenz.

        Die absolute Cooldown-Grenze soll eine Luecke ueberleben. Beobachtbar
        ist das durch die oeffentliche Schnittstelle nur eingeschraenkt, weil
        der Warmup-Gate nach einem Bruch ohnehin rund 260 Bars lang jedes
        Event unterdrueckt. Der Test haelt die Eigenschaft fest, beweist sie
        aber nicht unabhaengig vom Warmup.
        """
        bars = trending_bars()
        events = generate_events(bars, 'SYNTH')
        self.assertGreater(len(events), 1)
        first = events[0]['t']
        broken = bars.copy()
        broken.iloc[first + 2] = np.nan
        for event in generate_events(broken, 'SYNTH'):
            self.assertFalse(first < event['t'] <= first + COOLDOWN_BARS)

    def test_successful_features_have_the_expected_shapes(self):
        bars = trending_bars()
        events = generate_events(bars, 'SYNTH')
        for event in events:
            if event['feature_error'] is None:
                self.assertEqual(event['sequence'].shape, (60, 10))
                self.assertEqual(event['context'].shape, (4,))
            else:
                self.assertIsNone(event['sequence'])
                self.assertIsNone(event['context'])

    def test_a_feature_failure_keeps_the_event_in_the_stream(self):
        """Spec: FEATURE_INVALID wird protokolliert, NICHT aus dem Strom entfernt."""
        bars = trending_bars()
        with mock.patch('factor_lab.s4_events.build_features',
                        side_effect=ValueError('synthetic feature failure')):
            events = generate_events(bars, 'SYNTH')
        self.assertGreater(len(events), 0)
        for event in events:
            self.assertIsNone(event['sequence'])
            self.assertIsNone(event['context'])
            self.assertEqual(event['feature_error'], 'synthetic feature failure')
        clean = [e['t'] for e in generate_events(bars, 'SYNTH')]
        self.assertEqual([e['t'] for e in events], clean)


if __name__ == '__main__':
    unittest.main()
