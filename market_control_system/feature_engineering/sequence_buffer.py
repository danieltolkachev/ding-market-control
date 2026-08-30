"""
sequence_buffer.py
===================

Wandelt den Strom einzelner Feature-Vektoren (aus feature_pipeline.py) in
das Sequenz-Format um, das das LSTM erwartet:

    X in R^(Samples x Timesteps x Features)

Fuer den Live-Betrieb liefert SequenceBuffer pro Tick genau EIN Fenster
(Timesteps x Features), das dann fuer die Inferenz an das Modell geht.
Fuer Training/Backtest baut SequenceWindowBuilder aus einem kompletten
Feature-DataFrame alle Sample-Fenster auf einmal (vektorisiert).

Designentscheidung: feste `feature_names`-Liste als Vertrag.
--------------------------------------------------------------
Ein Python-dict hat in aelteren Versionen keine garantierte Ordnungs-
Semantik im Sinne eines Vertrags, und selbst mit Ordnungsgarantie
(Python 3.7+) ist es riskant, sich stillschweigend darauf zu verlassen,
wenn verschiedene Module (Live-Pfad, Batch-Pfad, spaetere Feature-
Erweiterungen) dicts bauen. Deshalb wird die Feature-Reihenfolge hier
EXPLIZIT als Liste uebergeben und bei jedem Push validiert. Das
verhindert den gefaehrlichsten Fehlerklasse in solchen Systemen: Modell
trainiert auf Reihenfolge [ret, vol, obi, spread], Live-System fuettert
[vol, ret, obi, spread] -> stiller Silent-Bug, keine Exception, nur
falsche Prognosen.
"""

from __future__ import annotations

from collections import deque
import numpy as np
import pandas as pd


class SequenceBuffer:
    """
    Ring-Buffer fuer den Live-/Streaming-Pfad. Haelt genau `timesteps` viele
    Feature-Vektoren vor und liefert bei Bedarf das aktuelle Fenster als
    numpy-Array mit Shape (timesteps, n_features).
    """

    def __init__(self, timesteps: int, feature_names: list[str]):
        if timesteps < 1:
            raise ValueError("timesteps muss >= 1 sein")
        self.timesteps = timesteps
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        self._buffer: deque[np.ndarray] = deque(maxlen=timesteps)

    def push(self, feature_dict: dict[str, float]) -> None:
        """
        Fuegt einen neuen Feature-Vektor hinzu. Erzwingt Vollstaendigkeit
        und korrekte Reihenfolge relativ zu self.feature_names.

        Raises:
            KeyError: falls ein erwartetes Feature fehlt (fail-fast statt
                      stillschweigend NaN einzufuegen, das spaeter im Modell
                      unbemerkt zu falschen Gradienten fuehren wuerde).
        """
        try:
            vector = np.array(
                [feature_dict[name] for name in self.feature_names],
                dtype=np.float32,
            )
        except KeyError as e:
            raise KeyError(
                f"Feature-Vektor unvollstaendig, fehlendes Feature: {e}. "
                f"Erwartet: {self.feature_names}"
            ) from e

        if np.any(np.isnan(vector)):
            raise ValueError(
                f"NaN im Feature-Vektor erkannt: {dict(zip(self.feature_names, vector))}. "
                "NaN darf nicht in den Sequence-Buffer gelangen -> Warm-up-Logik "
                "der vorgelagerten Pipeline pruefen."
            )

        self._buffer.append(vector)

    def is_ready(self) -> bool:
        """True, sobald `timesteps` viele Vektoren vorliegen."""
        return len(self._buffer) == self.timesteps

    def get_window(self) -> np.ndarray:
        """
        Liefert das aktuelle Fenster als (timesteps, n_features)-Array.
        Wirft, wenn der Buffer noch nicht voll ist -> explizites Fail-Fast
        statt eines stillschweigend unvollstaendigen/gepaddeten Fensters,
        das dem Modell einen falschen Kontext vortaeuschen wuerde.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer nicht bereit: {len(self._buffer)}/{self.timesteps} "
                "Timesteps gefuellt. is_ready() vor get_window() pruefen."
            )
        return np.stack(self._buffer, axis=0)

    def get_batch_window(self) -> np.ndarray:
        """
        Wie get_window(), aber mit zusaetzlicher Batch-Dimension der Groesse 1:
        Shape (1, timesteps, n_features) -- direkt kompatibel mit
        model.predict() bei den meisten Keras/PyTorch-Modellen.
        """
        return self.get_window()[np.newaxis, ...]

    def reset(self) -> None:
        self._buffer.clear()


class SequenceWindowBuilder:
    """
    Batch-Pendant zu SequenceBuffer: baut aus einem kompletten
    Feature-DataFrame ALLE gueltigen (X, y)-Sample-Fenster auf einmal,
    vektorisiert via numpy sliding_window_view.

    Fuer Training/Backtest -- nicht fuer den Live-Loop gedacht (dort:
    SequenceBuffer).
    """

    def __init__(self, timesteps: int, feature_names: list[str]):
        self.timesteps = timesteps
        self.feature_names = list(feature_names)

    def build(
        self, features: pd.DataFrame, target: pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, pd.Index]:
        """
        Args:
            features: DataFrame mit Spalten == self.feature_names (Reihenfolge
                       wird hier erzwungen, nicht angenommen).
            target: optionale Zielvariable (z.B. future_return), gleicher Index
                    wie features.

        Returns:
            X: Array (n_samples, timesteps, n_features)
            y: Array (n_samples,) oder None
            sample_end_index: pandas.Index der jeweils LETZTEN Zeitpunkte
                               jedes Fensters (fuer Debugging/Nachvollziehbarkeit,
                               welches Fenster zu welchem Ausgangszeitpunkt gehoert)
        """
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise KeyError(f"Fehlende Feature-Spalten im DataFrame: {missing}")

        ordered = features[self.feature_names]

        # Nur vollstaendige Zeilen (Warm-up-NaNs raus) -- aber Index behalten,
        # um Kontinuitaet der Fenster zu pruefen (siehe Hinweis unten).
        valid_mask = ordered.notna().all(axis=1)
        if target is not None:
            valid_mask &= target.notna()

        ordered_valid = ordered[valid_mask]
        arr = ordered_valid.to_numpy(dtype=np.float32)

        n_rows, n_features = arr.shape
        if n_rows < self.timesteps:
            raise ValueError(
                f"Zu wenige valide Zeilen ({n_rows}) fuer timesteps={self.timesteps}"
            )

        # Sliding-Window-View: erzeugt (n_samples, timesteps, n_features)
        # ohne die Daten zu kopieren (Performance bei grossen Datensaetzen)
        n_samples = n_rows - self.timesteps + 1
        window_view = np.lib.stride_tricks.sliding_window_view(
            arr, window_shape=self.timesteps, axis=0
        )  # shape: (n_samples, n_features, timesteps) -- Achsen noch vertauscht
        X = np.transpose(window_view, (0, 2, 1))  # -> (n_samples, timesteps, n_features)

        sample_end_index = ordered_valid.index[self.timesteps - 1 :]

        y = None
        if target is not None:
            target_valid = target[valid_mask]
            y = target_valid.to_numpy(dtype=np.float32)[self.timesteps - 1 :]

        return X, y, sample_end_index


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from feature_pipeline import FeaturePipeline, _generate_synthetic_market_data

    df = _generate_synthetic_market_data(n=300, seed=3)
    features, target = FeaturePipeline().transform_with_target(df, horizon=5)

    feature_names = list(features.columns)

    # --- Batch-Pfad ---
    builder = SequenceWindowBuilder(timesteps=20, feature_names=feature_names)
    X, y, end_idx = builder.build(features, target)
    print("=== SequenceWindowBuilder (Batch) ===")
    print(f"X shape: {X.shape}  (samples, timesteps, features)")
    print(f"y shape: {y.shape}")
    print(f"Erstes Fenster, letzter Timestep == Originaldaten an Index {end_idx[0]}? "
          f"{np.allclose(X[0, -1, :], features.loc[end_idx[0]].to_numpy(dtype=np.float32))}")

    # --- Live-Pfad ---
    print("\n=== SequenceBuffer (Live) ===")
    buf = SequenceBuffer(timesteps=20, feature_names=feature_names)
    ready_at = None
    for i, (_, row) in enumerate(features.dropna().iterrows()):
        buf.push(row.to_dict())
        if buf.is_ready() and ready_at is None:
            ready_at = i
    print(f"Buffer wurde nach {ready_at + 1} Pushes ready (erwartet: 20)")
    window = buf.get_batch_window()
    print(f"Live-Fenster-Shape: {window.shape}  (erwartet: (1, 20, {len(feature_names)}))")

    # Fehlerfall demonstrieren: unvollstaendiger Vektor
    print("\n=== Fail-Fast bei fehlendem Feature ===")
    try:
        buf.push({"log_return": 0.001})  # absichtlich unvollstaendig
    except KeyError as e:
        print(f"Korrekt abgefangen: {e}")
