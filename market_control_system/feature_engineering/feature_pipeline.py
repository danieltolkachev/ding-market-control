"""
feature_pipeline.py
====================

Feature-Engineering-Layer fuer das LSTM-zentrierte Market-Control-System.

Design-Prinzipien:
------------------
1. KAUSALITAET: Jedes Feature zum Zeitpunkt t darf ausschliesslich Informationen
   aus [t-n, t] verwenden. Kein zukuenftiges Wissen, kein globales Fitting
   (z.B. kein StandardScaler().fit(ganzer_datensatz)).
2. VEKTORISIERUNG: Fuer Batch-/Backtest-Verarbeitung werden pandas/numpy
   Rolling-Operationen genutzt (schnell, aber weiterhin kausal, da
   pandas .rolling() standardmaessig rueckwaertsschauend ist).
3. STREAMING-FAEHIGKEIT: Fuer den Live-Betrieb gibt es eine inkrementelle
   Variante (RollingStats/RollingSum), die O(1) pro neuem Datenpunkt
   aktualisiert wird, statt bei jedem Tick ueber das gesamte Fenster neu
   zu iterieren.
4. TRENNUNG: Feature-Berechnung (dieses Modul) ist strikt getrennt von
   Modell-Input-Formatierung (sequence_buffer.py) und von Skalierung,
   die selbst wieder kausal sein muss (RollingZScoreScaler).
5. PARITAET: Batch- (FeaturePipeline) und Live-Pfad (LiveFeatureEngine)
   muessen bei identischen Rohdaten identische Feature-Werte liefern.
   Siehe tests/test_consistency.py -- jede Erweiterung um ein neues
   Feature MUSS in beiden Pfaden UND im Test ergaenzt werden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Batch-Feature-Berechnung (fuer Training / Backtesting)
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Zentrale Konfiguration fuer alle Fenstergroessen und Parameter."""
    volatility_window: int = 30          # Bars fuer realized volatility
    vwap_window: int = 20                # Bars fuer VWAP
    trade_intensity_window: int = 50     # Bars fuer Trade-Intensity-Referenz
    zscore_window: int = 100             # Fenster fuer rolling z-score scaling
    epsilon: float = 1e-8                # numerische Stabilitaet


# Kanonische Feature-Reihenfolge -- MUSS exakt der Spaltenreihenfolge in
# FeaturePipeline.transform() entsprechen. Einzige Quelle der Wahrheit fuer
# Aufrufer (z.B. SequenceBuffer/SequenceWindowBuilder, orchestration/control_loop.py),
# die sonst die Liste an mehreren Stellen redundant und damit fehleranfaellig
# hardcoden muessten -- siehe Design-Hinweis zur Feature-Reihenfolge in
# sequence_buffer.py.
FEATURE_NAMES: tuple[str, ...] = (
    "log_return",
    "realized_vol",
    "orderbook_imbalance",
    "spread_norm",
    "vwap_deviation",
    "trade_intensity",
    "mid_price_return",
)

# Optionales erweitertes Feature-Set inkl. News-Sentiment (siehe
# data_layer/news_client.py). Additiv, NICHT von FeaturePipeline/
# LiveFeatureEngine selbst berechnet -- die News-Spalten werden separat
# (batch: compute_news_features(), live: LiveNewsFeatureEngine) berechnet
# und erst danach angehaengt. FeaturePipeline/LiveFeatureEngine bleiben
# dadurch unangetastet und rueckwaertskompatibel fuer alle Skripte, die
# kein News-Feature wollen.
NEWS_FEATURE_NAMES: tuple[str, ...] = ("news_intensity", "news_sentiment")
EXTENDED_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + NEWS_FEATURE_NAMES

# Marktrelatives Feature (siehe data_layer/market_relative.py) -- ebenfalls
# additiv, gleiches Prinzip wie NEWS_FEATURE_NAMES: extern berechnet und
# angehaengt, FeaturePipeline/LiveFeatureEngine bleiben unangetastet.
MARKET_RELATIVE_FEATURE_NAMES: tuple[str, ...] = ("market_relative_return",)
MARKET_RELATIVE_FEATURE_NAMES_SET: tuple[str, ...] = FEATURE_NAMES + MARKET_RELATIVE_FEATURE_NAMES


class FeaturePipeline:
    """
    Berechnet den vollstaendigen Feature-Satz aus rohen Marktdaten (Batch-Modus).

    Erwartetes Input-DataFrame (pro Zeile = 1 Bar/Tick-Aggregat):
        timestamp, price, volume, bid, ask, bid_volume, ask_volume, trade_count

    Output: DataFrame mit den Rohfeatures (noch NICHT skaliert).
    Skalierung erfolgt separat via RollingZScoreScaler, da Skalierung
    und Feature-Berechnung unterschiedliche Fenster-Logik brauchen koennen.
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.cfg = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erzeugt alle Features kausal (nur rolling/backward-looking Operationen)."""
        out = pd.DataFrame(index=df.index)

        # --- 1. Log-Returns ---
        out["log_return"] = np.log(df["price"]).diff()

        # --- 2. Realized Volatility (rolling std der log returns) ---
        out["realized_vol"] = out["log_return"].rolling(
            window=self.cfg.volatility_window, min_periods=self.cfg.volatility_window
        ).std()

        # --- 3. Orderbook Imbalance ---
        bid_vol = df["bid_volume"]
        ask_vol = df["ask_volume"]
        out["orderbook_imbalance"] = (bid_vol - ask_vol) / (
            bid_vol + ask_vol + self.cfg.epsilon
        )

        # --- 4. Normalisierter Spread ---
        mid = (df["bid"] + df["ask"]) / 2.0
        out["spread_norm"] = (df["ask"] - df["bid"]) / (mid + self.cfg.epsilon)

        # --- 5. VWAP-Deviation ---
        pv = df["price"] * df["volume"]
        vwap = (
            pv.rolling(self.cfg.vwap_window, min_periods=self.cfg.vwap_window).sum()
            / df["volume"].rolling(self.cfg.vwap_window, min_periods=self.cfg.vwap_window).sum()
        )
        out["vwap_deviation"] = (df["price"] - vwap) / (vwap + self.cfg.epsilon)

        # --- 6. Trade Intensity (relative Handelsaktivitaet) ---
        avg_trades = df["trade_count"].rolling(
            self.cfg.trade_intensity_window,
            min_periods=self.cfg.trade_intensity_window,
        ).mean()
        out["trade_intensity"] = df["trade_count"] / (avg_trades + self.cfg.epsilon)

        # --- 7. Mid-Preis-Return als zusaetzliches Robustheits-Feature ---
        out["mid_price_return"] = np.log(mid).diff()

        return out

    def transform_with_target(
        self, df: pd.DataFrame, horizon: int = 1
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Erzeugt Features UND das Zielvariable-Paar fuer Training.

        WICHTIG: Das Target ist der Return von t nach t+horizon -- die Zeile
        bei Index t darf also erst nach dem Shift verwendet werden, sonst
        entsteht Look-Ahead-Bias (das Modell wuerde sein eigenes Ziel sehen).
        """
        features = self.transform(df)
        future_return = (
            np.log(df["price"]).shift(-horizon) - np.log(df["price"])
        )
        # Letzte `horizon` Zeilen haben kein gueltiges Target -> NaN, spaeter droppen
        return features, future_return


# ---------------------------------------------------------------------------
# 2. Kausales Rolling Z-Score Scaling
# ---------------------------------------------------------------------------

class RollingZScoreScaler:
    """
    Skaliert Features kausal: z_t = (x_t - mu_{t-n:t-1}) / (sigma_{t-n:t-1} + eps)

    Im Gegensatz zu sklearn.StandardScaler wird NICHT einmalig auf dem
    gesamten Datensatz gefittet, sondern pro Zeitpunkt neu relativ zur
    Vergangenheit berechnet. Das verhindert, dass zukuenftige Verteilungs-
    eigenschaften (z.B. Volatilitaetsregime von uebermorgen) ins Training
    von heute durchsickern.
    """

    def __init__(self, window: int = 100, epsilon: float = 1e-8):
        self.window = window
        self.epsilon = epsilon

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        rolling_mean = features.rolling(self.window, min_periods=self.window).mean()
        rolling_std = features.rolling(self.window, min_periods=self.window).std()
        return (features - rolling_mean) / (rolling_std + self.epsilon)


def build_scaled_features_and_target(
    df: pd.DataFrame,
    horizon: int,
    zscore_window: int = 100,
    config: FeatureConfig = FeatureConfig(),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Bequemlichkeitsfunktion: FeaturePipeline.transform_with_target() +
    RollingZScoreScaler in einem Schritt.

    Existiert, damit Aufrufer (Offline-Vortraining in den orchestration/
    run_*.py-Skripten) nicht beides einzeln verdrahten muessen -- genau
    das Vergessen des zweiten Schritts war der Bug, den IncrementalZScore
    Scaler/diese Funktion beheben (siehe deren Docstrings).
    """
    features, target = FeaturePipeline(config).transform_with_target(df, horizon=horizon)
    scaled = RollingZScoreScaler(window=zscore_window).transform(features)
    return scaled, target


# ---------------------------------------------------------------------------
# 3. Streaming-faehige inkrementelle Bausteine (fuer Live/Paper-Trading)
# ---------------------------------------------------------------------------

class IncrementalStats:
    """
    O(1)-Update von Mittelwert und Varianz ueber ein gleitendes Fenster.
    Vermeidet O(n)-Neuberechnung bei jedem neuen Tick.
    """

    def __init__(self, window: int):
        self.window = window
        self.buffer: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, x: float) -> None:
        if len(self.buffer) == self.window:
            oldest = self.buffer[0]
            self._sum -= oldest
            self._sum_sq -= oldest ** 2
        self.buffer.append(x)
        self._sum += x
        self._sum_sq += x ** 2

    @property
    def mean(self) -> float:
        n = len(self.buffer)
        return self._sum / n if n > 0 else 0.0

    @property
    def std(self) -> float:
        """
        Stichprobenvarianz (ddof=1, Bessel-Korrektur), um konsistent mit
        pandas.rolling().std() (Batch-Pfad) zu sein. Ohne diese Korrektur
        divergieren Batch- und Live-Pfad systematisch um den Faktor
        sqrt(n/(n-1)) -> siehe tests/test_consistency.py.
        """
        n = len(self.buffer)
        if n < 2:
            return 0.0
        population_var = (self._sum_sq / n) - (self.mean ** 2)
        sample_var = population_var * n / (n - 1)
        return float(np.sqrt(max(sample_var, 0.0)))

    def is_ready(self) -> bool:
        return len(self.buffer) == self.window


class IncrementalZScoreScaler:
    """
    Live-Pendant zu RollingZScoreScaler: kausales Rolling-Z-Score-Scaling
    pro Feature, ein IncrementalStats je Feature-Name.

    BUGFIX-Kontext (2026-08-26): RollingZScoreScaler existierte bereits,
    wurde aber nirgendwo im tatsaechlichen Live-/Backtest-Pfad aufgerufen
    -- alle Features gingen unskaliert (log_return ~0.001, orderbook_
    imbalance -1..1, trade_intensity ~1.0, ...) direkt ins LSTM. Vermuteter
    Zusammenhang: mehrere additive Feature-Experimente (News-Sentiment,
    market_relative_return) verschlechterten das Ergebnis konsistent --
    plausible Erklaerung ist, dass ein unskalierter neuer Eingabekanal mit
    abweichender Groessenordnung das Modell eher stoert als informiert,
    unabhaengig vom tatsaechlichen Informationsgehalt des Features. Diese
    Klasse behebt die fehlende Skalierung fuer den Live-/Replay-Pfad;
    RollingZScoreScaler uebernimmt denselben Job im Batch-Pfad (Offline-
    Vortraining). Beide nutzen dieselbe Bessel-korrigierte Varianz
    (siehe IncrementalStats.std) und sind dadurch numerisch konsistent.
    """

    def __init__(self, feature_names: list[str], window: int = 100, epsilon: float = 1e-8):
        self.feature_names = list(feature_names)
        self.epsilon = epsilon
        self._stats = {name: IncrementalStats(window) for name in self.feature_names}

    def transform(self, features: dict[str, float]) -> dict[str, float] | None:
        """Aktualisiert die rollierenden Stats mit dem neuen Feature-Vektor
        und gibt die skalierte Version zurueck. None waehrend der Warm-up-
        Phase (Rolling-Fenster noch nicht voll fuer alle Features)."""
        for name in self.feature_names:
            self._stats[name].update(features[name])
        if not all(s.is_ready() for s in self._stats.values()):
            return None
        return {
            name: (features[name] - self._stats[name].mean) / (self._stats[name].std + self.epsilon)
            for name in self.feature_names
        }


class RollingSum:
    """
    O(1)-Update einer gleitenden Summe ueber ein festes Fenster (deque-basiert).

    Grundlage fuer VWAP (Summe von price*volume und volume) und
    Trade-Intensity (Summe/Mittelwert von trade_count) im Live-Pfad.

    Hinweis: Bei sehr lange laufenden Streams kann sich durch die
    fortlaufende Subtraktion/Addition Gleitkomma-Drift ansammeln. Fuer den
    MVP unkritisch (Restart pro Handelstag), fuer Produktion ggf.
    periodischer Resync aus dem Buffer noetig.
    """

    def __init__(self, window: int):
        self.window = window
        self.buffer: deque[float] = deque(maxlen=window)
        self._sum = 0.0

    def update(self, x: float) -> None:
        if len(self.buffer) == self.window:
            self._sum -= self.buffer[0]
        self.buffer.append(x)
        self._sum += x

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def mean(self) -> float:
        n = len(self.buffer)
        return self._sum / n if n > 0 else 0.0

    def is_ready(self) -> bool:
        return len(self.buffer) == self.window


@dataclass
class LiveFeatureState:
    """Haelt den laufenden Zustand fuer die Live-Feature-Berechnung pro Symbol."""
    cfg: FeatureConfig = field(default_factory=FeatureConfig)
    last_price: float | None = None
    last_mid: float | None = None
    return_stats: IncrementalStats = field(init=False)
    pv_sum: RollingSum = field(init=False)
    volume_sum: RollingSum = field(init=False)
    trade_count_sum: RollingSum = field(init=False)

    def __post_init__(self):
        self.return_stats = IncrementalStats(self.cfg.volatility_window)
        self.pv_sum = RollingSum(self.cfg.vwap_window)
        self.volume_sum = RollingSum(self.cfg.vwap_window)
        self.trade_count_sum = RollingSum(self.cfg.trade_intensity_window)


class LiveFeatureEngine:
    """
    Streaming-Pendant zu FeaturePipeline: verarbeitet einen Marktdaten-
    Event nach dem anderen und liefert sofort einen Feature-Vektor zurueck,
    ohne das gesamte historische Fenster neu zu iterieren.

    Liefert denselben 7-Feature-Satz wie FeaturePipeline.transform()
    (log_return, realized_vol, orderbook_imbalance, spread_norm,
    vwap_deviation, trade_intensity, mid_price_return) -- siehe
    tests/test_consistency.py fuer den Paritaetsnachweis. Vorherige
    Versionen dieser Klasse lieferten nur 4 der 7 Features; das war ein
    Blocker fuer den Live-Betrieb, da SequenceBuffer.push() bei fehlenden
    Features fail-fast eine KeyError wirft.

    Dies ist die Komponente, die im Live-Control-Loop pro eingehendem
    Tick/Bar aufgerufen wird, bevor der Vektor in den SequenceBuffer
    geschrieben wird.
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.cfg = config
        self.state = LiveFeatureState(cfg=config)

    def update(self, event: dict) -> dict | None:
        """
        event erwartet Keys: price, volume, bid, ask, bid_volume, ask_volume, trade_count

        Rueckgabe: dict mit allen 7 Rohfeatures, oder None solange das
        laengste Warm-up-Fenster (hier: trade_intensity_window) noch nicht
        voll ist.
        """
        price = event["price"]
        volume = event["volume"]
        bid = event["bid"]
        ask = event["ask"]
        bid_vol = event["bid_volume"]
        ask_vol = event["ask_volume"]
        trade_count = event["trade_count"]
        mid = (bid + ask) / 2.0

        st = self.state

        if st.last_price is None:
            st.last_price = price
            st.last_mid = mid
            return None  # erster Tick, noch kein Return berechenbar

        log_return = float(np.log(price) - np.log(st.last_price))
        mid_price_return = float(np.log(mid) - np.log(st.last_mid))

        st.return_stats.update(log_return)
        st.pv_sum.update(price * volume)
        st.volume_sum.update(volume)
        st.trade_count_sum.update(trade_count)

        st.last_price = price
        st.last_mid = mid

        if not (st.return_stats.is_ready() and st.pv_sum.is_ready() and st.trade_count_sum.is_ready()):
            return None  # Warm-up-Phase

        vwap = st.pv_sum.sum / (st.volume_sum.sum + self.cfg.epsilon)
        avg_trades = st.trade_count_sum.mean

        return {
            "log_return": log_return,
            "realized_vol": st.return_stats.std,
            "orderbook_imbalance": (bid_vol - ask_vol) / (bid_vol + ask_vol + self.cfg.epsilon),
            "spread_norm": (ask - bid) / (mid + self.cfg.epsilon),
            "vwap_deviation": (price - vwap) / (vwap + self.cfg.epsilon),
            "trade_intensity": trade_count / (avg_trades + self.cfg.epsilon),
            "mid_price_return": mid_price_return,
        }


# ---------------------------------------------------------------------------
# 4. Sanity-Check / Demo mit synthetischen Daten
# ---------------------------------------------------------------------------

def _generate_synthetic_market_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Erzeugt plausible synthetische Marktdaten fuer Pipeline-Tests."""
    rng = np.random.default_rng(seed)

    # Preis als geometrische Brownsche Bewegung mit Volatilitaets-Clustering
    # (GARCH-artig, damit realized_vol nicht konstant ist -> realistischerer Test)
    vol = np.zeros(n)
    vol[0] = 0.001
    innovations = rng.standard_normal(n)
    for t in range(1, n):
        vol[t] = 0.00002 + 0.85 * vol[t - 1] + 0.1 * (innovations[t - 1] ** 2) * 1e-4
    returns = innovations * np.sqrt(np.maximum(vol, 1e-8))
    price = 100 * np.exp(np.cumsum(returns))

    volume = rng.integers(100, 5000, size=n).astype(float)
    spread_base = rng.uniform(0.01, 0.05, size=n)
    bid = price - spread_base / 2
    ask = price + spread_base / 2
    bid_volume = rng.integers(50, 3000, size=n).astype(float)
    ask_volume = rng.integers(50, 3000, size=n).astype(float)
    trade_count = rng.integers(1, 200, size=n).astype(float)

    return pd.DataFrame(
        {
            "price": price,
            "volume": volume,
            "bid": bid,
            "ask": ask,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "trade_count": trade_count,
        }
    )


if __name__ == "__main__":
    # Batch-Pfad
    df = _generate_synthetic_market_data()
    pipeline = FeaturePipeline()
    scaler = RollingZScoreScaler(window=100)

    features, target = pipeline.transform_with_target(df, horizon=5)
    scaled = scaler.transform(features)

    print("=== Batch-Pipeline ===")
    print(f"Rohdaten shape:    {df.shape}")
    print(f"Feature shape:     {features.shape}")
    print(f"Nach Warm-up (dropna): {features.dropna().shape}")
    print("\nFeature-Statistiken (unskaliert):")
    print(features.describe().T[["mean", "std", "min", "max"]])
    print("\nSkalierte Features (letzte 3 Zeilen, sollten ~N(0,1)-artig sein):")
    print(scaled.dropna().tail(3))

    # Streaming-Pfad: Konsistenzcheck gegen Batch-Pfad
    print("\n=== Live-Feature-Engine (Streaming) ===")
    live_engine = LiveFeatureEngine()
    last_live_features = None
    for _, row in df.iterrows():
        event = row.to_dict()
        result = live_engine.update(event)
        if result is not None:
            last_live_features = result
    print("Letzter Live-Feature-Vektor:")
    print(last_live_features)
