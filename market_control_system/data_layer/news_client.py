"""
news_client.py
===============

Sentiment/News-Feature-Layer -- erweitert das 7-Feature-Set aus
feature_engineering/feature_pipeline.py um zwei zusaetzliche, optionale
Features:

    news_intensity: Anzahl Nachrichtenartikel im Rueckblickfenster
                     (window_minutes), log1p-skaliert.
    news_sentiment: einfacher Wortlisten-Score (-1..+1) ueber die
                     Schlagzeilen im selben Fenster, 0 wenn keine News.

Design-Entscheidung: additiv statt in FeaturePipeline/LiveFeatureEngine
integriert.
------------------------------------------------------------------------
Diese beiden Features werden bewusst NICHT in feature_pipeline.py
eingebaut, sondern als separate Spalten berechnet und erst danach an die
Markt-Features angehaengt (siehe compute_news_features() fuer Batch,
orchestration/control_loop.py fuer den Live-/Replay-Pfad). Grund: das
haelt FeaturePipeline/LiveFeatureEngine unangetastet und ruecken-
kompatibel -- alle bestehenden Skripte (Demos, Tests, der urspruengliche
7-Feature-Walk-Forward-Test) funktionieren unveraendert weiter, und nur
Skripte, die News explizit anfordern, bekommen das erweiterte 9-Feature-
Set (siehe feature_pipeline.EXTENDED_FEATURE_NAMES).

Kausalitaet: wie bei allen anderen Features gilt strikt nur Vergangenheit
-- zum Zeitpunkt t werden ausschliesslich Artikel beruecksichtigt, deren
created_at <= t liegt.

Sentiment-Ansatz ist BEWUSST simpel (Wortlisten-Scoring, kein NLP-Modell):
das ist schnell, ohne zusaetzliche Modell-Dependency, und ausreichend fuer
einen ersten Test, ob das Feature ueberhaupt Signalwert hat. Falls ja,
waere ein echtes Sentiment-Modell (z.B. FinBERT) der naheliegende naechste
Ausbauschritt -- hier bewusst nicht vorweggenommen (YAGNI).
"""

from __future__ import annotations

import sys
import os
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import numpy as np
import pandas as pd

from alpaca.data.historical.news import NewsClient
from alpaca.data.live.news import NewsDataStream
from alpaca.data.requests import NewsRequest

from settings import AlpacaConfig, load_alpaca_config


POSITIVE_WORDS = {
    "beat", "beats", "beating", "surge", "surges", "surging", "soar", "soars",
    "soaring", "rally", "rallies", "upgrade", "upgraded", "upgrades", "record",
    "growth", "strong", "outperform", "outperforms", "bullish", "gain", "gains",
    "profit", "profits", "profitable", "buy", "positive", "rise", "rises",
    "rising", "jump", "jumps", "boost", "boosts", "boosted", "win", "wins",
    "winning", "breakthrough", "expand", "expands", "expansion", "optimistic",
}
NEGATIVE_WORDS = {
    "miss", "misses", "missing", "plunge", "plunges", "plunging", "crash",
    "crashes", "crashing", "downgrade", "downgraded", "downgrades", "weak",
    "weakness", "underperform", "underperforms", "bearish", "loss", "losses",
    "sell", "selloff", "negative", "fall", "falls", "falling", "drop", "drops",
    "dropping", "decline", "declines", "declining", "concern", "concerns",
    "warning", "warns", "cut", "cuts", "lawsuit", "investigation", "recall",
    "layoff", "layoffs", "fraud", "probe", "slump", "slumps",
}
_WORD_RE = re.compile(r"[a-zA-Z']+")


def score_headline(headline: str) -> float:
    """Einfacher Wortlisten-Sentiment-Score einer einzelnen Schlagzeile,
    Bereich [-1, +1]. 0.0 wenn keine der Wortlisten trifft."""
    words = {w.lower() for w in _WORD_RE.findall(headline)}
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def fetch_historical_news(
    symbol: str,
    start: datetime,
    end: datetime,
    config: AlpacaConfig | None = None,
) -> pd.DataFrame:
    """Historische News fuer ein Symbol, sortiert nach Zeit. Spalten:
    created_at (UTC, tz-aware), headline."""
    cfg = config or load_alpaca_config()
    client = NewsClient(cfg.api_key, cfg.secret_key)
    news = client.get_news(
        NewsRequest(symbols=symbol, start=start, end=end, limit=50, include_content=False)
    )
    articles = news.data.get("news", []) if isinstance(news.data, dict) else news.data
    if not articles:
        return pd.DataFrame(columns=["created_at", "headline"])
    df = pd.DataFrame({
        "created_at": [pd.Timestamp(a.created_at) for a in articles],
        "headline": [a.headline for a in articles],
    }).sort_values("created_at").reset_index(drop=True)
    return df


def compute_news_features(
    bar_index: pd.DatetimeIndex,
    news_df: pd.DataFrame,
    window_minutes: int = 240,
) -> pd.DataFrame:
    """
    Berechnet news_intensity/news_sentiment fuer jeden Zeitpunkt in
    bar_index (kausal: nur Artikel mit created_at <= Bar-Zeitpunkt und
    created_at > Bar-Zeitpunkt - window_minutes werden beruecksichtigt).

    Implementiert als Sliding-Window mit zwei Zeigern (news_df ist nach
    Zeit sortiert) -- O(n_bars + n_news) statt O(n_bars * n_news), wichtig
    da n_bars im Backtest im 6-stelligen Bereich liegt.
    """
    window = pd.Timedelta(minutes=window_minutes)
    news_times = news_df["created_at"].to_numpy() if len(news_df) else np.array([], dtype="datetime64[ns]")
    news_scores = np.array([score_headline(h) for h in news_df["headline"]]) if len(news_df) else np.array([])

    bar_times = bar_index.to_numpy()
    intensity = np.zeros(len(bar_times))
    sentiment = np.zeros(len(bar_times))

    left = 0
    right = 0
    n_news = len(news_times)
    for i, t in enumerate(bar_times):
        window_start = t - window.to_numpy()
        # rechten Zeiger vorschieben: alle Artikel mit created_at <= t aufnehmen
        while right < n_news and news_times[right] <= t:
            right += 1
        # linken Zeiger vorschieben: Artikel ausserhalb des Fensters verwerfen
        while left < right and news_times[left] <= window_start:
            left += 1

        count = right - left
        intensity[i] = np.log1p(count)
        sentiment[i] = float(news_scores[left:right].mean()) if count > 0 else 0.0

    return pd.DataFrame(
        {"news_intensity": intensity, "news_sentiment": sentiment}, index=bar_index
    )


@dataclass
class LiveNewsFeatureState:
    window_minutes: int = 240
    _buffer: deque[tuple[pd.Timestamp, str]] = field(default_factory=deque)


class LiveNewsFeatureEngine:
    """
    Streaming-Pendant zu compute_news_features(): haelt einen Ring-Puffer
    aktueller Artikel (befuellt via on_news(), z.B. aus einem
    AlpacaLiveMarketDataStream-Handler fuer subscribe_news) und liefert
    per get_features(now) den aktuellen news_intensity/news_sentiment-Wert.

    Getrennt von LiveFeatureEngine (feature_pipeline.py) gehalten -- siehe
    Modul-Docstring, additive statt integrierte Architektur.
    """

    def __init__(self, window_minutes: int = 240):
        self.window_minutes = window_minutes
        self._buffer: deque[tuple[pd.Timestamp, str]] = deque()

    def on_news(self, created_at: pd.Timestamp, headline: str) -> None:
        self._buffer.append((created_at, headline))

    def get_features(self, now: pd.Timestamp) -> dict:
        cutoff = now - pd.Timedelta(minutes=self.window_minutes)
        while self._buffer and self._buffer[0][0] <= cutoff:
            self._buffer.popleft()
        relevant = [h for t, h in self._buffer if t <= now]
        count = len(relevant)
        sentiment = float(np.mean([score_headline(h) for h in relevant])) if count > 0 else 0.0
        return {"news_intensity": float(np.log1p(count)), "news_sentiment": sentiment}


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== score_headline Sanity-Check ===")
    cases = [
        "Apple Beats Earnings Expectations, Shares Surge",
        "Company Misses Estimates, Stock Plunges on Weak Guidance",
        "Apple Introduces New Chip",
    ]
    for h in cases:
        print(f"  {score_headline(h):+.2f}  {h}")

    print("\n=== fetch_historical_news + compute_news_features (echte Daten, AAPL, 7 Tage) ===")
    from datetime import timedelta, timezone
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=7)
    news_df = fetch_historical_news("AAPL", start, end)
    print(f"  {len(news_df)} Artikel geladen")

    bar_index = pd.date_range(start, end, freq="1h")
    features = compute_news_features(bar_index, news_df, window_minutes=240)
    print(features.describe())
    print("\nLetzte 5 Zeilen:")
    print(features.tail())
