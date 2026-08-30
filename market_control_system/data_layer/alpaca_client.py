"""
alpaca_client.py
=================

Layer 1 des Market-Control-Systems: echte Alpaca-Datenanbindung. Ersetzt
`_generate_synthetic_market_data` durch zwei Quellen desselben Schemas
(price, volume, bid, ask, bid_volume, ask_volume, trade_count):

1. fetch_historical_market_data() -- fuer Offline-Vortraining auf echten
   vergangenen Bars (statt synthetischen Daten).
2. AlpacaLiveMarketDataStream -- fuer den Live-/Paper-Betrieb, liefert
   pro abgeschlossenem 1-Minuten-Bar EIN raw_event-dict.

Design-Entscheidung: gleiches Schema wie synthetische Daten.
------------------------------------------------------------
FeaturePipeline/LiveFeatureEngine/ControlLoop.step() erwarten alle
dasselbe dict-/DataFrame-Schema, das feature_pipeline._generate_synthetic_
market_data() von Anfang an produziert hat. Dieses Modul haelt sich exakt
daran, damit KEIN nachgelagerter Code angefasst werden muss, um von
synthetisch auf echte Daten umzuschalten -- nur die Datenquelle wechselt.

Alpaca liefert Bars (open, high, low, close, volume, trade_count, vwap)
und Quotes (bid_price, bid_size, ask_price, ask_size) getrennt und mit
unterschiedlicher Taktrate (Quotes viel haeufiger als 1-Minuten-Bars).
Fuer historische Daten werden beide per pandas.merge_asof (backward)
zusammengefuehrt: jedem Bar wird die zuletzt bekannte Quote VOR seinem
Timestamp zugeordnet -- kausal, kein Look-Ahead. Fuer den Live-Pfad
uebernimmt AlpacaLiveMarketDataStream dieselbe Logik inkrementell (haelt
die letzte gesehene Quote im Speicher, kombiniert sie mit jedem neuen Bar).
"""

from __future__ import annotations

import sys
import os
from datetime import datetime
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame

from settings import AlpacaConfig, load_alpaca_config


def fetch_historical_market_data(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Minute,
    config: AlpacaConfig | None = None,
) -> pd.DataFrame:
    """
    Liefert historische Bars+Quotes im selben Schema wie
    feature_pipeline._generate_synthetic_market_data(): price, volume,
    bid, ask, bid_volume, ask_volume, trade_count -- indiziert nach
    Timestamp (aufsteigend sortiert).

    Fuer echtes Offline-Vortraining gedacht (siehe orchestration/
    control_loop.py) -- FeaturePipeline/SequenceWindowBuilder brauchen
    keine Anpassung, um dieses DataFrame statt synthetischer Daten zu
    verarbeiten.
    """
    cfg = config or load_alpaca_config()
    client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)

    bars_df = client.get_stock_bars(
        StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
    ).df
    if bars_df.empty:
        raise ValueError(f"Keine Bars fuer {symbol} im Zeitraum {start} - {end}")
    bars_df = bars_df.xs(symbol, level="symbol").reset_index().sort_values("timestamp")

    quotes_df = client.get_stock_quotes(
        StockQuotesRequest(symbol_or_symbols=symbol, start=start, end=end)
    ).df
    if quotes_df.empty:
        raise ValueError(f"Keine Quotes fuer {symbol} im Zeitraum {start} - {end}")
    quotes_df = quotes_df.xs(symbol, level="symbol").reset_index().sort_values("timestamp")

    merged = pd.merge_asof(
        bars_df,
        quotes_df[["timestamp", "bid_price", "bid_size", "ask_price", "ask_size"]],
        on="timestamp",
        direction="backward",
    )
    # Bars VOR der allerersten Quote im Zeitraum haben keinen Match -> raus
    # (kausal korrekt: fuer sie ist zum Bar-Zeitpunkt keine Quote bekannt).
    n_before = len(merged)
    merged = merged.dropna(subset=["bid_price", "ask_price"])
    if len(merged) < n_before:
        print(f"[alpaca_client] {n_before - len(merged)} Bar(s) ohne vorherige Quote verworfen "
              f"(kausal korrekt, meist nur ganz am Anfang des Zeitraums).")

    out = pd.DataFrame({
        "price": merged["close"].astype(float),
        "volume": merged["volume"].astype(float),
        "bid": merged["bid_price"].astype(float),
        "ask": merged["ask_price"].astype(float),
        "bid_volume": merged["bid_size"].astype(float),
        "ask_volume": merged["ask_size"].astype(float),
        "trade_count": merged["trade_count"].astype(float),
    })
    out.index = pd.DatetimeIndex(merged["timestamp"], name="timestamp")
    return out


def fetch_historical_bars_approximate(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Minute,
    config: AlpacaConfig | None = None,
) -> pd.DataFrame:
    """
    Schnelle Alternative zu fetch_historical_market_data() fuer GROSSE
    Zeitraeume (z.B. 1 Jahr) oder mehrere Symbole, bei denen echte
    Quote-Historie nicht mehr praktikabel ist.

    Gemessener Groessenordnungsunterschied (siehe Feasibility-Check vom
    24./25.08.2026): 1 Jahr Bars fuer ein Symbol ~8s, aber 1 einzelner TAG
    Trades bereits ~830.000 Zeilen in ~32s -- hochgerechnet auf 1 Jahr
    waeren das >200 Millionen Zeilen und mehrere Stunden allein fuers
    Laden, pro Symbol. Fuer einen Mehr-Symbol-Backtest ueber Nacht nicht
    praktikabel.

    Approximiert deshalb bid/ask/bid_volume/ask_volume aus dem OHLC-Bar
    selbst, statt sie aus echten historischen Quotes zu holen:
        half_spread = close * (spread_bps / 1e4) / 2
        ask = close + half_spread, bid = close - half_spread
        bid_volume = ask_volume = volume / 2  (neutral -- kein echtes
                                                 Imbalance-Signal mehr)

    BUGFIX (nach Run 1 vom 24./25.08.2026, siehe backtest_<id>-Logs):
    -----------------------------------------------------------------
    Die urspruengliche Version nutzte (high - low) / 2 als Spread-Proxy.
    Das ist die Preisspanne INNERHALB der Minute, nicht der tatsaechliche
    Bid-Ask-Spread -- und ueberschaetzt ihn massiv (bei liquiden Large-Caps
    wie AAPL/MSFT/GOOGL typischerweise 1-2 Basispunkte Spread, aber die
    High-Low-Spanne einer aktiven 1-Minuten-Kerze kann 10x groesser sein).
    Ergebnis im 1-Jahres-Backtest: 96,5% der Bars hatten eine (durch
    Forecast-Rauschen von Bar zu Bar bedingte) minimale Positionsaenderung,
    und JEDE davon kostete einen stark ueberschaetzten Spread -- kumuliert
    ueber 142.842 Bars summierten sich die Slippage-Kosten allein auf
    123% des Kapitals (nahezu der gesamte gemessene Verlust von -125%).
    Fix hier: fester, realistischer Spread in Basispunkten statt High-Low-
    Spanne. Der zweite Teil des Fixes (Rebalancing-Totband, damit nicht
    bei jedem Bar-zu-Bar-Rauschen neu positioniert wird) sitzt in
    controller/risk_overlay.py (RiskOverlayConfig.min_rebalance_threshold).

    Explizit akzeptierter Trade-off: orderbook_imbalance wird durch diese
    Naeherung praktisch immer ~0 (die Information steckt nicht mehr in
    den Daten), mid_price_return wird zur groben Naeherung statt echtem
    Wert. log_return, realized_vol, trade_intensity, vwap_deviation
    bleiben unveraendert exakt, da sie ohnehin nur price/volume/trade_count
    brauchen, keine Quotes.

    Fuer Live-Betrieb oder kurze/mittlere Zeitraeume bleibt
    fetch_historical_market_data() (echte Quotes, kausal per merge_asof
    zusammengefuehrt) die praezisere, richtige Wahl.
    """
    SPREAD_BPS = 2.0  # typische volle Spread-Groessenordnung liquider US-Large-Caps

    cfg = config or load_alpaca_config()
    client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)

    bars_df = client.get_stock_bars(
        StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
    ).df
    if bars_df.empty:
        raise ValueError(f"Keine Bars fuer {symbol} im Zeitraum {start} - {end}")
    bars_df = bars_df.xs(symbol, level="symbol").reset_index().sort_values("timestamp")

    half_spread = bars_df["close"] * (SPREAD_BPS / 1e4) / 2.0

    out = pd.DataFrame({
        "price": bars_df["close"].astype(float),
        "volume": bars_df["volume"].astype(float),
        "bid": (bars_df["close"] - half_spread).astype(float),
        "ask": (bars_df["close"] + half_spread).astype(float),
        "bid_volume": (bars_df["volume"] / 2.0).astype(float),
        "ask_volume": (bars_df["volume"] / 2.0).astype(float),
        "trade_count": bars_df["trade_count"].astype(float),
    })
    out.index = pd.DatetimeIndex(bars_df["timestamp"], name="timestamp")
    return out


class AlpacaLiveMarketDataStream:
    """
    Live-/Paper-Streaming-Adapter. Haelt die zuletzt gesehene Quote im
    Speicher und ruft `on_bar(raw_event)` SYNCHRON auf, sobald ein neues
    1-Minuten-Bar abgeschlossen ist -- raw_event hat exakt das Schema, das
    ControlLoop.step() erwartet (siehe orchestration/control_loop.py).

    on_bar sollte schnell zurueckkehren (typischerweise: ein ControlLoop.
    step()-Aufruf) -- laengere Arbeit (z.B. Logging in eine Datenbank)
    gehoert in einen separaten Thread/Task, sonst blockiert es die
    Websocket-Verarbeitung weiterer Nachrichten.
    """

    def __init__(
        self,
        symbol: str,
        on_bar: Callable[[dict], None],
        config: AlpacaConfig | None = None,
        feed: DataFeed = DataFeed.IEX,
    ):
        self.symbol = symbol
        self.on_bar = on_bar
        self.cfg = config or load_alpaca_config()
        self._stream = StockDataStream(self.cfg.api_key, self.cfg.secret_key, feed=feed)
        self._latest_quote: dict | None = None

        self._stream.subscribe_quotes(self._handle_quote, symbol)
        self._stream.subscribe_bars(self._handle_bar, symbol)

    async def _handle_quote(self, quote) -> None:
        self._latest_quote = {
            "bid": float(quote.bid_price),
            "ask": float(quote.ask_price),
            "bid_volume": float(quote.bid_size),
            "ask_volume": float(quote.ask_size),
        }

    async def _handle_bar(self, bar) -> None:
        if self._latest_quote is None:
            # Noch keine Quote empfangen (z.B. direkt nach Verbindungsaufbau) --
            # Bar ueberspringen statt mit falschem/fehlendem bid/ask weiterzumachen.
            print(f"[alpaca_client] Bar fuer {self.symbol} uebersprungen: noch keine Quote empfangen.")
            return

        raw_event = {
            "price": float(bar.close),
            "volume": float(bar.volume),
            "trade_count": float(bar.trade_count),
            **self._latest_quote,
        }
        self.on_bar(raw_event)

    def run(self) -> None:
        """Blockierend -- haelt die Websocket-Verbindung offen, bis der
        Prozess beendet wird (z.B. Ctrl+C)."""
        self._stream.run()


# ---------------------------------------------------------------------------
# Sanity-Check: historische Daten (kein Live-Stream -- der blockiert absichtlich)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import timedelta, timezone

    end = datetime.now(timezone.utc) - timedelta(minutes=20)  # Alpaca-Free-Tier: 15min Verzoegerung
    start = end - timedelta(days=3)

    print(f"=== Historische Daten AAPL, {start} bis {end} ===")
    df = fetch_historical_market_data("AAPL", start, end)
    print(f"Shape: {df.shape}")
    print(df.tail(5))
