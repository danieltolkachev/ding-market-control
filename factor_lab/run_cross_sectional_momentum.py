"""
run_cross_sectional_momentum.py — MECHANIK-Test fuer ein
Querschnitts-Momentum-Leg (12-2-Monats-Formation, Jegadeesh & Titman
1993), NICHT ein Promotion-Kandidat: kein Pre-Registration/Sealing/
Gates/Holdout-Apparat wie bei trend-etf-v1/v2 -- der Spike mit
Kenneth Frenchs CRSP-Faktordaten (siehe Chat-Verlauf) zeigte, dass die
rohe Momentum-Praemie seit 2000 statistisch nicht mehr von Null zu
unterscheiden ist; dieses Skript prueft NUR, ob eine vola-skalierte
Variante (Wiederverwendung von portfolio.py's bestehender inverse-Vola-
Gewichtung + Portfolio-Vola-Cap, siehe rebalance_weights) end-to-end
funktioniert -- kein Beweis fuer einen echten Edge.

Universum: heutige S&P-500-Mitglieder (Wikipedia, kostenlos) ueber
yfinance -- bekannter Survivorship-Bias-Kaveat (nur heutige
Ueberlebende), fuer eine Mechanik-Pruefung akzeptiert, fuer eine echte
Promotion-Entscheidung NICHT ausreichend (siehe Norgate/Sharadar-
Diskussion).

Ausfuehren: py -3.12 factor_lab/run_cross_sectional_momentum.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from factor_lab.cross_sectional_signals import cross_sectional_momentum_signal
from factor_lab.portfolio import ewma_annualized_vol, month_end_dates, trend_weight_provider, run_lagged_backtest
from factor_lab.stats import annualized_stats

LOOKBACK_DAYS = 252
SKIP_DAYS = 21
TOP_FRAC = 0.3
BOTTOM_FRAC = 0.3
WARMUP_DAYS = LOOKBACK_DAYS + SKIP_DAYS
MECHANICS_VOL_CAP = 0.15
UNCAPPED_VOL_CAP = 999.0  # Cap so hoch, dass er in der Praxis nie greift -- isoliert den Skalierungs-Effekt


def prepare_inputs(prices: pd.DataFrame, cash_daily: pd.Series) -> dict:
    returns = prices.pct_change().dropna()
    cash_daily = cash_daily.loc[returns.index]
    vols = ewma_annualized_vol(returns)
    signal = cross_sectional_momentum_signal(
        prices.loc[returns.index], lookback=LOOKBACK_DAYS, skip=SKIP_DAYS,
        top_frac=TOP_FRAC, bottom_frac=BOTTOM_FRAC,
    )
    warmup_end = returns.index[min(WARMUP_DAYS, len(returns.index) - 1)]
    ends = month_end_dates(returns.index)
    eval_decisions = ends[ends >= warmup_end]
    return {"returns": returns, "vols": vols, "signal": signal, "cash_daily": cash_daily,
            "eval_decisions": eval_decisions}


def run_variant(inputs: dict, mode: str, vol_cap: float, cost_bp: dict[str, float],
                borrow_bp_pa: float = 50.0) -> tuple[pd.Series, dict]:
    provider = trend_weight_provider(inputs["returns"], inputs["signal"], inputs["vols"], mode, vol_cap=vol_cap)
    return run_lagged_backtest(inputs["returns"], inputs["cash_daily"], inputs["eval_decisions"],
                               provider, cost_bp, borrow_bp_pa=borrow_bp_pa)


def run_mechanics_test(prices: pd.DataFrame, flat_cost_bp: float = 5.0) -> dict:
    cash_daily = pd.Series(0.0, index=prices.index)
    inputs = prepare_inputs(prices, cash_daily)
    cost_bp = {s: flat_cost_bp for s in inputs["returns"].columns}
    result = {}
    for mode in ("long_short", "long_flat"):
        result[mode] = {}
        for label, cap in (("vol_capped", MECHANICS_VOL_CAP), ("uncapped", UNCAPPED_VOL_CAP)):
            net, info = run_variant(inputs, mode, cap, cost_bp)
            result[mode][label] = {"stats": annualized_stats(net, inputs["cash_daily"]),
                                   "max_daily_gross": info["max_daily_gross"]}
    return result


def main() -> None:
    import io
    import urllib.request
    import yfinance as yf

    print("Lade S&P-500-Ticker von Wikipedia (heutige Mitglieder -- Survivorship-Bias-Kaveat)...")
    req = urllib.request.Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (research script, factor_lab momentum spike)"},
    )
    html = urllib.request.urlopen(req).read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    symbols = sorted(t.replace(".", "-") for t in tables[0]["Symbol"].tolist())
    print(f"  {len(symbols)} Ticker gefunden.")

    print("Lade Kurse via yfinance (kann mehrere Minuten dauern)...")
    raw = yf.download(symbols, start="2015-01-01", auto_adjust=True, progress=False)["Close"]
    prices = raw.dropna(axis=1, thresh=int(len(raw) * 0.9))
    print(f"  {prices.shape[1]} von {len(symbols)} Tickern mit ausreichender Historie.")

    result = run_mechanics_test(prices)
    for mode, variants in result.items():
        print(f"\n=== {mode} ===")
        for label, r in variants.items():
            s = r["stats"]
            print(f"  {label:12}: CAGR={s['cagr']:+.2%}  Vol={s['vol_pa']:.2%}  "
                  f"Sharpe={s['sharpe']:.2f}  MaxDD={s['max_drawdown']:.1%}  "
                  f"MaxGross={r['max_daily_gross']:.2f}")


if __name__ == "__main__":
    main()
