"""
test_portfolio.py — Vol-Schaetzung, Zielgewichte und (ab Task 4) der
taegliche Loop mit Lag-Konvention.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.portfolio import (
    ewma_annualized_vol,
    rebalance_weights,
    run_lagged_backtest,
    month_end_dates,
    trend_weight_provider,
    fixed_mix_provider,
)


def check_ewma_vol() -> None:
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    rets = pd.DataFrame({"A": [0.01, -0.01] * 40}, index=idx)
    vol = ewma_annualized_vol(rets, span=63)
    assert vol["A"].iloc[:62].isna().all(), "Vor min_periods muss NaN stehen"
    assert abs(vol["A"].iloc[-1] - 0.01 * np.sqrt(252)) < 0.02
    print("ewma_annualized_vol: OK")


def check_rebalance_weights() -> None:
    signal = pd.Series({"A": 1.0, "B": -1.0})
    vol = pd.Series({"A": 0.2, "B": 0.1})
    trailing = pd.DataFrame(0.0, index=range(63), columns=["A", "B"])

    # long_short: raw = [5, -10] -> base [1/3, -2/3], Gross exakt 1
    w = rebalance_weights(signal, vol, trailing, mode="long_short", vol_cap=0.10)
    assert abs(w["A"] - 1.0 / 3.0) < 1e-12 and abs(w["B"] + 2.0 / 3.0) < 1e-12
    assert abs(w.abs().sum() - 1.0) < 1e-12

    # long_flat: negatives Signal -> 0
    w = rebalance_weights(signal, vol, trailing, mode="long_flat", vol_cap=0.10)
    assert abs(w["A"] - 1.0) < 1e-12 and w["B"] == 0.0

    # Vol-Cap skaliert nur HERUNTER
    rng = np.random.default_rng(0)
    hot = pd.DataFrame({"A": rng.normal(0, 0.02, 63), "B": 0.0})
    w = rebalance_weights(pd.Series({"A": 1.0, "B": 0.0}), vol, hot, mode="long_flat", vol_cap=0.10)
    assert 0.0 < w["A"] < 1.0

    # NaN-Vol -> Gewicht 0; Alles-Null-Signal -> Cash
    w = rebalance_weights(pd.Series({"A": 1.0, "B": 1.0}), pd.Series({"A": 0.2, "B": np.nan}), trailing, mode="long_flat")
    assert w["B"] == 0.0 and abs(w["A"] - 1.0) < 1e-12
    w = rebalance_weights(pd.Series({"A": 0.0, "B": 0.0}), vol, trailing, mode="long_flat")
    assert (w == 0.0).all()
    print("rebalance_weights: OK")


def check_month_end_dates() -> None:
    idx = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2020-03-02"])
    ends = month_end_dates(idx)
    assert list(ends) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-02")]
    print("month_end_dates: OK")


def _lag_fixture():
    """5 Tage, 1 Instrument. Entscheidung an d0 setzt Gewicht 1.0.
    Erwartung (Lag-Konvention): d1 = Ausfuehrungstag (nur Kosten, alte
    Position 0 verdient nichts), d2 = erster Tag MIT Position."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    returns = pd.DataFrame({"SPY": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=idx)
    cash = pd.Series(0.0, index=idx)
    return idx, returns, cash


def check_future_poison_and_lag() -> None:
    idx, returns, cash = _lag_fixture()
    provider = fixed_mix_provider({"SPY": 1.0})
    net, info = run_lagged_backtest(
        returns, cash, decision_dates=pd.DatetimeIndex([idx[0]]),
        weight_provider=provider, cost_bp={"SPY": 1.5},
    )
    # PnL-Index beginnt am Ausfuehrungstag d1
    assert net.index[0] == idx[1], f"PnL muss am Ausfuehrungstag beginnen, bekam {net.index[0]}"
    # d1: KEINE Marktposition (Future-Poison-Test: r(d1)=0.02 darf NICHT
    # verdient werden), nur Kaufkosten 1.5bp
    assert abs(net.loc[idx[1]] - (0.0 - 1.5 / 10_000.0)) < 1e-12, (
        f"Ausfuehrungstag darf r(t+1) nicht verdienen, bekam {net.loc[idx[1]]}"
    )
    # d2: erster Tag mit Position -> r(d2)=0.03
    assert abs(net.loc[idx[2]] - 0.03) < 1e-12
    assert info["n_rebalances"] == 1 and abs(info["total_turnover"] - 1.0) < 1e-12
    print("run_lagged_backtest (Future-Poison + Fill-Lag): OK")


def check_reconciliation_and_cash() -> None:
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(3)
    returns = pd.DataFrame({"SPY": rng.normal(0.0005, 0.01, 60), "TLT": rng.normal(0.0, 0.008, 60)}, index=idx)
    cash = pd.Series(0.04 / 252, index=idx)  # 4% p.a. T-Bill
    provider = fixed_mix_provider({"SPY": 0.3, "TLT": 0.3})
    decisions = pd.DatetimeIndex([idx[0], *month_end_dates(idx)])
    net, info = run_lagged_backtest(returns, cash, decisions, provider, cost_bp={"SPY": 1.5, "TLT": 1.5})

    # Pflichttest Review: vollstaendige Netto = Brutto + Cash - Kosten - Borrow, pro Tag.
    per_day = info["per_day"]
    recon = per_day["gross_pnl"] + per_day["cash_pnl"] - per_day["trade_cost"] - per_day["borrow_cost"]
    assert np.allclose(net.to_numpy(), recon.to_numpy(), atol=1e-15), "Tagesgenaue Reconciliation verletzt"
    # Cash-Gewicht max(0, 1-Gross)=0.4 verdient Zins: am 2. Tag nach Aufbau
    day2 = net.index[2]
    assert per_day.loc[day2, "cash_pnl"] > 0
    print("run_lagged_backtest (Reconciliation + verzinstes Cash): OK")


def check_gross_drift_reporting() -> None:
    # +0.5/-0.5, beide Underlyings +10% an einem Tag -> Gross drifted auf 1.1.
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    returns = pd.DataFrame({"SPY": [0.0, 0.0, 0.10, 0.0], "TLT": [0.0, 0.0, -0.10, 0.0]}, index=idx)
    cash = pd.Series(0.0, index=idx)
    provider = fixed_mix_provider({"SPY": 0.5, "TLT": -0.5})
    net, info = run_lagged_backtest(returns, cash, pd.DatetimeIndex([idx[0]]), provider,
                                    cost_bp={"SPY": 1.5, "TLT": 1.5})
    # SPY: 0.5*1.1=0.55; TLT: -0.5*0.9=-0.45 -> bei Netto-PnL 0.5*0.1+(-0.5)*(-0.1)=0.1:
    # Gewichte /1.1 -> 0.5 & -0.409 -> Gross 0.909? Nein: Drift teilt durch (1+gross_pnl).
    # gross_pnl = 0.10 -> w_SPY = 0.55/1.1 = 0.5, w_TLT = -0.45/1.1 = -0.409, Gross 0.909.
    # Der REVIEW-Fall (beide +10%) braucht gleiche Vorzeichen der Returns:
    returns2 = pd.DataFrame({"SPY": [0.0, 0.0, 0.10, 0.0], "TLT": [0.0, 0.0, 0.10, 0.0]}, index=idx)
    net2, info2 = run_lagged_backtest(returns2, cash, pd.DatetimeIndex([idx[0]]), provider,
                                      cost_bp={"SPY": 1.5, "TLT": 1.5})
    # gross_pnl = 0.5*0.1 - 0.5*0.1 = 0 -> w = 0.55/-0.55 -> Gross 1.10
    assert info2["max_daily_gross"] > 1.09, f"Gross-Drift muss gemessen werden, bekam {info2['max_daily_gross']}"
    print("run_lagged_backtest (Max-Gross-Reporting): OK")


def check_missing_symbol_in_provider_output() -> None:
    """Regression (Review-Fund): ein weight_provider darf ein Symbol aus
    returns.columns auslassen (z.B. Signal=0 wird nicht zurueckgegeben).
    pending_target muss auf ALLE Symbole reindexiert werden (fill 0.0) --
    sonst erzeugt pandas-Index-Alignment ein NaN in deltas, das ueber
    trade_cost_fraction den GESAMTEN Tages-trade_cost (und damit net)
    stillschweigend vergiftet, statt einen Fehler zu werfen."""
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    returns = pd.DataFrame({"SPY": [0.0, 0.01], "TLT": [0.0, 0.01]}, index=idx)
    cash = pd.Series(0.0, index=idx)

    def provider(decision_date):
        return pd.Series({"SPY": 1.0})  # TLT fehlt absichtlich

    net, info = run_lagged_backtest(
        returns, cash, pd.DatetimeIndex([idx[0]]), provider, cost_bp={"SPY": 1.5, "TLT": 1.5},
    )
    assert not net.isna().any(), f"NaN in net durch fehlendes Symbol im Provider-Output: {net}"
    assert not info["per_day"]["trade_cost"].isna().any(), "NaN in trade_cost durch fehlendes Symbol"
    # Nur SPY-Kosten (1.5bp), TLT-Delta muss 0 sein (nicht NaN) -> Kosten exakt 1.5bp
    assert abs(info["per_day"]["trade_cost"].iloc[0] - 1.5 / 10_000.0) < 1e-12
    print("run_lagged_backtest (fehlendes Symbol im Provider-Output -> 0.0, kein NaN): OK")


def run_consistency_check() -> None:
    check_ewma_vol()
    check_rebalance_weights()
    check_month_end_dates()
    check_future_poison_and_lag()
    check_reconciliation_and_cash()
    check_gross_drift_reporting()
    check_missing_symbol_in_provider_output()
    print("\nAlle portfolio-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
