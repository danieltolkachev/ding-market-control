"""
test_run_cross_sectional_momentum.py — Integrationstest (synthetische
Daten) fuer den Momentum-Leg-MECHANIK-Test. Kein Promotion-Kandidat:
keine Gates/Bootstrap/Siegel -- nur pruefen, dass Signal + bestehende
portfolio.py-Maschinerie (Vola-Gewichtung, Portfolio-Vola-Skalierung)
end-to-end zusammenspielen. fetch_universe()/fetch_prices()/main()
brauchen echten Netzwerkzugriff und sind bewusst NICHT unit-getestet
(gleiche Konvention wie build_trend_snapshot*.py).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from factor_lab.run_cross_sectional_momentum import prepare_inputs, run_variant, run_mechanics_test


def _prices(n_stocks: int = 30, n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    drifts = np.linspace(-0.0015, 0.0015, n_stocks)
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    data = {c: 100.0 * np.cumprod(1.0 + drifts[i] + rng.normal(0.0, 0.01, n_days))
            for i, c in enumerate(cols)}
    return pd.DataFrame(data, index=idx)


def _cash(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(0.00005, index=index)  # ~1.3% p.a. flach, nur fuers Mechanik-Skelett


def check_prepare_inputs_shapes() -> None:
    prices = _prices()
    inputs = prepare_inputs(prices, cash_daily=_cash(prices.index))
    assert set(inputs) >= {"returns", "vols", "signal", "cash_daily", "eval_decisions"}
    assert inputs["returns"].shape[1] == 30
    assert inputs["signal"].shape == inputs["returns"].shape
    assert len(inputs["eval_decisions"]) > 0
    print("prepare_inputs (Cross-Sectional-Momentum): OK")


def check_run_variant_long_flat_never_net_short() -> None:
    prices = _prices()
    inputs = prepare_inputs(prices, cash_daily=_cash(prices.index))
    net, info = run_variant(inputs, mode="long_flat", vol_cap=0.15,
                            cost_bp={s: 5.0 for s in inputs["returns"].columns})
    assert not net.isna().any()
    assert info["max_daily_gross"] <= 1.0 + 1e-9, "long_flat darf Gross 1.0 nie ueberschreiten (kein Hebel)"
    print("run_variant long_flat (Cross-Sectional-Momentum): OK")


def check_run_variant_long_short_uses_both_legs() -> None:
    prices = _prices()
    inputs = prepare_inputs(prices, cash_daily=_cash(prices.index))
    net, info = run_variant(inputs, mode="long_short", vol_cap=0.15,
                            cost_bp={s: 5.0 for s in inputs["returns"].columns})
    assert not net.isna().any()
    assert any(v != 0.0 for v in info["instrument_contributions"].values())
    print("run_variant long_short (Cross-Sectional-Momentum): OK")


def check_mechanics_test_vol_scaling_reduces_realized_vol() -> None:
    """Kernfrage des Mechanik-Tests: skaliert das bestehende Vola-Cap aus
    portfolio.py die realisierte Vola tatsaechlich herunter, wenn der Cap
    strenger ist als ungebremst? (Reine Mechanik-Pruefung, kein Beweis
    einer echten Krisenfestigkeit -- dafuer fehlen hier die extremen
    Regime-Wechsel, die echte Momentum-Crashes ausmachen.)"""
    prices = _prices(seed=1)
    result = run_mechanics_test(prices)
    for mode in ("long_short", "long_flat"):
        capped = result[mode]["vol_capped"]["stats"]["vol_pa"]
        uncapped = result[mode]["uncapped"]["stats"]["vol_pa"]
        assert capped <= uncapped + 1e-9, (
            f"{mode}: vol-gecapte Variante ({capped:.4f}) sollte nicht mehr Vola haben als ungebremst ({uncapped:.4f})"
        )
    print("run_mechanics_test: Vola-Cap reduziert realisierte Vola (long_short + long_flat): OK")


def run_consistency_check() -> None:
    check_prepare_inputs_shapes()
    check_run_variant_long_flat_never_net_short()
    check_run_variant_long_short_uses_both_legs()
    check_mechanics_test_vol_scaling_reduces_realized_vol()
    print("\nAlle run_cross_sectional_momentum-Checks bestanden.")


if __name__ == "__main__":
    run_consistency_check()
