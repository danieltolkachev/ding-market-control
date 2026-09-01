"""
run_trend_baseline.py — Screening der Familie trend-etf-v1 (Spec v2
§5-§10): 8 Varianten + matched_long + Kontext-Benchmarks auf IDENTISCHEM
Evaluationsfenster, Kostenleiter, Stationary-Block-Inferenz auf monatlichen
Log-MEHRertraegen, Gates A-D (LOO-Reruns fuer A-C-Passer), candidate.json.
Das Screening ist ausdruecklich KEINE Bestaetigung (Spec §10).

Laufzeit-Hinweis: >20 min moeglich (LOO-Reruns) -> detacht starten
(Start-Process + PID-Datei + Monitor, siehe Tech-Stack-Memory).

Ausfuehren: py -3.12 factor_lab/run_trend_baseline.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from factor_lab.signals import momentum_sign, combo_signal
from factor_lab.costs import COST_BP, COST_LADDER
from factor_lab.portfolio import (
    ewma_annualized_vol, month_end_dates, run_lagged_backtest,
    trend_weight_provider, fixed_mix_provider,
)
from factor_lab.stats import (
    monthly_log_returns, stationary_block_bootstrap, monthly_sign_flip_pvalue,
    annualized_stats, full_year_excess, evaluate_screening_gates, screening_verdict,
)
from factor_lab.registration import REGISTRATION, config_hash, write_candidate, file_sha256
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.build_trend_snapshot import resolve_dev_end

LOOKBACKS = {"mom63": 63, "mom126": 126, "mom252": 252}
SLEEVES = {
    "us_equity": ["SPY", "QQQ", "IWM"], "intl_equity": ["EFA", "EEM"],
    "bonds": ["TLT", "IEF", "LQD"], "real_assets": ["GLD", "SLV", "DBC", "VNQ"],
}
VARIANT_NAMES = sorted(f"{sig}_{mode}" for sig in [*LOOKBACKS, "combo"] for mode in ["long_short", "long_flat"])
WARMUP_DAYS = REGISTRATION["warmup_days"]


def prepare_inputs(dfs: dict) -> dict:
    symbols = sorted(s for s in dfs if s != "IRX")
    common = dfs[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(dfs[s].index)
    common = common.sort_values()
    prices = pd.DataFrame({s: dfs[s].loc[common, "price"] for s in symbols})
    returns = prices.pct_change().dropna()
    cash_daily = (dfs["IRX"]["rate_pa_pct"].reindex(common).ffill().bfill() / 100.0 / 252.0).loc[returns.index]

    per_lb = {name: pd.DataFrame({s: momentum_sign(prices[s], lb) for s in symbols})
              for name, lb in LOOKBACKS.items()}
    signal_frames = {**per_lb, "combo": pd.DataFrame({
        s: combo_signal([per_lb[n][s] for n in LOOKBACKS]) for s in symbols})}
    signal_frames = {k: v.loc[returns.index] for k, v in signal_frames.items()}
    vols = ewma_annualized_vol(returns)

    warmup_end = returns.index[WARMUP_DAYS]
    ends = month_end_dates(returns.index)
    eval_decisions = ends[ends >= warmup_end]
    return {"symbols": symbols, "returns": returns, "cash_daily": cash_daily,
            "signal_frames": signal_frames, "vols": vols, "eval_decisions": eval_decisions,
            "dev_end": resolve_dev_end(dfs)}


def run_variant(inputs: dict, variant: str, cost_multiplier: float = 1.0,
                exclude: set = frozenset()) -> tuple[pd.Series, dict]:
    symbols = [s for s in inputs["symbols"] if s not in exclude]
    returns = inputs["returns"][symbols]
    vols = inputs["vols"][symbols]
    if variant == "matched_long":
        signals = pd.DataFrame(1.0, index=returns.index, columns=symbols)
        mode = "long_flat"
    else:
        sig_name = variant.rsplit("_", 2)[0]
        mode = "_".join(variant.rsplit("_", 2)[1:])
        signals = inputs["signal_frames"][sig_name][symbols]
    provider = trend_weight_provider(returns, signals, vols, mode,
                                     vol_cap=REGISTRATION["vol_cap"], vol_window=63)
    return run_lagged_backtest(returns, inputs["cash_daily"], inputs["eval_decisions"],
                               provider, {s: COST_BP.get(s, 3.0) for s in symbols},
                               cost_multiplier, REGISTRATION["borrow_bp_pa"])


def _breakeven_multiplier(excess_by_mult: dict[float, float]) -> float:
    """Linear interpolierter Kostensatz-Multiplikator, bei dem der
    compoundierte Mehrertrag 0 wird; inf, wenn selbst 5x positiv bleibt,
    0, wenn schon 1x negativ ist."""
    mults = sorted(excess_by_mult)
    if excess_by_mult[mults[0]] <= 0:
        return 0.0
    for lo, hi in zip(mults, mults[1:]):
        e_lo, e_hi = excess_by_mult[lo], excess_by_mult[hi]
        if e_hi <= 0 < e_lo:
            return float(lo + (hi - lo) * e_lo / (e_lo - e_hi))
    return float("inf")


def run_screening(dfs: dict, dev_end=None) -> tuple[dict, pd.DataFrame]:
    """dev_end: von main() uebergeben, weil dort die Daten bereits auf das
    Entwicklungsfenster geschnitten sind -- die 80%-Regel darf nicht ein
    zweites Mal auf die geschnittenen Daten angewendet werden."""
    inputs = prepare_inputs(dfs)
    if dev_end is None:
        dev_end = inputs["dev_end"]
    cash = inputs["cash_daily"]

    matched = {m: run_variant(inputs, "matched_long", cost_multiplier=m) for m in COST_LADDER}
    per_day = {"matched_long": matched[1.0][0]}

    summary, excess_sharpes, gates_by_variant = {}, {}, {}
    for variant in VARIANT_NAMES:
        runs = {m: run_variant(inputs, variant, cost_multiplier=m) for m in COST_LADDER}
        net, info = runs[1.0]
        per_day[variant] = net

        excess = net - matched[1.0][0]
        # Mehrertrag als Differenz der monatlichen Log-Ertraege (geometrisch sauber):
        monthly_excess = (monthly_log_returns(net) - monthly_log_returns(matched[1.0][0])).dropna()
        boot = stationary_block_bootstrap(monthly_excess.to_numpy(),
                                          REGISTRATION["bootstrap"]["expected_block_len_months"],
                                          REGISTRATION["bootstrap"]["n_boot"],
                                          REGISTRATION["bootstrap"]["seed"])
        sensitivity = {str(L): stationary_block_bootstrap(
            monthly_excess.to_numpy(), L, REGISTRATION["bootstrap"]["n_boot"],
            REGISTRATION["bootstrap"]["seed"])["ann_geom_lower_1s95"]
            for L in REGISTRATION["bootstrap"]["sensitivity_block_lens"]}
        perm = monthly_sign_flip_pvalue(monthly_excess.to_numpy(), REGISTRATION["permutation_n"],
                                        REGISTRATION["bootstrap"]["seed"])

        stats = annualized_stats(net, cash)
        stats_stress = annualized_stats(runs[2.0][0], cash)
        ladder = {}
        excess_by_mult = {}
        for m in COST_LADDER:
            ladder[str(m)] = annualized_stats(runs[m][0], cash)
            m_excess = (monthly_log_returns(runs[m][0]) - monthly_log_returns(matched[m][0])).dropna()
            excess_by_mult[m] = float(np.expm1(m_excess.sum()))

        # Gates A-C zuerst; LOO-Reruns (teuer) nur fuer A-C-Passer.
        pre = evaluate_screening_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                       stats_stress["cagr"],
                                       full_year_excess(excess),
                                       loo_excess_compounds={"pending": 1.0},
                                       dd_cap=REGISTRATION["dd_cap"],
                                       gate_c_floor=REGISTRATION["gate_c_floor"])
        loo = {}
        if pre["gate_a_excess_ci"] and pre["gate_b_drawdown"] and pre["gate_c_stressed_floor"]:
            loo_configs = {f"loo_{s}": {s} for s in inputs["symbols"]}
            loo_configs.update({f"loo_sleeve_{k}": set(v) for k, v in SLEEVES.items()})
            for loo_name, excl in loo_configs.items():
                v_net, _ = run_variant(inputs, variant, exclude=excl)
                m_net, _ = run_variant(inputs, "matched_long", exclude=excl)
                diff = (monthly_log_returns(v_net) - monthly_log_returns(m_net)).dropna()
                loo[loo_name] = float(np.expm1(diff.sum()))
        gates = evaluate_screening_gates(boot, stats["max_drawdown"], stats_stress["max_drawdown"],
                                         stats_stress["cagr"], full_year_excess(excess), loo,
                                         REGISTRATION["dd_cap"], REGISTRATION["gate_c_floor"])
        gates_by_variant[variant] = gates
        excess_sharpes[variant] = stats["excess_sharpe"]
        summary[variant] = {
            "stats": stats,
            "excess_bootstrap": boot,
            "excess_bootstrap_sensitivity": sensitivity,
            "permutation": perm,
            "gates": gates,
            "cost_ladder": ladder,
            "breakeven_cost_multiplier": _breakeven_multiplier(excess_by_mult),
            "loo_excess_compounds": loo,
            "yearly_excess_full_years": {str(k): v for k, v in full_year_excess(excess).items()},
            "max_daily_gross": info["max_daily_gross"],
            "total_turnover": info["total_turnover"],
            "instrument_contributions": info["instrument_contributions"],
            "labels": (["research_variant"] if variant.endswith("long_short") else []),
        }

    spy_provider = fixed_mix_provider({"SPY": 1.0}) if "SPY" in inputs["symbols"] else fixed_mix_provider({inputs["symbols"][0]: 1.0})
    spy_bh, _ = run_lagged_backtest(inputs["returns"], cash,
                                    pd.DatetimeIndex([inputs["eval_decisions"][0]]),
                                    spy_provider, COST_BP if "SPY" in inputs["symbols"] else {s: 1.5 for s in inputs["symbols"]})
    mix_symbols = ["SPY", "TLT"] if set(["SPY", "TLT"]) <= set(inputs["symbols"]) else inputs["symbols"][:2]
    mix, _ = run_lagged_backtest(inputs["returns"], cash, inputs["eval_decisions"],
                                 fixed_mix_provider({mix_symbols[0]: 0.6, mix_symbols[1]: 0.4}),
                                 # Kosten-Dict muss ALLE Spalten von inputs["returns"] abdecken
                                 # (trade_cost_fraction schlaegt Symbol-Keys unbedingt nach, auch
                                 # bei Delta 0 -- siehe test_portfolio.check_missing_symbol_in_provider_output).
                                 {s: COST_BP.get(s, 1.5) for s in inputs["symbols"]})
    per_day["bench_spy_bh"] = spy_bh
    per_day["bench_60_40"] = mix

    verdict, candidate = screening_verdict(gates_by_variant, excess_sharpes)
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git_sha = "unbekannt"
    provenance = {
        "registration": REGISTRATION,
        "config_hash": config_hash(),
        "snapshot_content_sha256": snapshot_content_sha256(dfs),
        "git_sha": git_sha,
        "dev_end": str(dev_end),
        "eval_start": str(per_day["matched_long"].index.min()),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    result = {
        "summary": summary,
        "benchmarks": {"matched_long": annualized_stats(per_day["matched_long"], cash),
                       "bench_spy_bh": annualized_stats(spy_bh, cash),
                       "bench_60_40": annualized_stats(mix, cash)},
        "verdict": verdict,
        "candidate": candidate,
        "provenance": provenance,
    }
    return result, pd.DataFrame(per_day)


def main() -> None:
    dfs = load_trend_snapshot()
    # Hash der VOLLEN (nicht getrimmten) Snapshot-Daten -- MUSS vor dem
    # Trunkieren berechnet werden. run_screening() haelt sonst nur den Hash
    # der auf dev_end geschnittenen Daten fest; Task 10 (Holdout-Runner)
    # laedt aber den VOLLEN Snapshot und hasht DEN, um candidate.json zu
    # verifizieren (read_and_verify_candidate vergleicht per Stringgleichheit)
    # -- ein Hash der getrimmten Daten kann diesem Vergleich niemals
    # entsprechen (Review-Fund, Amendment nach Plan-Fehler).
    full_snapshot_sha256 = snapshot_content_sha256(dfs)
    inputs_probe = prepare_inputs(dfs)
    dev_end = inputs_probe["dev_end"]
    dev_dfs = {name: df.loc[df.index <= dev_end] for name, df in dfs.items()}
    result, per_day = run_screening(dev_dfs, dev_end=dev_end)
    # Provenance-Feld auf den Voll-Snapshot-Hash ueberschreiben, BEVOR es
    # gespeichert oder an write_candidate() weitergereicht wird.
    result["provenance"]["snapshot_content_sha256"] = full_snapshot_sha256

    print(f"\n{result['verdict']}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "logs", f"trend_screening_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    per_day.to_csv(os.path.join(out_dir, "daily_net_returns.csv"))
    results_path = os.path.join(out_dir, "screening_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if result["candidate"] is not None:
        write_candidate(os.path.join(out_dir, "candidate.json"),
                        variant=result["candidate"],
                        snapshot_sha256=result["provenance"]["snapshot_content_sha256"],
                        git_sha=result["provenance"]["git_sha"],
                        dev_end=result["provenance"]["dev_end"],
                        results_sha256=file_sha256(results_path))
    print(f"Ergebnisse gespeichert: {out_dir}")


if __name__ == "__main__":
    main()
