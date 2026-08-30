"""
run_cross_sectional_backtest.py
=================================

Hauptskript des Cross-Sectional-Ansatzes (siehe Design-Spec
docs/superpowers/specs/2026-08-30-cross-sectional-portfolio-design.md):
laedt alle Symbole aus cross_sectional_universe.UNIVERSE, trainiert pro
Symbol ein unabhaengiges LSTM (wie run_backtest.py), und spielt dann einen
SYNCHRONISIERTEN Replay ueber alle Symbole, bei dem CrossSectionalPortfolio
pro gemeinsamem Zeitschritt long/short/flat pro Symbol entscheidet.

Ausrichtung ueber Symbole: nur Zeitstempel, die ALLE Symbole gemeinsam
haben (Inner-Join), gehen in den Replay ein -- vereinfacht gegenueber dem
Design-Spec-Vorschlag (Mark-to-Market bei fehlenden Symbolen), aber bei
12 liquiden Large Caps wird der Datenverlust dadurch als gering erwartet
und die Vereinfachung ist fuer die erste Iteration bewusst in Kauf
genommen (siehe Spec, "Nicht-Ziel").

Ausfuehren: py -3.12 orchestration/run_cross_sectional_backtest.py
"""
from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "execution"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import numpy as np
import pandas as pd
import torch

from feature_pipeline import FEATURE_NAMES, build_scaled_features_and_target
from sequence_buffer import SequenceWindowBuilder
from lstm_forecaster_torch import LSTMForecaster, train_epoch
from paper_execution import PaperExecutionEngine, ExecutionConfig
from alpaca_client import fetch_historical_bars_approximate
from backtest_stats import compute_period_statistics, format_statistics_report
from cross_sectional_portfolio import CrossSectionalPortfolio, CrossSectionalPortfolioConfig
from symbol_forecaster import SymbolForecaster
from cross_sectional_universe import UNIVERSE, BACKTEST_LOOKBACK_DAYS, PRETRAIN_FRACTION, TIMESTEPS, HORIZON

CAPITAL = 10_000.0
MIN_REBALANCE_THRESHOLD = 0.05  # Gewichtsaenderung unterhalb dieser Schwelle wird ignoriert (kein Re-Fill) --
                                 # dasselbe Deadband-Prinzip wie RiskOverlayConfig.min_rebalance_threshold im
                                 # Single-Symbol-System, hier pro Symbol angewendet statt ueber RiskOverlay,
                                 # weil CrossSectionalPortfolio.compute_target_weights() bei jeder Mitglieder-
                                 # aenderung EINES Symbols in einer Beinseite (Long/Short) ALLE gehaltenen
                                 # Symbole dieser Seite neu bepreist (1/len(longs) aendert sich) -- ohne Deadband
                                 # zahlt die gesamte Seite bei jedem Rang-Wackeln Transaktionskosten, obwohl die
                                 # Hysterese in cross_sectional_portfolio.py die Mitgliedschaft korrekt gehalten hat.


def fetch_all_symbols() -> dict:
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=BACKTEST_LOOKBACK_DAYS)
    dfs = {}
    for symbol in UNIVERSE:
        print(f"  Lade {symbol}...")
        dfs[symbol] = fetch_historical_bars_approximate(symbol, start, end)
        print(f"    {dfs[symbol].shape[0]} Bars")
    return dfs


def align_and_split(dfs: dict):
    """Inner-Join der Zeitindizes ueber alle Symbole, dann Cutoff-Zeitpunkt
    fuer den Pretrain/Replay-Split (PRETRAIN_FRACTION in den gemeinsamen
    Index hinein)."""
    aligned_index = dfs[UNIVERSE[0]].index
    for symbol in UNIVERSE[1:]:
        aligned_index = aligned_index.intersection(dfs[symbol].index)
    aligned_index = aligned_index.sort_values()

    if len(aligned_index) == 0:
        raise ValueError("Kein gemeinsamer Zeitindex ueber alle Symbole -- Universum/Zeitraum pruefen")

    split_idx = int(len(aligned_index) * PRETRAIN_FRACTION)
    cutoff = aligned_index[split_idx]
    print(f"  Gemeinsamer Zeitindex: {len(aligned_index)} Bars "
          f"({aligned_index.min()} bis {aligned_index.max()})")
    print(f"  Cutoff (Vortraining/Replay-Grenze): {cutoff}")
    return aligned_index, cutoff


def pretrain_symbol(symbol: str, df: pd.DataFrame, cutoff) -> LSTMForecaster:
    """df wird als bereits auf den gemeinsamen (ausgerichteten) Index VOR cutoff
    beschraenkt erwartet (siehe Aufrufer in run_backtest()) -- der Filter hier
    bleibt als Absicherung stehen, ist im Normalfall aber ein No-Op."""
    pretrain_df = df[df.index < cutoff]
    features, target = build_scaled_features_and_target(pretrain_df, horizon=HORIZON)
    builder = SequenceWindowBuilder(timesteps=TIMESTEPS, feature_names=list(FEATURE_NAMES))
    X, y, _ = builder.build(features, target)

    print(f"  {symbol}: {pretrain_df.shape[0]} Vortrainings-Bars, {X.shape[0]} Sequenzen")
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        stats = train_epoch(model, optimizer, X, y, batch_size=64)
        print(f"    {symbol} Epoch {epoch+1}: mean_loss={stats['mean_loss']:.4f}")
    return model


def run_backtest(seed: int | None = None, portfolio_config: CrossSectionalPortfolioConfig | None = None) -> dict:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"=== Lade {len(UNIVERSE)} Symbole ({BACKTEST_LOOKBACK_DAYS} Tage) ===")
    dfs = fetch_all_symbols()

    print("\n=== Zeitindizes ausrichten ===")
    aligned_index, cutoff = align_and_split(dfs)
    replay_index = aligned_index[aligned_index >= cutoff]
    pretrain_index = aligned_index[aligned_index < cutoff]

    print(f"\n=== Offline-Vortraining pro Symbol ===")
    forecasters = {}
    for symbol in UNIVERSE:
        # Vortraining auf dem AUSGERICHTETEN (Inner-Join-)Index, nicht der eigenen
        # dichten Bar-Serie des Symbols -- sonst bedeutet HORIZON/Rolling-Fenster im
        # Training etwas anderes als im (sparseren) synchronisierten Replay (z.B.
        # gemeinsamer Index 96988 Bars vs. AAPL allein 190630 Bars, ~2.4x dichter),
        # was Trainings-/Serving-Kadenz auseinanderlaufen laesst.
        model = pretrain_symbol(symbol, dfs[symbol].loc[pretrain_index], cutoff)
        forecasters[symbol] = SymbolForecaster(timesteps=TIMESTEPS, model=model, horizon=HORIZON)

    portfolio = CrossSectionalPortfolio(UNIVERSE, portfolio_config or CrossSectionalPortfolioConfig())
    executions = {symbol: PaperExecutionEngine(ExecutionConfig(slippage_bps=0.5)) for symbol in UNIVERSE}

    print(f"\n=== Synchronisierter Replay ({len(replay_index)} gemeinsame Bars) ===")
    timestamps_list = []
    returns_list = []
    n_active_steps = 0
    total_slippage_cost = 0.0
    total_turnover = 0.0
    previous_executed_weights = {symbol: 0.0 for symbol in UNIVERSE}

    for timestamp in replay_index:
        raw_events = {symbol: dfs[symbol].loc[timestamp].to_dict() for symbol in UNIVERSE}

        mus, sigmas = {}, {}
        all_ready = True
        for symbol in UNIVERSE:
            result = forecasters[symbol].step(raw_events[symbol])
            if result is None:
                all_ready = False
                continue
            mus[symbol] = result.mu
            sigmas[symbol] = result.sigma

        if not all_ready:
            continue  # noch nicht alle Symbole warmgelaufen (SequenceBuffer-Fuellphase)

        weights = portfolio.step(mus, sigmas)

        bar_return = 0.0
        for symbol in UNIVERSE:
            target_weight = weights[symbol]
            # Rebalance-Deadband: kleine Gewichtsaenderungen (z.B. weil ein
            # ANDERES Symbol im selben Bein neu ein-/austritt und dadurch
            # 1/len(longs) fuer alle Gehaltenen minimal verschiebt) werden
            # ignoriert -- die Hysterese in CrossSectionalPortfolio haelt die
            # MITGLIEDSCHAFT korrekt, aber ohne dieses Deadband wuerde trotzdem
            # jedes Bein bei jedem Mitgliederwechsel neu ausgefuehrt.
            if abs(target_weight - previous_executed_weights[symbol]) < MIN_REBALANCE_THRESHOLD:
                target_weight = previous_executed_weights[symbol]
            fill = executions[symbol].execute(target_weight, raw_events[symbol])
            previous_executed_weights[symbol] = target_weight
            bar_return += fill.realized_return
            total_slippage_cost += fill.slippage_cost
            total_turnover += abs(fill.position_delta)

        n_active_steps += 1
        timestamps_list.append(timestamp)
        returns_list.append(bar_return)

        if n_active_steps % 5000 == 0:
            print(f"  {n_active_steps}/{len(replay_index)} Bars verarbeitet ({timestamp.date()}) "
                  f"-- longs={sorted(portfolio.current_longs)} shorts={sorted(portfolio.current_shorts)}")

    print(f"\n=== Replay abgeschlossen: {n_active_steps} aktive Bars ===")

    period_stats = compute_period_statistics(
        pd.DatetimeIndex(timestamps_list), np.array(returns_list), period="W"
    )
    print(format_statistics_report(period_stats, period_label="Woche", capital=CAPITAL))
    print(f"Gesamt-Slippage-Kosten: {total_slippage_cost:.4f}  Gesamt-Turnover (Summe |Delta|): {total_turnover:.4f}")

    # Dieselbe Leer-Perioden-Maske wie compute_period_statistics() anwenden
    # (period_returns wird dort um Perioden OHNE Aktivitaet gekuerzt) -- sonst
    # kann period_timestamps laenger als period_returns werden und jedes Label
    # nach der Luecke verschiebt sich unbemerkt gegen die falsche Periode.
    step_series = pd.Series(returns_list, index=pd.DatetimeIndex(timestamps_list))
    period_active_mask = step_series.resample("W").count() > 0
    period_timestamps = [str(t) for t in step_series.resample("W").sum()[period_active_mask].index]

    return {
        "seed": seed,
        "universe": UNIVERSE,
        "n_active_steps": n_active_steps,
        "n_periods": period_stats.n_periods,
        "mean_period_return": period_stats.mean_return,
        "std_period_return": period_stats.std_return,
        "t_statistic": period_stats.t_statistic,
        "sharpe_like": period_stats.sharpe_like,
        "win_rate": period_stats.win_rate,
        "cumulative_return": period_stats.cumulative_return,
        "max_drawdown": period_stats.max_drawdown,
        "total_slippage_cost": total_slippage_cost,
        "total_turnover": total_turnover,
        "period_returns": period_stats.period_returns.tolist(),
        "period_timestamps": period_timestamps,
    }


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"cross_sectional_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Cross-Sectional-Backtest-Run-ID: {run_id}")

    summary = run_backtest()
    summary["run_id"] = run_id

    summary_path = os.path.join(results_dir, "portfolio_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
