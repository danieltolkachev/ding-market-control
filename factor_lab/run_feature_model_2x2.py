"""Vorab festgelegtes 2x2-Raster Features x Modell; vier Zellen, keine Suche.

Bekannte Historie, kein frisches Holdout. Vier Zellen kommen zu den bestehenden
141 unkorrigierten Vergleichen hinzu; keine Mehrfachtestkorrektur wird behauptet.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from factor_lab.daily_comparison import make_targets, simulate
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.evaluate_2x2 import CELLS, N_WINDOWS, expanding_windows, predict_over_windows
from factor_lab.features_2x2 import BASE_HORIZONS, EXTENDED_NAMES, build_2x2_dataset
from factor_lab.models_2x2 import GBM_PARAMS, GBM_ROUNDS, RIDGE_LAMBDA_PER_FEATURE
from factor_lab.run_daily_comparison import write_json
from factor_lab.stats import annualized_stats, monthly_log_returns, stationary_block_bootstrap

EXPECTED_SNAPSHOT_SHA256 = '36f1c3c0e5fca54c86a642d72f11efcd8ec1b5c6448c01a4865f9510a6b6bb89'
TEST_START = '2015-01-01'
HOLDING_INTERVAL = 21
COSTS = (3., 15.)
CONTROL = 'always_long'
LABEL_DELAY = 2
PRIMARY_COST = str(COSTS[0])
LIMITATIONS = [
    'Previously observed history and universe; new window cuts are not a fresh holdout',
    'Four preregistered cells added to 141 existing uncorrected comparisons; no multiplicity adjustment',
    'Descriptive bootstrap intervals on overlapping monthly data; not a significance test',
    'The selection pipeline falsification audit covers trend-etf-v2 screening only; it does not transfer to this grid',
    'No whole-share rounding, no broker minimum fees, no FX, no taxes, no financing',
    'Five extended channels are a narrow selection; a null result refutes these five in this pipeline, not cross-sectional information in general',
    'The SPY correlation channel is constant 1 for SPY itself; market coupling is uninformative for the market instrument',
    'Volatility ceiling and historical drawdown do not guarantee future risk',
    'The 2026 year is partial; gross exposure capped at 1.0x, below the permitted 1.25x',
]


def build_config(content_hash, symbols, dates, windows, pilot, forced):
    return {
        'classification': 'preregistered_2x2_on_previously_observed_history',
        'pilot': pilot,
        'forced': forced,
        'expected_snapshot_sha256': EXPECTED_SNAPSHOT_SHA256,
        'cells': [list(cell) for cell in CELLS],
        'n_windows': N_WINDOWS,
        'windows': [list(w) for w in windows],
        'base_channels': list(BASE_HORIZONS),
        'extended_channels': list(EXTENDED_NAMES),
        'ridge_lambda_per_feature': RIDGE_LAMBDA_PER_FEATURE,
        'gbm_params': GBM_PARAMS,
        'gbm_rounds': GBM_ROUNDS,
        'label': 'returns.shift(-2)/vol, clipped to [-10,10]',
        'label_delay': LABEL_DELAY,
        'holding_interval': HOLDING_INTERVAL,
        'execution': 'next close; drift between scheduled fills',
        'allocation': 'make_targets: inverse volatility, long/flat gate, 10% volatility cap',
        'cost_bp': list(COSTS),
        'primary_cost_bp': 3.0,
        'control': CONTROL,
        'primary_metric': 'annualized relative geometric growth vs control, pooled monthly deltas',
        'gross_ceiling': 1.,
        'user_leverage_ceiling': 1.25,
        'volatility_cap': .10,
        'return_target': .12,
        'drawdown_evaluation_limit': .15,
        'symbols': symbols,
        'test_start': str(dates[0]),
        'test_end': str(dates[-1]),
        'snapshot_content_sha256': content_hash,
        'bootstrap': {'block_months': 6, 'n_boot': 10000, 'seed': 7},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'lightgbm': lgb.__version__},
        'limitations': LIMITATIONS,
    }


def compute_verdict(primary, primary_cost=PRIMARY_COST):
    """Die vorab festgelegte Entscheidungsregel, Spec Abschnitt 7."""
    excluding = [name for name, _, _ in CELLS
                 if primary[name][primary_cost]['ci_low_95'] > 0
                 or primary[name][primary_cost]['ci_high_95'] < 0]
    return {'cells_with_interval_excluding_zero': excluding,
            'reading': ('null result' if not excluding
                        else 'hint, not evidence' if len(excluding) == 1
                        else 'check axis consistency: C-A vs D-B')}


def copy_sources(out):
    root = Path(__file__).resolve().parents[1]
    for relative in ['factor_lab/features_2x2.py', 'factor_lab/models_2x2.py',
                     'factor_lab/evaluate_2x2.py', 'factor_lab/run_feature_model_2x2.py',
                     'factor_lab/daily_comparison.py', 'factor_lab/stats.py',
                     'docs/superpowers/specs/2026-09-11-feature-model-2x2-design.md',
                     'docs/superpowers/plans/2026-09-13-feature-model-2x2-implementation.md']:
        source = root / relative
        dest = out / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--pilot', action='store_true',
                        help='one cell, one window, timing only')
    parser.add_argument('--force', action='store_true',
                        help='allow a snapshot whose content hash differs')
    args = parser.parse_args()

    dfs = load_trend_snapshot(args.snapshot)
    content_hash = snapshot_content_sha256(dfs)
    if content_hash != EXPECTED_SNAPSHOT_SHA256 and not args.force:
        raise ValueError(f'Snapshot content hash {content_hash} is not the sealed one')
    symbols = sorted(set(dfs) - {'IRX'})
    prices = pd.concat({s: dfs[s]['price'] for s in symbols}, axis=1, sort=True).dropna()
    data = build_2x2_dataset(prices)
    if data['columns'] != symbols:
        raise ValueError('Feature-matrix asset axis does not match symbols order')
    days = len(data['dates'])
    first = int(data['dates'].searchsorted(pd.Timestamp(TEST_START)))
    if first < 252 or first >= days - 252:
        raise ValueError('Insufficient training/evaluation data')
    windows = expanding_windows(first, days)
    if args.pilot:
        windows = windows[:1]
    dates = data['dates'][windows[0][0]:windows[-1][1]]
    cash = dfs['IRX']['rate_pa_pct'].reindex(prices.index).ffill().shift(1).loc[dates] / 100 / 252
    if cash.isna().any():
        raise ValueError('Missing causally available cash rate')

    stamp = datetime.now(timezone.utc).strftime('feature_2x2_%Y%m%d_%H%M%S_%f')
    out = Path(args.output_root) / (stamp + ('_pilot' if args.pilot else ''))
    out.mkdir(parents=True, exist_ok=False)
    config = build_config(content_hash, symbols, dates, windows, args.pilot, args.force)
    write_json(out / 'configuration.json', config)
    copy_sources(out)
    print(f'OUTPUT: {out}', flush=True)
    print(f'Rows {days}, evaluation {len(dates)} from {dates[0].date()} to {dates[-1].date()}',
          flush=True)

    cells = CELLS[:1] if args.pilot else CELLS
    predictions = {}
    for name, block, model in cells:
        started = datetime.now(timezone.utc)
        print(f'{name}: {model} on {data[block].shape[-1]} channels', flush=True)
        values = predict_over_windows(
            data[block], data['y'], windows, model, label_delay=LABEL_DELAY,
            progress=lambda text, cell=name: print(f'  {cell}: {text}', flush=True))
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
        print(f'{name}: fitted in {seconds:.1f}s', flush=True)
        if values.shape != (len(dates), len(symbols)) or not np.isfinite(values).all():
            raise ValueError(f'Invalid forecasts for {name}')
        frame = pd.DataFrame(values, index=dates, columns=symbols)
        frame.to_csv(out / f'{name}_predictions.csv')
        predictions[name] = frame

    if args.pilot:
        print('Pilot complete: timing only, no statistics, no COMPLETE marker.', flush=True)
        return

    signals = dict(predictions)
    signals[CONTROL] = pd.DataFrame(1., index=dates, columns=symbols)
    mask = pd.Series(np.arange(len(dates)) % HOLDING_INTERVAL == 0, index=dates)
    mask.to_csv(out / 'decisions.csv')
    summary = {'configuration': config, 'cells': {}, 'primary': {}, 'per_window': {},
               'prediction_metrics': {}}
    labels = data['y'][windows[0][0]:windows[-1][1]]
    valid = np.isfinite(labels)
    pd.DataFrame(labels, index=dates, columns=symbols).to_csv(out / 'labels.csv')
    paths = {}
    for name, values in signals.items():
        targets = make_targets(values, data['vol'].loc[dates], data['returns'])
        targets.to_csv(out / f'{name}_targets.csv')
        summary['cells'][name] = {}
        for cost in COSTS:
            net, detail = simulate(data['returns'].loc[dates], cash, targets, cost,
                                   decision_mask=mask)
            if not np.isfinite(net).all() or detail['gross'].max() > 1 + 1e-10:
                raise ValueError('Invalid equity or leverage path')
            paths[f'{name}/{cost}'] = net
            detail.to_csv(out / f'{name}_{cost:g}bp_daily.csv')
            stats = annualized_stats(net, cash)
            stats.update({
                'turnover_sum': float(detail['turnover'].sum()),
                'cost_sum': float(detail['cost'].sum()),
                'max_gross': float(detail['gross'].max()),
                'ending_10000': float(10000 * (1 + net).prod()),
                'historical_target_met': bool(stats['cagr'] >= .12
                                              and stats['max_drawdown'] >= -.15),
                'annual_returns': {str(year): float((1 + g).prod() - 1)
                                   for year, g in net.groupby(net.index.year)}})
            summary['cells'][name][str(cost)] = stats
        if name != CONTROL:
            forecasts = values.to_numpy()
            summary['prediction_metrics'][name] = {
                'n_labels': int(valid.sum()),
                'direction_accuracy': float(((forecasts > 0) == (labels > 0))[valid].mean()),
                'always_up_accuracy': float((labels[valid] > 0).mean()),
                'normalized_mse': float(((forecasts[valid] - labels[valid]) ** 2).mean()),
                'label_second_moment': float((labels[valid] ** 2).mean()),
                'long_share': float((forecasts[valid] > 0).mean())}
        print(f'{name}: portfolio calculations complete', flush=True)

    summary['cash_only'] = annualized_stats(cash, cash)
    paths['cash_only'] = cash
    pd.DataFrame(paths).to_csv(out / 'net_returns.csv')
    monthly = {key: monthly_log_returns(net) for key, net in paths.items()}
    for name, _, _ in CELLS:
        summary['primary'][name] = {}
        summary['per_window'][name] = {}
        for cost in COSTS:
            delta = monthly[f'{name}/{cost}'] - monthly[f'{CONTROL}/{cost}']
            summary['primary'][name][str(cost)] = stationary_block_bootstrap(
                delta.to_numpy(), 6, 10000, 7)
            per_window = {}
            for number, (start, end) in enumerate(windows, start=1):
                block = dates[start - windows[0][0]:end - windows[0][0]]
                piece = delta.loc[(delta.index >= block[0].to_period('M').to_timestamp())
                                  & (delta.index <= block[-1].to_period('M').to_timestamp())]
                per_window[str(number)] = {
                    'start': str(block[0].date()), 'end': str(block[-1].date()),
                    'n_months': int(len(piece)),
                    'ann_geom': float(np.exp(12 * piece.mean()) - 1) if len(piece) else float('nan')}
            summary['per_window'][name][str(cost)] = per_window
    print('Paired comparisons complete', flush=True)

    summary['verdict'] = compute_verdict(summary['primary'])
    write_json(out / 'summary.json', summary)
    write_report(out, summary, config, dates, windows)
    write_json(out / 'sha256.json',
               {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob('*') if p.is_file()})
    (out / 'COMPLETE').write_text(
        'Completed preregistered 2x2; all artifacts covered by sha256.json.\n', encoding='utf-8')


def write_report(out, summary, config, dates, windows):
    names = [name for name, _, _ in CELLS] + [CONTROL]
    lines = ['# 2x2 Features x Modell — vorab festgelegtes Raster', '',
             f'Zeitraum: {dates[0].date()} bis {dates[-1].date()}, {len(dates)} Handelstage, '
             f'{N_WINDOWS} expandierende Fenster. Alle Zahlen nach modellierten Kosten, vor Steuern.',
             '', '| Zelle | CAGR 3bp | Max DD 3bp | CAGR 15bp | Max DD 15bp | Turnover | Endwert 10.000 EUR |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        low, high = summary['cells'][name]['3.0'], summary['cells'][name]['15.0']
        lines.append(f"| {name} | {low['cagr']:.2%} | {low['max_drawdown']:.2%} | "
                     f"{high['cagr']:.2%} | {high['max_drawdown']:.2%} | "
                     f"{low['turnover_sum']:.2f} | {low['ending_10000']:,.0f} |")
    lines += ['', f"Nur Cash: {summary['cash_only']['cagr']:.2%} CAGR.", '',
              '## Primaermetrik — annualisiertes relatives geometrisches Wachstum gegen '
              f'`{CONTROL}` (deskriptives 95%-Intervall)', '']
    for name, _, _ in CELLS:
        for cost in COSTS:
            s = summary['primary'][name][str(cost)]
            lines.append(f"- {name} @ {cost:g}bp: {s['ann_geom']:.2%} "
                         f"[{s['ci_low_95']:.2%}, {s['ci_high_95']:.2%}]")
    lines += ['', '## Achsenbeitraege bei 3 bp (Differenz der Punktschaetzer)', '']
    point = {name: summary['primary'][name]['3.0']['ann_geom'] for name, _, _ in CELLS}
    lines += [f"- Feature-Achse: C-A = {point['C_ext_ridge'] - point['A_base_ridge']:.2%}, "
              f"D-B = {point['D_ext_gbm'] - point['B_base_gbm']:.2%}",
              f"- Modell-Achse: B-A = {point['B_base_gbm'] - point['A_base_ridge']:.2%}, "
              f"D-C = {point['D_ext_gbm'] - point['C_ext_ridge']:.2%}"]
    lines += ['', '## Fensterweise Aufschluesselung bei 3 bp (deskriptiv)', '',
              '| Zelle | ' + ' | '.join(f'F{i}' for i in range(1, N_WINDOWS + 1)) + ' |',
              '|---|' + '---:|' * N_WINDOWS]
    for name, _, _ in CELLS:
        per = summary['per_window'][name]['3.0']
        lines.append(f'| {name} | ' + ' | '.join(
            f"{per[str(i)]['ann_geom']:.2%}" for i in range(1, N_WINDOWS + 1)) + ' |')
    lines += ['', '## Prognosemetriken', '',
              '| Zelle | Trefferquote Richtung | Immer-aufwaerts | Norm. MSE | Long-Anteil |',
              '|---|---:|---:|---:|---:|']
    for name, _, _ in CELLS:
        m = summary['prediction_metrics'][name]
        lines.append(f"| {name} | {m['direction_accuracy']:.2%} | {m['always_up_accuracy']:.2%} | "
                     f"{m['normalized_mse']:.4f} | {m['long_share']:.2%} |")
    lines += ['', '## Urteil nach der vorab festgelegten Entscheidungsregel', '',
              f"Zellen mit Intervall ohne Null (3 bp): "
              f"{summary['verdict']['cells_with_interval_excluding_zero'] or 'keine'}.",
              f"Lesart: **{summary['verdict']['reading']}**.", '',
              '## Stueckelung und Gebuehren bei 10.000 EUR', '',
              'Ganze Stuecke, Broker-Mindestgebuehren, FX und Steuern sind **nicht modelliert**. '
              'Die Kosten sind ein reiner Basispunkt-Aufschlag auf den Umsatz (3 bp primaer, '
              f'15 bp Sensitivitaet). Bei 10.000 EUR auf {len(config["symbols"])} Instrumente '
              'liegen einzelne Positionen '
              'im niedrigen dreistelligen Bereich, wo Mindestgebuehren und Stueckelung real '
              'spuerbar waeren; die Endwerte oben sind insoweit optimistisch.', '',
              '## Grenzen', ''] + [f'- {item}' for item in config['limitations']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
