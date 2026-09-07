"""Fixed-recipe exploratory daily model comparison. No network or live orders."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import numpy as np
import pandas as pd
import torch
from factor_lab.daily_comparison import build_dataset, make_targets, simulate
from factor_lab.daily_models import fit_predict_models
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.stats import annualized_stats, monthly_log_returns, stationary_block_bootstrap


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    dfs = load_trend_snapshot(args.snapshot)
    symbols = sorted(set(dfs)-{'IRX'})
    prices = pd.concat({s:dfs[s]['price'] for s in symbols},axis=1).dropna()
    data = build_dataset(prices)
    first = int(data['dates'].searchsorted(pd.Timestamp('2015-01-01')))
    if first < 252 or first >= len(data['dates'])-252:
        raise ValueError('Insufficient training or evaluation history')
    dates = data['dates'][first:]
    # Rate known at the previous close; never backward-fill from the future.
    cash = dfs['IRX']['rate_pa_pct'].reindex(prices.index).ffill().shift(1).loc[dates]/100/252
    if cash.isna().any():
        raise ValueError('Cash rate unavailable causally')
    out = Path(args.output_root)/datetime.now(timezone.utc).strftime('daily_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    config = {'classification':'exploratory_only', 'seed':7, 'test_start':str(dates[0]),
              'test_end':str(dates[-1]), 'snapshot_content_sha256':snapshot_content_sha256(dfs),
              'symbols':symbols, 'first_test_index':first, 'sequence_days':20,
              'feature_horizons':[1,5,21,63,126], 'label':'return at t+2 / volatility at t',
              'cost_bp':[3.,6.,15.], 'target_cagr':.12, 'drawdown_evaluation_limit':.15,
              'leverage_ceiling':1.25, 'experiment_max_gross':1., 'vol_target':.10,
              'bootstrap':{'n':10000,'block_months':6,'seed':7},
              'versions':{'python':platform.python_version(),'numpy':np.__version__,
                          'pandas':pd.__version__,'torch':torch.__version__},
              'limitations':['Previously observed history; not independent validation',
                            'Fixed ETF universe; no asset-selection bias correction',
                            'No FX conversion, taxes, whole-share rounding or broker minimum fees',
                            'Daily close execution and flat per-notional costs are approximations',
                            'Volatility target and historical drawdown filter are not future guarantees',
                            'Terminal calendar year is partial; pooled accuracy is descriptive']}
    write_json(out/'configuration.json', config)
    for relative in ['factor_lab','market_control_system/controller/cross_sectional_signal_metrics.py',
                     'market_control_system/data_layer/frozen_snapshot.py',
                     'docs/superpowers/plans/2026-09-07-daily-model-comparison.md']:
        source = root/relative
        files = source.rglob('*.py') if source.is_dir() else [source]
        for file in files:
            dest = out/'source'/file.relative_to(root)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,dest)
    print(f'OUTPUT: {out}',flush=True)
    print(f'Training days: {first-1}; evaluation days: {len(dates)}; assets: {len(symbols)}',flush=True)
    predictions = fit_predict_models(data['X'],data['y'],first_test=first,seed=7)
    predictions['reference'] = np.where(data['reference'].iloc[first:].to_numpy(),1.,-1.)
    summary = {'configuration':config,'models':{},'paired':{}}
    paths = {}
    y = data['y'][first:]
    for name in ['reference','ridge','lstm_frozen','lstm_online']:
        pred = predictions[name]
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError(f'Invalid prediction shape/values: {name}')
        pd.DataFrame(pred,index=dates,columns=symbols).to_csv(out/f'{name}_predictions.csv')
        valid = np.isfinite(y)
        metrics = {'direction_accuracy':float(((pred>0)==(y>0))[valid].mean()),
                   'n_labels':int(valid.sum()),'normalized_mse':float(np.mean((pred[valid]-y[valid])**2)) if name!='reference' else None,
                   'always_up_accuracy':float((y[valid]>0).mean())}
        frame = pd.DataFrame(pred,index=dates,columns=symbols)
        targets = make_targets(frame,data['vol'].loc[dates],data['returns'])
        targets.to_csv(out/f'{name}_targets.csv')
        summary['models'][name] = {'prediction_metrics':metrics,'costs':{}}
        for cost in config['cost_bp']:
            net, detail = simulate(data['returns'].loc[dates],cash,targets,cost)
            detail.to_csv(out/f'{name}_{cost:g}bp_daily.csv')
            stats = annualized_stats(net,cash)
            stats.update({'turnover_sum':float(detail['turnover'].sum()),
                          'max_gross':float(detail['gross'].max()),
                          'ending_10000':float(10000*(1+net).prod()),
                          'historical_target_met':bool(stats['cagr']>=.12 and stats['max_drawdown']>=-.15),
                          'annual_returns':{str(year):float((1+group).prod()-1) for year,group in net.groupby(net.index.year)}})
            summary['models'][name]['costs'][str(cost)] = stats
            paths[f'{name}/{cost}'] = net
        print(f'{name}: portfolio calculations complete',flush=True)
    pd.DataFrame(paths).to_csv(out/'net_returns.csv')
    pd.DataFrame(y,index=dates,columns=symbols).to_csv(out/'labels.csv')
    for name, baseline in [('ridge','reference'),('lstm_frozen','reference'),
                           ('lstm_online','reference'),('lstm_online','lstm_frozen')]:
        delta = monthly_log_returns(paths[f'{name}/3.0'])-monthly_log_returns(paths[f'{baseline}/3.0'])
        summary['paired'][f'{name}_vs_{baseline}'] = stationary_block_bootstrap(delta.to_numpy(),6,10000,7)
    write_json(out/'summary.json',summary)
    lines = ['# Daily model comparison — exploratory', '',
             f'Period: {dates[0].date()} to {dates[-1].date()}. Net of trading costs, before tax. No independent holdout.', '',
             '| Model | CAGR (3bp) | Max DD | CAGR (6bp) | CAGR (15bp) | Direction accuracy |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in summary['models'].items():
        c = result['costs']
        lines.append(f"| {name} | {c['3.0']['cagr']:.2%} | {c['3.0']['max_drawdown']:.2%} | {c['6.0']['cagr']:.2%} | {c['15.0']['cagr']:.2%} | {result['prediction_metrics']['direction_accuracy']:.2%} |")
    lines += ['', '## Paired relative geometric growth (descriptive 95% bootstrap interval)', '']
    for name,result in summary['paired'].items():
        lines.append(f"- {name}: {result['ann_geom']:.2%} [{result['ci_low_95']:.2%}, {result['ci_high_95']:.2%}]")
    lines += ['', '## Limits', '']+[f'- {item}' for item in config['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes = {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    write_json(out/'sha256.json',hashes)
    (out/'COMPLETE').write_text('Run finished successfully. sha256.json covers research artifacts.\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
