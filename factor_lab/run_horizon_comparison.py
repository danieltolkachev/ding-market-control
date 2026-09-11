"""Fixed longer-horizon experiment; existing history is exploratory only."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shutil
import json
import numpy as np
import pandas as pd
import torch
from factor_lab.daily_comparison import build_dataset, make_targets, simulate
from factor_lab.daily_models import fit_predict_models
from factor_lab.horizon_data import horizon_labels
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.run_daily_comparison import write_json
from factor_lab.stats import annualized_stats, monthly_log_returns, stationary_block_bootstrap

CASES = ((1,1),(1,5),(1,21),(5,5),(21,21))
LEARNERS = ('ridge','lstm_frozen','lstm_online')
MODELS = ('reference','always_long')+LEARNERS
COSTS = (3.,6.,15.)


def case_name(horizon, interval):
    return f'h{horizon}_hold{interval}'


def assert_reproduction(actual, expected):
    """CSV date parsing may change timestamp precision, but not date values."""
    actual,expected = actual.copy(),expected.copy()
    actual.index = actual.index.as_unit('ns')
    expected.index = expected.index.as_unit('ns')
    if isinstance(actual,pd.Series):
        pd.testing.assert_series_equal(actual,expected,check_names=False,check_freq=False,
                                       check_exact=False,atol=1e-12,rtol=1e-10)
    else:
        pd.testing.assert_frame_equal(actual,expected,check_freq=False,
                                      check_exact=False,atol=1e-12,rtol=1e-10)


def verify_prior(prior, expected_snapshot_hash):
    if not (prior/'COMPLETE').exists():
        raise ValueError('Daily control is not marked complete')
    manifest = json.loads((prior/'sha256.json').read_text(encoding='utf-8'))
    names = ['configuration.json','net_returns.csv']+[f'{m}_predictions.csv' for m in LEARNERS]
    for name in names:
        if hashlib.sha256((prior/name).read_bytes()).hexdigest()!=manifest[name]:
            raise ValueError(f'Daily control hash mismatch: {name}')
    old_config = json.loads((prior/'configuration.json').read_text(encoding='utf-8'))
    if old_config['snapshot_content_sha256']!=expected_snapshot_hash:
        raise ValueError('Control used different snapshot')
    return {name:manifest[name] for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--daily-control',required=True)
    parser.add_argument('--output-root',required=True)
    args = parser.parse_args()
    dfs = load_trend_snapshot(args.snapshot)
    content_hash = snapshot_content_sha256(dfs)
    prior = Path(args.daily_control)
    control_hashes = verify_prior(prior,content_hash)
    symbols = sorted(set(dfs)-{'IRX'})
    prices = pd.concat({s:dfs[s]['price'] for s in symbols},axis=1,sort=True).dropna()
    data = build_dataset(prices)
    first = int(data['dates'].searchsorted(pd.Timestamp('2015-01-01')))
    if first < 252 or first >= len(data['dates'])-252:
        raise ValueError('Insufficient training/evaluation data')
    dates = data['dates'][first:]
    cash = dfs['IRX']['rate_pa_pct'].reindex(prices.index).ffill().shift(1).loc[dates]/100/252
    if cash.isna().any():
        raise ValueError('Missing causally available cash rate')
    out = Path(args.output_root)/datetime.now(timezone.utc).strftime('horizon_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    config = {'classification':'exploratory_only','cases':CASES,'primary_horizon':21,
              'secondary_horizon':5,'seed':7,'cost_bp':COSTS,'symbols':symbols,
              'test_start':str(dates[0]),'test_end':str(dates[-1]),
              'snapshot_content_sha256':content_hash,'daily_control_hashes':control_hashes,
              'label':'(P[t+H+1]/P[t+1]-1)/(vol[t]*sqrt(H)), clipped to [-10,10]',
              'maturity_delay':'H+1','execution':'next close; drift between scheduled fills',
              'decision_phase':'first evaluation date, then every holding_interval rows',
              'online_update_every':21,'online_mature_window_days':252,
              'return_target':.12,'drawdown_evaluation_limit':.15,'gross_ceiling':1.,
              'user_leverage_ceiling':1.25,'volatility_cap':.10,
              'bootstrap':{'block_months':6,'n_boot':10000,'seed':7},
              'versions':{'python':platform.python_version(),'numpy':np.__version__,
                          'pandas':pd.__version__,'torch':torch.__version__},
              'limitations':['Previously observed history; no independent holdout',
                            'Overlapping training labels; descriptive bootstrap intervals, no multiplicity adjustment',
                            'Single fixed decision phase; no phase robustness claim',
                            'No taxes, FX, whole-share rounding, broker minimum fees or financing',
                            'Volatility ceiling and historical drawdown do not guarantee future risk',
                            'Final partial holding period marked to market; no forced liquidation',
                            '2026 annual return is partial; model architecture and seed not searched']}
    write_json(out/'configuration.json',config)
    root = Path(__file__).resolve().parents[1]
    for relative in ['factor_lab','market_control_system/controller/cross_sectional_signal_metrics.py',
                     'market_control_system/data_layer/frozen_snapshot.py',
                     'docs/superpowers/plans/2026-09-08-horizon-comparison.md']:
        source = root/relative
        for file in source.rglob('*.py') if source.is_dir() else [source]:
            dest = out/'source'/file.relative_to(root)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,dest)
    print(f'OUTPUT: {out}',flush=True)
    predictions, labels = {},{}
    summary = {'configuration':config,'cases':{},'paired':{},'prediction_metrics':{}}
    for horizon in (1,5,21):
        # Preserve the old H=1 arithmetic exactly for a strict reproduction.
        y = data['y'] if horizon==1 else horizon_labels(prices,data['vol'],horizon).to_numpy(dtype='float32')
        labels[horizon] = y[first:]
        print(f'Horizon {horizon}: fit with maturity delay {horizon+1}',flush=True)
        predictions[horizon] = fit_predict_models(data['X'],y,first_test=first,label_delay=horizon+1,seed=7)
        pd.DataFrame(labels[horizon],index=dates,columns=symbols).to_csv(out/f'h{horizon}_labels.csv')
        summary['prediction_metrics'][str(horizon)] = {}
        for model in LEARNERS:
            values = predictions[horizon][model]
            if values.shape!=labels[horizon].shape or not np.isfinite(values).all():
                raise ValueError(f'Invalid forecasts for {horizon}/{model}')
            frame = pd.DataFrame(values,index=dates,columns=symbols)
            frame.to_csv(out/f'h{horizon}_{model}_predictions.csv')
            if horizon==1:
                original = pd.read_csv(prior/f'{model}_predictions.csv',index_col=0,parse_dates=True)
                assert_reproduction(frame,original)
            valid = np.isfinite(labels[horizon])
            summary['prediction_metrics'][str(horizon)][model] = {
                'n_labels':int(valid.sum()),
                'direction_accuracy':float(((values>0)==(labels[horizon]>0))[valid].mean()),
                'always_up_accuracy':float((labels[horizon][valid]>0).mean()),
                'normalized_mse':float(((values[valid]-labels[horizon][valid])**2).mean())}
        np.testing.assert_array_equal(predictions[horizon]['lstm_frozen'][:21],predictions[horizon]['lstm_online'][:21])
    references = {'reference':np.where(data['reference'].loc[dates].to_numpy(),1.,-1.),
                  'always_long':np.ones((len(dates),len(symbols)))}
    target_cache = {}
    for horizon in (1,5,21):
        for model,values in {**references,**predictions[horizon]}.items():
            key = (0 if model in references else horizon,model)
            if key not in target_cache:
                target_cache[key] = make_targets(pd.DataFrame(values,index=dates,columns=symbols),data['vol'].loc[dates],data['returns'])
                target_cache[key].to_csv(out/f'h{key[0]}_{model}_targets.csv')
    paths = {}
    old_net = pd.read_csv(prior/'net_returns.csv',index_col=0,parse_dates=True)
    for horizon,interval in CASES:
        name = case_name(horizon,interval)
        mask = pd.Series(np.arange(len(dates))%interval==0,index=dates)
        mask.to_csv(out/f'{name}_decisions.csv')
        summary['cases'][name] = {}
        for model in MODELS:
            targets = target_cache[(0 if model in references else horizon,model)]
            summary['cases'][name][model] = {}
            for cost in COSTS:
                net,detail = simulate(data['returns'].loc[dates],cash,targets,cost,decision_mask=mask)
                if not np.isfinite(net).all() or detail['gross'].max()>1+1e-10:
                    raise ValueError('Invalid equity or leverage path')
                if horizon==interval==1 and model!='always_long':
                    expected = old_net[f'{model}/{cost}']
                    assert_reproduction(net,expected)
                key = f'{name}/{model}/{cost}'
                paths[key] = net
                detail.to_csv(out/f'{name}_{model}_{cost:g}bp_daily.csv')
                stats = annualized_stats(net,cash)
                stats.update({'turnover_sum':float(detail['turnover'].sum()),
                              'cost_sum':float(detail['cost'].sum()),'max_gross':float(detail['gross'].max()),
                              'ending_10000':float(10000*(1+net).prod()),
                              'historical_target_met':bool(stats['cagr']>=.12 and stats['max_drawdown']>=-.15),
                              'annual_returns':{str(yr):float((1+g).prod()-1) for yr,g in net.groupby(net.index.year)}})
                summary['cases'][name][model][str(cost)] = stats
        print(f'{name}: portfolio calculations complete',flush=True)
    summary['cash_only'] = annualized_stats(cash,cash)
    paths['cash_only'] = cash
    pd.DataFrame(paths).to_csv(out/'net_returns.csv')
    monthly = {key:monthly_log_returns(net) for key,net in paths.items()}

    def paired(label,a,b):
        delta = monthly[a]-monthly[b]
        summary['paired'][label] = stationary_block_bootstrap(delta.to_numpy(),6,10000,7)

    for horizon,interval in CASES:
        name = case_name(horizon,interval)
        for cost in COSTS:
            for model in LEARNERS:
                for baseline in ('reference','always_long'):
                    paired(f'{name}/{model}_vs_{baseline}/{cost}',f'{name}/{model}/{cost}',f'{name}/{baseline}/{cost}')
            paired(f'{name}/online_vs_frozen/{cost}',f'{name}/lstm_online/{cost}',f'{name}/lstm_frozen/{cost}')
        print(f'{name}: paired comparisons complete',flush=True)
    for interval in (5,21):
        for model in LEARNERS:
            for cost in COSTS:
                paired(f'holding_effect/{interval}/{model}/{cost}',f'h1_hold{interval}/{model}/{cost}',f'h1_hold1/{model}/{cost}')
                paired(f'horizon_effect/{interval}/{model}/{cost}',f'h{interval}_hold{interval}/{model}/{cost}',f'h1_hold{interval}/{model}/{cost}')
    summary['daily_reproduction_verified'] = True
    write_json(out/'summary.json',summary)
    lines = ['# Longer horizon comparison — exploratory', '',
             f'Period: {dates[0].date()} through {dates[-1].date()}. All rows net of costs, before tax.',
             'H=forecast trading days; hold=decision interval. H21 is primary, H5 sensitivity.', '',
             '| H | Hold | Model | CAGR 3bp | Max DD | CAGR 6bp | CAGR 15bp | Turnover total |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    for horizon,interval in CASES:
        for model in MODELS:
            s = summary['cases'][case_name(horizon,interval)][model]
            lines.append(f"| {horizon} | {interval} | {model} | {s['3.0']['cagr']:.2%} | {s['3.0']['max_drawdown']:.2%} | {s['6.0']['cagr']:.2%} | {s['15.0']['cagr']:.2%} | {s['3.0']['turnover_sum']:.2f} |")
    lines += ['',f"Cash-only CAGR: {summary['cash_only']['cagr']:.2%}.", '',
              '## Paired relative geometric growth (descriptive 95% intervals)', '']
    for name,s in summary['paired'].items():
        lines.append(f"- {name}: {s['ann_geom']:.2%} [{s['ci_low_95']:.2%}, {s['ci_high_95']:.2%}]")
    lines += ['', 'Daily predictions and all12 original daily return paths reproduced within numerical tolerance.', '',
              '## Limits','']+[f'- {item}' for item in config['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'sha256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
    (out/'COMPLETE').write_text('Completed fixed experiment; all artifacts covered by sha256.json.\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
