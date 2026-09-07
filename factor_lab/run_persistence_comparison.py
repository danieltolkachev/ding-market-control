"""Explorativer gepaarter Vergleich; kein neuer Holdout, kein Datenabruf."""
import argparse
import hashlib
import json
import sys
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from factor_lab.data_snapshot import load_trend_snapshot, snapshot_content_sha256
from factor_lab.run_trend_baseline_v2 import prepare_inputs, run_variant, VARIANT_NAMES
from factor_lab.registration_v2 import REGISTRATION_V2
from factor_lab.portfolio import month_end_dates
from factor_lab.signal_persistence import daily_persistent_signals
from factor_lab.stats import annualized_stats, monthly_log_returns, stationary_block_bootstrap


def compare_period(inputs, start, end, variants=VARIANT_NAMES, costs=(1.,2.,5.), n_boot=10000):
    """Jeder Lauf startet cash; Filterzustand stammt kausal aus der Vorgeschichte."""
    raw = dict(inputs)
    raw['returns'] = inputs['returns'].loc[:end]
    decisions = inputs['eval_decisions']
    raw['eval_decisions'] = decisions[(decisions >= start) & (decisions < end)]
    prior = decisions[decisions < start]
    if len(prior):
        anchor = prior[-1]
        following = raw['returns'].index[raw['returns'].index > anchor]
        if len(following) and following[0] >= start:
            raw['eval_decisions'] = raw['eval_decisions'].union(pd.DatetimeIndex([anchor]))
        else:
            raise ValueError('Periodenstart muss direkt auf eine Monatsentscheidung folgen')
    if len(raw['eval_decisions']) == 0:
        raise ValueError('Keine ausfuehrbare Entscheidung in Periode')
    persistent = dict(raw)
    persistent['signal_frames'] = {k: daily_persistent_signals(v.loc[:end]) for k,v in inputs['signal_frames'].items()}
    records, paths = {}, {}
    benchmarks = {}
    for cost in costs:
        net, info = run_variant(raw, 'matched_long', cost_multiplier=cost)
        benchmarks[str(cost)] = annualized_stats(net, inputs['cash_daily'].loc[net.index])
        paths[f'matched_long/{cost}'] = net
    for variant in variants:
        records[variant] = {}
        for label, data in [('raw',raw), ('persistent',persistent)]:
            records[variant][label] = {}
            for cost in costs:
                net, info = run_variant(data, variant, cost_multiplier=cost)
                benchmark = paths[f'matched_long/{cost}']
                if not net.index.equals(benchmark.index) or net.isna().any() or (net <= -1).any():
                    raise ValueError('Ungueltiger gepaarter Renditepfad')
                paths[f'{variant}/{label}/{cost}'] = net
                excess = monthly_log_returns(net) - monthly_log_returns(benchmark)
                records[variant][label][str(cost)] = {
                    'stats': annualized_stats(net, inputs['cash_daily'].loc[net.index]),
                    'turnover':info['total_turnover'],
                    'trade_cost_sum':float(info['per_day']['trade_cost'].sum()),
                    'borrow_cost_sum':float(info['per_day']['borrow_cost'].sum()),
                    'max_daily_gross':info['max_daily_gross'],
                    'relative_growth_pa':float(np.expm1(excess.mean()*12)),
                }
        delta = monthly_log_returns(paths[f'{variant}/persistent/1.0']) - monthly_log_returns(paths[f'{variant}/raw/1.0'])
        records[variant]['paired_delta'] = stationary_block_bootstrap(delta.to_numpy(), 6., n_boot, 0)
        print(f'  {variant}: fertig', flush=True)
    frame = pd.DataFrame(paths)
    return {'start':str(frame.index.min()), 'end':str(frame.index.max()),
            'benchmarks':benchmarks, 'variants':records}, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    dfs = load_trend_snapshot(args.snapshot)
    inputs = prepare_inputs(dfs)
    common = dfs[inputs['symbols'][0]].index
    for symbol in inputs['symbols'][1:]:
        common = common.intersection(dfs[symbol].index)
    common = common.sort_values()
    cutoff = common[int(len(common)*.6)-1]
    ends = month_end_dates(common)
    split = ends[ends <= cutoff].max()
    out = Path(args.output_root) / datetime.now(timezone.utc).strftime('persistence_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    config = {'classification':'exploratory_only', 'snapshot_content_sha256':snapshot_content_sha256(dfs),
              'base_registration':REGISTRATION_V2, 'min_confirm_months':2, 'split_quantile':.6,
              'split':str(split), 'cost_multipliers':[1.,2.,5.], 'n_boot':10000, 'seed':0,
              'block_months':6, 'python':platform.python_version(), 'pandas':pd.__version__,
              'numpy':np.__version__, 'period_start':'cash; signal history retained',
              'limitations':['Entire history previously used; no independent confirmation',
              'Intervals descriptive; no multiple-testing correction',
              'Gross target 1.0; drift allowed; no guaranteed drawdown cap',
              'Shared v2 execution and cost approximations retained',
              'Terminal partial month conservatively omitted using weekday calendar']}
    (out/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    source_root = Path(__file__).resolve().parents[1]
    source_copy = out/'source'
    for folder in ['factor_lab','market_control_system']:
        for source in (source_root/folder).rglob('*.py'):
            if '__pycache__' not in source.parts:
                target = source_copy/source.relative_to(source_root)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
    results = {}
    periods = [('full',inputs['eval_decisions'].min(), common.max()),
               ('early',inputs['eval_decisions'].min(),split),
               ('late',split + pd.Timedelta(days=1),common.max())]
    table = ['# Exploratory monthly persistence comparison', '', 'All periods previously used. No holdout confirmation.', '',
             '| Period | Variant | Raw CAGR | Filter CAGR | Raw DD | Filter DD | Turnover change |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name, start, end in periods:
        print(f'{name}: {start} -- {end}',flush=True)
        result, daily = compare_period(inputs,start,end)
        results[name] = result
        daily.to_csv(out/f'{name}_daily.csv')
        monthly = daily.apply(monthly_log_returns)
        for variant in VARIANT_NAMES:
            monthly[f'{variant}/delta'] = monthly[f'{variant}/persistent/1.0'] - monthly[f'{variant}/raw/1.0']
            for label in ['raw','persistent']:
                monthly[f'{variant}/{label}/excess'] = monthly[f'{variant}/{label}/1.0'] - monthly['matched_long/1.0']
            pair = result['variants'][variant]
            a,b = pair['raw']['1.0'],pair['persistent']['1.0']
            table.append(f"| {name} | {variant} | {a['stats']['cagr']:.2%} | {b['stats']['cagr']:.2%} | {a['stats']['max_drawdown']:.2%} | {b['stats']['max_drawdown']:.2%} | {b['turnover']/a['turnover']-1:.2%} |")
        monthly.to_csv(out/f'{name}_monthly_log.csv')
        (out/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    (out/'report.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    hashes = {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(f'Ergebnisse: {out}',flush=True)


if __name__ == '__main__':
    main()
