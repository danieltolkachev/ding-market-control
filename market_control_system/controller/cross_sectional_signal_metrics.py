"""
cross_sectional_signal_metrics.py
====================================

Reine Statistikfunktionen fuer die Cross-Sectional-Signal-Diagnostik
(siehe Design-Spec docs/superpowers/specs/2026-08-31-cross-sectional-
signal-diagnostics-design.md). Bewusst OHNE jede Abhaengigkeit von
ControlLoop/PaperExecutionEngine/CrossSectionalPortfolio -- das hier
misst Signalqualitaet, nicht Handelsverhalten.

Spearman-Rangkorrelation wird ueber pandas' .rank() + Pearson-Korrelation
auf den Raengen selbst implementiert (Pearson-Korrelation der Raenge IST
per Definition die Spearman-Korrelation) -- keine scipy-Abhaengigkeit
noetig, die dieses Projekt bisher nicht hat (siehe requirements.txt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rank_ic(scores: dict[str, float], forward_returns: dict[str, float]) -> float:
    """Spearman-Rangkorrelation zwischen Score und tatsaechlichem
    Vorwaerts-Return ueber die gemeinsamen Symbole zu EINEM Zeitpunkt.
    NaN, wenn weniger als 3 gemeinsame Symbole vorliegen (Korrelation
    bei so wenigen Punkten nicht aussagekraeftig)."""
    common = sorted(set(scores) & set(forward_returns))
    if len(common) < 3:
        return float("nan")
    score_ranks = pd.Series([scores[s] for s in common]).rank()
    return_ranks = pd.Series([forward_returns[s] for s in common]).rank()
    return float(score_ranks.corr(return_ranks))


def compute_gross_spread(
    scores: dict[str, float], forward_returns: dict[str, float], n_long: int = 3, n_short: int = 3,
) -> float:
    """Long die hoechsten n_long Scores, Short die niedrigsten n_short,
    gleichgewichtet, OHNE Kosten, OHNE Deadband -- reiner Signal-Spread
    fuer EINEN Zeitpunkt. NaN, wenn nicht genug gemeinsame Symbole."""
    common = sorted(set(scores) & set(forward_returns))
    if len(common) < n_long + n_short:
        return float("nan")
    ranked = sorted(common, key=lambda s: scores[s], reverse=True)
    longs = ranked[:n_long]
    shorts = ranked[-n_short:]
    long_return = sum(forward_returns[s] for s in longs) / n_long
    short_return = sum(forward_returns[s] for s in shorts) / n_short
    return long_return - short_return


def compute_breakeven_cost(gross_spread_series: list[float], turnover_series: list[float]) -> float:
    """Welcher Pro-Einheit-Turnover-Kostensatz wuerde den mittleren
    Brutto-Spread auf 0 druecken. NaN, wenn mittlerer Turnover 0 ist."""
    mean_spread = float(np.mean(gross_spread_series))
    mean_turnover = float(np.mean(turnover_series))
    if mean_turnover == 0:
        return float("nan")
    return mean_spread / mean_turnover


def compound_return(returns: list[float]) -> float:
    """Echtes Compounding: prod(1+r) - 1, NICHT sum(r) -- siehe Design-
    Spec Review-Punkt 6 (additive Summe war der Fehler im bestehenden
    backtest_stats.py)."""
    result = 1.0
    for r in returns:
        result *= (1.0 + r)
    return result - 1.0


def equity_curve(returns: list[float], initial: float = 1.0) -> np.ndarray:
    """Echte Equity-Kurve: equity_t = equity_{t-1} * (1 + r_t)."""
    equity = [initial]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    return np.array(equity[1:])


def max_drawdown_from_returns(returns: list[float]) -> float:
    """Max Drawdown auf der ECHTEN (compoundenden) Equity-Kurve, nicht
    auf einer additiven cumsum-Kurve."""
    equity = equity_curve(returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def rolling_percentile_score(history: list[float], window: int = 500) -> float:
    """Perzentil-Rang des LETZTEN Werts in history relativ zu den
    VORHERIGEN `window` Werten (kausal -- der aktuelle Wert selbst geht
    nicht in sein eigenes Perzentil ein, und es werden nie zukuenftige
    Werte verwendet). Neutraler Default 0.5 bei zu wenig Historie."""
    if len(history) < 2:
        return 0.5
    lookback = history[max(0, len(history) - 1 - window):-1]
    if not lookback:
        return 0.5
    current = history[-1]
    return sum(1 for v in lookback if v <= current) / len(lookback)


def one_minute_transition_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True an Position p genau dann, wenn die NAECHSTE Zeile exakt 60
    Sekunden spaeter liegt -- also der horizon=1-Forward-Return von p ein
    echter 1-Minuten-Uebergang innerhalb derselben Session ist und keine
    Luecke (fehlende Bars, Overnight, Index-Schnittmengen-Loch)
    ueberspannt. Die letzte Position hat keinen Nachfolger -> False."""
    mask = np.zeros(len(index), dtype=bool)
    if len(index) >= 2:
        deltas = index[1:] - index[:-1]
        mask[:-1] = deltas == pd.Timedelta(minutes=1)
    return mask


def day_block_bootstrap(
    values: list[float], timestamps: pd.DatetimeIndex, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Tages-Block-Bootstrap fuer den Mittelwert einer per-Bar-Zeitreihe
    (z.B. Rank-IC pro Bar): es werden GANZE HANDELSTAGE mit Zuruecklegen
    resampelt, nicht einzelne Bars -- Intraday-Autokorrelation bleibt so
    innerhalb der Bloecke erhalten, statt die Stichprobe kuenstlich als
    IID zu behandeln (siehe Review-Praezisierung 3 zum urspruenglichen
    5x-Random-Schwellenwert-Verdikt).

    Liefert Punktschaetzer, 95%-Perzentil-CI, den Anteil der Bootstrap-
    Mittel <= 0 (einseitige Signifikanz fuer "Mittel > 0") und die rohe
    Bootstrap-Verteilung."""
    series = pd.Series(list(values), index=pd.DatetimeIndex(timestamps))
    day_blocks = [group.to_numpy() for _, group in series.groupby(series.index.date)]
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(day_blocks), size=len(day_blocks))
        boot_means[b] = float(np.concatenate([day_blocks[i] for i in chosen]).mean())
    return {
        "mean": float(series.mean()),
        "ci_low_95": float(np.percentile(boot_means, 2.5)),
        "ci_high_95": float(np.percentile(boot_means, 97.5)),
        "p_leq_zero": float((boot_means <= 0).mean()),
        "n_days": len(day_blocks),
        "n_boot": n_boot,
        "bootstrap_means": boot_means,
    }


def bootstrap_signal_verdict(bootstrap_by_variant: dict[str, dict], model_variant_names: list[str]) -> str:
    """Plain-Language-Verdikt auf Basis der Tages-Block-Bootstrap-CIs
    statt des frueheren 5x-|Random-IC|-Schwellenwerts (der war willkuerlich
    und stuetzte sich auf einen einzelnen Zufallspfad): ein Signal gilt
    erst dann als gezeigt, wenn das 95%-CI des mittleren Rank-IC einer
    MODELL-Variante vollstaendig ueber 0 liegt. Baselines (random/momentum/
    reversal) zaehlen absichtlich nicht -- sie sind Vergleichsmassstab,
    kein Kandidat."""
    significant = [
        name for name in model_variant_names
        if bootstrap_by_variant[name]["ci_low_95"] > 0
    ]
    if not significant:
        return (
            "VERDICT: kein Rank-IC-Signal gefunden -- das 95%-Tages-Block-"
            "Bootstrap-CI des mittleren Rank-IC schliesst fuer JEDE modellbasierte "
            "Score-Variante die 0 ein. Die compounded_gross_return-Werte sind "
            "KEIN Beleg fuer Profitabilitaet (siehe Rank-IC)."
        )
    return (
        f"VERDICT: {', '.join(significant)} zeigt ein 95%-Bootstrap-CI des "
        "mittleren Rank-IC vollstaendig ueber 0 -- naehere Pruefung noetig, "
        "bevor daraus ein echtes Signal abgeleitet wird (Multiple-Testing "
        "ueber die Varianten beachten)."
    )


def random_ranking_scores(symbols: list[str], seed: int) -> dict[str, float]:
    """Zufaellige Scores fuer die Random-Ranking-Baseline -- deterministisch
    pro Seed, damit Wiederholungen ueber mehrere Seeds gemittelt werden
    koennen."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=len(symbols))
    return {symbol: float(v) for symbol, v in zip(symbols, values)}


def momentum_scores(prices: dict[str, list[float]], lookback_bars: int) -> dict[str, float]:
    """Score = Return der letzten lookback_bars Bars. Symbole mit zu
    kurzer Preishistorie werden ausgelassen (kein kuenstlicher Default)."""
    scores = {}
    for symbol, series in prices.items():
        if len(series) <= lookback_bars:
            continue
        past = series[-1 - lookback_bars]
        current = series[-1]
        scores[symbol] = (current - past) / past
    return scores


def reversal_scores(prices: dict[str, list[float]], lookback_bars: int) -> dict[str, float]:
    """Exaktes Gegenteil von momentum_scores() -- dieselben Symbole,
    negiertes Vorzeichen."""
    return {symbol: -score for symbol, score in momentum_scores(prices, lookback_bars).items()}
