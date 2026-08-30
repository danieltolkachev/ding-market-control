"""
backtest_stats.py
==================

Statistische Auswertung eines Regelkreis-Laufs ueber MEHRERE unabhaengige
Perioden, statt nur eine einzelne kumulierte Return-Zahl zu berichten.

Hintergrund: eine einzelne Woche (wie im ersten Replay) sagt nichts
darueber aus, ob ein System echten Edge hat oder nur Rauschen zeigt.
Um das zu beurteilen, braucht es viele unabhaengige Perioden und einen
Signifikanztest (ist der mittlere Periodenreturn signifikant von 0
verschieden, relativ zu seiner Streuung?) statt einer einzelnen Zahl.

Dieses Modul nimmt eine Zeitreihe von Einzelschritt-Returns (wie sie
ControlLoop.step()/Fill.realized_return liefert) und aggregiert sie in
nicht ueberlappende Perioden (Standard: Kalenderwochen), dann berechnet
es Kennzahlen ueber diese Periodenreturns.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PeriodStatistics:
    """Kennzahlen ueber eine Menge nicht ueberlappender Perioden-Returns."""
    n_periods: int
    period_returns: np.ndarray          # Return pro Periode (additiv aufsummiert innerhalb der Periode)
    mean_return: float
    std_return: float
    t_statistic: float                  # mean / (std / sqrt(n)) -- H0: mittlerer Periodenreturn = 0
    sharpe_like: float                  # mean/std, nicht annualisiert (siehe annualization_factor)
    win_rate: float                     # Anteil Perioden mit Return > 0
    max_drawdown: float                 # ueber die kumulierte Periodenreturn-Kurve
    cumulative_return: float


def compute_period_statistics(
    timestamps: pd.DatetimeIndex,
    step_returns: np.ndarray,
    period: str = "W",
) -> PeriodStatistics:
    """
    Args:
        timestamps: Zeitpunkt jedes Einzelschritts (z.B. Bar-Timestamps).
        step_returns: realized_return pro Schritt, gleiche Laenge wie timestamps.
        period: pandas-Resample-Frequenz fuer die Periodenbildung
            (Standard "W" = Kalenderwoche; "M" fuer Monat etc.)

    Returns:
        PeriodStatistics mit Kennzahlen ueber die Perioden-Returns.

    Hinweis zur Interpretation: t_statistic ist ein GROBER Signifikanztest
    (Annahme: Perioden-Returns sind unabhaengig und annaehernd normal-
    verteilt -- bei Finanzzeitreihen nur naeherungsweise erfuellt, z.B.
    wegen Volatilitaets-Clustering). Als Faustregel: |t| > 2 entspricht
    grob einem 95%-Konfidenzniveau, ist hier aber ein Anhaltspunkt, kein
    strenger statistischer Beweis.
    """
    if len(timestamps) != len(step_returns):
        raise ValueError("timestamps und step_returns muessen gleich lang sein")
    if len(timestamps) == 0:
        raise ValueError("Leere Zeitreihe -- keine Statistik berechenbar")

    series = pd.Series(step_returns, index=pd.DatetimeIndex(timestamps))
    period_returns = series.resample(period).sum()
    # Perioden ganz ohne Aktivitaet (z.B. Boersen-Feiertage) rauswerfen --
    # ein Return von exakt 0 waere sonst irrefuehrend als "neutrale Periode"
    # gezaehlt statt als "keine Daten".
    period_returns = period_returns[series.resample(period).count() > 0]

    return summarize_period_returns(period_returns)


def summarize_period_returns(period_returns: pd.Series) -> PeriodStatistics:
    """
    Wie compute_period_statistics(), nimmt aber bereits fertig aggregierte
    Perioden-Returns entgegen (z.B. um mehrere Symbole zu einem gleich-
    gewichteten Portfolio zu kombinieren: je Symbol Wochen-Returns bilden,
    ueber die Symbole mitteln, dann hier zusammenfassen).
    """
    n = len(period_returns)
    mean_r = float(period_returns.mean())
    std_r = float(period_returns.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(mean_r / (std_r / np.sqrt(n))) if std_r > 0 and n > 1 else 0.0
    sharpe = float(mean_r / std_r) if std_r > 0 else 0.0
    win_rate = float((period_returns > 0).mean())

    cum = period_returns.cumsum()
    running_max = cum.cummax()
    max_dd = float((cum - running_max).min())

    return PeriodStatistics(
        n_periods=n,
        period_returns=period_returns.to_numpy(),
        mean_return=mean_r,
        std_return=std_r,
        t_statistic=t_stat,
        sharpe_like=sharpe,
        win_rate=win_rate,
        max_drawdown=max_dd,
        cumulative_return=float(period_returns.sum()),
    )


@dataclass
class SeedDistribution:
    """Verteilung einer einzelnen Kennzahl (z.B. cumulative_return) ueber
    mehrere unabhaengige Seed-Laeufe derselben Config."""
    metric_name: str
    n_seeds: int
    values: np.ndarray
    mean: float
    std: float
    min: float
    max: float


def summarize_seed_distribution(metric_name: str, values: list[float] | np.ndarray) -> SeedDistribution:
    """Fasst eine Kennzahl (z.B. portfolio_summary['cumulative_return'])
    ueber N Seed-Laeufe DERSELBEN Config zusammen. Beantwortet "wie sehr
    streut ein einzelner Lauf dieser Config allein durch den Seed" --
    genau die Frage, die ein einzelner Full-Year-Run nicht beantworten
    kann (siehe Docstring von paired_comparison() fuer den Vergleichsfall)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("Keine Seed-Werte -- Verteilung nicht berechenbar")
    return SeedDistribution(
        metric_name=metric_name,
        n_seeds=n,
        values=arr,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if n > 1 else 0.0,
        min=float(arr.min()),
        max=float(arr.max()),
    )


@dataclass
class PairedComparison:
    """Gepaarter Vergleich einer Kennzahl zwischen zwei Configs ueber
    DIESELBEN Seeds (Seed i lief fuer Config A und Config B mit demselben
    Zufalls-Startpunkt)."""
    metric_name: str
    n_pairs: int
    mean_diff: float   # mean(b - a)
    std_diff: float
    t_statistic: float  # H0: mean_diff = 0
    b_wins_rate: float  # Anteil Seeds, bei denen b > a


def paired_comparison(metric_a: list[float] | np.ndarray, metric_b: list[float] | np.ndarray, metric_name: str = "") -> PairedComparison:
    """
    Gepaarter t-Test ueber Seed-Differenzen (Config B minus Config A),
    fuer denselben Seed i in beiden Configs.

    Warum gepaart statt zwei unabhaengiger Verteilungen (Welch-t-Test):
    wenn Seed i fuer A und B denselben Zufalls-Startpunkt (Gewichtsinit,
    Batch-Reihenfolge) verwendet, teilen sich A_i und B_i einen Teil der
    Seed-bedingten Streuung -- die Differenz B_i - A_i hebt diesen
    gemeinsamen Anteil auf und hat dadurch weniger Varianz als A und B
    einzeln. Das macht den Test staerker (kleinere n reichen fuer dieselbe
    statistische Aussagekraft) als ein ungepaarter Vergleich.
    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("metric_a und metric_b muessen gleich viele (gepaarte) Seeds haben")
    n = len(a)
    if n == 0:
        raise ValueError("Keine Seed-Paare -- Vergleich nicht berechenbar")

    diffs = b - a
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(mean_diff / (std_diff / np.sqrt(n))) if std_diff > 0 and n > 1 else 0.0
    b_wins = float((diffs > 0).mean())

    return PairedComparison(
        metric_name=metric_name,
        n_pairs=n,
        mean_diff=mean_diff,
        std_diff=std_diff,
        t_statistic=t_stat,
        b_wins_rate=b_wins,
    )


def format_seed_distribution_report(dist: SeedDistribution) -> str:
    return (
        f"{dist.metric_name}: mean={dist.mean:+.4%}  std={dist.std:.4%}  "
        f"min={dist.min:+.4%}  max={dist.max:+.4%}  (n={dist.n_seeds} Seeds)"
    )


def format_paired_comparison_report(cmp: PairedComparison) -> str:
    return (
        f"{cmp.metric_name}: mean_diff(B-A)={cmp.mean_diff:+.4%}  std_diff={cmp.std_diff:.4%}  "
        f"t={cmp.t_statistic:+.2f}"
        + ("  [|t|>2, moeglich signifikant]" if abs(cmp.t_statistic) > 2 else "  [nicht signifikant von 0 unterscheidbar]")
        + f"  B-schlaegt-A-Rate={cmp.b_wins_rate:.1%}  (n={cmp.n_pairs} Paare)"
    )


def format_statistics_report(stats: PeriodStatistics, period_label: str = "Woche", capital: float = 10_000.0) -> str:
    """Formatiert PeriodStatistics als lesbaren Text-Report inkl. Dollar-Umrechnung."""
    lines = [
        f"Perioden ({period_label}): {stats.n_periods}",
        f"Mittlerer Periodenreturn: {stats.mean_return:+.4%}  ({stats.mean_return * capital:+,.2f} $)",
        f"Streuung (std) pro Periode: {stats.std_return:.4%}",
        f"t-Statistik (H0: Mittelwert=0): {stats.t_statistic:+.2f}"
        + ("  [|t|>2, moeglich signifikant]" if abs(stats.t_statistic) > 2 else "  [nicht signifikant von 0 unterscheidbar]"),
        f"Sharpe-artig (nicht annualisiert): {stats.sharpe_like:+.3f}",
        f"Win-Rate: {stats.win_rate:.1%}",
        f"Kumulierter Return: {stats.cumulative_return:+.4%}  ({stats.cumulative_return * capital:+,.2f} $)",
        f"Max Drawdown (Perioden-Kurve): {stats.max_drawdown:+.4%}  ({stats.max_drawdown * capital:+,.2f} $)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_steps = 100_000  # ~70 Tage bei 1-Minuten-Takt, genug fuer mehrere Wochen
    timestamps = pd.date_range("2026-01-01", periods=n_steps, freq="1min")
    # Reines Rauschen, leicht negativer Drift (simuliert Slippage-Kosten)
    step_returns = rng.normal(-0.00001, 0.001, size=n_steps)

    stats = compute_period_statistics(timestamps, step_returns, period="W")
    print("=== backtest_stats Sanity-Check (reines Rauschen, leicht negativer Drift) ===")
    print(format_statistics_report(stats, period_label="Woche", capital=10_000.0))

    print("\n=== Sanity-Check: summarize_seed_distribution / paired_comparison ===")
    seed_a = [-0.05, -0.03, -0.08, -0.02, -0.06]   # Config A ueber 5 Seeds
    seed_b = [0.01, 0.02, -0.01, 0.03, 0.00]        # Config B, DIESELBEN 5 Seeds -- durchgaengig besser
    dist_a = summarize_seed_distribution("cumulative_return (A)", seed_a)
    dist_b = summarize_seed_distribution("cumulative_return (B)", seed_b)
    print(format_seed_distribution_report(dist_a))
    print(format_seed_distribution_report(dist_b))
    cmp = paired_comparison(seed_a, seed_b, metric_name="cumulative_return")
    print(format_paired_comparison_report(cmp))
    assert cmp.mean_diff > 0 and cmp.b_wins_rate == 1.0, "B ist konstruktionsbedingt in jedem Seed besser -- Test fehlerhaft, wenn das nicht rauskommt"
