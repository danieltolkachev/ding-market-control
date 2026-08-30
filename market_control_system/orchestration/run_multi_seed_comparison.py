"""
run_multi_seed_comparison.py
=============================

Multi-Seed-Test-Infrastruktur: vergleicht zwei (oder mehr) Configs ueber
DIESELBEN mehreren Seeds hinweg, statt sich auf einen einzelnen
Full-Year-Multi-Symbol-Lauf pro Config zu verlassen.

Hintergrund (siehe Projekt-Notizen zum 2026-08-25-Overnight-Backtest):
einzelne volle 1-Jahres-Laeufe streuen allein durch den Zufalls-Seed sehr
stark -- derselbe Code, nur ein anderer Seed, hat Portfolio-Cumulative-
Return zwischen ca. -4% und -42% erzeugt. Ein einzelner Vergleich
"Run A vs. Run B" (wie run_news_comparison.py / run_market_relative_
comparison.py es fuer je EINEN Seed tun) kann Seed-Rauschen daher nicht
von einem echten Effekt der Config-Aenderung unterscheiden. Diese
Infrastruktur macht das: N Seeds pro Config, gepaarter Vergleich pro
Seed, Verteilungen statt Einzelpunkte.

Erster Lauf (2026-08-26, scaled vs. unscaled, 5 Seeds) hat die damals
offene Scaling-Konfundierung aufgeloest: |t|<1 auf allen 5 Kennzahlen,
kein signifikanter Unterschied -- die scheinbar grosse Run4/Run5-Differenz
war Seed-Rauschen (siehe Projekt-Notizen zum 2026-08-25-Overnight-Backtest
fuer Details). Gemessene Laufzeit dieses ersten Laufs: nur ~73 Minuten
fuer 5 Seeds x 2 Varianten x 3 Symbole (~7.3 Min/voller 3-Symbol-Lauf) --
deutlich schneller als die anfangs vorsichtige 8-15h-Schaetzung.

Aktuelle Belegung (2026-08-26, Folgefrage): mehrere Runs hatten gezeigt,
dass Slippage/Turnover den Loewenanteil der Netto-Verluste erklaert
(Gross-P&L oft nahe Buy-&-Hold, Netto zweistellig negativ) -- die Frage
ist, ob min_rebalance_threshold=0.15 (aktueller Default) noch zu
permissiv ist. Drei Varianten mit steigendem Deadband, alles andere
(inkl. scale_features=True, da statistisch neutral) konstant. Ziel ist
nicht Win-Rate, sondern Expected Value / kumulierter Return -- ein
hoeheres Deadband kann Turnover/Kosten senken, aber auch echte
Signal-Bars verpassen; das ist der eigentliche Trade-off, den der
Vergleich hier quantifiziert.

Laufzeit-Hinweis: bei ~7.3 Min/vollem 3-Symbol-Lauf (empirisch aus dem
ersten Multi-Seed-Lauf) sind 5 Seeds x 3 Varianten x 3 Symbole grob
1.5-2 Stunden. Gedacht fuer run_in_background / einen unbeaufsichtigten
Lauf. Checkpointing erfolgt auf drei Ebenen: pro Symbol (in
run_symbol_backtest, bestehend), pro (Variante, Seed) direkt danach, und
die Seed-Verteilung pro Variante wird nach JEDEM abgeschlossenen Seed neu
geschrieben -- ein Abbruch mitten im Lauf verliert also hoechstens den
gerade laufenden Einzel-Run.

Ausfuehren: py -3.12 orchestration/run_multi_seed_comparison.py
"""

from __future__ import annotations

import sys
import os
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

from risk_overlay import RiskOverlayConfig
from online_trainer import OnlineTrainerConfig
from backtest_stats import (
    summarize_seed_distribution, format_seed_distribution_report,
    paired_comparison, format_paired_comparison_report,
)

from run_backtest import (
    SYMBOLS, run_symbol_backtest, build_portfolio_summary,
    default_risk_config, default_online_trainer_config,
)


SEEDS = [0, 1, 2, 3, 4]

# Kennzahlen, fuer die Seed-Verteilungen und Paar-Vergleiche berichtet
# werden -- alles, was portfolio_summary bereits pro Lauf liefert.
COMPARISON_METRICS = ["cumulative_return", "t_statistic", "sharpe_like", "win_rate", "max_drawdown"]


@dataclass
class Variant:
    name: str
    risk_config: RiskOverlayConfig = field(default_factory=default_risk_config)
    online_trainer_config: OnlineTrainerConfig = field(default_factory=default_online_trainer_config)
    scale_features: bool = True


def _risk_config_with_deadband(threshold: float) -> RiskOverlayConfig:
    cfg = default_risk_config()
    cfg.min_rebalance_threshold = threshold
    return cfg


VARIANTS = [
    Variant(name="deadband_015_baseline", risk_config=_risk_config_with_deadband(0.15)),
    Variant(name="deadband_020", risk_config=_risk_config_with_deadband(0.20)),
    Variant(name="deadband_030", risk_config=_risk_config_with_deadband(0.30)),
]


def run_variant_seed(variant: Variant, seed: int, base_dir: str) -> dict | None:
    """Ein voller Multi-Symbol-Backtest fuer (Variante, Seed). Checkpointed
    wie run_backtest.py: pro Symbol sofort, danach das Portfolio-Summary."""
    run_dir = os.path.join(base_dir, variant.name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    summaries = {}
    for symbol in SYMBOLS:
        try:
            summaries[symbol] = run_symbol_backtest(
                symbol, run_dir,
                seed=seed,
                risk_config=variant.risk_config,
                online_trainer_config=variant.online_trainer_config,
                scale_features=variant.scale_features,
            )
        except Exception:
            print(f"\n!!! FEHLER bei {variant.name}/seed_{seed}/{symbol}, wird uebersprungen: !!!")
            traceback.print_exc()
            continue

    run_id = f"{variant.name}_seed{seed}"
    return build_portfolio_summary(summaries, run_id, run_dir)


def write_seed_distribution(variant_name: str, portfolio_summaries: list[dict], base_dir: str) -> dict:
    """Fasst alle bisher abgeschlossenen Seed-Laeufe einer Variante zu
    Verteilungen pro Kennzahl zusammen und schreibt sie sofort (nicht erst
    am Ende aller Seeds) -- Zwischenstand bleibt bei Abbruch nutzbar."""
    distributions = {}
    print(f"\n{'='*70}\n=== Seed-Verteilung: {variant_name} (n={len(portfolio_summaries)} Seeds) ===\n{'='*70}")
    for metric in COMPARISON_METRICS:
        values = [s[metric] for s in portfolio_summaries]
        dist = summarize_seed_distribution(metric, values)
        print(format_seed_distribution_report(dist))
        distributions[metric] = {
            "n_seeds": dist.n_seeds, "mean": dist.mean, "std": dist.std,
            "min": dist.min, "max": dist.max, "values": dist.values.tolist(),
        }

    out = {"variant": variant_name, "n_seeds": len(portfolio_summaries), "distributions": distributions}
    path = os.path.join(base_dir, variant_name, "seed_distribution_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def write_comparison(variant_a: str, variant_b: str, dist_a: dict, dist_b: dict, base_dir: str) -> None:
    """Gepaarter Vergleich (B relativ zu A) ueber die Seeds, die in BEIDEN
    Varianten bereits abgeschlossen sind."""
    n = min(dist_a["n_seeds"], dist_b["n_seeds"])
    if n == 0:
        return
    print(f"\n{'='*70}\n=== Vergleich: {variant_b} vs. {variant_a} (gepaart, n={n} gemeinsame Seeds) ===\n{'='*70}")

    comparisons = {}
    for metric in COMPARISON_METRICS:
        values_a = dist_a["distributions"][metric]["values"][:n]
        values_b = dist_b["distributions"][metric]["values"][:n]
        cmp = paired_comparison(values_a, values_b, metric_name=metric)
        print(format_paired_comparison_report(cmp))
        comparisons[metric] = {
            "n_pairs": cmp.n_pairs, "mean_diff": cmp.mean_diff, "std_diff": cmp.std_diff,
            "t_statistic": cmp.t_statistic, "b_wins_rate": cmp.b_wins_rate,
        }

    out = {"variant_a": variant_a, "variant_b": variant_b, "n_pairs": n, "comparisons": comparisons}
    path = os.path.join(base_dir, f"comparison_{variant_a}_vs_{variant_b}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nVergleich gespeichert: {path}")


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"multiseed_{run_id}")
    os.makedirs(base_dir, exist_ok=True)
    print(f"Multi-Seed-Run-ID: {run_id}")
    print(f"Varianten: {[v.name for v in VARIANTS]}, Seeds: {SEEDS}, Symbole: {SYMBOLS}")
    print(f"Ergebnisse werden nach {os.path.abspath(base_dir)} geschrieben.")

    all_distributions = {}
    for variant in VARIANTS:
        portfolio_summaries = []
        for seed in SEEDS:
            print(f"\n{'#'*70}\n# Variante '{variant.name}', Seed {seed}\n{'#'*70}")
            summary = run_variant_seed(variant, seed, base_dir)
            if summary is not None:
                portfolio_summaries.append(summary)
            if portfolio_summaries:
                all_distributions[variant.name] = write_seed_distribution(variant.name, portfolio_summaries, base_dir)

    variant_names = list(all_distributions.keys())
    for i in range(len(variant_names)):
        for j in range(i + 1, len(variant_names)):
            write_comparison(variant_names[i], variant_names[j],
                              all_distributions[variant_names[i]], all_distributions[variant_names[j]], base_dir)

    print(f"\nAlle Ergebnisse in: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    main()
