"""
cross_sectional_universe.py
==============================

Gemeinsames Symbol-Universum und Basis-Konstanten fuer alle Cross-
Sectional-Skripte (Baseline, Haupt-Backtest, Multi-Seed-Vergleich) -- ein
Ort, damit alle drei garantiert denselben Zeitraum/dasselbe Universum
verwenden (sonst waeren Vergleiche zwischen ihnen nicht aussagekraeftig).

Bewusst sektoruebergreifend gestreut (nicht 12x Tech), siehe Design-Spec:
die sqrt(N)-Diversifikationsannahme hinter dem Cross-Sectional-Ansatz
setzt vergleichsweise unkorrelierte Symbole voraus.
"""
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "JNJ", "XOM", "PG", "HD", "DIS", "KO", "CAT"]
BACKTEST_LOOKBACK_DAYS = 365
PRETRAIN_FRACTION = 0.25
TIMESTEPS = 20
HORIZON = 5
