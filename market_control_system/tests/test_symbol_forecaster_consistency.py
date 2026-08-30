"""
test_symbol_forecaster_consistency.py
========================================

Prueft, dass SymbolForecaster (Cross-Sectional-Pfad, orchestration/
symbol_forecaster.py) bei identischem Modell/Config/Daten exakt dieselbe
mu/sigma/p_up/train_stats-Sequenz produziert wie ControlLoop (Single-
Symbol-Pfad, orchestration/control_loop.py) -- SymbolForecaster ist als
Teilmenge von ControlLoop.step() gebaut (Feature/Sequence/Prognose/
Feedback/Online-Training identisch, nur Position/Risk/Ausfuehrung fehlt).
Divergenz hier wuerde bedeuten, dass die Extraktion versehentlich das
Prognose- oder Trainingsverhalten veraendert hat.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orchestration"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "execution"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feedback"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import numpy as np
import torch

from feature_pipeline import FEATURE_NAMES, _generate_synthetic_market_data
from lstm_forecaster_torch import LSTMForecaster
from online_trainer import OnlineTrainerConfig
from control_loop import ControlLoop
from exposure_controller import ControllerConfig
from risk_overlay import RiskOverlayConfig
from paper_execution import ExecutionConfig
from symbol_forecaster import SymbolForecaster

SEED = 123
N_BARS = 1500
TIMESTEPS = 20
HORIZON = 5


def run_control_loop_trace(df) -> list:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=16, num_layers=1)
    loop = ControlLoop(
        timesteps=TIMESTEPS, model=model, horizon=HORIZON,
        controller_config=ControllerConfig(k=0.01, max_position=1.0),
        risk_config=RiskOverlayConfig(max_step_change=0.1, max_sigma=0.08, drawdown_limit=-0.03, cooldown_steps=15, min_rebalance_threshold=0.30),
        execution_config=ExecutionConfig(slippage_bps=0.5),
        online_trainer_config=OnlineTrainerConfig(retrain_every=100, batch_size=32, max_training_window=256),
        recalibrate_k_on_retrain=False,   # kein Aequivalent in SymbolForecaster -- fuer den Vergleich deaktiviert
    )
    trace = []
    for _, row in df.iterrows():
        result = loop.step(row.to_dict())
        if result is None:
            trace.append(None)
        else:
            trace.append((result.mu, result.sigma, result.p_up,
                          result.train_stats["n_samples"] if result.train_stats else None))
    return trace


def run_symbol_forecaster_trace(df) -> list:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = LSTMForecaster(n_features=len(FEATURE_NAMES), hidden_size=16, num_layers=1)
    forecaster = SymbolForecaster(
        timesteps=TIMESTEPS, model=model, horizon=HORIZON,
        online_trainer_config=OnlineTrainerConfig(retrain_every=100, batch_size=32, max_training_window=256),
    )
    trace = []
    for _, row in df.iterrows():
        result = forecaster.step(row.to_dict())
        if result is None:
            trace.append(None)
        else:
            trace.append((result.mu, result.sigma, result.p_up,
                          result.train_stats["n_samples"] if result.train_stats else None))
    return trace


def run_consistency_check() -> None:
    df = _generate_synthetic_market_data(n=N_BARS, seed=7)

    control_loop_trace = run_control_loop_trace(df)
    forecaster_trace = run_symbol_forecaster_trace(df)

    assert len(control_loop_trace) == len(forecaster_trace) == N_BARS

    n_compared = 0
    for i, (a, b) in enumerate(zip(control_loop_trace, forecaster_trace)):
        if a is None or b is None:
            assert a is None and b is None, f"Warm-up-Divergenz bei Bar {i}: {a} vs {b}"
            continue
        mu_a, sigma_a, p_up_a, n_samples_a = a
        mu_b, sigma_b, p_up_b, n_samples_b = b
        assert abs(mu_a - mu_b) < 1e-9, f"mu weicht bei Bar {i} ab: {mu_a} vs {mu_b}"
        assert abs(sigma_a - sigma_b) < 1e-9, f"sigma weicht bei Bar {i} ab: {sigma_a} vs {sigma_b}"
        assert abs(p_up_a - p_up_b) < 1e-9, f"p_up weicht bei Bar {i} ab: {p_up_a} vs {p_up_b}"
        assert n_samples_a == n_samples_b, f"Online-Training-Zeitpunkt weicht bei Bar {i} ab: {n_samples_a} vs {n_samples_b}"
        n_compared += 1

    assert n_compared > 0, "Kein einziger vergleichbarer Schritt -- Warm-up-Logik pruefen"
    print(f"SymbolForecaster und ControlLoop stimmen ueber {n_compared} Schritte exakt ueberein "
          f"(mu, sigma, p_up, Online-Training-Zeitpunkte).")


if __name__ == "__main__":
    run_consistency_check()
