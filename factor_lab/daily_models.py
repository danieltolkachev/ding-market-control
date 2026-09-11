"""Bounded CPU forecasts for the fixed daily comparison recipe.

Caller supplies volatility-normalized, clipped targets; this module does not
shift, normalize or clip y. Label i becomes observable at close i+label_delay
(i+2 by default).
Ridge minimizes sum((z @ w + b - y)**2) + 100*||w||² with an
unpenalized intercept b. For centered Zc and yc, the solve is
(Zc.T Zc + 100 I) w = Zc.T yc; b = mean(y) - mean(Z) @ w.
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn


class _MeanLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 16, num_layers=1, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).squeeze(-1)


def fit_predict_models(X, y, first_test, update_every=21, seed=7, label_delay: int = 2):
    """Return ridge/frozen/online forecasts [D-first_test,A].

    Input X is [D,A,20,5]. Five channel means/population standard deviations
    pool the timesteps of finite-label initial examples (including overlapping
    sequences); they remain fixed. Standardized features are clipped to +/-10.
    Nonfinite labels are excluded, including from scaling. Online updates use
    the latest252 days with at least one finite label, through d-label_delay.
    Initial fitting includes labels through first_test-label_delay, inclusive.
    Each update shuffles eligible asset examples and retains Adam state between
    updates. CPU execution sets PyTorch's global thread/determinism settings.
    Only batch-sized feature copies are materialized; input arrays are untouched.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 4 or X.shape[2:] != (20, 5) or y.shape != X.shape[:2]:
        raise ValueError('expected X[D,A,20,5] and y[D,A]')
    days, assets = y.shape
    if (isinstance(label_delay, (bool, np.bool_))
            or not isinstance(label_delay, (int, np.integer)) or label_delay < 2):
        raise ValueError('label_delay must be an integer >=2')
    if (not isinstance(first_test, (int, np.integer))
            or not label_delay <= first_test <= days or assets == 0):
        raise ValueError('first_test must be an integer in [label_delay,D], with assets')
    if not isinstance(update_every, (int, np.integer)) or update_every < 1:
        raise ValueError('update_every must be a positive integer')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    valid = np.isfinite(y)
    initial = np.argwhere(valid[:first_test - label_delay + 1])
    if not len(initial):
        raise ValueError('no finite initial training labels')

    def raw(pairs):
        batch = np.asarray(X[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError('nonfinite features in eligible or prediction examples')
        return batch

    # Two passes avoid cancellation in variance without a full training copy.
    total = np.zeros(5)
    for start in range(0, len(initial), 256):
        total += raw(initial[start:start + 256]).sum(axis=(0, 1))
    mean = total / (len(initial) * 20)
    squares = np.zeros(5)
    for start in range(0, len(initial), 256):
        squares += ((raw(initial[start:start + 256]) - mean) ** 2).sum(axis=(0, 1))
    scale = np.sqrt(squares / (len(initial) * 20))
    scale[scale == 0] = 1

    def standardized(pairs):
        batch = (raw(pairs) - mean) / scale
        return np.clip(batch, -10, 10)

    gram, rhs = np.zeros((100, 100)), np.zeros(100)
    z_sum, y_sum = np.zeros(100), 0.0
    for start in range(0, len(initial), 256):
        pairs = initial[start:start + 256]
        z = standardized(pairs).reshape(len(pairs), 100)
        gram += z.T @ z
        targets = np.asarray(y[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
        rhs += z.T @ targets
        z_sum += z.sum(axis=0)
        y_sum += targets.sum()
    # Center after clipping: clipped training coordinates need not have mean0.
    z_mean, y_mean = z_sum / len(initial), y_sum / len(initial)
    ridge = np.linalg.solve(
        gram - np.outer(z_sum, z_mean) + 100 * np.eye(100),
        rhs - z_sum * y_mean,
    )
    intercept = y_mean - z_mean @ ridge

    def train(model, optimizer, pairs, epochs):
        model.train()
        for _ in range(epochs):
            order = rng.permutation(len(pairs))
            for start in range(0, len(pairs), 256):
                selected = pairs[order[start:start + 256]]
                inputs = torch.from_numpy(standardized(selected).astype(np.float32))
                targets = torch.as_tensor(y[selected[:, 0], selected[:, 1]], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(inputs), targets)
                loss.backward()
                optimizer.step()
        model.eval()

    frozen = _MeanLSTM()
    print(f'daily_models: initial fit, {len(initial)} examples, 5 epochs', flush=True)
    train(frozen, torch.optim.Adam(frozen.parameters(), lr=.001), initial, 5)
    online = deepcopy(frozen)
    optimizer = torch.optim.Adam(online.parameters(), lr=.0001)
    result = {name: np.empty((days - first_test, assets), dtype=np.float64)
              for name in ('ridge', 'lstm_frozen', 'lstm_online')}
    eligible_days = np.flatnonzero(valid.any(axis=1))
    for offset, d in enumerate(range(first_test, days)):
        if offset and offset % update_every == 0:
            end = np.searchsorted(eligible_days, d - label_delay, side='right')
            window = eligible_days[max(0, end - 252):end]
            local_day, asset = np.nonzero(valid[window])
            pairs = np.column_stack((window[local_day], asset))
            if len(pairs):
                train(online, optimizer, pairs, 1)
        with torch.inference_mode():
            for start in range(0, assets, 256):
                a = np.arange(start, min(start + 256, assets))
                z = standardized(np.column_stack((np.full(len(a), d), a)))
                inputs = torch.from_numpy(z.astype(np.float32))
                result['ridge'][offset, a] = z.reshape(len(a), 100) @ ridge + intercept
                result['lstm_frozen'][offset, a] = frozen(inputs).numpy()
                result['lstm_online'][offset, a] = online(inputs).numpy()
        if (offset + 1) % 252 == 0 or d == days - 1:
            print(f'daily_models: predicted {offset + 1}/{days - first_test} decisions', flush=True)
    return result
