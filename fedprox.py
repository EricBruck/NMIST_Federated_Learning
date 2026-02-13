# FedProx.py
"""
FedProx (Proposed Framework)

Local objective on client k in round t:
    minimize_w  F_k(w) + (mu/2) ||w - w_global||^2

Implementation details:
- Uses your softmax regression gradient: compute_gradient(Xb, yb, w)
- Adds proximal gradient term: mu * (w_local - w_global_snapshot)
- Server aggregates updated client models by weighted average (by local data size)
"""

import numpy as np
from sklearn.utils import shuffle
from utils import compute_gradient, cross_entropy_loss, compute_accuracy


def client_update_fedprox(
    w_init: np.ndarray,
    X_i: np.ndarray,
    y_i: np.ndarray,
    w_global: np.ndarray,
    lr: float,
    mu: float,
    local_epochs: int,
    batch_size: int,
    clip: float = 0.0,
):
    """
    One client's local FedProx update starting from w_init (typically w_global snapshot).
    """
    w = w_init.copy()
    n = len(X_i)
    if n == 0:
        return w

    for _ in range(local_epochs):
        X_i, y_i = shuffle(X_i, y_i)
        for s in range(0, n, batch_size):
            Xb = X_i[s:s+batch_size]
            yb = y_i[s:s+batch_size]
            if len(Xb) == 0:
                continue

            grad = compute_gradient(Xb, yb, w) + mu * (w - w_global)

            if clip and clip > 0:
                gnorm = np.linalg.norm(grad)
                if gnorm > clip:
                    grad = grad * (clip / (gnorm + 1e-12))

            w -= lr * grad

    return w


def fedprox_train(
    client_datasets,
    w_init: np.ndarray,
    R: int,
    local_epochs: int,
    lr: float,
    mu: float,
    batch_size: int = 64,
    client_fraction: float = 1.0,
    X_test=None,
    y_test=None,
    display_every: int = 1,
    clip: float = 0.0,
):
    """
    Returns:
      losses: global (weighted) train loss per round evaluated on w_global
      accs: test accuracy per round (if test provided)
      w_global: final global model
    """
    m = len(client_datasets)

    n_k = np.array([len(X) for X, _ in client_datasets], dtype=float)
    n_total = np.sum(n_k) if np.sum(n_k) > 0 else 1.0

    w_global = w_init.copy()
    losses, accs = [], []

    for r in range(R):
        # sample clients
        if client_fraction >= 1.0:
            S_r = np.arange(m)
        else:
            s = max(1, int(m * client_fraction))
            S_r = np.random.choice(m, size=s, replace=False)

        w_snapshot = w_global.copy()

        # local updates
        local_ws = []
        local_weights = []

        for i in S_r:
            X_i, y_i = client_datasets[i]
            w_i = client_update_fedprox(
                w_init=w_snapshot,
                X_i=X_i,
                y_i=y_i,
                w_global=w_snapshot,
                lr=lr,
                mu=mu,
                local_epochs=local_epochs,
                batch_size=batch_size,
                clip=clip,
            )
            local_ws.append(w_i)
            local_weights.append(n_k[i])

        # server aggregate (weighted)
        if len(local_ws) > 0:
            denom = float(np.sum(local_weights)) if np.sum(local_weights) > 0 else 1.0
            w_global = np.zeros_like(w_global)
            for w_i, wk in zip(local_ws, local_weights):
                w_global += (wk / denom) * w_i

        # evaluate global train loss
        total_loss = 0.0
        for i in range(m):
            X_i, y_i = client_datasets[i]
            if len(X_i) == 0:
                continue
            total_loss += (n_k[i] / n_total) * cross_entropy_loss(X_i, y_i, w_global)
        losses.append(total_loss)

        test_info = ""
        if X_test is not None and y_test is not None:
            acc = compute_accuracy(X_test, y_test, w_global)
            accs.append(acc)
            test_info = f", Test Acc={acc*100:.2f}%"

        if display_every and display_every > 0 and (r % display_every == 0):
            print(f"[FedProx] Round {r+1:3d}: Global Loss={total_loss:.4f}{test_info}")

    return np.array(losses), np.array(accs), w_global
