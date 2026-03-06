"""
Ditto for Personalized Federated Learning

Maintains:
- global model w (shared)
- personalized models v_k for each client k

Round t:
1) Global step: each selected client solves (approximately):
       min_w  F_k(w) + (mu/2)||w - w_global||^2
   starting from w_global, returns delta_k = w_k - w_global. 
   Server aggregates deltas to update global model.

2) Personalized step: each selected client does local SGD on:
       F_k(v_k) + (lam/2)||v_k - w_global||^2
   updating v_k (kept across rounds).
"""

import numpy as np
from sklearn.utils import shuffle
from utils import compute_gradient, cross_entropy_loss, compute_accuracy


# ============================================================
# PROXIMAL SGD HELPER
# ============================================================

def _sgd_prox_update(
    w_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    ref: np.ndarray,
    lr: float,
    prox_coeff: float,
    local_steps: int, 
    batch_size: int,
    clip: float = 0.0,
):
    """
    SGD on:  F(w) + (prox_coeff/2)||w - ref||^2
    gradient = ∇F(w) + prox_coeff*(w - ref)

    local_steps = number of mini-batch updates
    """

    w = w_init.copy()
    n = len(X)

    if n == 0:
        return w

    for step in range(local_steps):

        # sample minibatch
        idx = np.random.choice(n, batch_size, replace=n < batch_size)
        Xb = X[idx]
        yb = y[idx]

        grad = compute_gradient(Xb, yb, w) + prox_coeff * (w - ref)

        if clip and clip > 0:
            gnorm = np.linalg.norm(grad)
            if gnorm > clip:
                grad = grad * (clip / (gnorm + 1e-12))

        w -= lr * grad

    return w

# ============================================================
# DITTO TRAINING LOOP
# ============================================================

def ditto_train(
    client_datasets,
    w_init: np.ndarray,
    R: int,
    K: int,                     # local updates per round
    lr_global: float,
    lr_personal: float,
    mu: float,
    lam: float,
    batch_size: int = 64,
    client_fraction: float = 1.0,
    X_test=None,
    y_test=None,
    display_every: int = 1,
    clip: float = 0.0,
):
    """
    Ditto with K local mini-batch updates per round.

    Returns:
      global_losses
      global_accs
      w_global
      v_list
      personal_losses_rm
      personal_accs_rm
    """

    m = len(client_datasets)

    n_k = np.array([len(X) for X, _ in client_datasets], dtype=float)
    n_total = np.sum(n_k) if np.sum(n_k) > 0 else 1.0

    w_global = w_init.copy()

    v_list = [w_init.copy() for _ in range(m)]

    global_losses = []
    global_accs = []

    personal_losses_rm = np.full((R, m), np.nan)
    personal_accs_rm = np.full((R, m), np.nan)

    for r in range(R):

        # client sampling
        if client_fraction >= 1.0:
            S_r = np.arange(m)
        else:
            s = max(1, int(m * client_fraction))
            S_r = np.random.choice(m, size=s, replace=False)

        w_snapshot = w_global.copy()

        avg_delta = np.zeros_like(w_global)
        denom = 0.0

        # -----------------------------
        # GLOBAL STEP
        # -----------------------------
        for i in S_r:

            X_i, y_i = client_datasets[i]
            n = len(X_i)

            if n == 0:
                continue

            w_local = w_snapshot.copy()

            for step in range(K):

                idx = np.random.choice(n, size=min(batch_size, n), replace=False)
                Xb = X_i[idx]
                yb = y_i[idx]

                grad = compute_gradient(Xb, yb, w_local) + mu * (w_local - w_snapshot)

                if clip and clip > 0:
                    gnorm = np.linalg.norm(grad)
                    if gnorm > clip:
                        grad = grad * (clip / (gnorm + 1e-12))

                w_local -= lr_global * grad

            delta_i = w_local - w_snapshot

            weight = n_k[i]

            avg_delta += weight * delta_i
            denom += weight

        if denom > 0:
            avg_delta /= denom
            w_global = w_snapshot + avg_delta

        # -----------------------------
        # PERSONALIZED STEP
        # -----------------------------
        for i in S_r:

            X_i, y_i = client_datasets[i]
            n = len(X_i)

            if n == 0:
                continue

            v_local = v_list[i]

            for step in range(K):

                idx = np.random.choice(n, size=min(batch_size, n), replace=False)
                Xb = X_i[idx]
                yb = y_i[idx]

                grad = compute_gradient(Xb, yb, v_local) + lam * (v_local - w_snapshot)

                if clip and clip > 0:
                    gnorm = np.linalg.norm(grad)
                    if gnorm > clip:
                        grad = grad * (clip / (gnorm + 1e-12))

                v_local -= lr_personal * grad

            v_list[i] = v_local

        # -----------------------------
        # GLOBAL TRAIN LOSS
        # -----------------------------
        total_loss = 0.0

        for i in range(m):

            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            total_loss += (n_k[i] / n_total) * cross_entropy_loss(X_i, y_i, w_global)

        global_losses.append(total_loss)

        test_info = ""

        if X_test is not None and y_test is not None:

            acc = compute_accuracy(X_test, y_test, w_global)
            global_accs.append(acc)

            test_info = f", Test Acc={acc*100:.2f}%"

        # personalized metrics
        for i in range(m):

            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            personal_losses_rm[r, i] = cross_entropy_loss(X_i, y_i, v_list[i])

            if X_test is not None and y_test is not None:
                personal_accs_rm[r, i] = compute_accuracy(X_test, y_test, v_list[i])

        if display_every and display_every > 0 and (r % display_every == 0):
            print(f"[Ditto] Round {r+1:3d}: Global Loss={total_loss:.4f}{test_info}")

    return (
        np.array(global_losses),
        np.array(global_accs),
        w_global,
        v_list,
        personal_losses_rm,
        personal_accs_rm,
    )