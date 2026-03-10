# scaffold.py
"""
SCAFFOLD (Stochastic Controlled Averaging) for Federated Learning

Idea:
- Maintain control variates:
    c   : global control (same shape as model)
    c_i : per-client control for each client i

Client i local update (for K local steps):
    w <- w - lr * ( grad F_i(w) - c_i + c )

Client control update (standard SCAFFOLD update):
    c_i <- c_i - c + (1/(lr*K_steps)) * (w_global - w_local_final)

Server update:
    w_global <- w_global + weighted_avg( w_local_final - w_global )
    c <- c + avg_over_selected( c_i_new - c_i_old )

Notes:
- This is the classic control-variate mechanism to reduce client drift under non-IID data.
"""

import numpy as np
from sklearn.utils import shuffle
from utils import compute_gradient, cross_entropy_loss, compute_accuracy


def scaffold_train(
    client_datasets,
    w_init: np.ndarray,
    R: int,
    K: int,                 # number of local mini-batch updates
    lr: float,
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
    n_total = float(np.sum(n_k)) if np.sum(n_k) > 0 else 1.0

    # Global model and controls
    w_global = w_init.copy()
    c = np.zeros_like(w_global)                       # global control
    c_list = [np.zeros_like(w_global) for _ in range(m)]  # client controls

    losses, accs = [], []

    for r in range(R):

        # ---- Client sampling ----
        if client_fraction >= 1.0:
            S_r = np.arange(m)
        else:
            s = max(1, int(m * client_fraction))
            S_r = np.random.choice(m, size=s, replace=False)

        w_snapshot = w_global.copy()

        local_ws = []
        local_weights = []

        delta_c_sum = np.zeros_like(c)
        denom_selected = 0.0

        # ============================================================
        # CLIENT LOOP
        # ============================================================

        for i in S_r:

            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            w_local = w_snapshot.copy()
            c_i_old = c_list[i].copy()

            step_count = 0

            # ---- exactly K mini-batch updates ----
            while step_count < K:

                perm = np.random.permutation(len(X_i))
                X_i = X_i[perm]
                y_i = y_i[perm]

                for sidx in range(0, len(X_i), batch_size):

                    if step_count >= K:
                        break

                    Xb = X_i[sidx:sidx + batch_size]
                    yb = y_i[sidx:sidx + batch_size]

                    if len(Xb) == 0:
                        continue

                    grad = compute_gradient(Xb, yb, w_local)

                    # SCAFFOLD corrected gradient
                    grad_corr = grad - c_list[i] + c

                    if clip and clip > 0:
                        gnorm = np.linalg.norm(grad_corr)
                        if gnorm > clip:
                            grad_corr = grad_corr * (clip / (gnorm + 1e-12))

                    w_local -= lr * grad_corr

                    step_count += 1

            if step_count == 0:
                continue

            # ---- client control update ----
            c_i_new = c_i_old - c + (1.0 / (lr * step_count)) * (w_snapshot - w_local)
            c_list[i] = c_i_new

            delta_c_sum += (c_i_new - c_i_old)
            denom_selected += 1.0

            local_ws.append(w_local)
            local_weights.append(n_k[i])

        # ============================================================
        # SERVER AGGREGATION
        # ============================================================

        if len(local_ws) > 0:

            denom = float(np.sum(local_weights)) if np.sum(local_weights) > 0 else 1.0
            w_new = np.zeros_like(w_global)

            for w_i, wk in zip(local_ws, local_weights):
                w_new += (wk / denom) * w_i

            w_global = w_new

        # ---- update global control ----
        if denom_selected > 0:
            c = c + (1.0 / denom_selected) * delta_c_sum

        # ============================================================
        # EVALUATION
        # ============================================================

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

        if display_every and (r % display_every == 0):
            print(f"[SCAFFOLD] Round {r+1:3d}: Global Loss={total_loss:.4f}{test_info}")

    return np.array(losses), np.array(accs), w_global