"""
fedvi.py
--------
Implements the simplified FedVI algorithm:
- Local FedVI-style update
- Global block update rule
- Consensus regularization + variance scaling
"""

import numpy as np
from sklearn.utils import shuffle
from utils import compute_gradient, cross_entropy_loss, compute_accuracy


# ============================================================
# CLIENT UPDATE (FedVI)
# ============================================================

def fedVI_client_update(w_init, X, y, global_blocks, eta_r=0.0,
                        gamma_l=0.01, lambda_m=0.1, H=1, batch_size=32):
    """
    Performs FedVI local update on a single client.

    FedVI modification:
    - gradient scaled by (1 + eta_r)
    - regularization pulling model toward consensus block

    Parameters
    ----------
    global_blocks : list of weight matrices (1 per client)
    lambda_m : float
        Regularization strength toward block consensus.

    Returns
    -------
    w : np.ndarray
        Updated block for this client.
    """
    w = w_init.copy()
    n = len(X)

    block_avg = sum(global_blocks) / len(global_blocks)

    for _ in range(H):
        X, y = shuffle(X, y)

        for i in range(0, n, batch_size):
            X_b = X[i:i+batch_size]
            y_b = y[i:i+batch_size]
            if len(X_b) == 0:
                continue

            grad = compute_gradient(X_b, y_b, w)
            grad = (1 + eta_r) * grad                      # variance scaling
            reg  = lambda_m * (w - block_avg)             # FedVI consensus penalty

            w -= gamma_l * (grad + reg)

    return w


# ============================================================
# FEDVI TRAINING LOOP
# ============================================================

def fedVI(client_datasets, global_blocks, R, H, gamma_l, lambda_m,
          batch_size=32, client_fraction=1.0, eta_schedule=None,
          X_test=None, y_test=None, display_every=1):
    """
    FedVI global training loop.

    Parameters
    ----------
    global_blocks : list of local models (one per client)
    eta_schedule : list of eta values per round

    Returns
    -------
    losses : np.ndarray
    accs : np.ndarray
    global_blocks : final block set
    """
    num_clients = len(client_datasets)
    n_k = np.array([len(X) for X, _ in client_datasets])
    n_total = np.sum(n_k)

    losses, accs = [], []

    for r in range(R):

        # ---- Choose participating clients ----
        if client_fraction == 1.0:
            S_r = np.arange(num_clients)
        else:
            m = max(1, int(num_clients * client_fraction))
            S_r = np.random.choice(num_clients, size=m, replace=False)

        blocks_snapshot = [b.copy() for b in global_blocks]

        updated = {}

        if eta_schedule is not None:
            eta_r = eta_schedule[r]
        else:
            eta_r = 0.0


        # ---- Local FedVI updates ----
        for m in S_r:
            X_m, y_m = client_datasets[m]
            updated[m] = fedVI_client_update(
                w_init=blocks_snapshot[m],
                X=X_m, y=y_m,
                global_blocks=blocks_snapshot,
                eta_r=eta_r,
                gamma_l=gamma_l,
                lambda_m=lambda_m,
                H=H,
                batch_size=batch_size,
            )

        # ---- Update active blocks ----
        for m in S_r:
            global_blocks[m] = updated[m]

        # ---- Build global model by averaging blocks ----
        w_global = sum(global_blocks) / len(global_blocks)

        # ---- Global loss ----
        total_loss = 0
        for k in range(num_clients):
            X_k, y_k = client_datasets[k]
            total_loss += (n_k[k] / n_total) * cross_entropy_loss(X_k, y_k, w_global)
        losses.append(total_loss)

        # ---- test accuracy & loss ----
        test_info = ""
        if X_test is not None and y_test is not None:
            test_loss = cross_entropy_loss(X_test, y_test, w_global)
            test_acc  = compute_accuracy(X_test, y_test, w_global)
            accs.append(test_acc)
            test_info = f", Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%"

        # ---- print identical formatting ----
        if r % display_every == 0:
            print(f"[FedVI] Round {r:3d}: Global Loss={total_loss:.4f}{test_info}")


    return np.array(losses), np.array(accs), global_blocks
