"""
fedavg.py
---------
Implements the FedAvg algorithm:
- Local client SGD update
- Federated averaging server loop
"""

import numpy as np
from sklearn.utils import shuffle
from utils import compute_gradient, cross_entropy_loss, compute_accuracy


# ============================================================
# CLIENT UPDATE (FedAvg)
# ============================================================

def client_update(w_init, X, y, gamma=0.01, H=1, batch_size=32):
    """
    Performs local SGD at a single client.

    Parameters
    ----------
    w_init : np.ndarray
        Initial global model.

    gamma : float
        Learning rate.

    H : int
        Number of local epochs.

    Returns
    -------
    w : np.ndarray
        Updated local model.
    """
    w = w_init.copy()
    n = len(X)

    for _ in range(H):
        X, y = shuffle(X, y)

        for i in range(0, n, batch_size):
            X_b = X[i:i+batch_size]
            y_b = y[i:i+batch_size]
            if len(X_b) == 0:
                continue

            grad = compute_gradient(X_b, y_b, w)
            w -= gamma * grad

    return w


# ============================================================
# FEDERATED AVERAGING
# ============================================================

def federated_averaging(client_datasets, w_init, R, H, gamma, batch_size,
                        X_test=None, y_test=None, display_every=1):
    """
    Full FedAvg training loop.

    Parameters
    ----------
    client_datasets : list of (X, y)
    w_init : np.ndarray (global model)
    R : int (number of communication rounds)
    H : int (local epochs)
    gamma : float (learning rate)

    Returns
    -------
    losses : list of floats
    w_global : final model
    """
    num_clients = len(client_datasets)
    n_k = np.array([len(X) for X, _ in client_datasets])
    n_total = np.sum(n_k)

    w_global = w_init.copy()
    losses = []

    for r in range(R):

        # ---- Compute global loss ----
        global_loss = 0
        for k in range(num_clients):
            X_k, y_k = client_datasets[k]
            loss_k = cross_entropy_loss(X_k, y_k, w_global)
            global_loss += (n_k[k] / n_total) * loss_k
        losses.append(global_loss)

        # ---- Print progress ----
        if r % display_every == 0:
            test_info = ""
            if X_test is not None and y_test is not None:
                test_loss = cross_entropy_loss(X_test, y_test, w_global)
                test_acc  = compute_accuracy(X_test, y_test, w_global)
                test_info = f", Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%"

            print(f"[FedAvg] Round {r:3d}: Global Loss={total_loss:.4f}{test_info}")


        # ---- Local training ----
        client_updates = []
        for k in range(num_clients):
            X_k, y_k = client_datasets[k]
            w_k = client_update(w_global, X_k, y_k, gamma, H, batch_size)
            client_updates.append(w_k)

        # ---- Weighted aggregation ----
        w_global = sum((n_k[k] / n_total) * client_updates[k] for k in range(num_clients))

    return np.array(losses), w_global
