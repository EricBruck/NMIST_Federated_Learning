import numpy as np
from sklearn.utils import shuffle
from utils import compute_gradient, cross_entropy_loss, compute_accuracy


def h_i_smooth_dist(W_i: np.ndarray, W_bar: np.ndarray, eps: float = 1e-3) -> float:
    """
    h_i(x_i) = sqrt(||x_i - xbar||^2 + eps^2)
    Convex in x_i, C^1.
    """
    D = W_i - W_bar
    return float(np.sqrt(np.sum(D * D) + eps * eps))

def grad_h_i_smooth_dist(W_i: np.ndarray, W_bar: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    ∇_{x_i} h_i(x_i) = (x_i - xbar) / sqrt(||x_i-xbar||^2 + eps^2)
    """
    D = W_i - W_bar
    denom = np.sqrt(np.sum(D * D) + eps * eps)
    return D / denom

def grad_g_i(W_i: np.ndarray, W_bar: np.ndarray, rho_i: float, eps: float = 1e-3) -> np.ndarray:
    """
    g_i(x_i) = 1/2 * max(0, h_i(x_i) - rho_i)^2
    ∇g_i = ∇h_i(x_i) * max(0, h_i(x_i) - rho_i)
    """
    hval = h_i_smooth_dist(W_i, W_bar, eps=eps)
    hinge = max(0.0, hval - rho_i)
    return grad_h_i_smooth_dist(W_i, W_bar, eps=eps) * hinge


def client_update_pcfedavg(
    W_i_init: np.ndarray,
    X_i: np.ndarray,
    y_i: np.ndarray,
    W_bar_snapshot: np.ndarray,
    rho_i: float,
    eta_r: float,
    gamma_l: float,
    K: int,
    batch_size: int,
):
    """
    Implements the screenshot update direction:
      g_{i,t}^{r,eta} = ∇g_i(W_{i,t}) + eta_r * ∇ \tilde f_i(W_bar, xi_batch)
      W_{i,t+1} = W_{i,t} - gamma_l * g_{i,t}^{r,eta}
    Note: stochastic gradient is evaluated at W_bar_snapshot (global ref),
          constraint gradient at local W_{i,t}.
    """
    W = W_i_init.copy()
    n = len(X_i)
    if n == 0:
        return W

    for _ in range(K):
        X_i, y_i = shuffle(X_i, y_i)
        for s in range(0, n, batch_size):
            Xb = X_i[s:s+batch_size]
            yb = y_i[s:s+batch_size]
            if len(Xb) == 0:
                continue

            # constraint term at local variable
            grad_g = grad_g_i(W, W_bar_snapshot, rho_i, eps=1e-3)


            # stochastic gradient evaluated at global reference
            grad_f = compute_gradient(Xb, yb, W_bar_snapshot)

            g = grad_g + eta_r * grad_f

            # gradient clipping (L2)
            g_norm = np.linalg.norm(g)
            clip = 5.0
            if g_norm > clip:
               g = g * (clip / (g_norm + 1e-12))

            W -= gamma_l * g

    return W

def pcfedavg_blockwise(
    client_datasets,
    W_blocks,                 # list of per-client models W_i (these are the "blocks" x^i)
    R: int,
    K: int,
    gamma_l: float,
    batch_size: int = 64,
    client_fraction: float = 1.0,
    eta_schedule=None,
    rho_list=None,
    X_test=None,
    y_test=None,
    display_every: int = 1,
):
    """
    "Blockwise" here follows the paper's variable stacking x=(x_i):
      - each client i owns block x_i (=W_i)
      - global reference W_bar is the average of blocks
      - clients update their own block only
      - server keeps W_j unchanged if j not in S_r
      - when j in S_r, update W_j with weighted averaging (trivial here since only client j updates block j)
    """
    m = len(client_datasets)
    if rho_list is None:
        # default constraint thresholds: allow moderate norm
        rho_list = [25.0 for _ in range(m)]
    if len(rho_list) != m:
        raise ValueError("rho_list must have length = num_clients")

    n_k = np.array([len(X) for X, _ in client_datasets], dtype=float)
    n_total = np.sum(n_k) if np.sum(n_k) > 0 else 1.0

    losses, accs = [], []

    for r in range(R):
        # sample clients
        if client_fraction >= 1.0:
            S_r = np.arange(m)
        else:
            s = max(1, int(m * client_fraction))
            S_r = np.random.choice(m, size=s, replace=False)

        eta_r = eta_schedule[r] if eta_schedule is not None else 1.0

        # server snapshot
        W_snapshot = [W.copy() for W in W_blocks]
        W_bar = sum(W_snapshot) / m  # \bar{x}_t

        updated = {}

        # local updates (each client updates its own block)
        for i in S_r:
            X_i, y_i = client_datasets[i]
            updated[i] = client_update_pcfedavg(
                W_i_init=W_snapshot[i],
                X_i=X_i,
                y_i=y_i,
                W_bar_snapshot=W_bar,
                rho_i=rho_list[i],
                eta_r=eta_r,
                gamma_l=gamma_l,
                K=K,
                batch_size=batch_size,
            )

        # server update: keep old if not selected; update only selected blocks
        m_r = np.sum(n_k[S_r]) if len(S_r) > 0 else 1.0

        for j in range(m):
            if j in S_r:
                # weighted average (degenerates to just updated[j] if each block has one contributor)
                W_blocks[j] = (n_k[j] / m_r) * updated[j]
                # If, later, multiple clients can contribute to block j, you'd sum them here.
                # For the client-owned-block schedule, only client j contributes to block j.
                W_blocks[j] /= (n_k[j] / m_r)  # cancels now; keep structure explicit
            else:
                W_blocks[j] = W_snapshot[j]

        # evaluation: use global average model (typical reporting)
        W_bar_new = sum(W_blocks) / m

        # weighted train loss (across all client data)
        total_loss = 0.0
        for i in range(m):
            X_i, y_i = client_datasets[i]
            if len(X_i) == 0:
                continue
            total_loss += (n_k[i] / n_total) * cross_entropy_loss(X_i, y_i, W_bar_new)
        losses.append(total_loss)

        test_info = ""
        if X_test is not None and y_test is not None:
            test_loss = cross_entropy_loss(X_test, y_test, W_bar_new)
            test_acc  = compute_accuracy(X_test, y_test, W_bar_new)
            accs.append(test_acc)
            test_info = f", Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%"

        if r % display_every == 0:
            print(f"[PCFedAvg-Blockwise] Round {r:3d}: Global Loss={total_loss:.4f}{test_info}")

    return np.array(losses), np.array(accs), W_blocks
