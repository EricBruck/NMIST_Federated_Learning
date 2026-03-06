
"""
Constraint-Regularized Personalized Federated Averaging (PCFedAvg)

This module implements a constraint-regularized federated learning approach that:
1. Uses blockwise parameter distribution where each client owns one block of parameters
2. Imposes client-specific loss budgets / feasibility constraints (epsilon-based)
    to limit overfitting to client data
3. Uses constraint smoothing via a penalty function g_i for differentiable updates
4. Implements communication-efficient updates where each block is updated based on gradients
    from all clients who don't own that block


Key components:
- h_i_loss: Constrained loss function for client i (uses eps_i as a loss budget)
- grad_g_i: Smooth penalty function for constraint satisfaction
- estimate_epsilons: Heuristic to initialize per-client constraint thresholds (loss budgets)
- pcfedavg_blockwise_efficient: Main federated training loop
"""

# Import numpy for numerical operations and array manipulations
import numpy as np
# Import shuffle for randomizing data order during training
from sklearn.utils import shuffle
# Import utility functions: compute gradients, cross-entropy loss calculation, and accuracy evaluation
from utils import compute_gradient, cross_entropy_loss, compute_accuracy


def h_i_loss(W: np.ndarray, X: np.ndarray, y: np.ndarray, gamma_reg: float, eps_i: float) -> float:
    """
    Constrained loss function for client i.
    
    Formulation: h_i(W) = f_i(W) + (gamma/2)||W||^2 - eps_i
    
    Components:
    - f_i(W): Cross-entropy loss on client i's local data
    - (gamma/2)||W||^2: L2 regularization term to encourage small model weights
    - eps_i: Constraint threshold / loss budget (feasibility epsilon). Lower = stricter constraint.
    
    This function defines a constraint that h_i(W) <= 0. If violated (h_i > 0),
    the penalty function g_i will provide corrective gradient signals.
    
    Args:
        W: Model parameters
        X: Client's training data features
        y: Client's training data labels
        gamma_reg: L2 regularization coefficient (gamma)
        eps_i: Constraint threshold / loss budget for this client. Not a DP epsilon.
    
    Returns:
        The value of h_i(W), which should ideally be <= 0
    """
    # Compute the data loss term f_i(W) using cross-entropy on client's local data
    f_i = cross_entropy_loss(X, y, W)
    
    # Compute the L2 regularization term: (gamma/2) * ||W||^2
    # This penalizes large model weights to encourage smaller, more generalizable solutions
    reg = 0.5 * gamma_reg * float(np.sum(W * W))
    
    # Return constrained loss: f_i(W) + regularization - epsilon_i
    # The constraint is satisfied when h_i(W) <= 0, otherwise penalty signals are needed
    return f_i + reg - eps_i

def grad_h_i_loss(W: np.ndarray, X: np.ndarray, y: np.ndarray, gamma_reg: float) -> np.ndarray:
    """
    Gradient of the constrained loss function.
    
    Formulation: ∇h_i(W) = ∇f_i(W) + gamma * W
    
    This gradient is used by the penalty function g_i to adjust client updates
    when they would violate the constraint (h_i > 0).
    
    Args:
        W: Model parameters
        X: Client's training data features
        y: Client's training data labels
        gamma_reg: L2 regularization coefficient (gamma)
    
    Returns:
        Gradient vector of the constrained loss h_i(W)
    """
    # Gradient of f_i(W) (data loss) computed from cross-entropy
    # Plus gradient of L2 regularization term (gamma * W)
    # This gradient guides parameter updates toward lower loss and smaller weights
    return compute_gradient(X, y, W) + gamma_reg * W

def grad_g_i(W: np.ndarray, X: np.ndarray, y: np.ndarray, gamma_reg: float, eps_i: float, lam: float) -> np.ndarray:
    """
    Smooth penalty function gradient for constraint enforcement.
    
    This function implements constraint smoothing, providing differentiable penalty
    signals to keep h_i(W) <= 0. It uses a piecewise smooth approach:
    
    - If h_i(W) < 0 (constraint satisfied): gradient = 0 (no penalty)
    - If 0 <= h_i(W) < lambda (near boundary): gradient = (h_i/lambda) * ∇h_i
      → Linearly scales penalty from 0 to full gradient as constraint tightens
    - If h_i(W) >= lambda (constraint violated): gradient = ∇h_i (full penalty)
    
    The smoothing parameter lambda prevents sudden changes and enables stable convergence.
    This is key to making PCFedAvg trainable via gradient-based methods.
    
    Args:
        W: Model parameters
        X: Client's training data features
        y: Client's training data labels
        gamma_reg: L2 regularization coefficient
        eps_i: constraint parameter for this client
        lam: Smoothing parameter (lambda) controlling transition region width
    
    Returns:
        Gradient of smooth penalty function g_i(W)
    """
    # Evaluate the constraint function h_i(W) to check feasibility
    h_i = h_i_loss(W, X, y, gamma_reg=gamma_reg, eps_i=eps_i)
    
    # Case 1: Constraint is satisfied (h_i < 0) - no penalty needed
    # Client is fitting within their loss budget, so no corrective signal is required
    if h_i < 0:
        return np.zeros_like(W)
    
    # Compute gradient of constraint function for use in cases 2 and 3
    # This gradient indicates the direction of steepest increase in h_i
    grad_h = grad_h_i_loss(W, X, y, gamma_reg=gamma_reg)

    # Case 2: Near constraint boundary (0 <= h_i < lambda) - apply smooth linear scaling
    # This gradually increases penalty as we approach/cross the boundary
    # Ensures smooth transition and stable convergence without sudden jumps
    if h_i < lam:
        return (h_i / lam) * grad_h
    
    # Case 3: Constraint significantly violated (h_i >= lambda) - apply full gradient penalty
    # When constraint is clearly violated, apply maximum corrective force
    return grad_h

def g_value(h, lam):
    """
    Piecewise smooth penalty value function g_{i,λ}(h).
    
    This function maps the constraint value h to a penalty value that encourages h <= 0.
    It is differentiable (except at transitions) to enable stable gradient-based training.
    
    Args:
        h: The constraint value h_i(W) to evaluate
        lam: Smoothing parameter (transition region width)
    
    Returns:
        Penalty value for the given constraint violation h
    """
    # Case 1: Constraint satisfied (h < 0) - no penalty
    if h < 0:
        return 0.0
    # Case 2: Near boundary (0 <= h < lambda) - quadratic penalty that grows smoothly
    # Provides gentle encouragement to satisfy constraint near the boundary
    if h < lam:
        return (h * h) / (2.0 * lam)
    # Case 3: Constraint violated (h >= lambda) - linear penalty for larger violations
    # Stronger penalty for significant constraint violation
    return h - lam / 2.0

def grad_norm(mat: np.ndarray) -> float:
    """
    Compute the Euclidean (L2) norm of a matrix or vector.
    
    Used for gradient clipping and convergence diagnostics to measure
    the magnitude of gradient updates.
    
    Args:
        mat: Input matrix or vector
    
    Returns:
        The L2 norm (Euclidean length) of the input
    """
    # Compute and return the L2 norm as a Python float
    return float(np.linalg.norm(mat))


def estimate_epsilons(client_datasets, W_init, multiplier = 1.1, warmup_epochs=1, lr=0.01, batch_size=64):
    """
    Initialize constraint thresholds (feasibility epsilons / loss budgets) for each client.
    
    Strategy:
    1. Each client performs a short local warmup training using only cross-entropy loss
       (no regularization or penalties) to find their minimum achievable loss
    2. Sets eps_i = 1.1 * min_loss to allow some constraint slack
    
    Motivation:
    - eps_i values control how much each client can "fit" their data (loss budgets)
    - Lower eps_i = stricter constraint = prevents overfitting to local data
    - Higher eps_i = looser constraint = allows more data-specific learning
    - Setting eps_i slightly above minimum loss ensures initial feasibility
    
    Args:
        client_datasets: List of (X_i, y_i) tuples for each client
        W_init: Initial global model weights
        warmup_epochs: Number of local epochs for warmup (default: 1)
        lr: Learning rate for warmup training (default: 0.01)
        batch_size: Batch size for mini-batch SGD (default: 64)
    
    Returns:
        List of epsilon values [eps_0, eps_1, ..., eps_m-1], one per client
    """
    eps_list = []
    
    # For each client, estimate their minimum achievable loss during warmup
    for (X_i, y_i) in client_datasets:
        # Handle empty client datasets - assign a very large epsilon (no real constraint)
        if len(X_i) == 0:
            eps_list.append(1e9)  # Large epsilon = effectively no constraint for empty clients
            continue

        # Start with initial global weights for this client's warmup
        W = W_init.copy()
        best = float("inf")  # Track the minimum loss achieved during warmup

        # Local warmup training: plain SGD on cross-entropy loss (no constraints or penalties)
        # Goal: find what is the best (lowest) loss this client can achieve locally
        for _ in range(warmup_epochs):
            # Shuffle data for better stochastic gradient estimates
            X_i, y_i = shuffle(X_i, y_i)
            n = len(X_i)
            
            # Mini-batch SGD updates over the shuffled data
            for s in range(0, n, batch_size):
                Xb = X_i[s:s+batch_size]  # Extract current mini-batch
                yb = y_i[s:s+batch_size]
                if len(Xb) == 0:
                    continue
                
                # Compute gradient of cross-entropy loss and perform SGD update
                grad = compute_gradient(Xb, yb, W)
                W -= lr * grad
                
                # Track the best (minimum) loss achieved during warmup
                # This gives us the lowest loss this client can achieve
                L = cross_entropy_loss(Xb, yb, W)
                best = min(best, L)

        # Set epsilon to multiplier * minimum loss to allow some slack
        # This ensures the constraint is initially feasible while still limiting overfitting
        eps_list.append(multiplier * best)

    return eps_list


def pcfedavg_blockwise_efficient(
    client_datasets,
    W_blocks,
    R: int,                  # Global communication rounds
    K: int,                  # Local mini-batch update steps per round
    gamma_l: float,
    rho_base: float,
    lam: float,
    gamma_reg: float,
    eps_list,
    batch_size: int = 64,
    client_fraction: float = 1.0,
    X_test=None,
    y_test=None,
    display_every: int = 1,
    clip: float = 5.0,
    X_train=None,
    y_train=None,
):
    """
    PCFedAvg-CE (Blockwise Efficient Implementation)
    =================================================

    Communication-Efficient Blockwise Constraint-Regularized Federated Averaging.

    -------------------------------------------------------------------------
    ALGORITHM OVERVIEW
    -------------------------------------------------------------------------

    This algorithm implements a personalized, constraint-aware federated
    optimization scheme in which:

        • Each client owns exactly ONE parameter block.
        • Each communication round consists of:
            - K local update rounds
            - Each local round performs `epochs` full passes over data
        • Clients optimize a penalized local objective:
              f_i(z) + ρ_r g_i(h_i(W_i))
        • Only block-specific updates and accumulated gradient summaries
          are communicated to reduce bandwidth.

    The server reconstructs the globally consistent parameter vector
    using a communication-efficient correction rule derived from the
    blockwise formulation.

    -------------------------------------------------------------------------
    NOTATION MAPPING (Paper → Code)
    -------------------------------------------------------------------------

        R           → Number of global communication rounds
        K           → Number of local rounds per communication round
        epochs      → Passes over local dataset per local round
        γ_l         → Local learning rate
        ρ_r         → Round-dependent penalty weight
        h_i         → Constraint violation
        g_i         → Smooth penalty
        W_blocks    → Blockwise model parameters
        W_bar       → Averaged global model
        D_sum       → Accumulated block gradients

    IMPORTANT:
        K and epochs are fully independent.
        Neither depends on number of mini-batches.
        Mini-batch count = n_i / batch_size is irrelevant to the definition
        of a local round in this implementation.

    -------------------------------------------------------------------------
    RETURNS
    -------------------------------------------------------------------------

    losses                 → Weighted global loss per round
    accs                   → Test accuracy history
    W_blocks               → Final block parameters
    h_hist                 → Constraint violations
    g_hist                 → Penalty values
    metric_hist            → Combined convergence diagnostic
    gradnorm_hist          → Average gradient norms
    gmean_hist             → Mean penalty values
    local_loss_hist        → Per-client local losses
    local_acc_hist         → Per-client local accuracies
    global_train_acc_hist  → Global training accuracy
    avg_obj_hist           → Penalized objective values
    rho_hist               → Penalty schedule values
    """

    m = len(client_datasets)

    if len(W_blocks) != m:
        raise ValueError("W_blocks must have length = num_clients")

    if len(eps_list) != m:
        raise ValueError("eps_list must have length = num_clients")

    n_k = np.array([len(X) for X, _ in client_datasets], dtype=float)
    n_total = np.sum(n_k) if np.sum(n_k) > 0 else 1.0

    losses, accs = [], []

    h_hist = np.zeros((R, m))
    g_hist = np.zeros((R, m))

    local_loss_hist = np.full((R, m), np.nan)
    local_acc_hist = np.full((R, m), np.nan)

    global_train_acc_hist = np.full(R, np.nan)
    avg_obj_hist = np.full(R, np.nan)
    rho_hist = np.full(R, np.nan)

    metric_hist = np.zeros(R)
    gradnorm_hist = np.zeros(R)
    gmean_hist = np.zeros(R)

    W_bar_init = sum(W_blocks) / m

    init_loss = 0.0
    for i in range(m):
        X_i, y_i = client_datasets[i]
        if len(X_i) == 0:
            continue
        init_loss += (n_k[i] / n_total) * cross_entropy_loss(X_i, y_i, W_bar_init)

    print(f"[Before training] Global Loss={init_loss:.4f}")

    for r in range(R):

        rho_r = rho_base * (r + 10000) ** 0.25
        rho_hist[r] = rho_r

        if client_fraction >= 1.0:
            S_r = np.arange(m)
        else:
            s_size = max(1, int(m * client_fraction))
            S_r = np.random.choice(m, size=s_size, replace=False)

        W_snapshot = [W.copy() for W in W_blocks]

        sum_all = np.zeros_like(W_snapshot[0])
        for j in range(m):
            sum_all += W_snapshot[j]

        updated_block = {}
        D_sum = {}

        for i in S_r:

            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                updated_block[i] = W_snapshot[i].copy()
                D_sum[i] = np.zeros_like(W_snapshot[i])
                continue

            W_i = W_snapshot[i].copy()
            sum_other = sum_all - W_snapshot[i]

            D_acc = np.zeros_like(W_i)

            n = len(X_i)
            steps_per_epoch = int(np.ceil(n / batch_size))

            X_i, y_i = shuffle(X_i, y_i)

            for step in range(K):

                batch_idx = step % steps_per_epoch

                start = batch_idx * batch_size
                end = start + batch_size

                if batch_idx == 0 and step > 0:
                    X_i, y_i = shuffle(X_i, y_i)

                Xb = X_i[start:end]
                yb = y_i[start:end]

                if len(Xb) == 0:
                    continue

                z = (sum_other + W_i) / m

                grad_z = compute_gradient(Xb, yb, z)
                d_block = grad_z / m

                pen = grad_g_i(
                    W_i,
                    Xb,
                    yb,
                    gamma_reg=gamma_reg,
                    eps_i=eps_list[i],
                    lam=lam
                )

                v_i = d_block + rho_r * pen

                v_norm = np.linalg.norm(v_i)
                if v_norm > clip:
                    v_i *= clip / (v_norm + 1e-12)

                W_i -= gamma_l * v_i

                D_acc += d_block

            updated_block[i] = W_i
            D_sum[i] = D_acc

        new_blocks = [W_snapshot[j].copy() for j in range(m)]

        total_D = np.zeros_like(W_snapshot[0])
        for i in D_sum:
            total_D += D_sum[i]

        for j in range(m):

            x_snap_j = W_snapshot[j]
            x_owner = updated_block[j] if j in updated_block else x_snap_j

            sum_D_excl_j = total_D - (D_sum[j] if j in D_sum else 0.0)

            new_blocks[j] = (
                x_owner
                + (m - 1) * x_snap_j
                - gamma_l * sum_D_excl_j
            ) / m

        W_blocks = new_blocks

        for i in range(m):

            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            local_loss_hist[r, i] = cross_entropy_loss(X_i, y_i, W_blocks[i])
            local_acc_hist[r, i] = compute_accuracy(X_i, y_i, W_blocks[i])

        for i in range(m):

            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            h = h_i_loss(
                W_blocks[i],
                X_i,
                y_i,
                gamma_reg=gamma_reg,
                eps_i=eps_list[i]
            )

            h_hist[r, i] = h
            g_hist[r, i] = g_value(h, lam=lam)

        W_bar = sum(W_blocks) / m

        if X_train is not None and y_train is not None:
            global_train_acc_hist[r] = compute_accuracy(X_train, y_train, W_bar)

        grad_sum = np.zeros_like(W_bar)

        for i in range(m):
            X_i, y_i = client_datasets[i]
            if len(X_i) == 0:
                continue
            grad_sum += compute_gradient(X_i, y_i, W_bar)

        avg_grad = grad_sum / m
        avg_grad_norm = float(np.linalg.norm(avg_grad))

        g_mean = float(np.nanmean(g_hist[r, :]))

        gradnorm_hist[r] = avg_grad_norm
        gmean_hist[r] = g_mean
        metric_hist[r] = avg_grad_norm + rho_r * g_mean

        obj_vals = []

        for i in range(m):
            if not np.isnan(local_loss_hist[r, i]):
                obj_vals.append(
                    local_loss_hist[r, i] + rho_r * g_hist[r, i]
                )

        avg_obj_hist[r] = np.mean(obj_vals) if obj_vals else np.nan

        total_loss = 0.0

        for i in range(m):

            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            total_loss += (
                n_k[i] / n_total
            ) * cross_entropy_loss(X_i, y_i, W_bar)

        losses.append(total_loss)

        test_info = ""

        if X_test is not None and y_test is not None:

            test_acc = compute_accuracy(X_test, y_test, W_bar)

            accs.append(test_acc)

            test_info = f", Test Acc={test_acc*100:.2f}%"

        if display_every and (r % display_every == 0):

            print(
                f"[PCFedAvg-CE-Blockwise] "
                f"Round {r+1:3d}: Global Loss={total_loss:.4f}{test_info}"
            )

    return (
        np.array(losses),
        np.array(accs),
        W_blocks,
        h_hist,
        g_hist,
        metric_hist,
        gradnorm_hist,
        gmean_hist,
        local_loss_hist,
        local_acc_hist,
        global_train_acc_hist,
        avg_obj_hist,
        rho_hist
    )