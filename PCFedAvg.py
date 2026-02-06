
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

import numpy as np
from sklearn.utils import shuffle
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
    # Compute the data loss term f_i(W) using cross-entropy
    f_i = cross_entropy_loss(X, y, W)
    
    # Compute the L2 regularization term: (gamma/2) * ||W||^2
    reg = 0.5 * gamma_reg * float(np.sum(W * W))
    
    # Return constrained loss: f_i(W) + regularization - epsilon_i
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
    # Gradient of f_i(W) (data loss) + gradient of L2 regularization term
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
    # Evaluate the constraint function h_i(W)
    h_i = h_i_loss(W, X, y, gamma_reg=gamma_reg, eps_i=eps_i)
    
    # Case 1: Constraint is satisfied (h_i < 0) - no penalty needed
    if h_i < 0:
        return np.zeros_like(W)
    
    # Compute gradient of constraint function for use in cases 2 and 3
    grad_h = grad_h_i_loss(W, X, y, gamma_reg=gamma_reg)

    # Case 2: Near constraint boundary - apply smooth linear scaling
    # This gradually increases penalty as we approach/cross the boundary
    if h_i < lam:
        return (h_i / lam) * grad_h
    
    # Case 3: Constraint significantly violated - apply full gradient penalty
    return grad_h

def g_value(h, lam):
    # g_{i,λ}(x) piecewise scalar
    if h < 0:
        return 0.0
    if h < lam:
        return (h * h) / (2.0 * lam)
    return h - lam / 2.0


def estimate_epsilons(client_datasets, W_init, warmup_epochs=1, lr=0.01, batch_size=64):
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
    
    # For each client, estimate their minimum achievable loss
    for (X_i, y_i) in client_datasets:
        # Handle empty client datasets
        if len(X_i) == 0:
            eps_list.append(1e9)  # Large epsilon = no real constraint
            continue

        # Start with initial global weights
        W = W_init.copy()
        best = float("inf")  # Track minimum loss found

        # Local warmup training: plain SGD on cross-entropy loss
        for _ in range(warmup_epochs):
            # Shuffle data for better stochastic gradient estimates
            X_i, y_i = shuffle(X_i, y_i)
            n = len(X_i)
            
            # Mini-batch SGD updates
            for s in range(0, n, batch_size):
                Xb = X_i[s:s+batch_size]  # Current batch
                yb = y_i[s:s+batch_size]
                if len(Xb) == 0:
                    continue
                
                # Compute gradient of cross-entropy loss and update weights
                grad = compute_gradient(Xb, yb, W)
                W -= lr * grad
                
                # Track the best (minimum) loss achieved during warmup
                L = cross_entropy_loss(Xb, yb, W)
                best = min(best, L)

        # Set epsilon to 1.1x the minimum loss to allow some slack
        eps_list.append(1.1 * best)

    return eps_list


def pcfedavg_blockwise_efficient(
    client_datasets,
    W_blocks,                 # list length m, each block is (d x C)
    R: int,                   # Total number of communication rounds
    K: int,                   # Number of local SGD steps per client per round
    gamma_l: float,           # Client-side learning rate (γ_l)
    rho_base: float,          # Penalty weight for constraint enforcement (ρ)
    lam: float,               # Smoothing parameter (λ) for penalty function
    gamma_reg: float,         # L2 regularization coefficient (γ)
    eps_list,                 # constraint parameters [eps_0, ..., eps_m-1]
    batch_size: int = 64,
    client_fraction: float = 1.0,  # Fraction of clients sampled per round
    X_test=None,
    y_test=None,
    display_every: int = 1,
    clip: float = 5.0,        # Gradient clipping threshold
):
    """
    Communication-Efficient Blockwise PCFedAvg Implementation
    
    ALGORITHM OVERVIEW:
    This implements a constraint-regularized federated learning algorithm where:
    
    1. BLOCKWISE ARCHITECTURE: Each client i owns one parameter block x_i (size d x C)
       - The global model is the average: W_bar = (1/m) * Σ x_i
       - Each client trains locally using their own block
    
    2. LOCAL UPDATES (K epochs per round):
       - Each client computes gradients using the averaged model z = (1/m) * Σ x_j
       - For each block j, gradient is d_{i,t}^j = (1/m) * ∇f(z)
       - Augments data gradient with penalty: v_i = d_{i,t} + ρ * ∇g_i(x_i)
       - Updates own block: x_i ← x_i - γ_l * v_i
       - Accumulates gradients from OTHER blocks for later correction
    
    3. SERVER UPDATE (Communication-efficient):
       - Server receives each selected client’s updated owned block and its accumulated non-owner gradient summary D
       - For each block j:
         * If client j participated: use their updated block
         * Apply gradient correction from non-owners: x̄^j ← x_owner^j - (γ_l/m) * Σ_{k≠j} D_k^j
         * This incorporates gradient information from non-owners for each block while keeping communication low, and maintains consistency when only a subset of clients participate
    
     4. CONSTRAINT MECHANISM:
         - Constraints h_i(x_i) ≤ 0 limit how much client i can fit their data (loss budgets)
         - Penalty g_i enforces feasibility via smooth gradient signals
         - Epsilon parameters control constraint tightness per client (these are loss budgets)

    
    SPECIALIZED FORMULATION:
    Since loss depends on z = (1/m) Σ x^k, the gradient ∂f/∂x^j = (1/m) ∂f/∂z for all j.
    This means all blocks see the same scaled gradient, enabling efficient communication.
    
    Args:
        client_datasets: List of (X_i, y_i) tuples for each client
        W_blocks: Initial blockwise parameters [x_0, x_1, ..., x_m-1]
        R: Number of communication rounds
        K: Number of local SGD epochs per round
        gamma_l: Learning rate for local and server updates
        rho_base: Weighting factor for constraint penalty (typically ~0.1-1.0)
        lam: Smoothing threshold for penalty function (typically ~0.1-1.0)
        gamma_reg: L2 regularization strength (typically ~0.001-0.1)
        eps_list: Constraint thresholds / loss budgets (feasibility epsilons) (output of estimate_epsilons).
        batch_size: Mini-batch size for SGD
        client_fraction: Fraction of clients to sample each round (1.0 = all clients)
        X_test, y_test: Optional test set for evaluation
        display_every: Print metrics every N rounds
        clip: Gradient clipping norm threshold
    
    Returns:
        losses: Array of training losses per round
        accs: Array of test accuracies per round (if X_test provided)
        W_blocks: Final blockwise parameters
    """
    # === INITIALIZATION ===
    m = len(client_datasets)  # Number of clients
    
    # Validate inputs
    if len(W_blocks) != m:
        raise ValueError("W_blocks must have length = num_clients")
    if len(eps_list) != m:
        raise ValueError("eps_list must have length = num_clients")

    # Compute client data sizes for weighted loss aggregation
    n_k = np.array([len(X) for X, _ in client_datasets], dtype=float)
    n_total = np.sum(n_k) if np.sum(n_k) > 0 else 1.0

    # Lists to track metrics over rounds
    losses, accs = [], []

    h_hist = np.zeros((R, m), dtype=float)
    g_hist = np.zeros((R, m), dtype=float)

    # === MAIN FEDERATED LEARNING LOOP ===
    for r in range(R):

        # Round-dependent penalty weight (rho)
        rho_r = rho_base * (r + 10000) ** 0.25
        
        # --- CLIENT SAMPLING ---
        # Sample a subset of clients (potentially fewer than m) to participate in this round
        if client_fraction >= 1.0:
            S_r = np.arange(m)  # Use all clients
        else:
            s = max(1, int(m * client_fraction))  # Number of clients to select
            S_r = np.random.choice(m, size=s, replace=False)  # Random sampling without replacement

        # --- SERVER SNAPSHOT ---
        # Server maintains the current blockwise parameters
        W_snapshot = [W.copy() for W in W_blocks]
        
        # Precompute sum of all blocks for use in local computations
        sum_all = np.zeros_like(W_snapshot[0])
        for j in range(m):
            sum_all += W_snapshot[j]


        # --- CLIENT-LOCAL COMPUTATION ---
        # Storage for results from each participating client
        updated_block = {}   # i -> updated block x_i from client i's local training
        D_sum = {}           # i -> cumulative gradient sum Σ_t d_{i,t}^j (same for all j ≠ i)

        # Loop over sampled clients
        for i in S_r:
            # Get client i's data
            X_i, y_i = client_datasets[i]
            
            # Handle empty datasets
            if len(X_i) == 0:
                updated_block[i] = W_snapshot[i].copy()
                D_sum[i] = np.zeros_like(W_snapshot[i])
                continue

            # --- LOCAL MODEL COPY ---
            # Client i maintains a local copy of only their own block
            # Other blocks are fixed at server snapshot values (no local updates to others)
            W_i = W_snapshot[i].copy()  # Client i's own block (will be updated)
            
            # Precompute sum of all other clients' blocks (constant during local loop)
            sum_other = sum_all - W_snapshot[i]

            # Accumulator for gradient contributions that affect OTHER clients' blocks
            D_acc = np.zeros_like(W_i)  # Will store Σ_t d_{i,t}^j (same value for all j ≠ i)

            # --- LOCAL TRAINING LOOP (K epochs) ---
            for _ in range(K):
                # Shuffle data for better stochastic estimates
                X_i, y_i = shuffle(X_i, y_i)
                n = len(X_i)
                
                # Mini-batch SGD updates
                for s in range(0, n, batch_size):
                    Xb = X_i[s:s+batch_size]  # Current mini-batch
                    yb = y_i[s:s+batch_size]
                    if len(Xb) == 0:
                        continue

                    # --- COMPUTE AVERAGED MODEL ---
                    # z_{i,t} = (1/m) * Σ_j x_j = (1/m) * (sum_other + W_i)
                    # This is the model that all clients see and evaluate loss on
                    z = (sum_other + W_i) / m

                    # --- COMPUTE DATA GRADIENT ---
                    # Gradient of loss w.r.t. the averaged model z
                    grad_z = compute_gradient(Xb, yb, z)

                    # --- DISTRIBUTE GRADIENT TO BLOCKS ---
                    # Since z = (1/m) * Σ x_j, we have ∂f/∂x_j = (1/m) * ∂f/∂z for all j
                    # This means all blocks see the SAME scaled gradient contribution
                    d_block = grad_z / m

                    # --- UPDATE OWN BLOCK WITH PENALTY ---
                    # Compute constraint penalty term (only applied to own block)
                    pen = grad_g_i(W_i, Xb, yb, gamma_reg=gamma_reg, eps_i=eps_list[i], lam=lam)
                    
                    # Combined update direction: data gradient + rho * penalty
                    v_i = d_block + rho_r * pen

                    # --- GRADIENT CLIPPING ---
                    # Clip gradient norm to stabilize training and control update magnitudes
                    v_norm = np.linalg.norm(v_i)
                    if v_norm > clip:
                        v_i = v_i * (clip / (v_norm + 1e-12))

                    # Update own block using gradient descent
                    W_i -= gamma_l * v_i

                    # --- ACCUMULATE CROSS-BLOCK GRADIENTS ---
                    # Store gradient contribution from client i that will affect OTHER blocks
                    # This is the same d_block value that would apply to any j ≠ i
                    D_acc += d_block

            # Store client i's updated block and accumulated gradients for server update
            updated_block[i] = W_i
            D_sum[i] = D_acc

        # --- SERVER UPDATE PHASE ---
        # Server aggregates updates from participating clients and applies corrections
        # 
        # Update rule for each block j:
        #   x̄_{r+1}^j = x_{j,T_{r+1}}^j - (γ_l / m) * Σ_{k≠j} D_{k,r}^j
        #
        # Where:
        # - x_{j,T_{r+1}}^j: Client j's locally updated block (if j was selected, else snapshot)
        # - D_{k,r}^j: Gradient accumulated by client k affecting OTHER blocks
        #              Since all blocks see the same gradient, D_{k,r}^j = D_sum[k]
        # - Correction term: -(γ_l/m) * Σ_{k≠j} D_{k,r}^j
        #                   This counterbalances gradients from clients not updating block j
        #
        # Intuition: If block j is owned by client i but we sample other clients,
        #           those clients will push the averaged model z. We correct for this
        #           so block j still reflects client i's contribution properly.
        
        new_blocks = [W_snapshot[j].copy() for j in range(m)]

        # Precompute total gradient accumulation from all participating clients
        # This is used in the correction term for each block
        total_D = np.zeros_like(W_snapshot[0])
        for k in S_r:
            total_D += D_sum[k]

        # Update each block j
        for j in range(m):
            # Get the updated block from client j (if they participated)
            # Otherwise use server's snapshot as-is
            x_owner = updated_block[j] if j in updated_block else W_snapshot[j]

            # Compute correction gradient sum for block j
            # This is the sum of gradients from ALL other clients (all k ≠ j in S_r)
            # Computed as: (sum of all k in S_r) - (k=j if j in S_r)
            correction_grad_sum = total_D - (D_sum[j] if j in D_sum else 0.0)

            # Apply correction: subtract (γ_l/m) * Σ_{k≠j} D_{k}^j from owner's update
            # This prevents unselected clients' gradient pushes from dominating
            new_blocks[j] = x_owner - gamma_l * correction_grad_sum

        # Replace server blocks with newly aggregated blocks
        W_blocks = new_blocks

        # --- EVALUATION PHASE ---

        for i in range(m):
            X_i, y_i = client_datasets[i]
            if len(X_i) == 0:
                h_hist[r, i] = np.nan
                g_hist[r, i] = np.nan
                continue

            h = h_i_loss(W_blocks[i], X_i, y_i, gamma_reg=gamma_reg, eps_i=eps_list[i])
            h_hist[r, i] = h
            g_hist[r, i] = g_value(h, lam=lam)
        
        # Compute global averaged model for evaluation
        # This is what would be deployed in practice
        W_bar = sum(W_blocks) / m

        # Compute weighted training loss across all clients
        # Weighting by data size ensures larger clients contribute more to loss
        total_loss = 0.0
        for i in range(m):
            X_i, y_i = client_datasets[i]
            if len(X_i) == 0:
                continue
            # Weighted loss: (n_i / n_total) * f_i(W_bar)
            total_loss += (n_k[i] / n_total) * cross_entropy_loss(X_i, y_i, W_bar)
        losses.append(total_loss)

        # Evaluate on test set if provided
        test_info = ""
        if X_test is not None and y_test is not None:
            test_acc = compute_accuracy(X_test, y_test, W_bar)
            accs.append(test_acc)
            test_info = f", Test Acc={test_acc*100:.2f}%"

        # Print progress every display_every rounds
        if r % display_every == 0:
            print(f"[PCFedAvg-CE-Blockwise] Round {r:3d}: Global Loss={total_loss:.4f}{test_info}")

    # === RETURN RESULTS ===
    # Return training history and final model
    return np.array(losses), np.array(accs), W_blocks, h_hist, g_hist