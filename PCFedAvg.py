
"""
Constraint-Regularized Personalized Federated Averaging (PCFedAvg)

This module implements a constraint-regularized federated learning approach that:
1. Uses blockwise parameter distribution where each client owns one block of parameters
2. Imposes client-specific loss budgets / feasibility constraints (epsilon-based)
    to limit overfitting to client data
3. Uses constraint smoothing via a penalty function g_i for differentiable updates
4. Implements communication-efficient updates where each block is updated based on gradients
    from all clients who don't own that block


ALGORITHM ARCHITECTURE
======================

PCFedAvg operates on m clients, each owning one parameter block W_i of a larger model:

    Global model = W_1 || W_2 || ... || W_m  (concatenation of blocks)
    Global average = (W_1 + W_2 + ... + W_m) / m

Key Design Principles:

1. PERSONALIZATION: Each client's block W_i is optimized on local data with constraints.
   -> Allows heterogeneous local models adapted to each client's data distribution
   -> Prevents overfitting by imposing loss budget constraints

2. CONSTRAINT SATISFACTION: Each client must keep h_i(W_i) <= 0, where:
   h_i(W_i) = f_i(W_i) + (gamma/2)||W_i||^2 - eps_i
   - f_i(W_i) = local training loss
   - (gamma/2)||W_i||^2 = L2 regularization
   - eps_i = loss budget (lower eps_i = stricter constraint)
   -> Mechanism: penalty function g_i(h_i) provides gradient-based enforcement

3. COMMUNICATION EFFICIENCY: Only communicate:
   - Per-client updated block: W_i^{t+1}
   - Per-client gradient sums: D_sum_i = sum of mini-batch gradients
   -> Not full parameter blocks from server
   -> Server reconstructs global state using communication-efficient aggregation rule

4. CONSENSUS: Global blocks are updated using cross-client gradient information:
   x_j^{t+1} = (x_{owner,j}^{t+1} + (m-1)*x_snap_j - gamma_l * sum_{i!=j} D_i) / m
   -> Block j gets updated not just by client j's local training, but also by
      gradients from other clients' data (D_i terms from i != j)
   -> Creates soft consensus while preserving personalization


ALGORITHM FLOW (Per Communication Round)
==========================================

For each round r = 0, 1, ..., R-1:

   PHASE 1: CLIENT UPDATES (local training with constraints)
   ============================================================
   For each sampled client i:
     Initialize: W_i = W_i^{t}  (own parameter block)
     For K local steps:
       - Sample mini-batch (Xb, yb)
       - Compute z = (sum of other blocks + W_i) / m  (instantaneous average)
       - Compute gradients:
         * Data gradient: grad_z = ∇f(z)  (on mini-batch)
         * Constraint penalty: pen = ∇g_i(h_i(W_i))  (smoothed penalty signal)
       - Update: W_i -= gamma_l * (d_block + rho_r * pen)
         * d_block = (1/m) * grad_z  (scaled gradient)
         * rho_r = penalty weight (increases over rounds)
     Send to server: W_i^{t+1} and accumulated gradient sum D_i

   PHASE 2: SERVER AGGREGATION (communication-efficient block update)
   ==================================================================
   For each block j:
     Receive: x_{owner,j}^{t+1} from client j, gradient sums D_i from all clients
     Update rule (from Algorithm 2 Line 29):
       x_j^{t+1} = (x_{owner,j}^{t+1} + (m-1)*x_snap_j - gamma_l * sum_{i!=j} D_i) / m
     Ensures: blocks get updated by 1) local training, 2) cross-client gradients

   PHASE 3: EVALUATION
   ===================
   Compute and log:
   - Per-client loss/accuracy on each client's local data
   - Constraint satisfaction (h_i values)
   - Global averaged model performance
   - Convergence metrics (gradient norms, penalty values)


HYPERPARAMETER SUMMARY
======================

Optimization Hyperparameters:
  - R: Number of communication rounds (e.g., 1000)
  - K: Local update steps per round (e.g., 5-10)
  - gamma_l: Local learning rate (e.g., 0.01-0.1)
  - batch_size: Mini-batch size (e.g., 32, 64)

Constraint Hyperparameters:
  - gamma_reg: L2 regularization in constraint (prevents large weights)
  - eps_list: Per-client loss budgets (estimated via warmup or fixed)
  - lam (lambda): Smoothing parameter for penalty function g_i
    * Controls smooth transition from 0 penalty to full penalty
    * Affects constraint enforcement smoothness

Penalty Schedule:
  - rho_base: Base penalty weight for constraint enforcement
  - Actual: rho_r = rho_base * (r + 10000)^0.25
    * Offset 10000 prevents high penalties early (models unstable)
    * Exponent 0.25: polynomial growth (slow increase)
    * Strategy: start loose constraints, gradually tighten

Other:
  - clip: Gradient clipping threshold (stability, preventing divergence)
  - client_fraction: Client sampling fraction (<1.0 for partial participation)
  - display_every: Logging frequency (1 = every round)


KEY FUNCTIONS
==============

- h_i_loss: Evaluates constraint function h_i(W) = f_i(W) + (gamma/2)||W||^2 - eps_i
  -> Used to determine constraint satisfaction
  -> h_i < 0: satisfied, h_i > 0: violated

- grad_h_i_loss: Gradient of constraint function
  -> Used by penalty function to compute corrective signals

- grad_g_i: Smoothed penalty gradient (piecewise smooth)
  -> Returns 0 if h_i < 0 (satisfied)
  -> Returns scaled penalty if 0 <= h_i < lambda
  -> Returns full gradient if h_i >= lambda (violated)

- g_value: Penalty value function g_i(h) (for diagnostics)
  -> Maps constraint violation h to penalty magnitude
  -> Used in convergence metrics

- estimate_epsilons: Initialize per-client loss budgets eps_i
  -> Strategy: local warmup training to find min achievable loss
  -> Set eps_i = 1.1 * min_loss to allow slack

- pcfedavg_blockwise_efficient: Main training loop
  -> Implements Algorithms 1-2 from paper
  -> Returns training histories and final parameters


OUTPUT METRICS FOR ANALYSIS
============================

Convergence Diagnostics:
  - losses: Global weighted loss per round (should decrease)
  - gradnorm_hist: Average gradient norms (should decrease)
  - metric_hist: ||grad W_bar|| + rho_r * mean(g) (should decrease/plateau)

Constraint Enforcement:
  - h_hist: Constraint violations per client-round (should approach negative)
  - g_hist: Penalty values per client-round (should decrease)
  - gmean_hist: Mean penalty across clients (should decrease)

Personalization & Performance:
  - local_loss_hist: Per-client training losses on own data
  - local_acc_hist: Per-client accuracies on own data
  - global_train_acc_hist: Global model accuracy on full training set
  - accs: Test accuracy history (if test set provided)

Objective Values:
  - avg_obj_hist: Average penalized objective (data loss + penalty)
  - rho_hist: Penalty schedule values


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
    client_datasets,           # List of (X_i, y_i) tuples, one per client containing their local data
    W_blocks,                  # List of parameter blocks [W_1, W_2, ..., W_m], where client i owns W_i
    R: int,                    # Total number of global communication rounds
    K: int,                    # Local mini-batch update steps per client per communication round
    gamma_l: float,            # Local learning rate (γ_l in Algorithm 2) for client parameter updates
    rho_base: float,           # Base penalty weight; actual penalty ρ_r = rho_base * (r + 10000)^0.25 (schedule)
    lam: float,                # Smoothing parameter (λ) for constraint penalty function g_i (controls smooth transition)
    gamma_reg: float,          # L2 regularization coefficient (γ) for constraint loss h_i (limits parameter magnitude)
    eps_list,                  # Per-client constraint budgets [ε_1, ε_2, ..., ε_m] (loss thresholds)
    batch_size: int = 64,      # Mini-batch size for stochastic gradient computation during local updates
    client_fraction: float = 1.0,  # Fraction of clients sampled per round (1.0 = all clients, <1.0 for client sampling)
    X_test=None,              # Test set features for evaluation (optional)
    y_test=None,              # Test set labels for evaluation (optional)
    display_every: int = 1,    # Print progress every N rounds (1 = every round, 10 = every 10 rounds, etc.)
    clip: float = 5.0,         # Gradient clipping threshold: if ||v_i|| > clip, scale down to clip (prevents divergence)
    X_train=None,             # Full training set features for global accuracy monitoring (optional)
    y_train=None,             # Full training set labels for global accuracy monitoring (optional)
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

    # =========================================================================
    # INITIALIZATION PHASE
    # =========================================================================
    
    # Number of clients in the federated network
    m = len(client_datasets)

    # Validation: ensure we have one parameter block per client
    if len(W_blocks) != m:
        raise ValueError("W_blocks must have length = num_clients")

    # Validation: ensure we have one constraint budget (epsilon) per client
    if len(eps_list) != m:
        raise ValueError("eps_list must have length = num_clients")

    # Count samples per client: n_k[i] = number of samples for client i
    # Used for weighted averaging when computing global metrics
    n_k = np.array([len(X) for X, _ in client_datasets], dtype=float)
    # Total samples across all clients (used for normalization in global loss computation)
    n_total = np.sum(n_k) if np.sum(n_k) > 0 else 1.0

    # Storage for convergence metrics per round
    losses, accs = [], []  # Global loss and test accuracy history

    # CONSTRAINT TRACKING HISTORY:
    # h_hist[r, i] = constraint violation h_i(W_i) at round r for client i
    # Values < 0 indicate constraint satisfied, > 0 indicate violation
    h_hist = np.zeros((R, m))
    # g_hist[r, i] = penalty value g_i(h_i) at round r for client i
    # Tracks how much constraint violation was penalized
    g_hist = np.zeros((R, m))

    # PER-CLIENT METRICS (track personalized model performance):
    # local_loss_hist[r, i] = cross-entropy loss of client i's model on their own data at round r
    local_loss_hist = np.full((R, m), np.nan)
    # local_acc_hist[r, i] = accuracy of client i's model on their own data at round r
    local_acc_hist = np.full((R, m), np.nan)

    # GLOBAL METRICS (track overall system behavior):
    # global_train_acc_hist[r] = accuracy of averaged global model on full training set at round r
    global_train_acc_hist = np.full(R, np.nan)
    # avg_obj_hist[r] = average penalized objective value across all clients at round r
    # Objective = local_loss + ρ_r * g_i (measures training difficulty)
    avg_obj_hist = np.full(R, np.nan)
    # rho_hist[r] = penalty weight ρ_r used at round r (for algorithm analysis)
    rho_hist = np.full(R, np.nan)

    # CONVERGENCE DIAGNOSTICS:
    # metric_hist[r] = combined metric ||∇W_bar||_F + ρ_r * mean(g) at round r
    # Low values indicate convergence (small gradients and constraint satisfaction)
    metric_hist = np.zeros(R)
    # gradnorm_hist[r] = average gradient norm across all clients at round r
    gradnorm_hist = np.zeros(R)
    # gmean_hist[r] = mean penalty value across all clients at round r
    gmean_hist = np.zeros(R)

    # Compute initial averaged model (equally weighted average of all client blocks)
    # This serves as baseline for computing initial loss
    W_bar_init = sum(W_blocks) / m

    # Compute and display initial global loss (weighted average across all clients)
    # Loss weights are proportional to client dataset sizes for fair aggregation
    init_loss = 0.0
    for i in range(m):
        X_i, y_i = client_datasets[i]
        if len(X_i) == 0:
            continue
        init_loss += (n_k[i] / n_total) * cross_entropy_loss(X_i, y_i, W_bar_init)
        local_loss_hist[0, i] = cross_entropy_loss(X_i, y_i, W_blocks[i])



    print(f"[Before training] Global Loss={init_loss:.4f}")

    # =========================================================================
    # MAIN FEDERATED TRAINING LOOP (Algorithm 2, Lines 1-32)
    # =========================================================================
    for r in range(R):

        # ======================================================================
        # ROUND r INITIALIZATION
        # ======================================================================
        
        # PENALTY SCHEDULE: ρ_r = ρ_base * (r + 10000)^0.25
        # The schedule grows slowly with rounds to gradually enforce constraints
        # Adding 10000 offset prevents high penalties very early when models are unstable
        # Exponent 0.25 provides polynomial growth (slow increase) as rounds progress
        # Motivation: Start with loose constraints, tighten over time to guide convergence
        rho_r = rho_base * (r + 10000) ** 0.25
        rho_hist[r] = rho_r

        # CLIENT SAMPLING: Select subset of clients for this round
        # (Simulates realistic federated settings where not all clients participate each round)
        if client_fraction >= 1.0:
            # Use all m clients if client_fraction = 1.0
            S_r = np.arange(m)
        else:
            # Randomly sample ceil(m * client_fraction) clients without replacement
            # client_fraction < 1.0 models client stragglers/unavailability
            s_size = max(1, int(m * client_fraction))
            S_r = np.random.choice(m, size=s_size, replace=False)

        # Create snapshot of current model blocks at start of round
        # These snapshots are used as baseline for all clients in this round
        # Ensures synchronous updates (all clients compute with same global state)
        W_snapshot = [W.copy() for W in W_blocks]

        # Compute sum of all parameter blocks: sum_all = W_1 + W_2 + ... + W_m
        # This is used to calculate the global averaged model and "sum_other" for each client
        sum_all = np.zeros_like(W_snapshot[0])
        for j in range(m):
            sum_all += W_snapshot[j]

        # COMMUNICATION-EFFICIENT STORAGE:
        # updated_block[i] = W_i after local training (block owned by client i)
        # D_sum[i] = accumulated gradient sums for client i's block
        # These are the only values communicated from clients to server
        updated_block = {}
        D_sum = {}

        # ======================================================================
        # PHASE 1: CLIENT UPDATES (Algorithm 2, Lines 5-23: Clients Update)
        # ======================================================================
        
        for i in S_r:
            # Only optimize if client has data; skip empty clients
            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                # Handle empty client: copy block unchanged, zero gradients
                updated_block[i] = W_snapshot[i].copy()
                D_sum[i] = np.zeros_like(W_snapshot[i])
                continue

            # CLIENT i'S LOCAL BLOCK: W_i initially equals the snapshot (Algorithm 2 line 7)
            W_i = W_snapshot[i].copy()
            # SUM OF OTHER BLOCKS: sum_other = sum_j≠i W_j (used to compute global avg z)
            # This is communication-efficient: instead of communicating all blocks,
            # server can compute this as: sum_other = sum_all - W_i
            sum_other = sum_all - W_snapshot[i]

            # GRADIENT ACCUMULATOR: Track sum of gradient blocks across all K steps
            # This will be communicated to server for server-side block updates
            D_acc = np.zeros_like(W_i)

            # Compute mini-batch cycling parameters
            n = len(X_i)  # Number of samples for this client
            steps_per_epoch = int(np.ceil(n / batch_size))  # Mini-batches per full epoch

            # Shuffle data once before local training (improves stochastic gradient quality)
            X_i, y_i = shuffle(X_i, y_i)

            # LOCAL TRAINING LOOP: K update steps (Algorithm 2, Line 9)
            # Each step computes a mini-batch gradient and updates W_i
            for step in range(K):
                # MINI-BATCH SELECTION: Cycle through mini-batches without replacement
                # After returning to start of epoch, reshuffle data for next epoch
                batch_idx = step % steps_per_epoch  # Which mini-batch in current epoch

                start = batch_idx * batch_size
                end = start + batch_size

                # When starting new epoch (batch_idx == 0) and not first iteration,
                # reshuffle data to randomize mini-batch order in next epoch
                if batch_idx == 0 and step > 0:
                    X_i, y_i = shuffle(X_i, y_i)

                # Extract mini-batch
                Xb = X_i[start:end]
                yb = y_i[start:end]

                if len(Xb) == 0:
                    continue

                # GLOBAL AVERAGED MODEL (for this mini-batch, instant):
                # z = (sum_other + W_i) / m = average of all m client blocks at this step
                # Using this ensures clients compute gradients w.r.t. shared reference model
                z = (sum_other + W_i) / m

                # GRADIENT COMPUTATION (Algorithm 2, Line 10):
                # grad_z = \nabla f_i(z) computed on mini-batch (Xb, yb)
                # This gradient is w.r.t. the global model z
                grad_z = compute_gradient(Xb, yb, z)
                # BLOCK-SPECIFIC GRADIENT (Algorithm 2, Line 11):
                # d_block = (1/m) * grad_z
                # Scaling by 1/m is standard in federated averaging to handle number of clients
                d_block = grad_z / m

                # CONSTRAINT PENALTY GRADIENT (Algorithm 2, Line 12):
                # pen = \nabla g_i(h_i(W_i)) - smoothed penalty signal
                # This enforces client's constraint h_i(W_i) \leq 0 via gradient
                # Only provides penalty when constraint is violated (h_i > 0)
                pen = grad_g_i(
                    W_i,
                    Xb,
                    yb,
                    gamma_reg=gamma_reg,
                    eps_i=eps_list[i],
                    lam=lam
                )

                # PENALIZED UPDATE DIRECTION (Algorithm 2, Line 12):
                # v_i = d_block + ρ_r * pen
                # Combines data fitting (d_block) and constraint satisfaction (ρ_r * pen)
                # ρ_r weight increases over rounds to gradually enforce constraints
                v_i = d_block + rho_r * pen

                # GRADIENT CLIPPING (Stability):
                # If ||v_i|| > clip threshold, scale down to prevent large updates
                # Helps stabilize training when gradients become very large
                # Prevents exploding gradients that could destabilize convergence
                v_norm = np.linalg.norm(v_i)
                if v_norm > clip:
                    v_i *= clip / (v_norm + 1e-12)

                # LOCAL PARAMETER UPDATE (Algorithm 2, Line 15):
                # W_i ← W_i - γ_l * v_i
                # Standard gradient descent with learning rate γ_l
                # Updates the block owned by client i
                W_i -= gamma_l * v_i

                # ACCUMULATE GRADIENT (for server communication):
                # Track sum of d_block values for this client's owned block
                # Server will use this to update other clients' blocks
                D_acc += d_block

            # STORE CLIENT UPDATE RESULTS for server phase:
            # updated_block[i] = W_i after K local gradient steps
            # D_sum[i] = \sum_{l=1}^{K} d_i,l (accumulated gradient for client i)
            updated_block[i] = W_i
            D_sum[i] = D_acc

        # ======================================================================
        # PHASE 2: SERVER UPDATES (Algorithm 2, Lines 24-31: Server Update)
        # ======================================================================
        
        # Initialize new blocks (will be updated using communication-efficient rule)
        # new_blocks[j] = updated parameter block for client j after server aggregation
        new_blocks = [W_snapshot[j].copy() for j in range(m)]

        # COMMUNICATION-EFFICIENT AGGREGATION:
        # Compute total gradient sum across all clients (for non-owned block updates)
        # total_D = sum_i D_sum[i] = sum of all accumulated gradients
        # This is used in the server update rule to adjust blocks not owned by participating clients
        total_D = np.zeros_like(W_snapshot[0])
        for i in D_sum:
            total_D += D_sum[i]

        # SERVER BLOCK UPDATE RULE (Algorithm 2, Line 29):
        # For each block j ("non-owned" by server, but we update all):
        # x_j^{t+1} = (x_{owner,j}^{t+1} + (m-1)*x_snap_j - \gamma_l * \sum_{i\neq j} D_i) / m
        #
        # INTUITION:
        # - x_{owner,j}^{t+1}: The updated block from client who owns it (client j)
        # - (m-1)*x_snap_j: Keep contribution from other m-1 clients at snapshot value
        # - \gamma_l * \sum_{i\neq j} D_i: Apply gradient corrections from other clients
        # This ensures blocks updated by gradients from clients who don't own them,
        # while preserving update made by owning client. Communication-efficient because
        # only D_sum communication needed, not full parameter blocks.
        for j in range(m):
            # BLOCK OWNER'S UPDATE:
            # x_{owner,j} = the version of block j after client j's local training
            # If client j wasn't sampled (j not in S_r), use snapshot unchanged
            x_snap_j = W_snapshot[j]
            x_owner = updated_block[j] if j in updated_block else x_snap_j

            # GRADIENT SUM EXCLUDING BLOCK j's OWNER:
            # sum_D_excl_j = total_D - D_sum[j] = \sum_{i\neq j} D_i
            # These are gradients computed on data from all clients except j
            # Used for cross-client influence: client i's data shapes block j
            sum_D_excl_j = total_D - (D_sum[j] if j in D_sum else 0.0)

            # FINAL BLOCK UPDATE (Server aggregation rule from Algorithm 2 Line 29):
            # new_blocks[j] = (x_owner + (m-1)*x_snap + -\gamma_l * sum_D_excl_j) / m
            new_blocks[j] = (
                x_owner                           # Client j's updated block (if participated, else snapshot)
                + (m - 1) * x_snap_j              # Keep prior blocks from other m-1 clients
                - gamma_l * sum_D_excl_j          # Apply cross-client gradient influence
            ) / m  # Normalize by m clients

        # UPDATE GLOBAL STATE with newly computed blocks
        W_blocks = new_blocks

        # ======================================================================
        # PHASE 3: EVALUATION & METRICS COLLECTION
        # ======================================================================
        
        # COLLECT PER-CLIENT METRICS (personalized model performance on local data):
        # Each client's block forms a personalized model, evaluate it on their own data
        for i in range(m):
            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            # local_loss_hist[r, i] = cross-entropy loss of client i's model on client i's data
            # This measures how well client i's personalized model fits their local data
            local_loss_hist[r, i] = cross_entropy_loss(X_i, y_i, W_blocks[i])
            # local_acc_hist[r, i] = classification accuracy of client i's model on client i's data
            # Complements loss metric with accuracy (more interpretable metric)
            local_acc_hist[r, i] = compute_accuracy(X_i, y_i, W_blocks[i])

        # EVALUATE CONSTRAINT SATISFACTION (key metric for PCFedAvg):
        # For each client, compute how well their constraint h_i(W_i) \leq 0 is satisfied
        for i in range(m):
            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            # h_i(W_i) = f_i(W_i) + (\gamma/2)||W_i||^2 - \epsilon_i
            # Values < 0 = satisfied, > 0 = violated
            # This constraint limits overfitting by restricting per-client loss
            h = h_i_loss(
                W_blocks[i],
                X_i,
                y_i,
                gamma_reg=gamma_reg,
                eps_i=eps_list[i]
            )

            # Store constraint value and corresponding penalty value
            h_hist[r, i] = h
            # g_i(h) = penalty value for constraint violation h (smoothly approaches 0 when h \leq 0)
            g_hist[r, i] = g_value(h, lam=lam)

        # COMPUTE GLOBAL AVERAGED MODEL:
        # W_bar = (W_1 + W_2 + ... + W_m) / m = equally weighted average
        # This serves as the global model for evaluation
        W_bar = sum(W_blocks) / m

        # GLOBAL TRAINING ACCURACY (optional, if full training set provided):
        # Accuracy of global averaged model on entire training set
        # Shows how well the global consensus model generalizes across all clients
        if X_train is not None and y_train is not None:
            global_train_acc_hist[r] = compute_accuracy(X_train, y_train, W_bar)

        # COMPUTE GRADIENT-BASED CONVERGENCE DIAGNOSTICS:
        # Track gradient norms to monitor convergence (should decrease over rounds)
        grad_sum = np.zeros_like(W_bar)

        # Accumulate gradients across all clients (weighted by data size implicitly)
        for i in range(m):
            X_i, y_i = client_datasets[i]
            if len(X_i) == 0:
                continue
            grad_sum += compute_gradient(X_i, y_i, W_bar)

        # Average gradient across clients
        avg_grad = grad_sum / m
        # Frobenius norm of average gradient (Euclidean length of gradient vector)
        avg_grad_norm = float(np.linalg.norm(avg_grad))

        # Mean penalty across all clients (measure of constraint violation)
        # Low values indicate constraints are satisfied
        g_mean = float(np.nanmean(g_hist[r, :]))

        # Store diagnostic metrics
        gradnorm_hist[r] = avg_grad_norm  # Gradient magnitude
        gmean_hist[r] = g_mean             # Constraint violation severity
        # Combined metric for convergence: ||\nabla W_bar||_F + \rho_r * mean(g)
        # Lower values indicate better convergence (small gradients + satisfied constraints)
        metric_hist[r] = avg_grad_norm + rho_r * g_mean

        # COMPUTE PENALIZED OBJECTIVE VALUES (for training difficulty assessment):
        # Objective = data_loss + \rho_r * constraint_penalty at each client
        # Shows how hard the training problem is (high values = harder)
        obj_vals = []

        for i in range(m):
            # Only include clients with valid loss measurements
            if not np.isnan(local_loss_hist[r, i]):
                # Penalized objective at client i: f_i(W_i) + \rho_r * g_i(h_i(W_i))
                # Combines data fitting and constraint satisfaction into single metric
                obj_vals.append(
                    local_loss_hist[r, i] + rho_r * g_hist[r, i]
                )

        # Average penalized objective across all clients
        avg_obj_hist[r] = np.mean(obj_vals) if obj_vals else np.nan

        # GLOBAL LOSS COMPUTATION (weighted average across all clients):
        # This is the main metric for convergence: should monotonically decrease
        total_loss = 0.0

        for i in range(m):
            X_i, y_i = client_datasets[i]

            if len(X_i) == 0:
                continue

            # Weight by client dataset size: larger clients contribute more to global loss
            # Ensures loss metric reflects performance on full dataset, not just client average
            total_loss += (
                n_k[i] / n_total
            ) * cross_entropy_loss(X_i, y_i, W_bar)

        # Store global loss for round r
        losses.append(total_loss)

        # OPTIONAL: EVALUATE ON TEST SET
        test_info = ""

        if X_test is not None and y_test is not None:
            # Compute accuracy of global averaged model on held-out test set
            # Primary metric for generalization performance
            test_acc = compute_accuracy(X_test, y_test, W_bar)

            accs.append(test_acc)

            # Format test accuracy info for display
            test_info = f", Test Acc={test_acc*100:.2f}%"

        # PROGRESS LOGGING:
        # Print convergence info every `display_every` rounds
        if display_every and (r % display_every == 0):

            print(
                f"[PCFedAvg-CE-Blockwise] "
                f"Round {r+1:3d}: Global Loss={total_loss:.4f}{test_info}"
            )

    # =========================================================================
    # RETURN ALL TRAINING METRICS AND FINAL STATE
    # =========================================================================
    return (
        np.array(losses),                  # Shape (R,) - Global weighted loss per round [L_0, L_1, ..., L_R-1]
                                           #   Main convergence metric: should decrease over training
                                           #   Loss = (1/m) * sum_i (n_i/n_total) * f_i(W_bar_i)
        
        np.array(accs),                    # Shape (num_eval_rounds,) - Test set accuracy per evaluation
                                           #   Only populated if X_test provided
                                           #   Measures generalization on held-out test data
        
        W_blocks,                          # List of m arrays - Final personalized models per client
                                           #   W_blocks[i] = final parameters of client i's personalized model
                                           #   Each block initialized differently and trained on local constraints
        
        h_hist,                            # Shape (R, m) - Constraint violation h_i(W_i) per round and client
                                           #   h_hist[r, i] = h_i(W_i) at round r
                                           #   Negative = satisfied, Positive = violated
                                           #   Key metric: PCFedAvg should drive these toward negative (satisfied)
        
        g_hist,                            # Shape (R, m) - Penalty values g_i(h_i) per round and client
                                           #   g_hist[r, i] = g_i(h_i(W_i)) at round r
                                           #   Measures magnitude of constraint violation penalty
                                           #   Should decrease over time as constraints become satisfied
        
        metric_hist,                       # Shape (R,) - Combined convergence metric per round
                                           #   metric_hist[r] = ||grad W_bar||_F + rho_r * mean(g)
                                           #   Low values indicate good convergence (small gradients + satisfied constraints)
                                           #   Suitable for plotting: should decrease/plateau
        
        gradnorm_hist,                     # Shape (R,) - Average gradient norm per round
                                           #   gradnorm_hist[r] = ||grad W_bar||_F at round r
                                           #   Standard convergence metric from optimization
                                           #   Should monotonically decrease
        
        gmean_hist,                        # Shape (R,) - Mean constraint penalty across clients per round
                                           #   gmean_hist[r] = mean_i(g_i(h_i)) at round r
                                           #   Measures average severity of constraint violations
                                           #   Should decrease toward zero as training progresses
        
        local_loss_hist,                   # Shape (R, m) - Cross-entropy loss per client-round
                                           #   local_loss_hist[r, i] = f_i(W_i) at round r
                                           #   Personalized losses: how well each model fits its client's data
                                           #   NaN if client has no data
        
        local_acc_hist,                    # Shape (R, m) - Accuracy per client-round
                                           #   local_acc_hist[r, i] = accuracy of W_i on client i data at round r
                                           #   Complementary accuracy metric to loss
        
        global_train_acc_hist,             # Shape (R,) - Global model accuracy on full training set per round
                                           #   global_train_acc_hist[r] = accuracy of W_bar on all training data
                                           #   Only populated if X_train provided
        
        avg_obj_hist,                      # Shape (R,) - Average penalized objective per round
                                           #   avg_obj_hist[r] = mean_i(f_i(W_i) + rho_r * g_i)
                                           #   Combined data-fitting and constraint-satisfaction metric
                                           #   Lower = easier training problem
        
        rho_hist                           # Shape (R,) - Penalty schedule history
                                           #   rho_hist[r] = rho_r = penalty weight at round r
                                           #   Schedule: rho_r = rho_base * (r + 10000)^0.25
                                           #   Used for algorithm analysis and tuning
    )