import numpy as np
from utils import load_mnist, non_iid_split, plot_loss, compute_accuracy
from FedVI import pcfedavg_blockwise

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Clients
num_clients = 10
clients = non_iid_split(X_train, y_train, num_clients, alpha=0.5)

# Initialize per-client blocks (each client has its own model W_i)
d = X_train.shape[1]
C = 10
W_blocks = [np.random.randn(d, C) * 0.01 for _ in range(num_clients)]

R = 50
K = 3
gamma_l = 0.005
batch_size = 64

# eta schedule (paper uses eta_r multiplying the stochastic gradient term)
eta_schedule = np.linspace(1.0, 1.0, R)  # start constant; tune later

# rho thresholds (constraint): larger => weaker constraint
rho_list = [3.0 for _ in range(num_clients)]

losses, accs, final_blocks = pcfedavg_blockwise(
    client_datasets=clients,
    W_blocks=W_blocks,
    R=R,
    K=K,
    gamma_l=gamma_l,
    batch_size=batch_size,
    client_fraction=1.0,
    eta_schedule=eta_schedule,
    rho_list=rho_list,
    X_test=X_test,
    y_test=y_test,
    display_every=1,
)

# Report final global model (mean of blocks)
W_final = sum(final_blocks) / len(final_blocks)
acc = compute_accuracy(X_test, y_test, W_final)
print(f"[PCFedAvg-Blockwise] Final Accuracy = {acc*100:.2f}%")

plot_loss(losses, "PCFedAvg-Blockwise Loss Curve")
