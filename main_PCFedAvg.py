
#main_FedVI.py
import numpy as np
from utils import load_mnist, non_iid_split, plot_loss, plot_curve, plot_clients_curves, compute_accuracy
from PCFedAvg import pcfedavg_blockwise_efficient, estimate_epsilons

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Clients
num_clients = 4
clients = non_iid_split(X_train, y_train, num_clients, alpha=0.5)

# Initialize per-client blocks (each client has its own model W_i)
d = X_train.shape[1]
C = 10
W_blocks = [np.random.randn(d, C) * 0.01 for _ in range(num_clients)]

R = 60
K = 2
gamma_l = 0.005
rho_base=1.0
lam=0.5
gamma_reg=1e-4
batch_size = 128
client_fraction = 1.0

W_bar0 = sum(W_blocks) / len(W_blocks)
eps_list = estimate_epsilons(clients, W_init=W_bar0, warmup_epochs=1, lr=0.01, batch_size=64)

losses, accs, final_blocks, h_hist, g_hist = pcfedavg_blockwise_efficient(
    client_datasets = clients,
    W_blocks=W_blocks, 
    R=R,
    K=K,
    gamma_l=gamma_l,   
    rho_base=rho_base,        
    lam=lam,           
    gamma_reg=gamma_reg,        
    eps_list=eps_list,                
    batch_size=batch_size,
    client_fraction=client_fraction,
    X_test=X_test,
    y_test=y_test,
    display_every = 1
)

# Report final global model (mean of blocks)
W_final = sum(final_blocks) / len(final_blocks)
acc = compute_accuracy(X_test, y_test, W_final)
print(f"[PCFedAvg-Blockwise] Final Accuracy = {acc*100:.2f}%")

plot_curve(losses, "Global loss f over rounds", "Loss")
plot_clients_curves(h_hist, "Local constraint h_i(x_i) over rounds", "h_i(x_i)", max_clients=10)
plot_clients_curves(g_hist, "Infeasibility g_{i,λ}(x_i) over rounds", "g_{i,λ}(x_i)", max_clients=10)