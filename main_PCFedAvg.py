
#main_FedVI.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import load_mnist, non_iid_split, plot_curve, plot_clients_curves, compute_accuracy
from PCFedAvg import pcfedavg_blockwise_efficient, estimate_epsilons


# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Clients
num_clients = 4
clients = non_iid_split(X_train, y_train, num_clients, alpha=0.1)

# Initialize per-client blocks (each client has its own model W_i)
d = X_train.shape[1]
C = 10
W_blocks = [np.random.randn(d, C) * 0.01 for _ in range(num_clients)]

R = 100
K = 5
gamma_l = 0.01
rho_base=1.0
eps_multiplier = 1.25
lam=1.0
gamma_reg=5e-4
batch_size = 128
client_fraction = 1.0

W_bar0 = sum(W_blocks) / len(W_blocks)
eps_list = estimate_epsilons(clients, W_init=W_bar0, multiplier = eps_multiplier, warmup_epochs=2, lr=0.01, batch_size=64)

losses, accs, final_blocks, h_hist, g_hist, metric_hist, gradnorm_hist, gmean_hist = pcfedavg_blockwise_efficient(
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
print(f"[PCFedAvg-Blockwise] Final Global Loss = {losses[-1]:.4f}")

# Average across clients each round (ignore NaNs if any client is empty)
h_avg = np.nanmean(h_hist, axis=1)   # shape (R,)
g_avg = np.nanmean(g_hist, axis=1)   # shape (R,)



plot_curve(losses, "Global loss f over rounds", "Loss")
plot_curve(h_avg, "Average constraint value across clients", "avg h_i(x_i)")
plot_curve(g_avg, "Average infeasibility across clients", "avg g_{i,λ}(x_i)")


plot_curve(gradnorm_hist, "|| (1/m) Σ ∇f_i(W_bar) || over rounds", "norm")
plot_curve(metric_hist, "Gradient Norm Metric: ||avg grad|| + ρ·avg g(h)", "value")
plt.show()
