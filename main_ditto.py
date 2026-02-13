# main_Ditto.py
import numpy as np
from utils import load_mnist, non_iid_split, plot_curve, plot_clients_curves, compute_accuracy
from ditto import ditto_train

X_train, y_train, X_test, y_test = load_mnist()

num_clients = 4
clients = non_iid_split(X_train, y_train, num_clients, alpha=0.5)

d = X_train.shape[1]
C = 10
w0 = np.random.randn(d, C) * 0.01

R = 10
batch_size = 128
client_fraction = 1.0

# Ditto hyperparameters
lr_global = 0.01
lr_personal = 0.01
local_epochs_global = 1
local_epochs_personal = 1
mu = 0.01      # global prox weight
lam = 0.1      # personalization coupling strength

gloss, gacc, w_global, v_list, ploss_rm, pacc_rm = ditto_train(
    client_datasets=clients,
    w_init=w0,
    R=R,
    local_epochs_global=local_epochs_global,
    local_epochs_personal=local_epochs_personal,
    lr_global=lr_global,
    lr_personal=lr_personal,
    mu=mu,
    lam=lam,
    batch_size=batch_size,
    client_fraction=client_fraction,
    X_test=X_test,
    y_test=y_test,
    display_every=1,
)

print(f"[Ditto] Final Global Accuracy = {compute_accuracy(X_test, y_test, w_global)*100:.2f}%")

# Optional: average personalized accuracy at the end
final_personal_acc = np.nanmean(pacc_rm[-1])
print(f"[Ditto] Avg Personalized Acc (last round) = {final_personal_acc*100:.2f}%")

plot_curve(gloss, "Ditto: Global loss over rounds", "Loss")
plot_clients_curves(ploss_rm, "Ditto: Personalized loss (per client) over rounds", "Loss", max_clients=10, show_legend=True)

# If you want to see personalized accuracy curves too:
plot_clients_curves(pacc_rm, "Ditto: Personalized accuracy (per client) over rounds", "Accuracy", max_clients=10, show_legend=True)
