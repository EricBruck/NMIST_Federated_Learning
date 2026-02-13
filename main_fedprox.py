# main_FedProx.py
import numpy as np
from utils import load_mnist, non_iid_split, plot_curve, compute_accuracy
from fedprox import fedprox_train

X_train, y_train, X_test, y_test = load_mnist()

num_clients = 5
clients = non_iid_split(X_train, y_train, num_clients, alpha=0.5)

d = X_train.shape[1]
C = 10
w0 = np.random.randn(d, C) * 0.01

R = 80
local_epochs = 2
lr = 0.01
mu = 0.01
batch_size = 128
client_fraction = 1.0

losses, accs, w_final = fedprox_train(
    client_datasets=clients,
    w_init=w0,
    R=R,
    local_epochs=local_epochs,
    lr=lr,
    mu=mu,
    batch_size=batch_size,
    client_fraction=client_fraction,
    X_test=X_test,
    y_test=y_test,
    display_every=1,
)

print(f"[FedProx] Final Accuracy = {compute_accuracy(X_test, y_test, w_final)*100:.2f}%")
plot_curve(losses, "FedProx: Global loss over rounds", "Loss")
