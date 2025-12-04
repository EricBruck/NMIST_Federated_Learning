import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import torch
from torchvision import datasets, transforms

# ============================================================
# DATA LOADING (MNIST)
# ============================================================
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    X_train = train_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_test = test_dataset.targets.numpy()

    # Add bias term
    X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
    X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))

    return X_train, y_train, X_test, y_test


# ============================================================
# MODEL UTILS
# ============================================================
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(X, y, w):
    logits = X @ w
    probs = softmax(logits)
    probs = np.clip(probs, 1e-10, 1.0)

    n = len(X)
    correct_probs = probs[np.arange(n), y]
    return -np.mean(np.log(correct_probs))


def compute_accuracy(X, y, w):
    logits = X @ w
    preds = np.argmax(softmax(logits), axis=1)
    return np.mean(preds == y)


def compute_gradient(X, y, w):
    n = len(X)
    logits = X @ w
    probs = softmax(logits)

    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), y] = 1

    grad = (1.0 / n) * (X.T @ (probs - y_onehot))
    return grad


# ============================================================
# CLIENT UPDATE (FedAvg)
# ============================================================
def client_update(w_init, X, y, gamma=0.01, H=1, batch_size=32):
    w = w_init.copy()
    n = len(X)

    for epoch in range(H):
        X, y = shuffle(X, y)

        for i in range(0, n, batch_size):
            X_b = X[i : i + batch_size]
            y_b = y[i : i + batch_size]

            if len(X_b) == 0:
                continue

            grad = compute_gradient(X_b, y_b, w)
            w -= gamma * grad

    return w


# ============================================================
# CLIENT UPDATE (FedVI)
# ============================================================
def fedVI_client_update(
    w_init,
    X,
    y,
    global_blocks,
    eta_r=0.0,
    gamma_l=0.01,
    lambda_m=0.1,
    H=1,
    batch_size=32,
):
    w = w_init.copy()
    n = len(X)

    # Consensus model: average over blocks
    block_avg = sum(global_blocks) / len(global_blocks)

    for epoch in range(H):
        X, y = shuffle(X, y)

        for i in range(0, n, batch_size):
            X_b = X[i : i + batch_size]
            y_b = y[i : i + batch_size]

            if len(X_b) == 0:
                continue

            grad = compute_gradient(X_b, y_b, w)

            # variance scaling
            grad = (1.0 + eta_r) * grad

            # FedVI-style regularization (pull toward consensus)
            reg_term = lambda_m * (w - block_avg)

            total_grad = grad + reg_term
            w -= gamma_l * total_grad

    return w


# ============================================================
# FEDERATED AVERAGING (FedAvg)
# ============================================================
def federated_averaging(
    client_datasets,
    w_init,
    R,
    H,
    gamma,
    batch_size,
    X_test=None,
    y_test=None,
    display_every=5,
):
    m = len(client_datasets)
    n_k = np.array([len(df) for df in client_datasets])
    n_total = np.sum(n_k)

    w_global = w_init.copy()
    global_losses = []

    # Extract client data from DataFrames
    client_X = [df.iloc[:, :-1].values for df in client_datasets]
    client_y = [df.iloc[:, -1].values.astype(int) for df in client_datasets]

    for r in range(R + 1):
        # compute global loss
        total_loss = 0.0
        for k in range(m):
            loss_k = cross_entropy_loss(client_X[k], client_y[k], w_global)
            total_loss += (n_k[k] / n_total) * loss_k

        global_losses.append(total_loss)

        # display
        if r % display_every == 0:
            test_info = ""
            if X_test is not None and y_test is not None:
                test_loss = cross_entropy_loss(X_test, y_test, w_global)
                test_acc = compute_accuracy(X_test, y_test, w_global)
                test_info = f", Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%"
            print(f"[FedAvg] Round {r:3d}: Global Loss={total_loss:.4f}{test_info}")

        if r == R:
            break

        # client SGD
        client_models = []
        for k in range(m):
            w_k = client_update(
                w_global,
                client_X[k],
                client_y[k],
                gamma=gamma,
                H=H,
                batch_size=batch_size,
            )
            client_models.append(w_k)

        # FedAvg weighted aggregation
        w_global = sum((n_k[k] / n_total) * client_models[k] for k in range(m))

    return np.array(global_losses), w_global


# ============================================================
# FEDVI TRAINING LOOP
# ============================================================
def fedVI(
    client_datasets,
    global_blocks,
    R,
    H,
    gamma_l,
    lambda_m,
    batch_size=32,
    client_fraction=1.0,
    eta_schedule=None,
    X_test=None,
    y_test=None,
    display_every=1,
):
    
    num_clients = len(client_datasets)
    n_k = np.array([len(X) for (X, _) in client_datasets])

    global_losses = []
    test_accs = []

    for r in range(R):
        # choose active clients
        if client_fraction >= 1.0:
            S_r = np.arange(num_clients)
        else:
            m_active = max(1, int(client_fraction * num_clients))
            S_r = np.random.choice(num_clients, size=m_active, replace=False)

        # snapshot of global blocks
        blocks_snapshot = [b.copy() for b in global_blocks]
        updated_blocks = {}

        # per-round eta
        if eta_schedule is not None:
            eta_r = eta_schedule[r]
        else:
            eta_r = 0.0

        # client updates
        for m in S_r:
            X_m, y_m = client_datasets[m]

            w_m = fedVI_client_update(
                w_init=blocks_snapshot[m],
                X=X_m,
                y=y_m,
                global_blocks=blocks_snapshot,
                eta_r=eta_r,
                gamma_l=gamma_l,
                lambda_m=lambda_m,
                H=H,
                batch_size=batch_size,
            )
            updated_blocks[m] = w_m

        # update global blocks for active clients
        for m in S_r:
            global_blocks[m] = updated_blocks[m]

        # compute global model as average of blocks
        w_global = sum(global_blocks) / len(global_blocks)

        # compute global loss over all client data
        total_loss = 0.0
        n_total = np.sum(n_k)
        for k in range(num_clients):
            X_k, y_k = client_datasets[k]
            loss_k = cross_entropy_loss(X_k, y_k, w_global)
            total_loss += (n_k[k] / n_total) * loss_k

        global_losses.append(total_loss)

        test_info = ""
        if X_test is not None and y_test is not None:
            test_loss = cross_entropy_loss(X_test, y_test, w_global)
            test_acc = compute_accuracy(X_test, y_test, w_global)
            test_accs.append(test_acc)
            test_info = f", Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%"

        if r % display_every == 0:
            print(f"[FedVI] Round {r:3d}: Global Loss={total_loss:.4f}{test_info}")

    return np.array(global_losses), np.array(test_accs), global_blocks


# ============================================================
# DATA SPLITTING
# ============================================================
def create_IID_clients(X, y, num_clients):
    n = len(X)
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]

    size = n // num_clients
    clients = []

    for i in range(num_clients):
        s, e = i * size, (i + 1) * size
        clients.append((X[s:e], y[s:e]))

    return clients


def non_iid_split(X, y, num_clients, alpha=0.5):
    """
    Dirichlet-based non-IID split.
    """
    clients = [[] for _ in range(num_clients)]
    classes = np.unique(y)

    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)

        proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha)

        split_points = (np.cumsum(proportions) * len(idx)).astype(int)
        split_indices = np.split(idx, split_points[:-1])

        for i in range(num_clients):
            clients[i].extend(split_indices[i])

    return [(X[np.array(idx)], y[np.array(idx)]) for idx in clients]


# ============================================================
# VISUALIZATION
# ============================================================
def plot_loss(losses, title="Global Loss per Round"):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # load data once
    X_train, y_train, X_test, y_test = load_mnist()

    num_clients = 3
    client_splits = non_iid_split(X_train, y_train, num_clients)


    # ===================== FedVI =============================
    # re-use the same data splits but as (X, y) tuples
    client_datasets_fedvi = client_splits  

    np.random.seed(42)
    global_blocks_init = [np.random.randn(X_train.shape[1], 10) * 0.01 for _ in range(num_clients)]

    R_fedvi = 20
    eta_schedule = [0.1 for _ in range(R_fedvi)] 

    losses_fedvi, accs_fedvi, global_blocks_final = fedVI(
        client_datasets=client_datasets_fedvi,
        global_blocks=[b.copy() for b in global_blocks_init],
        R=R_fedvi,
        H=20,
        gamma_l=0.05,
        lambda_m=0.01,
        batch_size=64,
        client_fraction=0.40,
        eta_schedule=eta_schedule,
        X_test=X_test,
        y_test=y_test,
        display_every=1,
    )

    W_fedvi = sum(global_blocks_final) / len(global_blocks_final)
    final_acc = compute_accuracy(X_test, y_test, W_fedvi)
    print(f"[FedVI] Final Test Accuracy: {final_acc*100:.2f}%")

    plot_loss(losses_fedvi, title="FedVI Global Loss")


