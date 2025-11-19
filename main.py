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

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train = train_dataset.data.numpy().reshape(-1, 28*28) / 255.0
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy().reshape(-1, 28*28) / 255.0
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
    probs = np.clip(probs, 1e-10, 1)

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

    grad = (1/n) * (X.T @ (probs - y_onehot))
    return grad



# ============================================================
# CLIENT UPDATE (CORRECT FEDAVG VERSION)
# ============================================================
def client_update(w_init, X, y, gamma=0.01, H=1, batch_size=32):
    w = w_init.copy()
    n = len(X)

    for epoch in range(H):
        X, y = shuffle(X, y)

        for i in range(0, n, batch_size):
            X_b = X[i:i+batch_size]
            y_b = y[i:i+batch_size]

            grad = compute_gradient(X_b, y_b, w)
            w -= gamma * grad

    return w



# ============================================================
# FEDERATED AVERAGING (Corrected Colab version)
# ============================================================
def federated_averaging(client_datasets, w_init, R, H, gamma, batch_size,
                        X_test=None, y_test=None, display_every=5):

    m = len(client_datasets)
    n_k = np.array([len(df) for df in client_datasets])
    n_total = np.sum(n_k)

    w_global = w_init.copy()
    global_losses = []

    # Extract client data from DataFrames
    client_X = [df.iloc[:, :-1].values for df in client_datasets]
    client_y = [df.iloc[:, -1].values.astype(int) for df in client_datasets]

    for r in range(R+1):

        # compute global loss
        total_loss = 0
        for k in range(m):
            loss_k = cross_entropy_loss(client_X[k], client_y[k], w_global)
            total_loss += (n_k[k] / n_total) * loss_k

        global_losses.append(total_loss)

        # display
        if r % display_every == 0:
            test_info = ""
            if X_test is not None:
                test_loss = cross_entropy_loss(X_test, y_test, w_global)
                test_acc = compute_accuracy(X_test, y_test, w_global)
                test_info = f", Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%"
            print(f"Round {r:3d}: Global Loss={total_loss:.4f}{test_info}")

        if r == R:
            break

        # client SGD
        client_models = []
        for k in range(m):
            w_k = client_update(w_global, client_X[k], client_y[k], gamma, H, batch_size)
            client_models.append(w_k)

        # FedAvg weighted aggregation
        w_global = sum((n_k[k]/n_total) * client_models[k] for k in range(m))

    return np.array(global_losses), w_global



# ============================================================
# DATA SPLITTING (IID)
# ============================================================
def create_IID_clients(X, y, num_clients):
    n = len(X)
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]

    size = n // num_clients
    clients = []

    for i in range(num_clients):
        s, e = i * size, (i+1) * size
        clients.append((X[s:e], y[s:e]))

    return clients



# ============================================================
# VISUALIZATION
# ============================================================
def plot_loss(losses):
    plt.plot(losses)
    plt.title("Global Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def plot_accuracy(train_acc, test_acc):
    rounds = len(train_acc)
    plt.plot(range(rounds), train_acc, label="Train")
    plt.plot(range(rounds), test_acc, label="Test")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Rounds")
    plt.grid(True)
    plt.legend()
    plt.show()



# ============================================================
# MAIN
# ============================================================
# load data
X_train, y_train, X_test, y_test = load_mnist()

# convert to DataFrame (Colab-style)
df = pd.DataFrame(np.column_stack((X_train, y_train)))

# split into clients
num_clients = 3
client_splits = create_IID_clients(X_train, y_train, num_clients)
client_datasets = [pd.DataFrame(np.column_stack((X, y))) for X, y in client_splits]

# init model
np.random.seed(42)
W_init = np.random.randn(X_train.shape[1], 10) * 0.01

# run FL
losses, W_final = federated_averaging(
    client_datasets,
    W_init,
    R=5,
    H=1,
    gamma=0.1,
    batch_size=32,
    X_test=X_test,
    y_test=y_test
)

# final accuracy
print(f"Final Test Accuracy: {compute_accuracy(X_test, y_test, W_final)*100:.2f}%")

# visualization
plot_loss(losses)
