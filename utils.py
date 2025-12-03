"""
utils.py
--------
Utility functions for:
- MNIST loading
- Softmax, loss, accuracy, gradients
- IID + non-IID dataset splitting
- Plotting loss curves
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch
from torchvision import datasets, transforms


# ============================================================
# DATA LOADING (MNIST)
# ============================================================

def load_mnist():
    """
    Loads MNIST dataset and returns flattened images with bias term.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    X_train = train_set.data.numpy().reshape(-1, 784) / 255.0
    X_test  = test_set.data.numpy().reshape(-1, 784) / 255.0

    # Append bias term = 1
    X_train = np.column_stack((X_train, np.ones(len(X_train))))
    X_test  = np.column_stack((X_test,  np.ones(len(X_test))))

    y_train = train_set.targets.numpy()
    y_test  = test_set.targets.numpy()

    return X_train, y_train, X_test, y_test


# ============================================================
# MATH UTILITIES
# ============================================================

def softmax(z):
    """
    Stable softmax function.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(X, y, w):
    """
    Computes cross-entropy loss for softmax regression.
    """
    logits = X @ w
    probs = np.clip(softmax(logits), 1e-10, 1)
    correct = probs[np.arange(len(X)), y]
    return -np.mean(np.log(correct))


def compute_accuracy(X, y, w):
    """
    Computes accuracy = (# correct) / total.
    """
    preds = np.argmax(softmax(X @ w), axis=1)
    return np.mean(preds == y)


def compute_gradient(X, y, w):
    """
    Computes gradient of multinomial logistic regression model.
    """
    logits = X @ w
    probs = softmax(logits)

    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(X)), y] = 1

    return (1 / len(X)) * (X.T @ (probs - y_onehot))


# ============================================================
# DATA SPLITTING FUNCTIONS
# ============================================================

def create_IID_clients(X, y, num_clients):
    """
    Splits data evenly and randomly (IID).
    """
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    size = len(X) // num_clients
    return [
        (X[i*size:(i+1)*size], y[i*size:(i+1)*size])
        for i in range(num_clients)
    ]


def non_iid_split(X, y, num_clients, alpha=0.5):
    """
    Dirichlet-based non-IID data split.
    """
    clients = [[] for _ in range(num_clients)]

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)
        shards = np.split(idx, cuts[:-1])

        for i in range(num_clients):
            clients[i].extend(shards[i])

    return [(X[np.array(idx)], y[np.array(idx)]) for idx in clients]


# ============================================================
# VISUALIZATION
# ============================================================

def plot_loss(losses, title="Loss Curve"):
    """
    Plots training loss vs round.
    """
    plt.plot(losses)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.show()
