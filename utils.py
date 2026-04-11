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
from sklearn.utils import shuffle
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


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
# VISUALIZATION (POSTER READY)
# ============================================================

def setup_poster_style():
    """
    Configures matplotlib global parameters for large, uniform poster plots.
    Call this once at the beginning of your main scripts.
    """
    plt.rcParams.update({
        "font.size": 30,              # Base text size
        "axes.titlesize": 40,         # Title size (stands out more)
        "axes.labelsize": 35,         # Axis label size
        "legend.fontsize": 24,        # Legend text size
        "xtick.labelsize": 30,        # X-axis tick numbers
        "ytick.labelsize": 30,        # Y-axis tick numbers
        "lines.linewidth": 3.5,       # Thicker lines for visibility from a distance
        "figure.figsize": (10, 7),    # Uniform aspect ratio and size
        "figure.autolayout": True,    # Prevents labels from getting cut off
        "axes.grid": True,            # Standardize grid overlay
        "grid.alpha": 0.6,            # Slightly softer grid lines
        "text.usetex": False          # Keep False unless LaTeX is installed on your machine
    })

def plot_curve(values, title, ylabel, xlabel="Round"):
    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig

def plot_clients_curves(values_rm, title, ylabel, xlabel="Round", max_clients=10, show_legend=True):
    values_rm = np.asarray(values_rm)
    R, m = values_rm.shape

    fig, ax = plt.subplots()
    for i in range(min(m, max_clients)):
        ax.plot(values_rm[:, i], label=f"client {i}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_legend and m <= max_clients:
        ax.legend(loc="best", frameon=True, ncol=2 if m > 5 else 1)
    return fig

def plot_compare_curves(curves, title, ylabel, xlabel="Round", show_legend=True):
    # Notice we removed figsize and hardcoded fontsizes here.
    # It will now inherit from setup_poster_style() automatically.
    fig, ax = plt.subplots()

    for name, vals in curves.items():
        # X-axis is generated dynamically based on length of vals so it scales correctly
        x_vals = np.arange(1, len(vals) + 1)
        ax.plot(x_vals, vals, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_legend:
        ax.legend(loc="best", frameon=True)

    return fig