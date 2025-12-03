from utils import load_mnist, create_IID_clients, plot_loss, compute_accuracy
from fedavg import federated_averaging
import numpy as np

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Create IID clients
num_clients = 3
clients = create_IID_clients(X_train, y_train, num_clients)

# Initialize global model
np.random.seed(42)
w_init = np.random.randn(X_train.shape[1], 10) * 0.01

# Run FedAvg
losses, w_final = federated_averaging(
    clients, w_init,
    R=5, H=1, gamma=0.1, batch_size=32,
    X_test=X_test, y_test=y_test
)

# Evaluate
acc = compute_accuracy(X_test, y_test, w_final)
print(f"[FedAvg] Final Accuracy = {acc*100:.2f}%")

# Plot loss
plot_loss(losses, "FedAvg Loss Curve")
