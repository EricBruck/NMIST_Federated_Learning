from utils import load_mnist, create_IID_clients, non_iid_split, plot_loss, compute_accuracy
from fedvi import fedVI
import numpy as np

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Create IID clients
num_clients = 3
clients = non_iid_split(X_train, y_train, num_clients)

# Initialize FedVI blocks
global_blocks = [
    np.random.randn(X_train.shape[1], 10) * 0.01
    for _ in range(num_clients)
]

# Eta schedule
eta_schedule = [0.0] * 10

# Run FedVI
losses, accs, final_blocks = fedVI(
    clients, global_blocks,
    R=10, H=1,
    gamma_l=0.05, lambda_m=0.01,
    eta_schedule=eta_schedule,
    X_test=X_test, y_test=y_test
)

# Convert blocks → global model
w_final = sum(final_blocks) / len(final_blocks)

# Evaluate
acc = compute_accuracy(X_test, y_test, w_final)
print(f"[FedVI] Final Accuracy = {acc*100:.2f}%")

# Plot
plot_loss(losses, "FedVI Loss Curve")
