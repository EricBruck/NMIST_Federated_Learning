# Federated Learning on MNIST

This project implements Federated Averaging (FedAvg) using a NumPy-based
logistic regression model and MNIST loaded through PyTorch.


## 📦 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/NMIST_Federated_Learning.git
cd NMIST_Federated_Learning

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
# Linux / macOS / ChromeOS:
source venv/bin/activate
# Windows (PowerShell):
# venv\Scripts\Activate.ps1
# Windows (CMD):
# venv\Scripts\activate.bat

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Run the Federated Learning project
python main.py
