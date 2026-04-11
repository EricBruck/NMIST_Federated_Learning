# main_compare.py
import numpy as np
import matplotlib.pyplot as plt

# Import the new setup_poster_style and the unified plot_compare_curves
from utils import load_mnist, non_iid_split, compute_accuracy, setup_poster_style, plot_compare_curves
from PCFedAvg import pcfedavg_blockwise_efficient, estimate_epsilons
from fedprox import fedprox_train
from ditto import ditto_train
from scaffold import scaffold_train

def main():
    # -------------------------
    # Apply poster styling globally
    # -------------------------
    setup_poster_style()

    # -------------------------
    # Shared experiment settings
    # -------------------------
    seed = 0
    np.random.seed(seed)

    num_clients = 5
    alpha = 0.1

    R = 120
    K = 500
    batch_size = 64
    client_fraction = 1.0

    d = 785  # MNIST flattened 784 + bias
    C = 10

    # -------------------------
    # Load data + split clients
    # -------------------------
    X_train, y_train, X_test, y_test = load_mnist()
    clients = non_iid_split(X_train, y_train, num_clients, alpha=alpha)

    # Use the SAME initial global model for FedProx/Ditto
    w0 = np.random.randn(d, C) * 0.01

    # Use SAME initial blocks for PCFedAvg (each client block starts near w0)
    W_blocks0 = [w0.copy() for _ in range(num_clients)]

    # -------------------------
    # Run PCFedAvg
    # -------------------------
    pcf_gamma_l = 0.01  
    rho_base = 1.0       
    eps_multiplier = 1.25  
    lam = 0.5
    gamma_reg = 1e-4

    eps_list = estimate_epsilons(clients, W_init=w0, multiplier=eps_multiplier, warmup_epochs=1, lr=0.01, batch_size=64)

    losses_pcf, accs_pcf, final_blocks_pcf, h_hist, g_hist, metric_hist, gradnorm_hist, gmean_hist, local_loss_hist, local_acc_hist, global_train_acc_hist, avg_obj_hist, rho_hist = pcfedavg_blockwise_efficient(
        client_datasets=clients,
        W_blocks=[W.copy() for W in W_blocks0],
        R=R,
        K=K,
        gamma_l=pcf_gamma_l,
        rho_base=rho_base,
        lam=lam,
        gamma_reg=gamma_reg,
        eps_list=eps_list,
        batch_size=batch_size,
        client_fraction=client_fraction,
        X_test=X_test,
        y_test=y_test,
        display_every=1,
        X_train=None,
        y_train=None,
    )
    w_pcf = sum(final_blocks_pcf) / len(final_blocks_pcf)
    final_acc_pcf = compute_accuracy(X_test, y_test, w_pcf)
    print(f"[PCFedAvg] Final Test Acc = {final_acc_pcf*100:.2f}%")
    print(f"[PCFedAvg-Blockwise] Final Global Loss = {losses_pcf[-1]:.4f}")


    # -------------------------
    # Run FedProx
    # -------------------------
    fedprox_lr = 0.01
    fedprox_mu = 0.01

    losses_fp, accs_fp, w_fp = fedprox_train(
        client_datasets=clients,
        w_init=w0.copy(),
        R=R,
        K=K,
        lr=fedprox_lr,
        mu=fedprox_mu,
        batch_size=batch_size,
        client_fraction=client_fraction,
        X_test=X_test,
        y_test=y_test,
        display_every=0,
    )
    final_acc_fp = compute_accuracy(X_test, y_test, w_fp)
    print(f"[FedProx] Final Test Acc = {final_acc_fp*100:.2f}%")
    print(f"[FedProx] Final Global Loss = {losses_fp[-1]:.4f}")


    # -------------------------
    # Run Ditto
    # -------------------------
    ditto_lr_global = 0.01
    ditto_lr_personal = 0.01
    ditto_mu = 0.01
    ditto_lam = 0.1

    losses_dt, accs_dt, w_dt, v_list, per_loss_rm, per_acc_rm = ditto_train(
        client_datasets=clients,
        w_init=w0.copy(),
        R=R,
        K=K,
        lr_global=ditto_lr_global,
        lr_personal=ditto_lr_personal,
        mu=ditto_mu,
        lam=ditto_lam,
        batch_size=batch_size,
        client_fraction=client_fraction,
        X_test=X_test,
        y_test=y_test,
        display_every=0,
    )
    final_acc_dt = compute_accuracy(X_test, y_test, w_dt)
    print(f"[Ditto] Final Global Test Acc = {final_acc_dt*100:.2f}%")
    print(f"[Ditto] Final Global Loss = {losses_dt[-1]:.4f}")


    # -------------------------
    # Run SCAFFOLD
    # -------------------------
    scaffold_lr = 0.01
    scaffold_clip = 5.0  

    losses_sc, accs_sc, w_sc = scaffold_train(
        client_datasets=clients,
        w_init=w0.copy(),
        R=R,
        K=K,
        lr=scaffold_lr,
        batch_size=batch_size,
        client_fraction=client_fraction,
        X_test=X_test,
        y_test=y_test,
        display_every=0,   
        clip=scaffold_clip,
    )
    final_acc_sc = compute_accuracy(X_test, y_test, w_sc)
    print(f"[SCAFFOLD] Final Test Acc = {final_acc_sc*100:.2f}%")
    print(f"[SCAFFOLD] Final Global Loss = {losses_sc[-1]:.4f}")


    # Personalized mean acc (optional)
    per_acc_mean = np.nanmean(per_acc_rm, axis=1)

    # -------------------------
    # Plot overlays (using the shared function from utils.py)
    # -------------------------
    
    # We no longer need to pass `rounds` manually as `x` because plot_compare_curves 
    # handles dynamic x-axis generation internally now.
    
    plot_compare_curves(
        curves={
            "PNCFedAvg": losses_pcf,
            "FedProx": losses_fp,
            "Ditto": losses_dt,
            "SCAFFOLD": losses_sc
        },
        title=r"Global Training Loss vs Round",
        ylabel=r"Loss $f(x)$",
    )

    if len(accs_pcf) > 0 and len(accs_fp) > 0 and len(accs_dt) > 0 and len(accs_sc) > 0:
        plot_compare_curves(
            curves={
                "PNCFedAvg": accs_pcf,
                "FedProx": accs_fp,
                "Ditto": accs_dt,
                "SCAFFOLD": accs_sc,
            },
            title=r"Global Test Accuracy vs Round",
            ylabel=r"Accuracy",
        )
        
    plt.show()

if __name__ == "__main__":
    main()