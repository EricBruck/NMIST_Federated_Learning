import numpy as np

from utils import load_mnist, non_iid_split, plot_compare_curves, compute_accuracy
from PCFedAvg import pcfedavg_blockwise_efficient, estimate_epsilons


def run_pcfedavg_for_K(
    clients, X_test, y_test, d, C,
    R, K, gamma_l, rho_base, lam, gamma_reg, batch_size, client_fraction, eps_mult=1.25,
    init_scale=0.01,
    seed=42,
):
    # Make runs comparable
    rng = np.random.default_rng(seed)

    # Same init distribution (but different random draw per K unless you fix it externally)
    W_blocks0 = [rng.standard_normal((d, C)) * init_scale for _ in range(len(clients))]
    W_bar0 = sum(W_blocks0) / len(W_blocks0)

    # eps depends on init + clients; recompute per run for consistency
    eps_list = estimate_epsilons(
        clients, W_init=W_bar0, multiplier = eps_mult, warmup_epochs=2, lr=0.01, batch_size=64
    )

    losses, accs, final_blocks, h_hist, g_hist, metric_hist, gradnorm_hist, gmean_hist = pcfedavg_blockwise_efficient(
        client_datasets=clients,
        W_blocks=W_blocks0,
        R=R,
        K=K,
        gamma_l=gamma_l,
        rho_base=rho_base,
        lam=lam,
        gamma_reg=gamma_reg,
        eps_list=eps_list,
        batch_size=batch_size,
        client_fraction=client_fraction,
        X_test=X_test,
        y_test=y_test,
        display_every=1,   # set to 0 or a big number if you don't want spam prints
    )

    W_final = sum(final_blocks) / len(final_blocks)
    final_acc = compute_accuracy(X_test, y_test, W_final)

    return losses, accs, final_acc


def main():
    # --------------------------
    # Shared experiment settings
    # --------------------------
    seed = 42
    np.random.seed(seed)

    X_train, y_train, X_test, y_test = load_mnist()

    num_clients = 4
    alpha = 0.5
    clients = non_iid_split(X_train, y_train, num_clients, alpha=alpha)

    d = X_train.shape[1]
    C = 10

    # Keep these FIXED across K
    R = 100
    gamma_l = 0.005
    rho_base = 1.0
    eps_multiplier = 1.25
    lam = 0.5
    gamma_reg = 1e-4
    batch_size = 128
    client_fraction = 1.0

    # --------------------------
    # Sweep K
    # --------------------------
    Ks = [1, 5, 10]

    loss_curves = {}
    acc_curves = {}

    for K in Ks:
        print(f"\nRun {Ks.index(K)+1} of PCFedAvg with K={K} local steps per round...")
        losses, accs, final_acc = run_pcfedavg_for_K(
            clients=clients,
            X_test=X_test, y_test=y_test,
            d=d, C=C,
            R=R, K=K,
            gamma_l=gamma_l,
            rho_base=rho_base,
            lam=lam,
            gamma_reg=gamma_reg,
            batch_size=batch_size,
            client_fraction=client_fraction,
            eps_mult=eps_multiplier,
            init_scale=0.01,
            seed=seed,   # same seed so init is comparable
        )

        loss_curves[f"K={K} (loss)"] = losses
        acc_curves[f"K={K} (acc)"] = accs
        print(f"[PCFedAvg K={K}] Final Acc = {final_acc*100:.2f}%")

    # --------------------------
    # Plot comparisons
    # --------------------------
    plot_compare_curves(
        loss_curves,
        title="PCFedAvg: Global Train Loss vs Round for Different K",
        ylabel="Loss",
        xlabel="Round",
        show_legend=True,
    )

    plot_compare_curves(
        acc_curves,
        title="PCFedAvg: Test Accuracy vs Round for Different K",
        ylabel="Accuracy",
        xlabel="Round",
        show_legend=True,
    )

    # keep windows open at end when running as a script
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
