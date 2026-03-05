import numpy as np

from utils import (
    load_mnist, non_iid_split, plot_compare_curves,
    cross_entropy_loss, compute_accuracy, compute_gradient
)
from PCFedAvg import (
    pcfedavg_blockwise_efficient, estimate_epsilons,
    h_i_loss, g_value
)


# ----------------------------
# Helpers
# ----------------------------

def weighted_global_train_loss(client_datasets, w_global):
    n_k = np.array([len(X) for X, _ in client_datasets], dtype=float)
    n_total = np.sum(n_k) if np.sum(n_k) > 0 else 1.0
    total = 0.0
    for i, (X_i, y_i) in enumerate(client_datasets):
        if len(X_i) == 0:
            continue
        total += (n_k[i] / n_total) * cross_entropy_loss(X_i, y_i, w_global)
    return float(total)


def _mean_std_stack(curve_list):
    """
    curve_list: list of 1D arrays (same length)
    returns: (mean, std) 1D arrays
    """
    A = np.vstack([np.asarray(c) for c in curve_list])
    return A.mean(axis=0), A.std(axis=0)


def _mean_std_stack_2d(mat_list):
    """
    mat_list: list of 2D arrays (same shape)
    returns: (mean, std) arrays of same shape
    """
    A = np.stack([np.asarray(M) for M in mat_list], axis=0)  # (S, T, m)
    return A.mean(axis=0), A.std(axis=0)


def initial_metrics_for_run(
    clients, W_blocks0, W_bar0,
    X_train, y_train, X_test, y_test,
    rho_base, lam, gamma_reg, eps_list
):
    """
    Computes *round 0* metrics BEFORE training.
    Returns:
      init_loss, init_test_acc, init_train_acc,
      init_local_loss_vec (m,),
      init_h_vec (m,),
      init_g_vec (m,),
      init_avg_obj_scalar,
      init_metric_scalar
    """
    m = len(clients)

    # rho at r=0 (match your schedule)
    rho0 = rho_base * (0 + 10000) ** 0.25

    # global loss/acc at W_bar0
    init_loss = weighted_global_train_loss(clients, W_bar0)
    init_test_acc = compute_accuracy(X_test, y_test, W_bar0)
    init_train_acc = compute_accuracy(X_train, y_train, W_bar0) if X_train is not None and y_train is not None else np.nan

    # local loss per client at its OWN block x_i
    init_local_loss = np.full(m, np.nan, dtype=float)
    init_h = np.full(m, np.nan, dtype=float)
    init_g = np.full(m, np.nan, dtype=float)

    for i in range(m):
        X_i, y_i = clients[i]
        if len(X_i) == 0:
            continue
        init_local_loss[i] = cross_entropy_loss(X_i, y_i, W_blocks0[i])
        h = h_i_loss(W_blocks0[i], X_i, y_i, gamma_reg=gamma_reg, eps_i=eps_list[i])
        init_h[i] = h
        init_g[i] = g_value(h, lam=lam)

    # avg objective at round 0: mean_i [ f_i(x_i) + rho0*g_i ]
    obj_vals = []
    for i in range(m):
        if np.isnan(init_local_loss[i]) or np.isnan(init_g[i]):
            continue
        obj_vals.append(init_local_loss[i] + rho0 * init_g[i])
    init_avg_obj = float(np.mean(obj_vals)) if len(obj_vals) > 0 else np.nan

    # metric at round 0: ||avg grad|| + rho0*avg g
    grad_sum = np.zeros_like(W_bar0)
    for i in range(m):
        X_i, y_i = clients[i]
        if len(X_i) == 0:
            continue
        grad_sum += compute_gradient(X_i, y_i, W_bar0)
    avg_grad = grad_sum / m
    avg_grad_norm = float(np.linalg.norm(avg_grad))
    g_mean0 = float(np.nanmean(init_g))
    init_metric = avg_grad_norm + rho0 * g_mean0

    return (
        init_loss, init_test_acc, init_train_acc,
        init_local_loss,
        init_h, init_g,
        init_avg_obj,
        init_metric
    )


# ----------------------------
# Core runners
# ----------------------------

def run_pcfedavg_for_K(
    clients, X_train, y_train, X_test, y_test, d, C,
    R, K, epochs, gamma_l, rho_base, lam, gamma_reg, batch_size, client_fraction,
    eps_mult=1.25, init_scale=0.01, seed=42,
    display_every=0,
):
    rng = np.random.default_rng(seed)
    m = len(clients)

    # init blocks + averaged model
    W_blocks0 = [rng.standard_normal((d, C)) * init_scale for _ in range(m)]
    W_bar0 = sum(W_blocks0) / m

    # eps list from warmup
    eps_list = estimate_epsilons(
        clients, W_init=W_bar0, multiplier=eps_mult,
        warmup_epochs=2, lr=0.01, batch_size=64
    )

    # round 0 metrics (BEFORE training)
    (
        init_loss, init_test_acc, init_train_acc,
        init_local_loss_vec,
        init_h_vec, init_g_vec,
        init_avg_obj,
        init_metric
    ) = initial_metrics_for_run(
        clients=clients,
        W_blocks0=W_blocks0,
        W_bar0=W_bar0,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        rho_base=rho_base,
        lam=lam,
        gamma_reg=gamma_reg,
        eps_list=eps_list,
    )

    # run training (R rounds)
    (
        losses,
        accs,
        final_blocks,
        h_hist,
        g_hist,
        metric_hist,
        gradnorm_hist,
        gmean_hist,
        local_loss_hist,
        local_acc_hist,
        global_train_acc_hist,
        avg_obj_hist,
        rho_hist,
    ) = pcfedavg_blockwise_efficient(
        client_datasets=clients,
        W_blocks=W_blocks0,
        R=R,
        K=K,
        epochs=epochs,
        gamma_l=gamma_l,
        rho_base=rho_base,
        lam=lam,
        gamma_reg=gamma_reg,
        eps_list=eps_list,
        batch_size=batch_size,
        client_fraction=client_fraction,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        display_every=display_every,
    )

    # final acc on test
    W_final = sum(final_blocks) / m
    final_acc = compute_accuracy(X_test, y_test, W_final)

    # ---- prepend round-0 to global curves (now length R+1) ----
    losses = np.concatenate(([init_loss], losses))
    accs   = np.concatenate(([init_test_acc], accs))

    # global train acc curve: returned length R, prepend init
    global_train_acc = np.concatenate(([init_train_acc], global_train_acc_hist))

    # h/g: returned shape (R, m), prepend init vec -> (R+1, m)
    h_full = np.vstack([init_h_vec[None, :], h_hist])
    g_full = np.vstack([init_g_vec[None, :], g_hist])

    # local loss: returned shape (R, m), prepend init local loss vec
    local_loss_full = np.vstack([init_local_loss_vec[None, :], local_loss_hist])

    # avg objective: returned length R, prepend init
    avg_obj_full = np.concatenate(([init_avg_obj], avg_obj_hist))

    # metric: returned length R, prepend init
    metric_full = np.concatenate(([init_metric], metric_hist))

    return {
        "losses": losses,                       # (R+1,)
        "accs": accs,                           # (R+1,)
        "global_train_acc": global_train_acc,   # (R+1,)
        "h_full": h_full,                       # (R+1,m)
        "g_full": g_full,                       # (R+1,m)
        "local_loss_full": local_loss_full,     # (R+1,m)
        "avg_obj": avg_obj_full,                # (R+1,)
        "metric": metric_full,                  # (R+1,)
        "final_acc": float(final_acc),
    }


def run_pcfedavg_multiple_seeds(
    seeds,
    clients, X_train, y_train, X_test, y_test, d, C,
    R, K, epochs, gamma_l, rho_base, lam, gamma_reg, batch_size, client_fraction,
    eps_mult=1.25, init_scale=0.01,
):
    """
    Returns mean/std curves for K using multiple runs.
    Also returns mean/std per-client local loss matrix (R+1, m).
    """
    outs = []
    for sd in seeds:
        outs.append(run_pcfedavg_for_K(
            clients=clients,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            d=d, C=C,
            R=R, K=K, epochs=epochs,
            gamma_l=gamma_l,
            rho_base=rho_base,
            lam=lam,
            gamma_reg=gamma_reg,
            batch_size=batch_size,
            client_fraction=client_fraction,
            eps_mult=eps_mult,
            init_scale=init_scale,
            seed=sd,
            display_every=0,   # IMPORTANT: no per-run spam
        ))

    # mean/std for 1D curves
    keys_1d = ["losses", "accs", "global_train_acc", "avg_obj", "metric"]
    mean_1d, std_1d = {}, {}
    for k in keys_1d:
        mean_1d[k], std_1d[k] = _mean_std_stack([o[k] for o in outs])

    # mean/std for per-client local losses
    local_loss_mean, local_loss_std = _mean_std_stack_2d([o["local_loss_full"] for o in outs])
    h_mean, h_std = _mean_std_stack_2d([o["h_full"] for o in outs])
    g_mean, g_std = _mean_std_stack_2d([o["g_full"] for o in outs])

    mean_final_acc = float(np.mean([o["final_acc"] for o in outs]))
    std_final_acc  = float(np.std([o["final_acc"] for o in outs]))

    return (
        mean_1d, std_1d,
        local_loss_mean, local_loss_std,
        h_mean, h_std,
        g_mean, g_std,
        mean_final_acc, std_final_acc
    )


# ----------------------------
# Main
# ----------------------------

def main():
    seed = 42
    np.random.seed(seed)

    X_train, y_train, X_test, y_test = load_mnist()

    num_clients = 5
    alpha = 0.5
    clients = non_iid_split(X_train, y_train, num_clients, alpha=alpha)

    d = X_train.shape[1]
    C = 10

    # fixed across K
    R = 120
    epochs = 1
    gamma_l = 0.005
    rho_base = 1.0
    eps_multiplier = 1.25
    lam = 0.5
    gamma_reg = 1e-4
    batch_size = 64
    client_fraction = 1.0

    Ks = [1, 5, 10]

    # global comparison curves
    loss_curves = {}
    test_acc_curves = {}
    train_acc_curves = {}
    infeas_curves = {}
    constraint_curves = {}
    avg_obj_curves = {}
    metric_curves = {}

    # store per-client local loss matrices for separate plots
    local_loss_by_K = {}   # K -> (T, m) with T=R+1

    for K in Ks:
        if K >= 8:
            seeds_for_k10 = [10, 11, 12]  # increase for smoother

            (
                mean_1d, std_1d,
                local_loss_mean, local_loss_std,
                h_mean, h_std,
                g_mean, g_std,
                mean_final_acc, std_final_acc
            ) = run_pcfedavg_multiple_seeds(
                seeds=seeds_for_k10,
                clients=clients,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                d=d, C=C,
                R=R, K=K, epochs=epochs,
                gamma_l=gamma_l,
                rho_base=rho_base,
                lam=lam,
                gamma_reg=gamma_reg,
                batch_size=batch_size,
                client_fraction=client_fraction,
                eps_mult=eps_multiplier,
                init_scale=0.01,
            )

            tag = f"K={K} (mean of {len(seeds_for_k10)})"

            loss_curves[tag] = mean_1d["losses"]
            test_acc_curves[tag] = mean_1d["accs"]
            train_acc_curves[tag] = mean_1d["global_train_acc"]

            # single-line h/g uses mean over clients each round
            constraint_curves[tag] = np.nanmean(h_mean, axis=1)
            infeas_curves[tag] = np.nanmean(g_mean, axis=1)

            avg_obj_curves[tag] = mean_1d["avg_obj"]
            metric_curves[tag] = mean_1d["metric"]

            # per-client local loss curves (mean across seeds)
            local_loss_by_K[K] = local_loss_mean

            print(f"\n[PCFedAvg K={K}] Final Acc mean = {mean_final_acc*100:.2f}% (std={std_final_acc*100:.2f}%)")

            # PRINT ONLY ONE MEAN LINE PER ROUND (loss + acc)
            print(f"\n[PCFedAvg K={K}] Mean per-round metrics:")
            for t in range(len(mean_1d["losses"])):  # 0..R
                print(f"  Round {t:3d}: Mean Loss={mean_1d['losses'][t]:.4f}, Mean Test Acc={mean_1d['accs'][t]*100:.2f}%")

        else:
            print(f"\nRunning PCFedAvg with K={K}...")

            out = run_pcfedavg_for_K(
                clients=clients,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                d=d, C=C,
                R=R, K=K, epochs=epochs,
                gamma_l=gamma_l,
                rho_base=rho_base,
                lam=lam,
                gamma_reg=gamma_reg,
                batch_size=batch_size,
                client_fraction=client_fraction,
                eps_mult=eps_multiplier,
                init_scale=0.01,
                seed=seed,
                display_every=1,  # keep prints for single run
            )

            tag = f"K={K}"
            loss_curves[tag] = out["losses"]
            test_acc_curves[tag] = out["accs"]
            train_acc_curves[tag] = out["global_train_acc"]

            # single-line h/g uses mean over clients each round
            constraint_curves[tag] = np.nanmean(out["h_full"], axis=1)
            infeas_curves[tag] = np.nanmean(out["g_full"], axis=1)

            avg_obj_curves[tag] = out["avg_obj"]
            metric_curves[tag] = out["metric"]

            local_loss_by_K[K] = out["local_loss_full"]

            print(f"[PCFedAvg K={K}] Final Acc = {out['final_acc']*100:.2f}%")

    # --------------------------
    # Global overlay plots
    # --------------------------
    plot_compare_curves(
        loss_curves,
        title="PCFedAvg: Global Train Loss vs Round",
        ylabel="Loss",
        xlabel="Round",
        show_legend=True,
    )

    plot_compare_curves(
        test_acc_curves,
        title="PCFedAvg: Global Test Accuracy vs Round",
        ylabel="Accuracy",
        xlabel="Round",
        show_legend=True,
    )

    plot_compare_curves(
        infeas_curves,
        title="PCFedAvg: Avg Infeasibility g (mean over clients) vs Round",
        ylabel="avg g_{i,λ}(x_i)",
        xlabel="Round",
        show_legend=True,
    )

    plot_compare_curves(
        avg_obj_curves,
        title="PCFedAvg: Avg Objective mean_i[f_i(x_i)+ρg_i] vs Round",
        ylabel="avg objective",
        xlabel="Round",
        show_legend=True,
    )

    # --------------------------
    # NEW: Local client loss plots (NOT mean) — separate plot per K
    # --------------------------
    for K in Ks:
        M = local_loss_by_K[K]  # (R+1, m)
        curves = {f"client {i}": M[:, i] for i in range(M.shape[1])}
        plot_compare_curves(
            curves,
            title=f"PCFedAvg: Local Client Losses f_i(x_i) vs Round (K={K})",
            ylabel="local loss",
            xlabel="Round",
            show_legend=True,
        )

    # keep windows open
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()