"""
subgradientiter1_variants.py - method-selectable deterministic subgradient
solver. Same Lagrangian relaxation as subgradientiter1.py, but the projected
ascent step can be any method from optimizer_updates.py.

The deterministic batched Bellman recursion can also use the paper-faithful
``mu * n_{m,j}`` term (use_n_cost=True) -- with use_n_cost=False and
method="harmonic" the dual closely matches the legacy subgradientiter1 on the
same problem (it differs only because the harmonic update here uses a single
projected ascent step per outer iter rather than the MATLAB Episode1 M-step
closed-form -- they coincide asymptotically and agree on tiny problems within
a couple of percent).
"""
import os
import time
import numpy as np
import scipy.io as sio

from optimizer_updates import (
    init_optimizer_state,
    projected_update,
    get_default_subgradient_methods,
)
from Episode1 import Episode1
from paths import DATA_DETERMINISTIC_DIR


_HERE = DATA_DETERMINISTIC_DIR

# Methods that drive multipliers via Episode1's M-step closed-form projection
# (MATLAB-equivalent behaviour).
_EPISODE_MSTEP_METHODS = {"episode1_mstep"}


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def _batched_gain_table_variants(lambdasource, mu, B, T, gamma, w, p,
                                 M, km, n=None, use_n_cost=False):
    """Deterministic batched Bellman recursion (one (m,j) per row).

    Q_no(d)       = P + gamma * V[next, aged]
    Q_schedule(d) = P + lambda + (mu * n_mj if use_n_cost else mu)
                    + gamma * V[next, reset]
    """
    p = np.asarray(p)[:, :B]
    w = np.asarray(w, dtype=float)
    P = (w[:, :, None] * p[None, :, :]).reshape(M * km, B)
    lam = np.repeat(lambdasource, km, axis=0)
    mu_arr = np.asarray(mu, dtype=float).flatten()
    if use_n_cost:
        if n is None:
            raise ValueError("use_n_cost=True requires n to be provided")
        n_flat = np.asarray(n, dtype=float).reshape(M * km)
    else:
        n_flat = None

    V = np.zeros((M * km, T + 1, B + 1))
    a = np.zeros((M * km, T, B))

    age_next = np.minimum(np.arange(B) + 1, B - 1)

    for i in range(T - 1, -1, -1):
        V_next_age = V[:, i + 1, age_next]
        V_next_reset = V[:, i + 1, 0:1]

        Q_no = P + gamma * V_next_age
        if use_n_cost:
            Q_sched = (P + lam[:, i:i + 1]
                       + mu_arr[i] * n_flat[:, None]
                       + gamma * V_next_reset)
        else:
            Q_sched = (P + lam[:, i:i + 1]
                       + mu_arr[i]
                       + gamma * V_next_reset)

        V[:, i, :B] = np.minimum(Q_no, Q_sched)
        a[:, i, :] = Q_no - Q_sched
        V[:, i, B] = V[:, i, B - 1]

    return a.reshape(M, km, T, B).transpose(0, 2, 3, 1)


def compute_relaxed_actions_and_residuals_deterministic(
    asource1, M, T, B, gamma, N, km, n, c
):
    """Relaxed attempted action pi* = 1{gain > 0} starting from Delta=0.

    Returns
    -------
    g_lambda : (M, T) discounted compute residual
                 gamma^t * (sum_j pi[m,j,t] - c[m]).
    g_mu     : (T,) discounted channel residual
                 gamma^t * (sum_{m,j} pi[m,j,t] * n[m,j] - N).
    diagnostics : dict with mean positive residuals and pi sum.
    pi       : (M, km, T) bool tensor of attempted actions.
    """
    n_arr = np.asarray(n, dtype=float)
    c_use = _as_1d_c(c, M)

    Delta = np.zeros((M, km), dtype=np.int64)
    m_idx_grid, task_idx_grid = np.meshgrid(np.arange(M), np.arange(km),
                                            indexing='ij')

    g_lambda = np.zeros((M, T))
    g_mu = np.zeros(T)
    pi_out = np.zeros((M, km, T), dtype=bool)

    pos_compute = 0.0
    pos_channel = 0.0

    for t in range(T):
        gainindex = asource1[m_idx_grid, t, Delta, task_idx_grid]
        pi = (gainindex > 0)
        pi_out[:, :, t] = pi

        gsum = pi.sum(axis=1).astype(float)
        chan_use = float((pi.astype(float) * n_arr).sum())

        gt = gamma ** t
        compute_residual = gsum - c_use
        channel_residual = chan_use - N

        g_lambda[:, t] = gt * compute_residual
        g_mu[t] = gt * channel_residual

        pos_compute += float(np.clip(compute_residual, 0.0, None).mean())
        pos_channel += max(0.0, channel_residual)

        # Deterministic AoI evolution: scheduled -> reset, else age.
        Delta = np.where(pi, 0,
                         np.minimum(Delta + 1, B - 1)).astype(np.int64)

    diagnostics = {
        "mean_compute_positive_residual": pos_compute / T,
        "mean_channel_positive_residual": pos_channel / T,
    }
    return g_lambda, g_mu, diagnostics, pi_out


def subgradientiter1_variants(
    M, N, T, B, gamma, p, km, w, n, c,
    titer=10000,
    method="harmonic",
    beta=0.9,
    seed=0,
    verbose=False,
    history=False,
    save=False,
    use_n_cost=False,
):
    """Method-selectable deterministic subgradient driver.

    Returns
    -------
    lambdasource : (M, T) nonneg per-source multipliers.
    mu           : (T,)   nonneg channel multipliers.
    history_dict : dict (always returned; empty arrays when history=False
                   except for runtime and method).

    Notes
    -----
    Does NOT overwrite multipliers.mat. With save=True, writes
    multipliers_deterministic_{method}.mat.
    """
    if method not in get_default_subgradient_methods() \
            and method not in _EPISODE_MSTEP_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Choose from "
            f"{get_default_subgradient_methods()} + {sorted(_EPISODE_MSTEP_METHODS)}"
        )

    n_arr = np.asarray(n, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    w_arr = np.asarray(w, dtype=float)

    lambdasource = np.zeros((M, T))
    mu = np.zeros(T)

    z = np.concatenate([lambdasource.ravel(), mu.ravel()])
    state = init_optimizer_state(z.shape, method)
    block_slices = [(0, M * T), (M * T, M * T + T)]

    hist = {
        "max_lambda": [],
        "mean_lambda": [],
        "max_mu": [],
        "mean_mu": [],
        "residual_norm": [],
        "mean_compute_positive_residual": [],
        "mean_channel_positive_residual": [],
        "step_size": [],
        "method": method,
    } if history else {"method": method}

    t0 = time.perf_counter()

    for j in range(1, titer + 1):
        asource1 = _batched_gain_table_variants(
            lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km,
            n=n_arr, use_n_cost=use_n_cost,
        )
        g_lambda, g_mu, diag, _ = (
            compute_relaxed_actions_and_residuals_deterministic(
                asource1, M, T, B, gamma, N, km, n_arr, c,
            )
        )
        g = np.concatenate([g_lambda.ravel(), g_mu.ravel()])
        if method in _EPISODE_MSTEP_METHODS:
            # MATLAB-equivalent Episode1 closed-form M-step projection.
            A_step = Episode1(asource1, M, T, B, gamma, N, beta / j,
                              lambdasource, mu, km, w_arr, n_arr, c)
            lambdasource = A_step[:M, :]
            mu = A_step[M, :]
            z = np.concatenate([lambdasource.ravel(), mu.ravel()])
            upd_diag = {"step": float(beta / j), "method": method}
        else:
            z, state, upd_diag = projected_update(
                z, g, j, method, state,
                beta=beta, block_slices=block_slices,
            )
            lambdasource = z[:M * T].reshape(M, T)
            mu = z[M * T:].reshape(T)

        if history:
            hist["max_lambda"].append(float(np.max(np.abs(lambdasource))))
            hist["mean_lambda"].append(float(np.mean(lambdasource)))
            hist["max_mu"].append(float(np.max(np.abs(mu))))
            hist["mean_mu"].append(float(np.mean(mu)))
            hist["residual_norm"].append(float(np.linalg.norm(g)))
            hist["mean_compute_positive_residual"].append(
                float(diag["mean_compute_positive_residual"]))
            hist["mean_channel_positive_residual"].append(
                float(diag["mean_channel_positive_residual"]))
            hist["step_size"].append(float(upd_diag["step"]))

        if verbose and (j % max(1, titer // 20) == 0 or j == 1):
            print(f"  sgd_v[{method}] iter {j}/{titer}  "
                  f"step={upd_diag['step']:.3e}  "
                  f"max|lam|={np.max(np.abs(lambdasource)):.4f}  "
                  f"max|mu|={np.max(np.abs(mu)):.4f}  "
                  f"||g||={np.linalg.norm(g):.3e}")

    runtime = time.perf_counter() - t0
    hist["runtime"] = float(runtime)

    if save:
        out_path = os.path.join(_HERE, f"multipliers_deterministic_{method}.mat")
        sio.savemat(out_path,
                    {"lambdasource": lambdasource,
                     "mu": mu.reshape(1, -1),
                     "method": method,
                     "titer": int(titer),
                     "seed": int(seed),
                     "use_n_cost": int(use_n_cost)})

    return lambdasource, mu, hist
