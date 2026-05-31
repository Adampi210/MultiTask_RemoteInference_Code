"""
subgradientiter1_probabilistic.py - probabilistic-transmission dual learning.

Trains (lambdasource, mu) for the probabilistic-link Lagrangian relaxation.
Supports the full method-selectable update set from optimizer_updates.py:

    harmonic, sqrt, normalized_global, normalized_blocks, adagrad, rmsprop,
    adam, deflected_sqrt

Legacy method aliases (preserved for backward compatibility with the previous
version of this file):

    constant     -> harmonic with step held at beta (via custom path)
    normalized   -> normalized_global
    polyak_like  -> normalized_global

Dual learning uses the EXPECTED probabilistic Bellman transitions (q-weighted
reset) -- not sampled successes. The supergradient comes from a deterministic
relaxed-action rollout from Delta=0, so multipliers learned here are
independent of the rollout noise that appears later when the online policy is
evaluated.
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
from paths import data_path, DATA_PROBABILISTIC_DIR


_HERE = DATA_PROBABILISTIC_DIR
_MULTIPLIERS_PATH = data_path("multipliers_probabilistic.mat", probabilistic=True)

# Methods that drive multipliers via Episode1's M-step closed-form projection
# (MATLAB-equivalent behaviour). All other methods use a single projected
# ascent step per outer iter (subgradientiter1_variants-style).
_EPISODE_MSTEP_METHODS = {"episode1_mstep"}


_LEGACY_ALIAS = {
    "constant": "harmonic",      # constant beta path is handled specially
    "normalized": "normalized_global",
    "polyak_like": "normalized_global",
}


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


# ---------------------------------------------------------------------------
# Batched probabilistic Bellman
# ---------------------------------------------------------------------------

def _batched_gain_table_probabilistic(lambdasource, mu, B, T, gamma, w, p,
                                      M, km, q, n):
    """Probabilistic Bellman, batched over (m, j)."""
    if q is None:
        raise ValueError("q must be provided (use q=ones for deterministic)")
    p = np.asarray(p)[:, :B]
    w = np.asarray(w, dtype=float)
    q = np.asarray(q, dtype=float)
    n = np.asarray(n, dtype=float)

    if q.shape != (M, km):
        raise ValueError(f"q shape {q.shape} != ({M},{km})")
    if not ((q >= 0.0).all() and (q <= 1.0).all()):
        raise ValueError("q entries must lie in [0, 1]")

    P = (w[:, :, None] * p[None, :, :]).reshape(M * km, B)
    q_flat = q.reshape(M * km)
    n_flat = n.reshape(M * km)
    lam = np.repeat(lambdasource, km, axis=0)
    mu_arr = np.asarray(mu, dtype=float).flatten()

    V = np.zeros((M * km, T + 1, B + 1))
    a = np.zeros((M * km, T, B))

    age_next = np.minimum(np.arange(B) + 1, B - 1)

    for i in range(T - 1, -1, -1):
        V_next_age = V[:, i + 1, age_next]
        V_next_reset = V[:, i + 1, 0:1]

        Q_no = P + gamma * V_next_age
        Q_sched = (
            P
            + lam[:, i:i + 1]
            + mu_arr[i] * n_flat[:, None]
            + gamma * (q_flat[:, None] * V_next_reset
                       + (1.0 - q_flat[:, None]) * V_next_age)
        )

        V[:, i, :B] = np.minimum(Q_no, Q_sched)
        a[:, i, :] = Q_no - Q_sched
        V[:, i, B] = V[:, i, B - 1]

    return a.reshape(M, km, T, B).transpose(0, 2, 3, 1)


# Backwards-compatible re-export.
batched_gain_table_probabilistic = _batched_gain_table_probabilistic


# ---------------------------------------------------------------------------
# Deterministic-rollout supergradient (EXPECTED transitions; not sampled)
# ---------------------------------------------------------------------------

def compute_relaxed_actions_and_residuals_probabilistic(
    asource1, M, T, B, gamma, N, km, n, c,
):
    """Relaxed attempted action pi* = 1{gain > 0} starting from Delta=0.

    Resource residuals are computed from attempted pi (not delivery success).
    """
    n_arr = np.asarray(n, dtype=float)
    c_use = _as_1d_c(c, M)

    Delta = np.zeros((M, km), dtype=np.int64)
    m_idx_grid, task_idx_grid = np.meshgrid(np.arange(M), np.arange(km),
                                            indexing='ij')

    g_lambda = np.zeros((M, T))
    g_mu = np.zeros(T)
    pos_compute = 0.0
    pos_channel = 0.0
    pi_out = np.zeros((M, km, T), dtype=bool)

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

        # AoI evolution for the deterministic relaxed rollout: scheduled ->
        # reset (q=1 surrogate). The dual learning uses expected transitions
        # already through the value function; the rollout is just a way to get
        # the action pi* at each time index.
        Delta = np.where(pi, 0,
                         np.minimum(Delta + 1, B - 1)).astype(np.int64)

    diagnostics = {
        "mean_compute_positive_residual": pos_compute / T,
        "mean_channel_positive_residual": pos_channel / T,
    }
    return g_lambda, g_mu, diagnostics, pi_out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _resolve_method(method):
    """Map legacy aliases to canonical names, return (canonical, is_constant)."""
    is_constant = (method == "constant")
    canonical = _LEGACY_ALIAS.get(method, method)
    if canonical in _EPISODE_MSTEP_METHODS:
        return canonical, False
    if canonical not in get_default_subgradient_methods():
        raise ValueError(
            f"Unknown method {method!r}. Choose from "
            f"{get_default_subgradient_methods()} + {sorted(_EPISODE_MSTEP_METHODS)} "
            f"(or legacy constant/normalized/polyak_like)"
        )
    return canonical, is_constant


def subgradientiter1_probabilistic(
    M, N, T, B, gamma, p, km, w, n, c, q,
    titer=10000,
    method="harmonic",
    beta=0.9,
    seed=0,
    verbose=False,
    history=False,
    save=False,
    save_path=None,
):
    """Method-selectable probabilistic subgradient driver.

    Returns
    -------
    A : (M+1, T) packed [lambdasource; mu] (same shape as Episode1 output, for
        backward compat with callers that read A directly).
    history_dict : if history=True, also returned as a 2-tuple (A, hist).

    Notes
    -----
    Does NOT overwrite multipliers.mat. With save=True, writes
    multipliers_probabilistic_{method}.mat. save_path overrides the filename.
    """
    canonical, constant_step = _resolve_method(method)

    n_arr = np.asarray(n, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    lambdasource = np.zeros((M, T))
    mu = np.zeros(T)

    z = np.concatenate([lambdasource.ravel(), mu.ravel()])
    state = init_optimizer_state(z.shape, canonical)
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
        asource1 = _batched_gain_table_probabilistic(
            lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km, q_arr, n_arr,
        )

        if canonical in _EPISODE_MSTEP_METHODS:
            # MATLAB-equivalent: Episode1's M-step closed-form projection. The
            # resource residuals (used by the methods below for history) are
            # computed once per outer iter from the same gain table -- they're
            # diagnostics here, not driving the update.
            g_lambda, g_mu, diag, _ = (
                compute_relaxed_actions_and_residuals_probabilistic(
                    asource1, M, T, B, gamma, N, km, n_arr, c,
                )
            )
            g = np.concatenate([g_lambda.ravel(), g_mu.ravel()])

            A_step = Episode1(asource1, M, T, B, gamma, N, beta / j,
                              lambdasource, mu, km, w_arr, n_arr, c)
            lambdasource = A_step[:M, :]
            mu = A_step[M, :]
            z = np.concatenate([lambdasource.ravel(), mu.ravel()])
            upd_diag = {"step": float(beta / j), "method": canonical}
        else:
            g_lambda, g_mu, diag, _ = (
                compute_relaxed_actions_and_residuals_probabilistic(
                    asource1, M, T, B, gamma, N, km, n_arr, c,
                )
            )
            g = np.concatenate([g_lambda.ravel(), g_mu.ravel()])

            if constant_step:
                # Legacy "constant": single-step projected ascent with step=beta.
                z = np.maximum(0.0, z + beta * g)
                upd_diag = {"step": float(beta), "method": "constant"}
            else:
                z, state, upd_diag = projected_update(
                    z, g, j, canonical, state,
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
            print(f"  sgd_p[{method}] iter {j}/{titer}  "
                  f"step={upd_diag['step']:.3e}  "
                  f"max|lam|={np.max(np.abs(lambdasource)):.4f}  "
                  f"max|mu|={np.max(np.abs(mu)):.4f}  "
                  f"||g||={np.linalg.norm(g):.3e}")

    runtime = time.perf_counter() - t0
    hist["runtime"] = float(runtime)

    A = np.zeros((M + 1, T))
    A[:M, :] = lambdasource
    A[M, :] = mu

    if save:
        out_path = save_path or os.path.join(
            _HERE, f"multipliers_probabilistic_{method}.mat"
        )
        sio.savemat(out_path,
                    {"lambdasource": lambdasource,
                     "mu": mu.reshape(1, -1),
                     "method": method,
                     "titer": int(titer),
                     "seed": int(seed)})

    if history:
        return A, hist
    return A


def load_multipliers_probabilistic(path=None):
    path = path or _MULTIPLIERS_PATH
    data = sio.loadmat(path)
    return data["lambdasource"], data["mu"].flatten()
