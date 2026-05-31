"""
dual_oracle_deterministic.py - dual oracle for the deterministic Lagrangian
relaxation (q == 1 everywhere). Mirrors dual_oracle_probabilistic.py so the
cutting-plane / bundle methods can be reused unchanged for both modes.
"""
import numpy as np

from subgradientiter1_variants import (
    _batched_gain_table_variants,
    compute_relaxed_actions_and_residuals_deterministic,
)


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def _expand_simplified(z, M, T):
    if z.shape[0] != M + 1:
        raise ValueError(f"simplified z must have length {M + 1}, got {z.shape[0]}")
    lambdasource = np.broadcast_to(z[:M, None], (M, T)).copy()
    mu = np.full(T, float(z[M]))
    return lambdasource, mu


def _expand_full(z, M, T):
    if z.shape[0] != M * T + T:
        raise ValueError(f"full z must have length {M*T + T}, got {z.shape[0]}")
    lambdasource = z[:M * T].reshape(M, T).copy()
    mu = z[M * T:].copy()
    return lambdasource, mu


def _value_at_Delta0_sum(lambdasource, mu, B, T, gamma, w, p, M, km, n,
                         use_n_cost):
    p = np.asarray(p)[:, :B]
    w = np.asarray(w, dtype=float)
    P = (w[:, :, None] * p[None, :, :]).reshape(M * km, B)
    lam = np.repeat(lambdasource, km, axis=0)
    mu_arr = np.asarray(mu, dtype=float).flatten()
    if use_n_cost:
        n_flat = np.asarray(n, dtype=float).reshape(M * km)

    V = np.zeros((M * km, T + 1, B + 1))
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
        V[:, i, B] = V[:, i, B - 1]

    return float(V[:, 0, 0].sum())


def dual_oracle_deterministic(
    z, M, N, T, B, gamma, p, km, w, n, c,
    simplified=True,
    return_actions=False,
    use_n_cost=False,
):
    z = np.asarray(z, dtype=float).flatten()
    if simplified:
        lambdasource, mu = _expand_simplified(z, M, T)
    else:
        lambdasource, mu = _expand_full(z, M, T)

    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    p_arr = np.asarray(p, dtype=float)

    asource1 = _batched_gain_table_variants(
        lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km,
        n=n_arr, use_n_cost=use_n_cost,
    )
    g_lambda_t, g_mu_t, diag, pi_out = (
        compute_relaxed_actions_and_residuals_deterministic(
            asource1, M, T, B, gamma, N, km, n_arr, c,
        )
    )
    relaxed_value = _value_at_Delta0_sum(
        lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km, n_arr, use_n_cost,
    )

    c_use = _as_1d_c(c, M)

    if gamma == 1.0:
        gamma_disc = float(T)
    else:
        gamma_disc = (1.0 - gamma ** T) / (1.0 - gamma)

    if simplified:
        budget_penalty = float(
            gamma_disc * (float((z[:M] * c_use).sum()) + float(z[M]) * float(N))
        )
        dual_value = relaxed_value - budget_penalty
        s_lambda = g_lambda_t.sum(axis=1)
        s_mu = float(g_mu_t.sum())
        supergradient = np.concatenate([s_lambda, [s_mu]])
    else:
        gamma_vec = gamma ** np.arange(T)
        per_t_pen = (lambdasource * c_use[:, None] * gamma_vec[None, :]).sum() \
                    + (mu * gamma_vec * N).sum()
        dual_value = relaxed_value - float(per_t_pen)
        supergradient = np.concatenate([g_lambda_t.ravel(), g_mu_t.ravel()])

    diagnostics = {
        "relaxed_value": float(relaxed_value),
        "mean_compute_positive_residual":
            float(diag["mean_compute_positive_residual"]),
        "mean_channel_positive_residual":
            float(diag["mean_channel_positive_residual"]),
        "supergradient_norm": float(np.linalg.norm(supergradient)),
        "z_norm": float(np.linalg.norm(z)),
        "simplified": bool(simplified),
        "dual_value_approx": False,
    }
    if return_actions:
        diagnostics["pi"] = pi_out
    return float(dual_value), supergradient, diagnostics
