"""
dual_oracle_probabilistic.py - dual oracle for the probabilistic Lagrangian
relaxation. Used by cutting-plane / bundle methods.

Two modes:

* simplified=True (default):
    z = [lambda_1, ..., lambda_M, mu], length M + 1, all entries shared
    across time. The expanded multipliers are
        lambdasource[m, t] = z[m]    for all t
        mu[t]              = z[M]    for all t
    Cutting-plane methods typically start with this form because the dual
    objective is concave in the small vector z and the supergradient is
    immediately interpretable.

* simplified=False:
    z is the flattened time-indexed vector
        z = [lambdasource.ravel(), mu.ravel()]   length M*T + T
    Used as a stress test; runtime grows with T.

The dual value returned is the expected probabilistic relaxed DP value
(computed via the batched Bellman recursion starting from Delta=0) plus the
multiplier-side budget terms gamma^t * (lambda * c + mu * N) summed over t.

Returns
-------
dual_value : float
supergradient : ndarray, same shape as z
diagnostics : dict
"""
import numpy as np

from subgradientiter1_probabilistic import (
    _batched_gain_table_probabilistic,
    compute_relaxed_actions_and_residuals_probabilistic,
)


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def _expand_simplified(z, M, T):
    """z = [lambda_1..lambda_M, mu] -> (lambdasource (M,T), mu (T,))."""
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


def _value_at_Delta0_sum(lambdasource, mu, B, T, gamma, w, p, M, km, q, n):
    """Sum over (m, j) of the relaxed expected DP value at Delta=0, time 0.

    The relaxed value at (t=0, Delta=0) for pair (m, j) is V[m,j,0,0] from
    the probabilistic Bellman. Returns sum_{m, j} V[m, j, 0, 0]. The cutting-
    plane LP needs this in the same units as the supergradient g = sum_t
    gamma^t * (sum_j pi - c) (unnormalized residual).
    """
    p = np.asarray(p)[:, :B]
    w = np.asarray(w, dtype=float)
    q = np.asarray(q, dtype=float)
    n = np.asarray(n, dtype=float)

    P = (w[:, :, None] * p[None, :, :]).reshape(M * km, B)
    q_flat = q.reshape(M * km)
    n_flat = n.reshape(M * km)
    lam = np.repeat(lambdasource, km, axis=0)
    mu_arr = np.asarray(mu, dtype=float).flatten()

    V = np.zeros((M * km, T + 1, B + 1))
    age_next = np.minimum(np.arange(B) + 1, B - 1)

    for i in range(T - 1, -1, -1):
        V_next_age = V[:, i + 1, age_next]
        V_next_reset = V[:, i + 1, 0:1]

        Q_no = P + gamma * V_next_age
        Q_sched = (P + lam[:, i:i + 1]
                   + mu_arr[i] * n_flat[:, None]
                   + gamma * (q_flat[:, None] * V_next_reset
                              + (1.0 - q_flat[:, None]) * V_next_age))
        V[:, i, :B] = np.minimum(Q_no, Q_sched)
        V[:, i, B] = V[:, i, B - 1]

    V0 = V[:, 0, 0]                         # (M*km,)
    return float(V0.sum())


def dual_oracle_probabilistic(
    z, M, N, T, B, gamma, p, km, w, n, c, q,
    simplified=True,
    return_actions=False,
):
    """Probabilistic dual oracle. See module docstring."""
    z = np.asarray(z, dtype=float).flatten()

    if simplified:
        lambdasource, mu = _expand_simplified(z, M, T)
    else:
        lambdasource, mu = _expand_full(z, M, T)

    c_use = _as_1d_c(c, M)
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    asource1 = _batched_gain_table_probabilistic(
        lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km, q_arr, n_arr,
    )
    g_lambda_t, g_mu_t, diag, pi_out = (
        compute_relaxed_actions_and_residuals_probabilistic(
            asource1, M, T, B, gamma, N, km, n_arr, c,
        )
    )

    # Relaxed expected DP value (sum over pairs -- raw units).
    relaxed_value = _value_at_Delta0_sum(
        lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km, q_arr, n_arr,
    )

    # gamma^t budget discount factor.
    if gamma == 1.0:
        gamma_disc = float(T)
    else:
        gamma_disc = (1.0 - gamma ** T) / (1.0 - gamma)

    # Multiplier-side constant penalty. With simplified z (lambda_m, mu shared
    # over t), the per-step penalty is constant, so we factor out gamma_disc.
    if simplified:
        budget_penalty = float(
            gamma_disc * (float((z[:M] * c_use).sum()) + float(z[M]) * float(N))
        )
        dual_value = relaxed_value - budget_penalty
        # Supergradient: collapse residuals over t.
        # grad D / d lambda_m = sum_t gamma^t (sum_j pi^*_{m,j,t} - c_m).
        # grad D / d mu       = sum_t gamma^t (sum_{mj} pi^*_{mj,t} * n_mj - N).
        s_lambda = g_lambda_t.sum(axis=1)        # (M,)
        s_mu = float(g_mu_t.sum())               # scalar
        supergradient = np.concatenate([s_lambda, [s_mu]])
    else:
        # Time-indexed: each lam[m,t] contributes -gamma^t c_m and each mu[t]
        # contributes -gamma^t N.
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
        # Dual value is exact in the simplified mode because the batched
        # Bellman directly computes the Lagrangian DP optimum at Delta=0
        # for the given (lambdasource, mu). We flag it anyway for callers.
        "dual_value_approx": False,
    }
    if return_actions:
        diagnostics["pi"] = pi_out
    return float(dual_value), supergradient, diagnostics
