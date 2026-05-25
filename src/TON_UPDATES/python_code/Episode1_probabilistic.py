"""
Episode1_probabilistic.py - probabilistic-transmission Episode1.

Differences vs. Episode1.py:
  * Per-link success probability q[m, j]. AoI evolves stochastically:
        if action[m,j] == 1:
            success ~ Bernoulli(q[m,j])
            Delta_new = 0 if success else min(Delta + 1, B - 1)
        else:
            Delta_new = min(Delta + 1, B - 1)
  * The Lagrangian subgradient still uses the *attempted* action pi, not the
    realized u — resources are consumed at the attempted transmission.
  * Channel-cost coefficient n[m, j] is propagated into the multiplier update
    (consistent with the paper's mu * n_{m,j} term).

The closed-form M-step projection identities from Episode1.py carry over
unchanged: only the AoI rollout differs.
"""
import numpy as np


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def _project_M_steps(A_init, s_vec, dp_vec, m_vec, M):
    """Closed form of M projected non-negativity steps. See Episode1.py."""
    S_M = -m_vec * s_vec + (M - m_vec) * dp_vec
    min_S = np.where(dp_vec >= 0,
                     -m_vec * s_vec,
                     -m_vec * s_vec + (M - 1 - m_vec) * dp_vec)
    return np.maximum.reduce([
        np.zeros_like(S_M),
        S_M - min_S,
        A_init + S_M,
    ])


def _project_channel(A_init_mu, delta_mu):
    S_mu = np.concatenate([[0.0], np.cumsum(delta_mu)])
    min_S_partial = S_mu[:-1].min()
    S_total = S_mu[-1]
    return max(0.0, S_total - min_S_partial, A_init_mu + S_total)


def Episode1_probabilistic(
    asource1, M, T, B, gamma, N, beta,
    lambdasource, mu, km, w, n, c, q,
    rng=None,
    transition_mode="stochastic",
    return_violations=False,
):
    """Probabilistic Episode1: one subgradient outer iteration.

    Preserves the MATLAB Episode1 pattern verbatim (M closed-form projected
    ascent steps per t via the `for m=1:M` quirk), with two changes:
      * AoI evolves stochastically with success probability q[m, j].
      * Channel-cost coefficient n[m, j] enters the mu update.

    Parameters
    ----------
    asource1     : (M, T, B, km) per-pair gain table.
    M, T, B, km  : problem dimensions.
    gamma        : discount factor.
    N            : total channel capacity.
    beta         : current step size (e.g. beta_outer / j for harmonic).
    lambdasource : (M, T) per-source multipliers.
    mu           : (T,)   channel multipliers.
    w            : (M, km) weights (unused; kept for parity with Episode1.m).
    n            : (M, km) channel cost coefficients.
    c            : (M,) or (M, km) per-source compute cap.
    q            : (M, km) per-link success probabilities, in [0, 1].
    rng          : np.random.Generator; if None, a default is built.
    transition_mode :
        - "stochastic" (default): sample Bernoulli(q) per scheduled pair.
        - "deterministic_q1": ignore q, always reset (matches Episode1.py).
    return_violations :
        If True, also return a dict with mean per-iter compute / channel
        positive constraint residuals (useful for subgradient diagnostics).

    Returns
    -------
    A : (M + 1, T) updated multipliers, with A[:M] = lambdasource, A[M] = mu.
    stats (optional) : dict with 'compute_violation' and 'channel_violation'.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    A = np.zeros((M + 1, T))
    A[:M, :] = lambdasource
    A[M, :] = np.asarray(mu).flatten()
    Delta = np.zeros((M, km), dtype=np.int64)

    c_use = _as_1d_c(c, M)
    n = np.asarray(n, dtype=float)
    q = np.asarray(q, dtype=float)

    m_idx_grid, task_idx_grid = np.meshgrid(np.arange(M), np.arange(km),
                                            indexing='ij')
    m_vec = np.arange(M, dtype=float)

    compute_viol = 0.0
    channel_viol = 0.0

    for t in range(T):
        gainindex1 = asource1[m_idx_grid, t, Delta, task_idx_grid]
        # Attempted action pi (used for subgradient updates and resource use).
        pi = (gainindex1 > 0)
        g_final = pi.astype(float)

        # AoI evolution (stochastic delivery).
        if transition_mode == "stochastic":
            success_draws = rng.random(size=(M, km)) < q
        elif transition_mode == "deterministic_q1":
            success_draws = np.ones((M, km), dtype=bool)
        else:
            raise ValueError(f"Unknown transition_mode {transition_mode!r}")

        reset_mask = pi & success_draws
        Delta_new = np.where(reset_mask, 0,
                             np.minimum(Delta + 1, B - 1)).astype(np.int64)

        # Subgradient update: based on attempted scheduling.
        gt = gamma ** t
        gsum = g_final.sum(axis=1)
        s_vec = beta * c_use * gt
        dp_vec = beta * (gsum - c_use) * gt

        A[:M, t] = _project_M_steps(A[:M, t], s_vec, dp_vec, m_vec, M)

        g_row_n = (g_final * n).sum(axis=1)
        Ncurr_vec = np.cumsum(g_row_n)
        delta_mu = beta * (Ncurr_vec - N) * gt
        A[M, t] = _project_channel(A[M, t], delta_mu)

        if return_violations:
            compute_viol += float(np.clip(gsum - c_use, 0.0, None).mean())
            channel_viol += max(0.0, float(Ncurr_vec[-1] - N))

        Delta = Delta_new

    if return_violations:
        stats = {
            "compute_violation": compute_viol / T,
            "channel_violation": channel_viol / T,
        }
        return A, stats
    return A
