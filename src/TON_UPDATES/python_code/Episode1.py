"""
Episode1.py - 1:1 Python port of Episode1.m  (vectorized over the inner M loop)

The MATLAB code at lines 15-49 wraps the "Resources used" and "Lagrangian
update" blocks inside an outer `for m=1:M` loop with m shadowed by the inner
loops. Effect: those updates run M times per t with progressively-filled g.

We reproduce that exactly. We close form the M iterations of
    A^{(k+1)} = max(0, A^{(k)} + delta_k)
via
    A_M = max(0, S_M - min_{j=0..M-1} S_j, A_init + S_M),  S_k = sum_{i<k} d_i
which is the standard identity for max(0, .) iterated.

For each source m the d-sequence is
    d_k = -s_m          for k = 0..m-1      (Ccurr_k[m] = 0)
    d_k = dp_m          for k = m..M-1      (Ccurr_k[m] = g_final[m,:].sum())
with  s_m  = beta * c_use[m] * gamma^t
      dp_m = beta * (g_final[m,:].sum() - c_use[m]) * gamma^t.

For the channel multiplier A[M, t] the d-sequence is delta_mu_k =
beta * (Ncurr_k - N) * gamma^t with Ncurr_k = cumsum_k(g_final * n).
"""
import numpy as np


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def _project_M_steps(A_init, s_vec, dp_vec, m_vec, M):
    """Closed form of running M projected updates on each source.

    For source m, the M-step `d` sequence is m copies of -s_m followed by
    (M - m) copies of dp_m. Returns the final A value for every source.
    """
    # S_M (total sum) per source
    S_M = -m_vec * s_vec + (M - m_vec) * dp_vec

    # min_{j=0..M-1} S_j per source. S_j decreases linearly with slope -s for
    # j <= m (reaches -m*s at j=m). For j > m it changes by dp each step.
    #   If dp >= 0 -> min stays at -m*s.
    #   If dp <  0 -> min is at j=M-1, value = -m*s + (M-1-m)*dp.
    min_S = np.where(dp_vec >= 0,
                     -m_vec * s_vec,
                     -m_vec * s_vec + (M - 1 - m_vec) * dp_vec)

    return np.maximum.reduce([
        np.zeros_like(S_M),
        S_M - min_S,
        A_init + S_M,
    ])


def _project_channel(A_init_mu, delta_mu):
    """Closed form for the channel multiplier: same iteration on a single scalar."""
    # S_mu[k] = sum of first k deltas, k = 0..M.
    S_mu = np.concatenate([[0.0], np.cumsum(delta_mu)])
    min_S_partial = S_mu[:-1].min()  # j = 0..M-1
    S_total = S_mu[-1]
    return max(0.0, S_total - min_S_partial, A_init_mu + S_total)


def Episode1(asource1, M, T, B, gamma, N, beta, lambdasource, mu, km, w, n, c):
    """1:1 port of Episode1.m with the inner M loop closed-formed.

    Returns A of shape (M+1, T) with A[:M] = updated lambdasource,
    A[M] = updated mu.
    """
    A = np.zeros((M + 1, T))
    A[:M, :] = lambdasource
    A[M, :] = np.asarray(mu).flatten()
    Delta = np.zeros((M, km), dtype=np.int64)

    c_use = _as_1d_c(c, M)
    n = np.asarray(n, dtype=float)

    m_idx_grid, task_idx_grid = np.meshgrid(np.arange(M), np.arange(km),
                                            indexing='ij')
    m_vec = np.arange(M, dtype=float)

    for t in range(T):
        gainindex1 = asource1[m_idx_grid, t, Delta, task_idx_grid]
        g_final = (gainindex1 > 0).astype(float)
        Delta_new = np.where(gainindex1 > 0, 0,
                             np.minimum(Delta + 1, B - 1)).astype(np.int64)

        gt = gamma ** t
        gsum = g_final.sum(axis=1)
        s_vec = beta * c_use * gt
        dp_vec = beta * (gsum - c_use) * gt

        A[:M, t] = _project_M_steps(A[:M, t], s_vec, dp_vec, m_vec, M)

        g_row_n = (g_final * n).sum(axis=1)
        Ncurr_vec = np.cumsum(g_row_n)
        delta_mu = beta * (Ncurr_vec - N) * gt
        A[M, t] = _project_channel(A[M, t], delta_mu)

        Delta = Delta_new

    return A
