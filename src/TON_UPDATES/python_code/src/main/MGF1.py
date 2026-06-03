"""
MGF1.py - 1:1 Python port of MGF1.m  (Max-Gain-First / index policy)

1. Calls subgradientiter1 to recompute Lagrange multipliers.
2. Builds the gain-index lookup `asource1` via valuefunction1.
3. Greedily schedules the (source, task) pair with the largest positive gain
   index, gated by resource constraints.

The multipliers (lambdasource, mu) are obtained directly from the in-memory
return value of `subgradientiter1`. The original MATLAB flow round-tripped
through multipliers.mat; we still write that file for compatibility, but we
do NOT read it back -- two MGF1 calls running concurrently would race on the
file otherwise.
"""
import numpy as np

from valuefunction1 import valuefunction1
from subgradientiter1 import subgradientiter1


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def MGF1(M, N, km, T, B, K, n, c, w, gamma, p, titer=10000, verbose=True,
         save_multipliers=True, m_step_repeats=1):
    """Run the MGF policy and return the discounted sum of weighted errors.

    save_multipliers=False skips subgradientiter1's write to multipliers.mat
    (used when many MGF1 calls run concurrently, e.g. the steps sweep).

    m_step_repeats controls how many times Episode1's M-pass projection block
    is applied per timestep (total projected sub-updates = m_step_repeats * M).
    m_step_repeats=1 reproduces MATLAB exactly.
    """
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float)
    c_use = _as_1d_c(c, M)

    # 1) Subgradient method. Returned A is shape (M+1, T) with A[:M] =
    # lambdasource and A[M] = mu. subgradientiter1 also writes multipliers.mat
    # for compatibility but we don't read it back here.
    A = subgradientiter1(M, N, T, B, gamma, p, km, w, n, c, titer=titer,
                         save=save_multipliers, m_step_repeats=m_step_repeats)
    lambdasource = A[:M, :]
    mu = A[M, :]

    # 2) Build gain-index lookup table.
    asource1 = np.zeros((M, T, B, km))
    for m in range(M):
        lambda_m = lambdasource[m, :]
        for task in range(km):
            a = valuefunction1(lambda_m, mu, B,
                               w[m, task] * p[task, :B], T, gamma)
            asource1[m, :, :, task] = a

    # 3) Run greedy policy with gain indices.
    Delta = np.zeros((M, km), dtype=np.int64)
    pavg = np.zeros(K)
    presult = 0.0

    for t in range(K):
        g = np.zeros((M, km))
        m_idx, task_idx = np.meshgrid(np.arange(M), np.arange(km), indexing='ij')
        gainindex1 = asource1[m_idx, t, Delta, task_idx].copy()
        Ccurr = np.zeros(M)
        Ncurr = 0.0
        Change = np.zeros((M, km), dtype=bool)

        while gainindex1.max() > 0:
            col_max = gainindex1.max(axis=0)
            col_argmax = gainindex1.argmax(axis=0)
            column = int(col_max.argmax())
            row = int(col_argmax[column])
            n1 = Ncurr + n[row, column]
            if n1 <= N and Ccurr[row] + 1 <= c_use[row]:
                g[row, column] = 1
                Delta[row, column] = 0
                Ccurr[row] += 1
                Ncurr = n1
                Change[row, column] = True
            gainindex1[row, column] = 0

        for m in range(M):
            for j in range(km):
                if not Change[m, j]:
                    if Delta[m, j] + 1 > B - 1:
                        Delta[m, j] = B - 1
                    else:
                        Delta[m, j] = Delta[m, j] + 1
                pavg[t] += (w[m, j] * p[j, Delta[m, j]]) / (km * M)

        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    if verbose:
        print(f"MGF1 presult = {presult}")
    return presult
