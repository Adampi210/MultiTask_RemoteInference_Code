"""
MAF1.py - 1:1 Python port of MAF1.m  (Max-AoI-First scheduling policy)

Greedy: at every time step, repeatedly pick the (source, task) pair with the
largest current AoI; schedule it if resource constraints allow. The greedy
resource-feasibility loop is shared with MIEF1 via greedy_scheduler.
"""
import numpy as np

from greedy_scheduler import greedy_select, as_1d_c


def MAF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=True):
    """
    Returns
    -------
    presult : float, the discounted sum of weighted average errors over t=0..T-2.
    """
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float)
    c_use = as_1d_c(c, M)

    # State (AoI): MATLAB Delta starts at 1; we keep it 0-indexed so it can
    # directly index `p`. The priority `Delta + 1` matches MATLAB's `D = Delta`
    # and ensures the greedy loop enters on the first step when Delta=0.
    Delta = np.zeros((M, km), dtype=np.int64)
    pavg = np.zeros(K)
    presult = 0.0

    for t in range(K):
        priority = Delta.astype(float) + 1.0
        _, Change, _, _ = greedy_select(priority, n, c_use, N, M, km)

        # Scheduled pairs reset to AoI=0; others age by 1 (capped at B-1).
        Delta = np.where(Change, 0, np.minimum(Delta + 1, B - 1))

        for m in range(M):
            for j in range(km):
                pavg[t] += (w[m, j] * p[j, Delta[m, j]]) / (km * M)

        # MATLAB: if t<T (where t=1..K). Python: t=0..K-1, so condition is t+1<T.
        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    if verbose:
        print(f"MAF1 presult = {presult}")
    return presult
