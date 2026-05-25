"""
MIEF1.py - Maximum Instantaneous Error First (MIEF) scheduling policy.

At every time step t:
  1. For every (source, task) pair (m, j), read the current AoI Delta[m, j].
  2. Compute the priority score
         score[m, j] = w[m, j] * p[j, Delta[m, j]]
     where w is the task weight and p is the task-specific AoI/error penalty
     function already used in the objective.
  3. Greedily schedule pairs in descending order of score subject to the same
     constraints as the other baselines:
         per-source compute:  sum_j action[m, j]            <= c[m]
         total channel:       sum_{m,j} action[m,j] * n[m,j] <= N
  4. Return action[m, j] in {0, 1}.

The greedy resource-feasibility loop is shared with MAF1 via greedy_scheduler,
so this file only owns the priority-score construction.
"""
import numpy as np

from greedy_scheduler import greedy_select, as_1d_c


def mief_priority(Delta, w, p, km):
    """Per-pair priority score score[m, j] = w[m, j] * p[j, Delta[m, j]]."""
    Delta = np.asarray(Delta, dtype=np.int64)
    task_idx = np.broadcast_to(np.arange(km), Delta.shape)
    return np.asarray(w, dtype=float) * np.asarray(p, dtype=float)[task_idx, Delta]


def mief_select(Delta, M, km, n, c_use, N, w, p):
    """Build the priority matrix and run the shared greedy selection.

    Returns (action, Change, Ccurr, Ncurr); action is the M-by-km 0/1 matrix.
    """
    # Build the priority matrix.
    priority = mief_priority(Delta, w, p, km)
    return greedy_select(priority, n, c_use, N, M, km)

# M - number of sources
# N - total channel capacity
# km - number of tasks per source
# T - total time steps
# B - AoI buffer size (max AoI is B-1)
# K - number of episodes to average over
# n - M-by-km matrix of per-pair channel costs
# c - M-length vector of per-source compute limits
# w - M-by-km matrix of per-pair task weights
# gamma - discount factor for the objective
# p - km-by-B matrix of per-pair AoI penalty functions
def MIEF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=True):
    """Run the MIEF policy and return the discounted sum of weighted errors."""
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float)
    c_use = as_1d_c(c, M)

    # Initialize AoI Delta[m, j] = 0 for all (m, j) pairs. 
    Delta = np.zeros((M, km), dtype=np.int64)
    pavg = np.zeros(K)
    presult = 0.0

    # Run K episodes (similar to MGF1, K is for simulation, T is for scoring)
    for t in range(K):
        _, Change, _, _ = mief_select(Delta, M, km, n, c_use, N, w, p)

        # Increment AoI for all pairs, but reset to 0 if the pair was scheduled (Change=True).
        Delta = np.where(Change, 0, np.minimum(Delta + 1, B - 1))

        # Compute the average penalty pavg[t] across all pairs at this time step,
        # weighted by w and normalized by km and M.
        for m in range(M):
            for j in range(km):
                pavg[t] += (w[m, j] * p[j, Delta[m, j]]) / (km * M)
        
        # Accumulate the discounted sum of average penalties.
        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    if verbose:
        print(f"MIEF1 presult = {presult}")
    return presult
