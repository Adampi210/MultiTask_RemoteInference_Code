"""
greedy_scheduler.py - Shared greedy resource-feasibility loop used by the
priority-based scheduling baselines (MAF, MIEF, and MGF-style policies).

At each call, repeatedly pick the (source, task) pair with the largest
positive priority and schedule it iff
    sum_j action[m, j]            <= c_use[m]    (per-source compute)
    sum_{m,j} action[m,j] * n[m,j] <= N         (total channel)
Tie-breaking matches the existing MAF/MGF loops: column-wise argmax first,
then argmax across columns. Picked entries are zeroed so the loop terminates
in at most M*km iterations.
"""
import numpy as np


def as_1d_c(c, M):
    """Accept c as either (M,) or (M, km) and return the (M,) per-source cap."""
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def greedy_select(priority, n, c_use, N, M, km):
    """Run the shared MAF/MIEF/MGF greedy resource-gated selection.

    Parameters
    ----------
    priority : (M, km) array. Only strictly-positive entries are candidates.
    n        : (M, km) channel cost per (source, task).
    c_use    : (M,)    per-source compute capacity.
    N        : scalar  total channel capacity.
    M, km    : ints, problem dimensions.

    Returns
    -------
    action : (M, km) int8 0/1 schedule.
    Change : (M, km) bool view of action (kept for parity with MAF1.m code).
    Ccurr  : (M,) per-source compute used.
    Ncurr  : float total channel used.
    """
    n = np.asarray(n, dtype=float)
    P = np.asarray(priority, dtype=float).copy()
    action = np.zeros((M, km), dtype=np.int8)
    Change = np.zeros((M, km), dtype=bool)
    Ccurr = np.zeros(M)
    Ncurr = 0.0

    while P.max() > 0:
        col_max = P.max(axis=0)
        col_argmax = P.argmax(axis=0)
        column = int(col_max.argmax())
        row = int(col_argmax[column])
        n1 = Ncurr + n[row, column]
        if n1 <= N and Ccurr[row] + 1 <= c_use[row]:
            action[row, column] = 1
            Change[row, column] = True
            Ccurr[row] += 1
            Ncurr = n1
        P[row, column] = 0

    return action, Change, Ccurr, Ncurr
