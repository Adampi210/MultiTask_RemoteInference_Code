"""
checkmief.py - Stand-alone sanity check of the MIEF policy with a synthetic
penalty. Mirrors checkmaf.py / checkmgf.py.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MIEF1 import MIEF1, mief_select
from greedy_scheduler import as_1d_c


def main():
    M = 10
    N = 10
    km = 2
    B = 20
    T = 10

    n = np.ones((M, km))
    c = np.ones(M)
    w = np.zeros((M, km))
    for m in range(M):
        for j in range(km):
            w[m, j] = (m + 1) / km

    K = T
    gamma = 0.9

    p = np.zeros((km, B))
    p[0, :B] = np.log(np.arange(1, B + 1))
    p[1, :B] = np.arange(1, B + 1)

    Delta = np.zeros((M, km), dtype=np.int64)
    pavg = np.zeros(K)
    presult = 0.0
    c_use = as_1d_c(c, M)

    for t in range(K):
        action, Change, Ccurr, Ncurr = mief_select(Delta, M, km, n, c_use, N, w, p)

        # Constraint check (sanity): per-source compute & total channel.
        assert np.all(action.sum(axis=1) <= c_use + 1e-9), "compute cap violated"
        assert (action * n).sum() <= N + 1e-9, "channel cap violated"

        Delta = np.where(Change, 0, np.minimum(Delta + 1, B - 1))

        for m in range(M):
            for j in range(km):
                pavg[t] += (w[m, j] * p[j, Delta[m, j]]) / (km * M)

        print(f"t={t+1}  Ncurr={Ncurr}")
        print("Delta =\n", Delta + 1)
        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    print(f"\npresult = {presult}")

    # Also confirm the wrapper returns the same value.
    np.random.seed(0)
    full = MIEF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=False)
    print(f"MIEF1 wrapper presult = {full}")
    assert abs(full - presult) < 1e-9, "wrapper / step-by-step mismatch"


if __name__ == "__main__":
    main()
