"""
randpolicy.py - 1:1 Python port of randpolicy.m

Randomized scheduling baseline: pick min(N, M) sources uniformly at random
without replacement, assign each a uniformly random task. No capacity gating.
Average the discounted sum of errors over Titer=100 Monte-Carlo trials.
"""
import numpy as np


def randpolicy(M, N, km, T, B, K, n, c, w, gamma, p, titer=100, verbose=True):
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float)

    presultiter = np.zeros(titer)
    for it in range(titer):
        Delta = np.zeros((M, km), dtype=np.int64)
        pavg = np.zeros(K)
        for t in range(K):
            Change = np.zeros((M, km), dtype=bool)

            n_select = min(N, M)
            source = np.random.permutation(M)[:n_select]
            for row in source:
                column = np.random.randint(0, km)  # MATLAB randperm(km, 1)
                Delta[row, column] = 0
                Change[row, column] = True

            for m in range(M):
                for j in range(km):
                    if not Change[m, j]:
                        if Delta[m, j] + 1 > B - 1:
                            Delta[m, j] = B - 1
                        else:
                            Delta[m, j] = Delta[m, j] + 1
                    pavg[t] += (w[m, j] * p[j, Delta[m, j]]) / (km * M)

            if t + 1 < T:
                presultiter[it] += (gamma ** t) * pavg[t]

    presult = float(np.mean(presultiter))
    if verbose:
        print(f"randpolicy presult = {presult}")
    return presult
