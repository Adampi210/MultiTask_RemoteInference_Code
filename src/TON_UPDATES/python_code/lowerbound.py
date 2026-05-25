"""
lowerbound.py - 1:1 Python port of lowerbound.m

Computes the relaxed Lagrangian lower bound: greedy single-source decisions
(no joint resource gating) plus the contribution of the multipliers.
"""
import numpy as np

from valuefunction1 import valuefunction1
from subgradientiter1 import subgradientiter1, load_multipliers


def lowerbound(M, N, km, T, B, K, n, c, w, gamma, p, titer=10000, verbose=True):
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float)

    subgradientiter1(M, N, T, B, gamma, p, km, w, n, c, titer=titer)
    lambdasource, mu = load_multipliers()

    asource1 = np.zeros((M, T, B, km))
    for m in range(M):
        lambda_m = lambdasource[m, :]
        for task in range(km):
            a = valuefunction1(lambda_m, mu, B,
                               w[m, task] * p[task, :B], T, gamma)
            asource1[m, :, :, task] = a

    Delta = np.zeros((M, km), dtype=np.int64)
    pavg = np.zeros(K)
    presult = 0.0

    for t in range(K):
        g = np.zeros((M, km))
        m_idx, task_idx = np.meshgrid(np.arange(M), np.arange(km), indexing='ij')
        gainindex1 = asource1[m_idx, t, Delta, task_idx]

        for m in range(M):
            for j in range(km):
                if gainindex1[m, j] > 0:
                    g[m, j] = 1
                    Delta[m, j] = 0
                else:
                    Delta[m, j] = Delta[m, j] + 1
                    if Delta[m, j] > B - 1:
                        Delta[m, j] = B - 1
                pavg[t] += ((w[m, j] * p[j, Delta[m, j]]) / (km * M)
                            + lambdasource[m, t] * g[m, j]
                            + mu[t] * g[m, j])

        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    if verbose:
        print(f"lowerbound presult = {presult}")
    return presult
