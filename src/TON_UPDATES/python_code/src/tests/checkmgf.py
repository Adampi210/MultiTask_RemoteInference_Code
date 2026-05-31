"""
checkmgf.py - 1:1 Python port of checkmgf.m

Stand-alone sanity check of the MGF policy with a synthetic penalty.
"""
import os
import numpy as np

from _bootstrap import paths   # noqa: F401  (bootstraps sys.path)

from valuefunction1 import valuefunction1
from subgradientiter1 import subgradientiter1, load_multipliers


TITER = int(os.environ.get("INFOCOM_TITER", "10000"))


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

    # Gain index calculation via subgradient method
    subgradientiter1(M, N, T, B, gamma, p, km, w, n, c, titer=TITER)
    lambdasource, mu = load_multipliers()
    print("mu =\n", mu)

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
            if n1 <= N and Ccurr[row] + 1 <= c[row]:
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
        print("Delta =\n", Delta + 1)

    print(f"\npresult = {presult}")


if __name__ == "__main__":
    main()
