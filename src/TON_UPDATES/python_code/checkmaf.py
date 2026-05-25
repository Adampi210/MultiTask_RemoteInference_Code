"""
checkmaf.py - 1:1 Python port of checkmaf.m

Stand-alone sanity check of the MAF policy with a synthetic penalty.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    M = 10
    N = 10
    km = 2
    B = 20
    T = 10

    # n(m, j) = 1 for all m, j  (both MATLAB branches set 1)
    n = np.ones((M, km))

    # c(m) = 1 for all m (both MATLAB branches set 1) - 1D vector
    c = np.ones(M)

    # w(m, j) = m/km   (MATLAB m=1..M so w(m,j) = m/km)
    w = np.zeros((M, km))
    for m in range(M):
        for j in range(km):
            w[m, j] = (m + 1) / km

    K = T
    gamma = 0.9

    # Penalty function
    p = np.zeros((km, B))
    p[0, :B] = np.log(np.arange(1, B + 1))
    p[1, :B] = np.arange(1, B + 1)

    Delta = np.zeros((M, km), dtype=np.int64)  # AoI=1 in MATLAB
    pavg = np.zeros(K)
    presult = 0.0

    for t in range(K):
        g = np.zeros((M, km))
        Ccurr = np.zeros(M)
        Ncurr = 0.0
        Change = np.zeros((M, km), dtype=bool)
        # Shift by +1: MATLAB Delta starts at 1 (AoI=1); our 0-indexed Delta needs
        # this offset so the priority equals MATLAB's `D = Delta` and the loop runs.
        D = Delta.astype(float).copy() + 1.0

        while D.max() > 0:
            col_max = D.max(axis=0)
            col_argmax = D.argmax(axis=0)
            column = int(col_max.argmax())
            row = int(col_argmax[column])
            n1 = Ncurr + n[row, column]
            if n1 <= N and Ccurr[row] + 1 <= c[row]:
                g[row, column] = 1
                Delta[row, column] = 0
                Ccurr[row] += 1
                Ncurr = n1
                Change[row, column] = True
            D[row, column] = 0

        for m in range(M):
            for j in range(km):
                if not Change[m, j]:
                    if Delta[m, j] + 1 > B - 1:
                        Delta[m, j] = B - 1
                    else:
                        Delta[m, j] = Delta[m, j] + 1
                pavg[t] += (w[m, j] * p[j, Delta[m, j]]) / (km * M)

        print(f"t={t+1}  Ncurr={Ncurr}")
        print("Delta =\n", Delta + 1)  # show 1-based for readability
        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    print(f"\npresult = {presult}")


if __name__ == "__main__":
    main()
