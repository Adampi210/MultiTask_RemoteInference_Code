"""
checkrand.py - 1:1 Python port of checkrand.m

Stand-alone sanity check of the random policy with a synthetic penalty.
"""
import numpy as np

from _bootstrap import paths   # noqa: F401  (bootstraps sys.path)


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

    for t in range(K):
        g = np.zeros((M, km))
        Ccurr = np.zeros(M)
        Ncurr = 0.0
        Change = np.zeros((M, km), dtype=bool)

        source = np.random.permutation(M)[:min(N, M)]
        # MATLAB while-loop: fill until Ncurr >= N, indexing into `source`
        i = 0
        while Ncurr < N and i < len(source):
            row = int(source[i])
            # MATLAB randperm(km, min(c(row), km)) -> random subset of tasks
            n_pick = int(min(c[row], km))
            columnrand = np.random.permutation(km)[:n_pick]
            for column in columnrand:
                g[row, column] = 1
                Delta[row, column] = 0
                Ccurr[row] += 1
                Ncurr += n[row, column]
                Change[row, column] = True
            i += 1

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

    print(f"presult = {presult}")


if __name__ == "__main__":
    main()
