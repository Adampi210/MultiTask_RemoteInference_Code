"""
ErrorVsSources.py - 1:1 Python port of ErrorVsSources.m

Note: The MATLAB ErrorVsSources.m references `M` on line 25 BEFORE M is defined.
That `n=ones(M,km)` will use the workspace value from a prior session (or fail
on a clean start). We faithfully reproduce the loop body; we initialize n and c
inside the `sources` loop (where M is known) so the script runs.

Writes: data/deterministic/ErrorVsSources_data.csv
        plots/deterministic/ErrorVsSources.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from _bootstrap import paths

from MGF1 import MGF1
from MAF1 import MAF1
from MIEF1 import MIEF1
from randpolicy import randpolicy


TITER = int(os.environ.get("INFOCOM_TITER", "10000"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))


def main():
    np.random.seed(SEED)
    B = 20
    km = 9
    p = np.zeros((km, B))
    for j in range(1, km + 1):
        if j % 3 == 0:
            p[j - 1, :B] = np.arange(1, B + 1)
        elif j % 3 == 1:
            p[j - 1, :B] = 10 * np.log(np.arange(1, B + 1))
        else:
            p[j - 1, :B] = np.exp(0.5 * np.arange(1, B + 1))

    N = 10
    T = 100

    sources = np.arange(2, 21, 2)
    K = T
    gamma = 0.9

    pmgf = np.zeros(len(sources))
    pmaf = np.zeros(len(sources))
    pmief = np.zeros(len(sources))
    prand = np.zeros(len(sources))

    for ch in range(len(sources)):
        M = int(sources[ch])
        print(f"\n=== M = {M} ===")
        n = np.ones((M, km))
        c = np.ones((M, km)) * 2

        w = np.zeros((M, km))
        for m in range(M):
            for j in range(km):
                if (m + 1) > M / 2:
                    w[m, j] = 0.01
                else:
                    w[m, j] = 1

        pmgf[ch] = MGF1(M, N, km, T, B, K, n, c, w, gamma, p, titer=TITER)
        pmaf[ch] = MAF1(M, N, km, T, B, K, n, c, w, gamma, p)
        pmief[ch] = MIEF1(M, N, km, T, B, K, n, c, w, gamma, p)
        prand[ch] = randpolicy(M, N, km, T, B, K, n, c, w, gamma, p)

    csv_out = paths.data_path('ErrorVsSources_data.csv', probabilistic=False)
    np.savetxt(csv_out,
               np.column_stack([sources, prand, pmaf, pmgf, pmief]),
               delimiter=',', header='sources,prand,pmaf,pmgf,pmief', comments='')
    print(f"\nSaved data to {csv_out}")

    plt.figure()
    plt.plot(sources, prand, 'ro-', label='Random Policy',
             linewidth=2, markersize=10)
    plt.plot(sources, pmaf, 'b*-', label='MAF Policy',
             linewidth=2, markersize=10)
    plt.plot(sources, pmgf, 'g-', label='MGF Policy',
             linewidth=2, markersize=10)
    plt.plot(sources, pmief, 'ms--', label='MIEF (Max Instantaneous Error First)',
             linewidth=2, markersize=10)
    plt.xlabel('Number of Sources')
    plt.ylabel('Discounted Sum of Errors')
    plt.yscale('log')
    plt.legend()
    plt.title('ErrorVsSources')
    png_out = paths.plot_path('ErrorVsSources.png', probabilistic=False)
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {png_out}")
    plt.show()


if __name__ == "__main__":
    main()
