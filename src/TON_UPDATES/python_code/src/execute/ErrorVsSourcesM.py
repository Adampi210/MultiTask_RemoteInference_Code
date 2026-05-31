"""
ErrorVsSourcesM.py - 1:1 Python port of ErrorVsSourcesM.m

Faithful quirk:
    In MATLAB, after the outer `for j=1:km` (penalty init), `j` retains its
    last value (km). The inner weight loop then writes only `w(m, j)` for that
    leftover `j`, leaving every other column of w at zero. We replicate this
    exactly so the discounted error values match MATLAB.

Writes: data/deterministic/ErrorVsSourcesM_data.csv
        plots/deterministic/ErrorVsSourcesM.png
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
    N = 10
    km = 9
    B = 20
    T = 100
    K = T
    gamma = 0.9

    p = np.zeros((km, B))
    leftover_j = None
    for j in range(1, km + 1):
        if j % 3 == 0:
            p[j - 1, :B] = np.arange(1, B + 1)
        elif j % 3 == 1:
            p[j - 1, :B] = 10 * np.log(np.arange(1, B + 1))
        else:
            p[j - 1, :B] = np.exp(0.5 * np.arange(1, B + 1))
        leftover_j = j

    sources = np.arange(3, 22, 3)
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
        j_idx = leftover_j - 1
        for m in range(M):
            if (m + 1) > 0.5 * M:
                w[m, j_idx] = 0.01
            else:
                w[m, j_idx] = 1

        pmgf[ch] = MGF1(M, N, km, T, B, K, n, c, w, gamma, p, titer=TITER)
        pmaf[ch] = MAF1(M, N, km, T, B, K, n, c, w, gamma, p)
        pmief[ch] = MIEF1(M, N, km, T, B, K, n, c, w, gamma, p)
        prand[ch] = randpolicy(M, N, km, T, B, K, n, c, w, gamma, p)

    csv_out = paths.data_path('ErrorVsSourcesM_data.csv', probabilistic=False)
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
    plt.ylabel('Discounted Sum of Avg. Error')
    plt.yscale('log')
    plt.legend()
    plt.title('ErrorVsSourcesM')
    png_out = paths.plot_path('ErrorVsSourcesM.png', probabilistic=False)
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {png_out}")
    plt.show()


if __name__ == "__main__":
    main()
