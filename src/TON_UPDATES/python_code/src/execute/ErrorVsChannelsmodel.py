"""
ErrorVsChannelsmodel.py - 1:1 Python port of ErrorVsChannelsmodel.m

Sweeps over channels with km=9 model-based penalty functions:
  - tasks with j % 3 == 0 : linear (1..B)
  - tasks with j % 3 == 1 : 10*log(1..B)
  - tasks with j % 3 == 2 : exp(0.5*(1..B))
(1-indexed j in MATLAB; we replicate exactly.)

Writes: data/deterministic/ErrorVsChannelsmodel_data.csv
        plots/deterministic/ErrorVsChannelsmodel.png
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

    M = 20
    N = 10
    T = 100

    n = np.ones((M, km))
    c = np.ones((M, km)) * 2

    w = np.zeros((M, km))
    for m in range(M):
        for j in range(km):
            if (m + 1) > M / 2:
                w[m, j] = 0.01
            else:
                w[m, j] = 1

    channel = np.arange(2, 21, 2)
    K = T
    gamma = 0.9

    pmgf = np.zeros(len(channel))
    pmaf = np.zeros(len(channel))
    pmief = np.zeros(len(channel))
    prand = np.zeros(len(channel))

    for ch in range(len(channel)):
        N = int(channel[ch])
        print(f"\n=== N = {N} ===")
        pmgf[ch] = MGF1(M, N, km, T, B, K, n, c, w, gamma, p, titer=TITER)
        pmaf[ch] = MAF1(M, N, km, T, B, K, n, c, w, gamma, p)
        pmief[ch] = MIEF1(M, N, km, T, B, K, n, c, w, gamma, p)
        prand[ch] = randpolicy(M, N, km, T, B, K, n, c, w, gamma, p)

    csv_out = paths.data_path('ErrorVsChannelsmodel_data.csv', probabilistic=False)
    np.savetxt(csv_out,
               np.column_stack([channel, prand, pmaf, pmgf, pmief]),
               delimiter=',', header='channel,prand,pmaf,pmgf,pmief', comments='')
    print(f"\nSaved data to {csv_out}")

    plt.figure()
    plt.plot(channel, prand, 'ro-', label='Random Policy',
             linewidth=2, markersize=10)
    plt.plot(channel, pmaf, 'b*-', label='MAF Policy',
             linewidth=2, markersize=10)
    plt.plot(channel, pmgf, 'g-', label='MGF Policy',
             linewidth=2, markersize=10)
    plt.plot(channel, pmief, 'ms--', label='MIEF (Max Instantaneous Error First)',
             linewidth=2, markersize=10)
    plt.xlabel('channel')
    plt.ylabel('Discounted Sum of Errors')
    plt.legend()
    plt.title('ErrorVsChannelsmodel')
    png_out = paths.plot_path('ErrorVsChannelsmodel.png', probabilistic=False)
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {png_out}")
    plt.show()


if __name__ == "__main__":
    main()
