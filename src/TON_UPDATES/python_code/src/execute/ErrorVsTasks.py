"""
ErrorVsTasks.py - 1:1 Python port of ErrorVsTasks.m

Sweep over number of tasks km = 3..15 step 3 with the same model-based
penalties as ErrorVsChannelsmodel.

Writes: data/deterministic/ErrorVsTasks_data.csv
        plots/deterministic/ErrorVsTasks.png
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
    p_full = np.zeros((15, B))
    for j in range(1, 15 + 1):
        if j % 3 == 0:
            p_full[j - 1, :B] = np.arange(1, B + 1)
        elif j % 3 == 1:
            p_full[j - 1, :B] = 10 * np.log(np.arange(1, B + 1))
        else:
            p_full[j - 1, :B] = np.exp(0.5 * np.arange(1, B + 1))

    M = 20
    N = 10
    T = 100
    Tasks = np.arange(3, 16, 3)
    K = T
    gamma = 0.9

    pmgf = np.zeros(len(Tasks))
    pmaf = np.zeros(len(Tasks))
    pmief = np.zeros(len(Tasks))
    prand = np.zeros(len(Tasks))

    for ch in range(len(Tasks)):
        km = int(Tasks[ch])
        print(f"\n=== km = {km} ===")
        n = np.ones((M, km))
        c = np.ones((M, km)) * 2
        w = np.zeros((M, km))
        for m in range(M):
            for j in range(km):
                if (m + 1) > M / 2:
                    w[m, j] = 0.01
                else:
                    w[m, j] = 1

        p = p_full[:km, :]
        pmgf[ch] = MGF1(M, N, km, T, B, K, n, c, w, gamma, p, titer=TITER)
        pmaf[ch] = MAF1(M, N, km, T, B, K, n, c, w, gamma, p)
        pmief[ch] = MIEF1(M, N, km, T, B, K, n, c, w, gamma, p)
        prand[ch] = randpolicy(M, N, km, T, B, K, n, c, w, gamma, p)

    csv_out = paths.data_path('ErrorVsTasks_data.csv', probabilistic=False)
    np.savetxt(csv_out,
               np.column_stack([Tasks, prand, pmaf, pmgf, pmief]),
               delimiter=',', header='tasks,prand,pmaf,pmgf,pmief', comments='')
    print(f"\nSaved data to {csv_out}")

    plt.figure()
    plt.plot(Tasks, prand, 'ro-', label='Random Policy',
             linewidth=2, markersize=10)
    plt.plot(Tasks, pmaf, 'b*-', label='MAF Policy',
             linewidth=2, markersize=10)
    plt.plot(Tasks, pmgf, 'g-', label='MGF Policy',
             linewidth=2, markersize=10)
    plt.plot(Tasks, pmief, 'ms--', label='MIEF (Max Instantaneous Error First)',
             linewidth=2, markersize=10)
    plt.xlabel('Number of Tasks')
    plt.ylabel('Discounted Sum of Errors')
    plt.yscale('log')
    plt.legend()
    plt.title('ErrorVsTasks')
    png_out = paths.plot_path('ErrorVsTasks.png', probabilistic=False)
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {png_out}")
    plt.show()


if __name__ == "__main__":
    main()
