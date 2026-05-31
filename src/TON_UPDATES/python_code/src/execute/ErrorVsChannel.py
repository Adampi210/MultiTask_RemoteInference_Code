"""
ErrorVsChannel.py - 1:1 Python port of ErrorVsChannel.m

Sweep over number of channels (N = 2..20 step 2) and plot the discounted
sum of errors for Random / MAF / MGF policies. Uses penalties loaded from
loss.mat (sub-sampled p1, p2 from the inference-error data).

Set TITER below to reduce the subgradient iteration count for quick smoke
tests. MATLAB default is 10000.

Reads:  data/deterministic/loss.mat
Writes: data/deterministic/ErrorVsChannel_data.csv
        plots/deterministic/ErrorVsChannel.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from _bootstrap import paths  # bootstraps sys.path to find src/main/

from MGF1 import MGF1
from MAF1 import MAF1
from MIEF1 import MIEF1
from randpolicy import randpolicy


TITER = int(os.environ.get("INFOCOM_TITER", "10000"))  # MATLAB default
SEED = int(os.environ.get("INFOCOM_SEED", "0"))


def main():
    np.random.seed(SEED)  # makes randpolicy reproducible across Python runs
    M = 20
    N = 10
    km = 2
    B = 20
    T = 100

    n = np.ones((M, km))
    c = np.ones((M, km)) * 2

    # MATLAB branches set w(m,j)=0.01 then override specific cells.
    w = np.full((M, km), 0.01)
    w[0, 1] = 1   # MATLAB w(1, 2) = 1
    w[4, 0] = 1   # MATLAB w(5, 1) = 1

    K = T
    gamma = 0.9

    # Penalty function: subsample p1, p2 every 5 indices to length B.
    loss = sio.loadmat(paths.data_path('loss.mat', probabilistic=False))
    p1 = loss['p1'].flatten()
    p2 = loss['p2'].flatten()
    p = np.zeros((km, B))
    i = 0
    for j_mat in range(1, 101, 5):
        if i >= B:
            break
        j_py = j_mat - 1
        if j_py < p1.size:
            p[0, i] = p1[j_py]
        if j_py < p2.size:
            p[1, i] = p2[j_py]
        i += 1

    channel = np.arange(2, 21, 2)
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

    csv_out = paths.data_path('ErrorVsChannel_data.csv', probabilistic=False)
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
    plt.xlabel('Number of Channels')
    plt.ylabel('Discounted Sum of Errors')
    plt.legend()
    plt.title('ErrorVsChannel')
    png_out = paths.plot_path('ErrorVsChannel.png', probabilistic=False)
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {png_out}")
    plt.show()


if __name__ == "__main__":
    main()
