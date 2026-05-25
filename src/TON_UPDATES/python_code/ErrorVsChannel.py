"""
ErrorVsChannel.py - 1:1 Python port of ErrorVsChannel.m

Sweep over number of channels (N = 2..20 step 2) and plot the discounted
sum of errors for Random / MAF / MGF policies. Uses penalties loaded from
loss.mat (sub-sampled p1, p2 from the inference-error data).

Set TITER below to reduce the subgradient iteration count for quick smoke
tests. MATLAB default is 10000.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Allow importing local helpers when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MGF1 import MGF1
from MAF1 import MAF1
from MIEF1 import MIEF1
from randpolicy import randpolicy


TITER = int(os.environ.get("INFOCOM_TITER", "10000"))  # MATLAB default


SEED = int(os.environ.get("INFOCOM_SEED", "0"))


def main():
    np.random.seed(SEED)  # makes randpolicy reproducible across Python runs
    # Number of sources, channels, bound, and time
    M = 20
    N = 10
    km = 2
    B = 20
    T = 100

    # Channel needed
    n = np.ones((M, km))

    # Computation resources
    c = np.ones((M, km)) * 2

    # Different weight (both MATLAB branches set w(m,j)=0.01, then specific cells overridden)
    w = np.full((M, km), 0.01)
    w[0, 1] = 1   # MATLAB w(1, 2) = 1
    w[4, 0] = 1   # MATLAB w(5, 1) = 1

    K = T
    gamma = 0.9

    # Penalty function: subsample p1, p2 every 5 indices to length B
    loss = sio.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'loss.mat'))
    p1 = loss['p1'].flatten()
    p2 = loss['p2'].flatten()
    p = np.zeros((km, B))
    i = 0
    for j_mat in range(1, 101, 5):  # MATLAB j = 1:5:100, 20 values
        if i >= B:
            break
        j_py = j_mat - 1
        if j_py < p1.size:
            p[0, i] = p1[j_py]
        if j_py < p2.size:
            p[1, i] = p2[j_py]
        i += 1

    channel = np.arange(2, 21, 2)  # MATLAB 2:2:20
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

    here = os.path.dirname(os.path.abspath(__file__))
    # Save raw data as CSV so the comparison script can pick it up. The MIEF
    # column is appended last so the existing column layout used by
    # compare_with_matlab.py (prand, pmaf, pmgf at indices 1..3) is preserved.
    np.savetxt(os.path.join(here, 'ErrorVsChannel_data.csv'),
               np.column_stack([channel, prand, pmaf, pmgf, pmief]),
               delimiter=',', header='channel,prand,pmaf,pmgf,pmief', comments='')

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
    out = os.path.join(here, 'ErrorVsChannel.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out}")
    plt.show()


if __name__ == "__main__":
    main()
