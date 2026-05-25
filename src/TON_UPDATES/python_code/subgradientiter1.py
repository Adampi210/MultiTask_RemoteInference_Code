"""
subgradientiter1.py - 1:1 Python port of subgradientiter1.m

Runs the projected subgradient method for Titer iterations to learn the
Lagrange multipliers (lambdasource, mu). Writes the result to multipliers.mat.

The expensive bit is the M*km independent value-function recursions per outer
iter, all of which share the same mu and the same lambda (per source). We
batch them into a single tensorized Bellman backward pass for a big speedup.
"""
import os
import numpy as np
import scipy.io as sio

from valuefunction1 import valuefunction1  # kept for compare/check scripts
from Episode1 import Episode1


_MULTIPLIERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'multipliers.mat')


def _batched_gain_table(lambdasource, mu, B, T, gamma, w, p, M, km):
    """Compute asource1[m, t, d, task] via a single batched Bellman recursion.

    Identical algorithm to looping valuefunction1(...) over (m, task), but
    M*km copies are evaluated in parallel.
    """
    p = np.asarray(p)[:, :B]                          # (km, B)
    w = np.asarray(w)                                 # (M, km)
    P = (w[:, :, None] * p[None, :, :]).reshape(M * km, B)
    lam = np.repeat(lambdasource, km, axis=0)         # (M*km, T)
    mu_arr = np.asarray(mu).flatten()                 # (T,)

    V = np.zeros((M * km, T + 1, B + 1))
    a = np.zeros((M * km, T, B))

    for i in range(T - 1, -1, -1):
        Q1 = P + gamma * V[:, i + 1, 1:B + 1]                          # (MK, B)
        Q2 = P + lam[:, i:i + 1] + mu_arr[i] + gamma * V[:, i + 1, 0:1]
        V[:, i, :B] = np.minimum(Q1, Q2)
        a[:, i, :] = Q1 - Q2
        V[:, i, B] = V[:, i, B - 1]

    # a.shape = (M*km, T, B) -> asource1.shape = (M, T, B, km)
    return a.reshape(M, km, T, B).transpose(0, 2, 3, 1)


def subgradientiter1(M, N, T, B, gamma, p, km, w, n, c, titer=10000, verbose=False):
    lambdasource = np.zeros((M, T))
    mu = np.zeros(T)
    beta = 0.9
    p = np.asarray(p)
    w = np.asarray(w)

    A = None
    for j in range(1, titer + 1):  # MATLAB j = 1..Titer
        asource1 = _batched_gain_table(lambdasource, mu, B, T, gamma, w, p, M, km)
        A = Episode1(asource1, M, T, B, gamma, N, beta / j,
                     lambdasource, mu, km, w, n, c)
        lambdasource = A[:M, :]
        mu = A[M, :]

        if verbose and (j % max(1, titer // 20) == 0 or j == 1):
            print(f"  subgradient iter {j}/{titer}  "
                  f"max|lambda|={np.max(np.abs(lambdasource)):.4f}  "
                  f"max|mu|={np.max(np.abs(mu)):.4f}")

    sio.savemat(_MULTIPLIERS_PATH,
                {'lambdasource': lambdasource,
                 'mu': mu.reshape(1, -1)})
    return A


def load_multipliers():
    data = sio.loadmat(_MULTIPLIERS_PATH)
    return data['lambdasource'], data['mu'].flatten()
