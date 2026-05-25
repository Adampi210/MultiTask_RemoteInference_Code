"""
valuefunction1.py - 1:1 Python port of valuefunction1.m

Computes Bellman value function V and gain index 'a' for a single source/task
via backward induction.

MATLAB original:
    V is (T+1, B+1), a is (T, B).
    For i = T downto 1, d = 1..B:
        Q1 = p(d) + gamma * V(i+1, d+1)
        Q2 = p(d) + lambda(i) + mu(i) + gamma * V(i+1, 1)
        V(i, d) = min(Q1, Q2)
        a(i, d) = Q1 - Q2
    V(i, B+1) = V(i, B)
"""
import numpy as np


def valuefunction1(lambda_, mu, B, p, T, gamma):
    """
    Parameters
    ----------
    lambda_ : 1D array of length T (Lagrange multipliers for the source).
    mu      : 1D array of length T (Lagrange multipliers for the channel).
    B       : int, AoI bound.
    p       : 1D array of length B (per-AoI penalty values, already weighted).
    T       : int, horizon.
    gamma   : float, discount factor.

    Returns
    -------
    a : (T, B) numpy array, where a[i, d] = Q1 - Q2 (gain index).
    """
    lambda_ = np.asarray(lambda_).flatten()
    mu = np.asarray(mu).flatten()
    p = np.asarray(p).flatten()

    V = np.zeros((T + 1, B + 1))
    a = np.zeros((T, B))

    for i in range(T - 1, -1, -1):  # MATLAB i = T..1 maps to Python i = T-1..0
        # Vectorize over d = 0..B-1 (MATLAB d = 1..B)
        Q1 = p[:B] + gamma * V[i + 1, 1:B + 1]
        Q2 = p[:B] + lambda_[i] + mu[i] + gamma * V[i + 1, 0]
        V[i, :B] = np.minimum(Q1, Q2)
        a[i, :] = Q1 - Q2
        # Mirror MATLAB: V(i, B+1) = V(i, B) so an out-of-bound AoI stays capped.
        V[i, B] = V[i, B - 1]

    return a
