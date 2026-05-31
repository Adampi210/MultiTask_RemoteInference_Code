"""
experiment_utils.py - shared utilities for precheck/experiment scripts.

Builds the synthetic penalty matrix used throughout the paper/MATLAB code in
a single place, plus small dimensional helpers (per-source compute cap
broadcasting, discounted-budget bound).
"""
import numpy as np


def make_synthetic_p(km, B):
    """Synthetic penalty matrix p of shape (km, B).

    Uses the paper / MATLAB 1-indexed task convention:
        j % 3 == 0 -> linear:        p[j-1, d] = d+1       for d=0..B-1
        j % 3 == 1 -> 10 * log:      p[j-1, d] = 10*log(d+1)
        j % 3 == 2 -> exponential:   p[j-1, d] = exp(0.5*(d+1))

    AoI levels in code are 0..B-1 but the penalty arguments are evaluated at
    1..B (matching the existing ErrorVsSources.py / ErrorVsSourcesM.py code).
    """
    km = int(km)
    B = int(B)
    p = np.zeros((km, B))
    for j in range(1, km + 1):
        if j % 3 == 0:
            p[j - 1, :B] = np.arange(1, B + 1)
        elif j % 3 == 1:
            p[j - 1, :B] = 10.0 * np.log(np.arange(1, B + 1))
        else:
            p[j - 1, :B] = np.exp(0.5 * np.arange(1, B + 1))
    return p


def as_1d_c(c, M):
    """Accept c as either shape (M,) or (M, km), return shape (M,) float."""
    c = np.asarray(c)
    if c.ndim == 1:
        if c.shape[0] != M:
            raise ValueError(f"c has length {c.shape[0]}, expected {M}")
        return c.astype(float)
    return c[:, 0].astype(float)


def discounted_budget(T, gamma, budget):
    """Return sum_{t=0}^{T-1} gamma^t * budget."""
    T = int(T)
    gamma = float(gamma)
    if gamma == 1.0:
        return float(T) * float(budget)
    return float(budget) * (1.0 - gamma ** T) / (1.0 - gamma)


def make_base_problem(
    M=6,
    N=4,
    km=6,
    B=20,
    T=50,
    gamma=0.9,
    weights="ones",
    c_value=2,
    n_value=1,
):
    """Build a self-contained problem dict for precheck/experiment use.

    Parameters
    ----------
    M, N, km, B, T : int problem dimensions.
    gamma          : discount factor.
    weights        : "ones" (default) -> w = np.ones((M, km)).
                     "errorvssources" -> the original heterogeneous w used in
                                         ErrorVsSources.py (1 for m+1 <= M/2,
                                         else 0.01). Provided for parity tests
                                         only; probabilistic experiments use
                                         "ones" per the spec.
    c_value, n_value : scalar compute cap and channel cost (broadcast to (M,)
                       and (M, km) respectively).

    Returns
    -------
    dict with keys M, N, km, B, T, K, gamma, p, n, c, w.
    """
    M = int(M)
    N = int(N)
    km = int(km)
    B = int(B)
    T = int(T)
    K = T

    p = make_synthetic_p(km, B)
    n = np.full((M, km), float(n_value))
    c = np.full(M, float(c_value))

    if weights == "ones":
        w = np.ones((M, km))
    elif weights == "errorvssources":
        w = np.zeros((M, km))
        for m in range(M):
            w[m, :] = 1.0 if (m + 1) <= M / 2 else 0.01
    else:
        raise ValueError(f"Unknown weights mode {weights!r}")

    return {
        "M": M, "N": N, "km": km, "B": B, "T": T, "K": K,
        "gamma": float(gamma),
        "p": p, "n": n, "c": c, "w": w,
    }
