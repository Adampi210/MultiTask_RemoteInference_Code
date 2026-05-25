"""
lowerbound_probabilistic.py - approximate relaxed lower bound for the
probabilistic-transmission Lagrangian relaxation.

This is the natural relaxed analogue of lowerbound.py: greedy per-pair
scheduling decisions WITHOUT joint resource gating, with the Lagrangian
contributions added back. The Bellman is the probabilistic one (with q-reset
and mu * n_{m, j}). The AoI evolution uses *expected* values rather than
sampling, so the output is deterministic given the learned multipliers:

    if pi == 1:
        Delta(t+1) = q * 0 + (1 - q) * min(Delta + 1, B - 1)
                   -> use expectation in the penalty term.

Note: this is the relaxed lower-bound *approximation*; the exact lower bound
for the unreliable-channel WCMDP requires more involved analysis.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subgradientiter1_probabilistic import (
    subgradientiter1_probabilistic,
    load_multipliers_probabilistic,
    _batched_gain_table_probabilistic,
)


def lowerbound_probabilistic(
    M, N, km, T, B, K, n, c, w, gamma, p, q,
    titer=10000,
    subgradient_method="harmonic",
    seed=0,
    verbose=True,
):
    """Approximate relaxed lower bound (probabilistic variant)."""
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    subgradientiter1_probabilistic(
        M, N, T, B, gamma, p_arr, km, w_arr, n_arr, c, q_arr,
        titer=titer, method=subgradient_method, seed=seed,
        verbose=False,
    )
    lambdasource, mu = load_multipliers_probabilistic()

    asource1 = _batched_gain_table_probabilistic(
        lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km, q_arr, n_arr,
    )

    # Per-pair state. Use expected AoI in [0, B-1] (need not be integer for
    # penalty evaluation; we interpolate p linearly).
    Delta = np.zeros((M, km), dtype=float)
    pavg = np.zeros(K)
    presult = 0.0
    m_idx, task_idx = np.meshgrid(np.arange(M), np.arange(km), indexing='ij')

    def _expected_penalty(D, j):
        """Linear interp of p[j, :] at non-integer expected AoI D in [0, B-1]."""
        d0 = np.floor(D).astype(int)
        d1 = np.minimum(d0 + 1, B - 1)
        frac = D - d0
        return (1.0 - frac) * p_arr[j, d0] + frac * p_arr[j, d1]

    for t in range(K):
        # Gain index at floor(Delta) (relaxed policy uses integer state lookup).
        d_int = np.minimum(np.floor(Delta).astype(int), B - 1)
        gainindex = asource1[m_idx, t, d_int, task_idx]
        pi = (gainindex > 0).astype(float)

        # Expected next AoI (no resource gating: each source decides independently).
        aged = np.minimum(Delta + 1.0, B - 1)
        Delta_next = pi * ((1.0 - q_arr) * aged + q_arr * 0.0) + (1.0 - pi) * aged

        # Expected weighted penalty after the update.
        per_pair_pen = np.zeros((M, km))
        for j in range(km):
            per_pair_pen[:, j] = w_arr[:, j] * _expected_penalty(Delta_next[:, j], j)
        pavg[t] = float(per_pair_pen.sum() / (km * M))

        # Lagrangian contributions (per-source compute and channel).
        lam_contrib = float((lambdasource[:, t][:, None] * pi).sum())
        mu_contrib = float(mu[t] * (pi * n_arr).sum())

        if t + 1 < T:
            presult += (gamma ** t) * (pavg[t]
                                        + lam_contrib / (km * M)
                                        + mu_contrib / (km * M))

        Delta = Delta_next

    if verbose:
        print(f"lowerbound_probabilistic[{subgradient_method}] "
              f"presult = {presult}")
    return float(presult)
