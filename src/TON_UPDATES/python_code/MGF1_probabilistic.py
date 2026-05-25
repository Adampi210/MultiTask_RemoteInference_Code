"""
MGF1_probabilistic.py - probabilistic-transmission MGF (Max-Gain-First).

Same control flow as MGF1.py with two changes:
  1. Multipliers and gain table use the probabilistic Bellman (mu * n_{m,j}
     and q-weighted reset).
  2. AoI evolution after greedy scheduling is stochastic: scheduled pairs
     reset only if Bernoulli(q[m, j]) draws success; failed transmissions age
     by 1.

Constraints (per-source compute c[m], total channel N) are enforced on the
attempted action regardless of delivery success.

Multiplier learning supports any first-order method from optimizer_updates
(``harmonic``, ``sqrt``, ``normalized_global``, ``normalized_blocks``,
``adagrad``, ``rmsprop``, ``adam``, ``deflected_sqrt``) and the cutting-plane
methods (``kelley_bounded``, ``trust_region_kelley``, ``proximal_bundle``).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subgradientiter1_probabilistic import (
    subgradientiter1_probabilistic,
    _batched_gain_table_probabilistic,
)
from optimizer_updates import get_default_subgradient_methods


_CUTTING_PLANE_METHODS = {"kelley_bounded", "trust_region_kelley",
                          "proximal_bundle"}


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def _greedy_pick_with_gating(gainindex, n, c_use, N, M, km):
    """MGF greedy resource-gated picker."""
    G = gainindex.copy()
    Change = np.zeros((M, km), dtype=bool)
    Ccurr = np.zeros(M)
    Ncurr = 0.0
    while G.max() > 0:
        col_max = G.max(axis=0)
        col_argmax = G.argmax(axis=0)
        column = int(col_max.argmax())
        row = int(col_argmax[column])
        n1 = Ncurr + n[row, column]
        if n1 <= N and Ccurr[row] + 1 <= c_use[row]:
            Change[row, column] = True
            Ccurr[row] += 1
            Ncurr = n1
        G[row, column] = 0
    return Change, Ccurr, Ncurr


def _single_rollout(asource1, Delta_init, M, T, K, B, km, n, c_use, N,
                    w, p, gamma, q, rng):
    """One stochastic rollout. Returns discounted weighted-error scalar."""
    Delta = Delta_init.copy()
    pavg = np.zeros(K)
    presult = 0.0
    m_idx, task_idx = np.meshgrid(np.arange(M), np.arange(km), indexing='ij')

    for t in range(K):
        gainindex = asource1[m_idx, t, Delta, task_idx]
        Change, _, _ = _greedy_pick_with_gating(gainindex, n, c_use, N, M, km)

        success = rng.random(size=(M, km)) < q
        reset_mask = Change & success
        Delta = np.where(reset_mask, 0,
                         np.minimum(Delta + 1, B - 1)).astype(np.int64)

        pavg[t] = float((w * p[np.arange(km)[None, :], Delta]).sum()
                        / (km * M))
        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    return presult


def _learn_multipliers(
    M, N, T, B, gamma, p, km, w, n, c, q,
    titer, subgradient_method, beta, seed, verbose,
):
    """Run the chosen multiplier-learning routine and return (lam, mu, hist)."""
    if subgradient_method in get_default_subgradient_methods() or \
            subgradient_method in ("constant", "normalized", "polyak_like"):
        A, hist = subgradientiter1_probabilistic(
            M, N, T, B, gamma, p, km, w, n, c, q,
            titer=titer, method=subgradient_method,
            beta=beta, seed=seed, verbose=verbose,
            history=True, save=False,
        )
        return A[:M, :].copy(), A[M, :].copy(), hist

    if subgradient_method in _CUTTING_PLANE_METHODS:
        from cuttingplaneiter_probabilistic import cuttingplaneiter_probabilistic
        return cuttingplaneiter_probabilistic(
            M, N, T, B, gamma, p, km, w, n, c, q,
            max_iter=titer, method=subgradient_method, seed=seed,
            simplified=True, verbose=verbose, history=True,
        )

    raise ValueError(
        f"Unknown subgradient_method {subgradient_method!r}"
    )


def MGF1_probabilistic(
    M, N, km, T, B, K, n, c, w, gamma, p, q,
    titer=10000,
    subgradient_method="harmonic",
    beta=0.9,
    seed=0,
    mc_trials=1,
    verbose=True,
    return_diagnostics=False,
):
    """Probabilistic MGF.

    Returns
    -------
    If return_diagnostics=False:
        mc_trials == 1 -> float (single-trial discounted weighted error).
        mc_trials  > 1 -> dict with keys mean, std, trials.
    If return_diagnostics=True:
        dict with objective_mean / objective_std / objective_trials,
        lambdasource, mu, subgradient_history, runtime_sec, q_summary.
    """
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    c_use = _as_1d_c(c, M)

    t0 = time.perf_counter()

    lambdasource, mu, hist = _learn_multipliers(
        M, N, T, B, gamma, p_arr, km, w_arr, n_arr, c, q_arr,
        titer, subgradient_method, beta, seed,
        verbose and mc_trials == 1,
    )

    asource1 = _batched_gain_table_probabilistic(
        lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km, q_arr, n_arr,
    )

    base_rng = np.random.default_rng(seed)
    trial_seeds = base_rng.integers(0, 2**31 - 1, size=mc_trials)

    results = np.zeros(mc_trials)
    Delta_init = np.zeros((M, km), dtype=np.int64)
    for trial in range(mc_trials):
        rng = np.random.default_rng(int(trial_seeds[trial]))
        results[trial] = _single_rollout(
            asource1, Delta_init, M, T, K, B, km, n_arr, c_use, N,
            w_arr, p_arr, gamma, q_arr, rng,
        )

    runtime = time.perf_counter() - t0

    if return_diagnostics:
        out = {
            "objective_mean": float(results.mean()),
            "objective_std": (float(results.std(ddof=1)) if mc_trials > 1
                              else 0.0),
            "objective_trials": results.tolist(),
            "lambdasource": lambdasource,
            "mu": mu,
            "subgradient_history": hist,
            "runtime_sec": float(runtime),
            "q_summary": {
                "q_min": float(q_arr.min()),
                "q_max": float(q_arr.max()),
                "q_mean": float(q_arr.mean()),
                "q_std": float(q_arr.std()),
            },
        }
        if verbose:
            print(f"MGF1_probabilistic[{subgradient_method}] "
                  f"objective = {out['objective_mean']:.6g} "
                  f"+- {out['objective_std']:.3g} ({mc_trials} trials)")
        return out

    if mc_trials == 1:
        val = float(results[0])
        if verbose:
            print(f"MGF1_probabilistic[{subgradient_method}] presult = {val}")
        return val

    out = {
        "mean": float(results.mean()),
        "std": float(results.std(ddof=1)) if mc_trials > 1 else 0.0,
        "trials": results,
    }
    if verbose:
        print(f"MGF1_probabilistic[{subgradient_method}] presult = "
              f"{out['mean']:.6g} +- {out['std']:.3g} ({mc_trials} trials)")
    return out
