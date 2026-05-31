"""
MGF1_variants.py - method-selectable deterministic MGF.

Same control flow as MGF1.py but the multiplier learning method can be any
first-order method from optimizer_updates.get_default_subgradient_methods()
OR any cutting-plane method from cuttingplaneiter_variants.

Online (greedy) rollout is fully deterministic (q=1 implicit). For
probabilistic transmission, use MGF1_probabilistic.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subgradientiter1_variants import (
    subgradientiter1_variants,
    _batched_gain_table_variants,
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
    """MGF greedy resource-gated picker. Returns (Change, Ccurr, Ncurr)."""
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


def MGF1_variants(
    M, N, km, T, B, K, n, c, w, gamma, p,
    titer=10000,
    subgradient_method="harmonic",
    beta=0.9,
    seed=0,
    verbose=True,
    return_diagnostics=False,
    use_n_cost=False,
):
    """Method-selectable deterministic MGF."""
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    c_use = _as_1d_c(c, M)

    t0 = time.perf_counter()

    _FIRST_ORDER = set(get_default_subgradient_methods()) | {"episode1_mstep"}
    if subgradient_method in _FIRST_ORDER:
        lambdasource, mu, hist = subgradientiter1_variants(
            M, N, T, B, gamma, p_arr, km, w_arr, n_arr, c,
            titer=titer, method=subgradient_method,
            beta=beta, seed=seed, verbose=verbose,
            history=True, save=False, use_n_cost=use_n_cost,
        )
    elif subgradient_method in _CUTTING_PLANE_METHODS:
        from cuttingplaneiter_variants import cuttingplaneiter_variants
        lambdasource, mu, hist = cuttingplaneiter_variants(
            M, N, T, B, gamma, p_arr, km, w_arr, n_arr, c,
            max_iter=titer, method=subgradient_method, seed=seed,
            simplified=True, verbose=verbose, history=True,
            use_n_cost=use_n_cost,
        )
    else:
        raise ValueError(
            f"Unknown subgradient_method {subgradient_method!r}"
        )

    # Build gain-index table from the deterministic Bellman with learned (lam, mu).
    asource1 = _batched_gain_table_variants(
        lambdasource, mu, B, T, gamma, w_arr, p_arr, M, km,
        n=n_arr, use_n_cost=use_n_cost,
    )

    # Online greedy rollout (deterministic).
    Delta = np.zeros((M, km), dtype=np.int64)
    pavg = np.zeros(K)
    presult = 0.0
    m_idx, j_idx = np.meshgrid(np.arange(M), np.arange(km), indexing='ij')

    for t in range(K):
        gainindex = asource1[m_idx, t, Delta, j_idx].copy()
        Change, _, _ = _greedy_pick_with_gating(
            gainindex, n_arr, c_use, N, M, km,
        )
        Delta = np.where(Change, 0,
                         np.minimum(Delta + 1, B - 1)).astype(np.int64)

        pavg[t] = float((w_arr * p_arr[np.arange(km)[None, :], Delta]).sum()
                        / (km * M))
        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    runtime = time.perf_counter() - t0

    if verbose:
        print(f"MGF1_variants[{subgradient_method}] presult = {presult}")

    if not return_diagnostics:
        return float(presult)

    return {
        "objective": float(presult),
        "lambdasource": lambdasource,
        "mu": mu,
        "subgradient_history": hist,
        "runtime_sec": float(runtime),
    }
