"""
MIEF1_probabilistic.py - probabilistic-transmission MIEF
(Max Instantaneous Error First).

Default priority is q[m,j] * w[m,j] * p[j, Delta[m,j]], i.e., expected
instantaneous usefulness. Pass reliability_aware=False to recover the pure
MIEF priority w * p(Delta) for direct comparison.

AoI evolution after greedy scheduling is stochastic.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from greedy_scheduler import greedy_select, as_1d_c


def _priority(Delta, w, p, km, q, reliability_aware):
    task_idx = np.broadcast_to(np.arange(km), Delta.shape)
    base = w * p[task_idx, Delta]
    if reliability_aware:
        base = q * base
    return base


def _single_rollout(M, T, K, B, km, n, c_use, N, w, p, gamma, q, rng,
                    reliability_aware):
    Delta = np.zeros((M, km), dtype=np.int64)
    pavg = np.zeros(K)
    presult = 0.0

    for t in range(K):
        priority = _priority(Delta, w, p, km, q, reliability_aware)
        _, Change, _, _ = greedy_select(priority, n, c_use, N, M, km)

        success = rng.random(size=(M, km)) < q
        reset_mask = Change & success
        Delta = np.where(reset_mask, 0,
                         np.minimum(Delta + 1, B - 1)).astype(np.int64)

        pavg[t] = float((w * p[np.arange(km)[None, :], Delta]).sum()
                        / (km * M))

        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    return presult


def MIEF1_probabilistic(
    M, N, km, T, B, K, n, c, w, gamma, p, q,
    seed=0,
    mc_trials=1,
    reliability_aware=True,
    verbose=True,
):
    """Probabilistic MIEF. Default reliability_aware=True."""
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    c_use = as_1d_c(c, M)

    base_rng = np.random.default_rng(seed)
    trial_seeds = base_rng.integers(0, 2**31 - 1, size=mc_trials)

    results = np.zeros(mc_trials)
    for trial in range(mc_trials):
        rng = np.random.default_rng(int(trial_seeds[trial]))
        results[trial] = _single_rollout(
            M, T, K, B, km, n_arr, c_use, N, w_arr, p_arr, gamma, q_arr, rng,
            reliability_aware,
        )

    if mc_trials == 1:
        out = float(results[0])
        if verbose:
            label = "MIEF1_p(rel)" if reliability_aware else "MIEF1_p(pure)"
            print(f"{label} presult = {out}")
        return out

    out = {
        'mean': float(results.mean()),
        'std': float(results.std(ddof=1)) if mc_trials > 1 else 0.0,
        'trials': results,
    }
    if verbose:
        label = "MIEF1_p(rel)" if reliability_aware else "MIEF1_p(pure)"
        print(f"{label} presult = {out['mean']:.6g} +- {out['std']:.3g} "
              f"({mc_trials} trials)")
    return out
