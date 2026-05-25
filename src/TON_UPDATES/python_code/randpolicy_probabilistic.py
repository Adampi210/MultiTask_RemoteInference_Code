"""
randpolicy_probabilistic.py - probabilistic-transmission random baseline.

Two variants:
  * gated=True  (default): respect per-source compute and total channel by
                            attempting uniformly-random feasible additions
                            until no more can be added.
  * gated=False:            mimic randpolicy.py -- pick min(N, M) sources, each
                            picks a uniformly random task, ignoring constraints.

In both cases the AoI update is stochastic: scheduled-and-successful resets
AoI to 0; otherwise AoI ages by 1.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def _ungated_select(M, km, N, rng):
    """Old randpolicy: pick min(N, M) sources, each picks a uniform task."""
    Change = np.zeros((M, km), dtype=bool)
    n_select = min(N, M)
    sources = rng.permutation(M)[:n_select]
    for row in sources:
        column = int(rng.integers(0, km))
        Change[row, column] = True
    return Change


def _gated_select(M, km, n, c_use, N, rng):
    """Sample uniformly random feasible additions until no more fit."""
    Change = np.zeros((M, km), dtype=bool)
    Ccurr = np.zeros(M)
    Ncurr = 0.0

    # Candidate (m, j) pairs in random order.
    flat_order = rng.permutation(M * km)
    for idx in flat_order:
        m, j = divmod(int(idx), km)
        if Change[m, j]:
            continue
        if Ccurr[m] + 1 > c_use[m]:
            continue
        cost = n[m, j]
        if Ncurr + cost > N:
            continue
        Change[m, j] = True
        Ccurr[m] += 1
        Ncurr += cost

    return Change


def _single_rollout(M, T, K, B, km, n, c_use, N, w, p, gamma, q, rng, gated):
    Delta = np.zeros((M, km), dtype=np.int64)
    pavg = np.zeros(K)
    presult = 0.0

    for t in range(K):
        if gated:
            Change = _gated_select(M, km, n, c_use, N, rng)
        else:
            Change = _ungated_select(M, km, N, rng)

        success = rng.random(size=(M, km)) < q
        reset_mask = Change & success
        Delta = np.where(reset_mask, 0,
                         np.minimum(Delta + 1, B - 1)).astype(np.int64)

        pavg[t] = float((w * p[np.arange(km)[None, :], Delta]).sum()
                        / (km * M))

        if t + 1 < T:
            presult += (gamma ** t) * pavg[t]

    return presult


def randpolicy_probabilistic(
    M, N, km, T, B, K, n, c, w, gamma, p, q,
    seed=0,
    mc_trials=1,
    gated=True,
    verbose=True,
):
    """Probabilistic random baseline. Default gated=True."""
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    c_use = _as_1d_c(c, M)

    base_rng = np.random.default_rng(seed)
    trial_seeds = base_rng.integers(0, 2**31 - 1, size=mc_trials)

    results = np.zeros(mc_trials)
    for trial in range(mc_trials):
        rng = np.random.default_rng(int(trial_seeds[trial]))
        results[trial] = _single_rollout(
            M, T, K, B, km, n_arr, c_use, N, w_arr, p_arr, gamma, q_arr, rng,
            gated,
        )

    if mc_trials == 1:
        out = float(results[0])
        if verbose:
            label = "rand_p(gated)" if gated else "rand_p(ungated)"
            print(f"{label} presult = {out}")
        return out

    out = {
        'mean': float(results.mean()),
        'std': float(results.std(ddof=1)) if mc_trials > 1 else 0.0,
        'trials': results,
    }
    if verbose:
        label = "rand_p(gated)" if gated else "rand_p(ungated)"
        print(f"{label} presult = {out['mean']:.6g} +- {out['std']:.3g} "
              f"({mc_trials} trials)")
    return out
