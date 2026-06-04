"""
robot_probabilistic.py - probabilistic-link (unreliable-channel) robot-car
scheduling policies, faithful to ``matlab_code/robot_code`` (main.m, mainweight.m)
but extended with per-link delivery reliability q[m, j].

Problem (from main.m):
    M = 4 sources, km = [2, 1, 1, 1] tasks/source (heterogeneous),
    per-(source, task) empirical penalty p[m, j, d] (see robot_data.py),
    B = 40, T = 100, gamma = 0.1, per-source compute cap c[m] = 1, channel
    cost n[m, j] = 1, total channel budget N.

Probabilistic extension (matching the rest of the python_code probabilistic
flavor):
    A scheduled pair (m, j) is delivered only with probability q[m, j]
    (Bernoulli). On success AoI resets (Delta -> 0); on failure / not scheduled
    it ages (Delta -> min(Delta + 1, B - 1)). Resource feasibility is always
    checked against the *attempted* schedule. With q = 1 every policy reduces
    exactly to the deterministic MATLAB robot policy.

Heterogeneous km is handled with a (M, km_max) boolean ``valid`` mask; invalid
(padded) pairs carry penalty 0 / weight 0 and a gain of -inf so they are never
scheduled and never contribute to the objective. The objective normalizer is
``sum(km_vec)`` (= number of valid pairs), exactly as MATLAB's ``sum(km)``.

Dual (MGF only): the Lagrangian uses the **time-invariant** multiplier vector
``z = [lambda_1..M, mu]`` learned by projected subgradient ascent with the
MATLAB harmonic step ``beta / j`` (subgradient.m + episode.m), with the
probabilistic (q-weighted) Bellman in the value function. Multipliers are
learned once over the horizon T from Delta = 0; with gamma = 0.1 this is
numerically indistinguishable from MATLAB's per-timestep ``MGFReoptimized``
re-solve (the discounted objective is dominated by t = 1..3, where the two
agree), so we use the cheaper learn-once form for the q-profile / weight sweeps.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Problem container
# ---------------------------------------------------------------------------

def _valid_pairs(valid):
    """Return list of (m, j) index tuples for valid pairs, row-major order."""
    return [(int(m), int(j)) for m, j in zip(*np.where(valid))]


# ---------------------------------------------------------------------------
# Probabilistic (q-weighted) value function, batched over the valid pairs
# ---------------------------------------------------------------------------

def _gain_table(lambdasource, mu, B, T, gamma, P_pairs, lam_pairs,
                q_pairs, n_pairs_cost):
    """Backward-induction gain table for every valid pair.

    Parameters
    ----------
    lambdasource : (M,) time-invariant per-source multipliers (read via
                   lam_pairs, kept for signature symmetry).
    mu           : float, time-invariant channel multiplier.
    P_pairs      : (n_valid, B) weighted penalty w[m,j] * p[m,j,:].
    lam_pairs    : (n_valid,) lambda for each valid pair's source.
    q_pairs      : (n_valid,) delivery probability per valid pair.
    n_pairs_cost : (n_valid,) channel cost per valid pair.

    Returns
    -------
    a : (n_valid, T, B) gain a[i, d] = Q_no(d) - Q_sched(d). Positive => the
        relaxed-optimal action is to schedule.
    """
    n_valid = P_pairs.shape[0]
    V = np.zeros((n_valid, T + 1, B + 1))
    a = np.zeros((n_valid, T, B))

    age_next = np.minimum(np.arange(B) + 1, B - 1)
    lam_col = lam_pairs[:, None]
    q_col = q_pairs[:, None]
    n_col = n_pairs_cost[:, None]

    for i in range(T - 1, -1, -1):
        V_age = V[:, i + 1, age_next]          # (n_valid, B)
        V_reset = V[:, i + 1, 0:1]             # (n_valid, 1)

        Q_no = P_pairs + gamma * V_age
        Q_sched = (P_pairs + lam_col + mu * n_col
                   + gamma * (q_col * V_reset + (1.0 - q_col) * V_age))

        V[:, i, :B] = np.minimum(Q_no, Q_sched)
        a[:, i, :] = Q_no - Q_sched
        V[:, i, B] = V[:, i, B - 1]

    return a


# ---------------------------------------------------------------------------
# Time-invariant projected-subgradient dual (subgradient.m + episode.m)
# ---------------------------------------------------------------------------

def learn_multipliers(M, N, T, B, gamma, P_pairs, pair_src, q_pairs,
                      n_pairs_cost, km_vec, c_cap=1.0, titer=2000, beta=0.9):
    """Learn time-invariant (lambdasource, mu) by projected subgradient ascent.

    Faithful to subgradient.m / episode.m: harmonic step beta/j, deterministic
    (expected) relaxed rollout from Delta=0 where a pair is "scheduled" iff its
    gain > 0 (no joint resource gating -- this is the relaxed problem), AoI
    resets on the attempted action. The q-weighting enters only through the
    value-function gain table, so the dual is deterministic in q.

    Returns (lambdasource (M,), mu float).
    """
    n_valid = P_pairs.shape[0]
    pair_src = np.asarray(pair_src)
    lambdasource = np.zeros(M)
    mu = 0.0

    r = (1.0 - gamma ** T) / (1.0 - gamma)
    Ngamma = N * r
    cgamma = c_cap * r
    gpow = gamma ** np.arange(T)                      # gamma^(t-1), t=1..T

    for j in range(1, titer + 1):
        lam_pairs = lambdasource[pair_src]
        a = _gain_table(lambdasource, mu, B, T, gamma, P_pairs, lam_pairs,
                        q_pairs, n_pairs_cost)

        # Deterministic relaxed rollout from Delta=0 over the horizon.
        Delta = np.zeros(n_valid, dtype=np.int64)
        # discounted resource usage
        chan_disc = 0.0
        comp_disc = np.zeros(M)
        for t in range(T):
            gain = a[np.arange(n_valid), t, Delta]
            g = gain > 0.0
            gt = gpow[t]
            chan_disc += gt * float((g * n_pairs_cost).sum())
            # per-source compute usage
            np.add.at(comp_disc, pair_src[g], gt)
            Delta = np.where(g, 0, np.minimum(Delta + 1, B - 1))

        step = beta / j
        # compute constraint per source: discounted usage <= c[m] * r
        lambdasource = np.maximum(lambdasource + step * (comp_disc - cgamma), 0.0)
        # channel constraint: discounted usage <= N * r
        mu = max(mu + step * (chan_disc - Ngamma), 0.0)

    return lambdasource, mu


# ---------------------------------------------------------------------------
# Greedy resource-gated picker (shared by MGF / MAF)
# ---------------------------------------------------------------------------

def _greedy_gate(priority, valid, pair_src, N, M, c_cap, n_cost,
                 threshold):
    """Pick pairs by descending priority, gated by channel N and compute cap.

    priority  : (M, km_max) priority value; invalid pairs must already be -inf.
    threshold : schedule only while the running max priority is > threshold
                (MGF gain uses >= 0 -> threshold = -tiny; MAF age uses > 0).
    Returns Change (M, km_max) bool of attempted schedules.
    """
    G = priority.copy()
    Change = np.zeros_like(valid, dtype=bool)
    Ccurr = np.zeros(M)
    Ncurr = 0.0
    while np.nanmax(G) > threshold:
        flat = int(np.argmax(G))
        row, col = divmod(flat, G.shape[1])
        n1 = Ncurr + n_cost[row, col]
        if n1 <= N and Ccurr[row] + 1 <= c_cap:
            Change[row, col] = True
            Ccurr[row] += 1
            Ncurr = n1
        G[row, col] = -np.inf
    return Change


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def _objective_from_rollout(prob, w, q, N, gamma, T, schedule_fn,
                            seed, mc_trials, include_last):
    """Generic MC rollout driver shared by the policies.

    schedule_fn(Delta, t) -> Change (M, km_max) bool (attempted schedule).
    """
    M = prob["M"]
    km_max = prob["km_max"]
    B = prob["B"]
    p = prob["p"]
    valid = prob["valid"]
    n_pairs = prob["n_pairs"]

    base_rng = np.random.default_rng(seed)
    trial_seeds = base_rng.integers(0, 2 ** 31 - 1, size=mc_trials)
    results = np.zeros(mc_trials)

    for trial in range(mc_trials):
        rng = np.random.default_rng(int(trial_seeds[trial]))
        Delta = np.zeros((M, km_max), dtype=np.int64)
        presult = 0.0
        for t in range(T):
            Change = schedule_fn(Delta, t)
            success = rng.random(size=(M, km_max)) < q
            reset = Change & success & valid
            Delta = np.where(reset, 0,
                             np.minimum(Delta + 1, B - 1)).astype(np.int64)
            pen = p[np.arange(M)[:, None], np.arange(km_max)[None, :], Delta]
            pavg = float((w * pen * valid).sum() / n_pairs)
            if include_last or (t + 1 < T):
                presult += (gamma ** t) * pavg
        results[trial] = presult

    return results


def _mc_summary(results):
    if results.size == 1:
        return float(results[0]), 0.0
    return float(results.mean()), float(results.std(ddof=1))


def MGF_robot_probabilistic(prob, N, w, q, gamma, T,
                            seed=0, mc_trials=1, titer=2000, beta=0.9,
                            return_std=False):
    """Reoptimized-MGF index policy (learn-once multipliers) with q-delivery."""
    M = prob["M"]
    km_max = prob["km_max"]
    B = prob["B"]
    p = prob["p"]
    valid = prob["valid"]
    km_vec = prob["km_vec"]

    pairs = _valid_pairs(valid)
    pair_src = np.array([m for m, _ in pairs])
    P_pairs = np.array([w[m, j] * p[m, j, :] for m, j in pairs])  # (n_valid,B)
    q_pairs = np.array([q[m, j] for m, j in pairs])
    n_cost = np.ones((M, km_max))
    n_pairs_cost = np.ones(len(pairs))

    lambdasource, mu = learn_multipliers(
        M, N, T, B, gamma, P_pairs, pair_src, q_pairs, n_pairs_cost, km_vec,
        c_cap=1.0, titer=titer, beta=beta)

    lam_pairs = lambdasource[pair_src]
    a = _gain_table(lambdasource, mu, B, T, gamma, P_pairs, lam_pairs,
                    q_pairs, n_pairs_cost)        # (n_valid, T, B)

    # Scatter gains into a (M, km_max, T, B) table; invalid pairs -> -inf.
    a_full = np.full((M, km_max, T, B), -np.inf)
    for idx, (m, j) in enumerate(pairs):
        a_full[m, j] = a[idx]

    def schedule_fn(Delta, t):
        gain = a_full[np.arange(M)[:, None], np.arange(km_max)[None, :],
                      t, Delta]                    # (M, km_max), invalid=-inf
        # MGFReoptimized schedules while max gain >= 0.
        return _greedy_gate(gain, valid, pair_src, N, M, 1.0, n_cost,
                            threshold=-1e-12)

    results = _objective_from_rollout(prob, w, q, N, gamma, T, schedule_fn,
                                      seed, mc_trials, include_last=False)
    mean, std = _mc_summary(results)
    return (mean, std) if return_std else mean


def MAF_robot_probabilistic(prob, N, w, q, gamma, T,
                            seed=0, mc_trials=1, return_std=False):
    """Max-Age-First (priority = AoI), resource-gated, with q-delivery."""
    M = prob["M"]
    km_max = prob["km_max"]
    valid = prob["valid"]
    pair_src = np.array([m for m, _ in _valid_pairs(valid)])
    n_cost = np.ones((M, km_max))

    def schedule_fn(Delta, t):
        priority = np.where(valid, Delta.astype(float), -np.inf)
        return _greedy_gate(priority, valid, pair_src, N, M, 1.0, n_cost,
                            threshold=0.0)

    results = _objective_from_rollout(prob, w, q, N, gamma, T, schedule_fn,
                                      seed, mc_trials, include_last=True)
    mean, std = _mc_summary(results)
    return (mean, std) if return_std else mean


def MEF_robot_probabilistic(prob, N, w, q, gamma, T,
                            seed=0, mc_trials=1, reliability_aware=True,
                            return_std=False):
    """Maximum (Instantaneous) Error First, resource-gated, with q-delivery.

    Priority = w[m,j] * p[m,j,Delta] (the current weighted inference error);
    reliability_aware=True scales it by q[m,j] (= q*w*p), so a less reliable
    link must promise more error reduction to win a channel. This is the robot
    counterpart of MIEF1_probabilistic (no MATLAB original).
    """
    M = prob["M"]
    km_max = prob["km_max"]
    B = prob["B"]
    p = prob["p"]
    valid = prob["valid"]
    pair_src = np.array([m for m, _ in _valid_pairs(valid)])
    n_cost = np.ones((M, km_max))
    rows = np.arange(M)[:, None]
    cols = np.arange(km_max)[None, :]

    def schedule_fn(Delta, t):
        pen = p[rows, cols, Delta]                 # (M, km_max)
        priority = w * pen
        if reliability_aware:
            priority = priority * q
        priority = np.where(valid, priority, -np.inf)
        return _greedy_gate(priority, valid, pair_src, N, M, 1.0, n_cost,
                            threshold=0.0)

    results = _objective_from_rollout(prob, w, q, N, gamma, T, schedule_fn,
                                      seed, mc_trials, include_last=True)
    mean, std = _mc_summary(results)
    return (mean, std) if return_std else mean


def randpolicy_robot_probabilistic(prob, N, w, q, gamma, T,
                                   seed=0, mc_trials=100, return_std=False):
    """Random baseline: pick min(N, M) sources uniformly, a random valid task
    per chosen source, with q-delivery. MC-averaged (MATLAB Titer=100)."""
    M = prob["M"]
    km_max = prob["km_max"]
    valid = prob["valid"]
    km_vec = np.asarray(prob["km_vec"])
    valid_tasks = [np.where(valid[m])[0] for m in range(M)]

    # Per-trial RNG seeded reproducibly; the policy itself draws here.
    base_rng = np.random.default_rng(seed)
    trial_seeds = base_rng.integers(0, 2 ** 31 - 1, size=mc_trials)
    B = prob["B"]
    p = prob["p"]
    n_pairs = prob["n_pairs"]
    results = np.zeros(mc_trials)

    n_pick = min(N, M)
    for trial in range(mc_trials):
        rng = np.random.default_rng(int(trial_seeds[trial]))
        Delta = np.zeros((M, km_max), dtype=np.int64)
        presult = 0.0
        for t in range(T):
            Change = np.zeros((M, km_max), dtype=bool)
            sources = rng.choice(M, size=n_pick, replace=False)
            for m in sources:
                col = int(rng.choice(valid_tasks[m]))
                Change[m, col] = True
            success = rng.random(size=(M, km_max)) < q
            reset = Change & success & valid
            Delta = np.where(reset, 0,
                             np.minimum(Delta + 1, B - 1)).astype(np.int64)
            pen = p[np.arange(M)[:, None], np.arange(km_max)[None, :], Delta]
            pavg = float((w * pen * valid).sum() / n_pairs)
            if t + 1 < T:
                presult += (gamma ** t) * pavg
        results[trial] = presult

    mean, std = _mc_summary(results)
    return (mean, std) if return_std else mean
