"""
valuefunction1_probabilistic.py - probabilistic-transmission Bellman recursion.

Paper model (1-indexed in the paper, 0-indexed here):

    Q_no(d)       = p(d) + gamma * V(d_aged)
    Q_schedule(d) = p(d) + lambda + mu * n_mj
                  + gamma * ( q * V(0) + (1 - q) * V(d_aged) )
    V(d)          = min( Q_no(d), Q_schedule(d) )
    gain(d)       = Q_no(d) - Q_schedule(d)

where d_aged = min(d + 1, B - 1).

With q = 1 and n_cost = 1 this reduces to the deterministic Bellman in
valuefunction1.py (the legacy port).
"""
import numpy as np


def valuefunction1_probabilistic(lambda_, mu, B, p, T, gamma,
                                 q=1.0, n_cost=1.0):
    """Probabilistic-transmission Bellman recursion for a single (m, j) pair.

    Parameters
    ----------
    lambda_ : 1D array of length T (per-source Lagrange multipliers).
    mu      : 1D array of length T (channel Lagrange multipliers).
    B       : int, AoI bound. AoI is in {0, ..., B-1}.
    p       : 1D array of length >= B (per-AoI penalty, already weighted if
              the caller multiplied by w[m, j]).
    T       : int, horizon.
    gamma   : float in (0, 1), discount factor.
    q       : float in [0, 1], transmission success probability for this pair.
    n_cost  : float, channel cost coefficient n_{m, j}.

    Returns
    -------
    a : (T, B) gain matrix where a[i, d] = Q_no(d) - Q_schedule(d).
        Positive => schedule is the relaxed-optimal action.
    """
    q = float(q)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q!r}")

    lambda_ = np.asarray(lambda_, dtype=float).flatten()
    mu = np.asarray(mu, dtype=float).flatten()
    p = np.asarray(p, dtype=float).flatten()
    n_cost = float(n_cost)

    V = np.zeros((T + 1, B + 1))
    a = np.zeros((T, B))

    # age_next[d] = min(d + 1, B - 1) for d in 0..B-1
    age_next = np.minimum(np.arange(B) + 1, B - 1)

    for i in range(T - 1, -1, -1):
        V_next_age = V[i + 1, age_next]   # (B,)
        V_next_reset = V[i + 1, 0]        # scalar

        Q_no = p[:B] + gamma * V_next_age
        Q_sched = (
            p[:B]
            + lambda_[i]
            + mu[i] * n_cost
            + gamma * (q * V_next_reset + (1.0 - q) * V_next_age)
        )

        V[i, :B] = np.minimum(Q_no, Q_sched)
        a[i, :] = Q_no - Q_sched
        # Preserve cap convention V(i, B) = V(i, B-1) so out-of-range AoI is
        # idempotent (matches the deterministic port).
        V[i, B] = V[i, B - 1]

    return a
