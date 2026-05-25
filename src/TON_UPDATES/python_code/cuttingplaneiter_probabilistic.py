"""
cuttingplaneiter_probabilistic.py - cutting-plane / bundle methods for the
probabilistic Lagrangian dual.

Supports:
    kelley_bounded         - Kelley's cutting-plane method with box bounds.
    trust_region_kelley    - Kelley + shrinking inf-norm trust region.
    proximal_bundle        - quadratic-prox bundle method, limited bundle.

simplified=True (default) is strongly recommended; the dual variable is
z = [lambda_1, ..., lambda_M, mu], shape (M + 1,). The output multipliers are
expanded to lambdasource (M, T) and mu (T,) so that downstream MGF rollout can
consume them like the first-order methods.
"""
import numpy as np

from dual_oracle_probabilistic import dual_oracle_probabilistic
from _cutting_plane_core import cutting_plane_loop


def _default_z_max_for_problem(M, T, gamma, p, w, c, simplified=True):
    """Build a generous per-coordinate upper bound for z.

    The dual is bounded by primal optimal value; we use a multiple of the
    worst-case un-discounted primal cost as an upper bound, which is overly
    safe but easy to compute. Returns shape (M + 1,) for simplified.
    """
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    if simplified:
        scale = 10.0 * float(p.max() * w.max() + 1.0)
        return np.full(M + 1, scale)
    raise ValueError("Non-simplified default z_max is not implemented")


def cuttingplaneiter_probabilistic(
    M, N, T, B, gamma, p, km, w, n, c, q,
    max_iter=100,
    method="proximal_bundle",
    seed=0,
    simplified=True,
    z_max=None,
    verbose=False,
    history=True,
):
    """Cutting-plane / bundle method for the probabilistic dual.

    Returns
    -------
    lambdasource : (M, T) per-source multipliers (expanded from simplified z).
    mu           : (T,)   channel multipliers.
    history_dict : iteration-level diagnostics.
    """
    if not simplified:
        raise NotImplementedError(
            "Full time-indexed cutting-plane is not implemented. "
            "Use simplified=True."
        )

    if z_max is None:
        z_max = _default_z_max_for_problem(M, T, gamma, p, w, c, simplified)

    rng = np.random.default_rng(seed)
    # Warm start at zero (matches first-order baseline) -- the oracle is
    # well-defined at z=0 (just the relaxed unconstrained DP value).
    z_init = np.zeros(M + 1)
    if seed != 0:
        # Small random perturbation to break degenerate behaviour at z=0.
        z_init = rng.uniform(0.0, 1e-3, size=M + 1)

    def oracle(z):
        return dual_oracle_probabilistic(
            z, M, N, T, B, gamma, p, km, w, n, c, q,
            simplified=True, return_actions=False,
        )

    z_best, hist = cutting_plane_loop(
        oracle, z_init, z_max,
        max_iter=max_iter, method=method,
        verbose=verbose, history=history,
    )

    # Expand simplified z into (lambdasource, mu).
    lambdasource = np.broadcast_to(z_best[:M, None], (M, T)).copy()
    mu = np.full(T, float(z_best[M]))
    return lambdasource, mu, hist
