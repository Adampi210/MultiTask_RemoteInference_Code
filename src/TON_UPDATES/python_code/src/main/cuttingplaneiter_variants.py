"""
cuttingplaneiter_variants.py - cutting-plane / bundle methods for the
deterministic (q == 1) Lagrangian dual. Mirrors cuttingplaneiter_probabilistic.
"""
import numpy as np

from dual_oracle_deterministic import dual_oracle_deterministic
from _cutting_plane_core import cutting_plane_loop


def _default_z_max_for_problem(M, T, gamma, p, w, c, simplified=True):
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    if simplified:
        scale = 10.0 * float(p.max() * w.max() + 1.0)
        return np.full(M + 1, scale)
    raise ValueError("Non-simplified default z_max is not implemented")


def cuttingplaneiter_variants(
    M, N, T, B, gamma, p, km, w, n, c,
    max_iter=100,
    method="proximal_bundle",
    seed=0,
    simplified=True,
    z_max=None,
    verbose=False,
    history=True,
    use_n_cost=False,
):
    """Cutting-plane / bundle method for the deterministic dual.

    Returns (lambdasource, mu, history_dict). lambdasource and mu are expanded
    from the simplified z (length M+1).
    """
    if not simplified:
        raise NotImplementedError(
            "Full time-indexed cutting-plane is not implemented. "
            "Use simplified=True."
        )

    if z_max is None:
        z_max = _default_z_max_for_problem(M, T, gamma, p, w, c, simplified)

    rng = np.random.default_rng(seed)
    z_init = np.zeros(M + 1)
    if seed != 0:
        z_init = rng.uniform(0.0, 1e-3, size=M + 1)

    def oracle(z):
        return dual_oracle_deterministic(
            z, M, N, T, B, gamma, p, km, w, n, c,
            simplified=True, return_actions=False,
            use_n_cost=use_n_cost,
        )

    z_best, hist = cutting_plane_loop(
        oracle, z_init, z_max,
        max_iter=max_iter, method=method,
        verbose=verbose, history=history,
    )

    lambdasource = np.broadcast_to(z_best[:M, None], (M, T)).copy()
    mu = np.full(T, float(z_best[M]))
    return lambdasource, mu, hist
