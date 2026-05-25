"""
optimizer_updates.py - centralized projected-ascent update rules used by the
method-selectable deterministic and probabilistic subgradient drivers.

All updates operate on a flattened vector z >= 0 with a corresponding
supergradient g; the projection keeps z >= 0 component-wise. block_slices is
an optional list of (start, stop) tuples used by ``normalized_blocks`` to
normalize lambda and mu blocks separately. config is an optional dict of
hyperparameters for the method (e.g. b1, b2 for adam; rho for rmsprop).
"""
import numpy as np


_FIRST_ORDER_METHODS = [
    "harmonic",
    "sqrt",
    "normalized_global",
    "normalized_blocks",
    "adagrad",
    "rmsprop",
    "adam",
    "deflected_sqrt",
]


def get_default_subgradient_methods():
    """List the first-order projected-ascent methods exposed here."""
    return list(_FIRST_ORDER_METHODS)


class OptimizerState:
    """Per-method state for adaptive / momentum methods.

    All fields are sized to match z. Unused fields stay at zero.
    """

    __slots__ = ("G", "v", "m", "d_prev", "method")

    def __init__(self, shape, method):
        shape = tuple(np.atleast_1d(shape).astype(int))
        self.method = method
        self.G = np.zeros(shape)         # adagrad: sum of squared grads
        self.v = np.zeros(shape)         # rmsprop / adam: 2nd moment
        self.m = np.zeros(shape)         # adam: 1st moment
        self.d_prev = np.zeros(shape)    # deflected_sqrt: previous direction


def init_optimizer_state(shape, method):
    """Public constructor for OptimizerState."""
    return OptimizerState(shape, method)


def _normalize(g, eps):
    nrm = float(np.linalg.norm(g))
    if nrm < eps:
        return np.zeros_like(g)
    return g / (nrm + eps)


def projected_update(
    z,
    g,
    k,
    method,
    state,
    beta=0.9,
    eps=1e-8,
    block_slices=None,
    config=None,
):
    """Apply one projected-ascent step on z >= 0 with supergradient g.

    Parameters
    ----------
    z : ndarray, current iterate (flattened, nonneg).
    g : ndarray, supergradient direction (same shape as z).
    k : int, 1-indexed iteration (used by harmonic/sqrt/adam bias correction).
    method : str, one of get_default_subgradient_methods().
    state : OptimizerState (use init_optimizer_state to build).
    beta : float, base step size.
    eps : float, small constant for division safety.
    block_slices : optional list of (start, stop) tuples. Used by
                   normalized_blocks to normalize subsets of g separately.
    config : optional dict overriding hyperparameters
                 (rho for rmsprop, b1/b2 for adam, alpha for deflected_sqrt).

    Returns
    -------
    z_new : ndarray, updated nonneg iterate.
    state : OptimizerState (updated in place; returned for convenience).
    diagnostics : dict with effective step size info.
    """
    config = config or {}
    z = np.asarray(z, dtype=float)
    g = np.asarray(g, dtype=float)
    k = int(max(1, k))

    if z.shape != g.shape:
        raise ValueError(f"z shape {z.shape} != g shape {g.shape}")

    if method == "harmonic":
        step = beta / k
        z_new = np.maximum(0.0, z + step * g)
        diag = {"step": step, "method": method}

    elif method == "sqrt":
        step = beta / np.sqrt(k)
        z_new = np.maximum(0.0, z + step * g)
        diag = {"step": step, "method": method}

    elif method == "normalized_global":
        step = beta / np.sqrt(k)
        direction = _normalize(g, eps)
        z_new = np.maximum(0.0, z + step * direction)
        diag = {"step": step, "method": method}

    elif method == "normalized_blocks":
        if not block_slices:
            # Without explicit blocks fall back to global normalization.
            return projected_update(z, g, k, "normalized_global", state,
                                    beta=beta, eps=eps, config=config)
        step = beta / np.sqrt(k)
        direction = np.zeros_like(g)
        for sl in block_slices:
            lo, hi = sl
            block = g[lo:hi]
            direction[lo:hi] = _normalize(block, eps)
        z_new = np.maximum(0.0, z + step * direction)
        diag = {"step": step, "method": method, "n_blocks": len(block_slices)}

    elif method == "adagrad":
        state.G += g * g
        z_new = np.maximum(0.0, z + beta * g / (np.sqrt(state.G) + eps))
        diag = {"step": float(beta), "method": method,
                "G_max": float(state.G.max())}

    elif method == "rmsprop":
        rho = float(config.get("rho", 0.9))
        state.v = rho * state.v + (1.0 - rho) * g * g
        z_new = np.maximum(0.0, z + beta * g / (np.sqrt(state.v) + eps))
        diag = {"step": float(beta), "method": method, "rho": rho}

    elif method == "adam":
        b1 = float(config.get("b1", 0.9))
        b2 = float(config.get("b2", 0.999))
        state.m = b1 * state.m + (1.0 - b1) * g
        state.v = b2 * state.v + (1.0 - b2) * g * g
        bc1 = 1.0 - b1 ** k
        bc2 = 1.0 - b2 ** k
        m_hat = state.m / bc1
        v_hat = state.v / bc2
        z_new = np.maximum(0.0, z + beta * m_hat / (np.sqrt(v_hat) + eps))
        diag = {"step": float(beta), "method": method, "b1": b1, "b2": b2}

    elif method == "deflected_sqrt":
        alpha = float(config.get("alpha", 0.5))
        d_new = alpha * state.d_prev + (1.0 - alpha) * g
        state.d_prev = d_new
        step = beta / np.sqrt(k)
        z_new = np.maximum(0.0, z + step * d_new)
        diag = {"step": step, "method": method, "alpha": alpha}

    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from "
            f"{get_default_subgradient_methods()}"
        )

    return z_new, state, diag
