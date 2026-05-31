"""
_cutting_plane_core.py - shared cutting-plane / bundle implementations used
by cuttingplaneiter_probabilistic and cuttingplaneiter_variants.

The maximize-the-concave-dual problem is:
    max_z D(z)   s.t.   0 <= z <= z_max

with a supergradient oracle that returns (D_i, s_i) at each query z_i. The
linearization upper-bound for concave D is
    D(z) <= D_i + s_i^T (z - z_i)     (cut i)
so Kelley's outer-approximation LP is
    maximize_{theta, z} theta
    s.t.   theta - s_i^T z <= D_i - s_i^T z_i     for each cut i
            0 <= z <= z_max

Three methods are exposed:
    kelley_bounded         - vanilla Kelley with box bounds.
    trust_region_kelley    - Kelley with a (shrinking) infinity-norm trust region.
    proximal_bundle        - Quadratic prox term around z_center, limited bundle.

All three call a generic oracle :: z -> (D, s, diagnostics).
"""
import time
import numpy as np

try:
    from scipy.optimize import linprog, minimize, LinearConstraint
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _default_z_max(M, simplified, scale):
    """Default per-coordinate upper bound for the dual variable z.

    The bound is generous; cutting-plane methods just need a finite box.
    """
    if simplified:
        return np.full(M + 1, float(scale))
    return None  # caller must provide for non-simplified


def _solve_kelley_lp(cuts, z_max, n_dim):
    """Solve Kelley's outer-approximation LP. Returns (z_new, theta).

    cuts: list of (D_i, s_i, z_i).
    Variables in linprog: [theta, z_1, ..., z_n_dim].
    Minimize -theta s.t. theta - s_i^T z <= D_i - s_i^T z_i for each cut.
    Bounds: theta in (-inf, +inf); z in [0, z_max].
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for cutting-plane methods")
    c = np.zeros(n_dim + 1)
    c[0] = -1.0
    A_ub = []
    b_ub = []
    for D_i, s_i, z_i in cuts:
        row = np.zeros(n_dim + 1)
        row[0] = 1.0
        row[1:] = -np.asarray(s_i, dtype=float)
        A_ub.append(row)
        b_ub.append(float(D_i) - float(np.dot(s_i, z_i)))
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None)] + [(0.0, float(zm)) for zm in z_max]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        # Fallback: return centre of the box.
        z_new = np.array([zm / 2.0 for zm in z_max])
        return z_new, None
    theta = -res.fun
    z_new = res.x[1:]
    return z_new, theta


def _solve_trust_region_lp(cuts, z_max, n_dim, z_center, radius):
    """Kelley LP with an additional inf-norm trust region:
        ||z - z_center||_inf <= radius.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for cutting-plane methods")
    c = np.zeros(n_dim + 1)
    c[0] = -1.0
    A_ub = []
    b_ub = []
    for D_i, s_i, z_i in cuts:
        row = np.zeros(n_dim + 1)
        row[0] = 1.0
        row[1:] = -np.asarray(s_i, dtype=float)
        A_ub.append(row)
        b_ub.append(float(D_i) - float(np.dot(s_i, z_i)))
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = [(None, None)]
    for i in range(n_dim):
        lo = max(0.0, z_center[i] - radius)
        hi = min(float(z_max[i]), z_center[i] + radius)
        if hi < lo:
            hi = lo
        bounds.append((lo, hi))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return z_center.copy(), None
    theta = -res.fun
    return res.x[1:], theta


def _solve_proximal_bundle(cuts, z_max, n_dim, z_center, rho):
    """Solve  max theta - (rho/2) ||z - z_center||^2  s.t. cuts and 0<=z<=zmax.

    We use scipy.optimize.minimize with linear constraints. To keep it stable,
    we use the SLSQP method with the negated objective.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for cutting-plane methods")

    n_vars = n_dim + 1   # theta + z

    def neg_obj(x):
        theta = x[0]
        z = x[1:]
        return -(theta - 0.5 * rho * float(np.dot(z - z_center,
                                                  z - z_center)))

    def neg_obj_grad(x):
        z = x[1:]
        g = np.zeros(n_vars)
        g[0] = -1.0
        g[1:] = rho * (z - z_center)
        return g

    # Cuts: theta - s_i^T z <= D_i - s_i^T z_i   -> linear_constraint upper bound.
    n_cuts = len(cuts)
    A_lin = np.zeros((n_cuts, n_vars))
    ub = np.zeros(n_cuts)
    for i, (D_i, s_i, z_i) in enumerate(cuts):
        A_lin[i, 0] = 1.0
        A_lin[i, 1:] = -np.asarray(s_i, dtype=float)
        ub[i] = float(D_i) - float(np.dot(s_i, z_i))
    lc = LinearConstraint(A_lin, lb=-np.inf, ub=ub)

    bounds = [(None, None)] + [(0.0, float(zm)) for zm in z_max]

    # Warm start at z_center, theta = max over cuts evaluated at z_center.
    theta0 = max(D_i + float(np.dot(s_i, z_center - z_i))
                 for (D_i, s_i, z_i) in cuts)
    x0 = np.concatenate([[theta0], z_center])
    # Project z_center into bounds.
    x0[1:] = np.clip(x0[1:], 0.0, np.array(z_max, dtype=float))

    res = minimize(neg_obj, x0, jac=neg_obj_grad, method="SLSQP",
                   bounds=bounds, constraints=[lc],
                   options={"maxiter": 200, "ftol": 1e-6})
    if not res.success:
        return z_center.copy(), theta0
    return res.x[1:], res.x[0]


def cutting_plane_loop(
    oracle,
    z_init,
    z_max,
    max_iter=100,
    method="proximal_bundle",
    bundle_size=50,
    radius_init=None,
    rho=1.0,
    verbose=False,
    history=True,
):
    """Generic cutting-plane / bundle loop.

    oracle :: z -> (D, s, diag). z is shape (n_dim,).
    Returns (z_best, history_dict).
    """
    z_init = np.asarray(z_init, dtype=float)
    z_max = np.asarray(z_max, dtype=float)
    n_dim = z_init.shape[0]

    # Initial trust-region radius scales with the bound.
    if radius_init is None:
        radius_init = max(1.0, float(z_max.max()) / 2.0)

    cuts = []           # list of (D_i, s_i, z_i)
    z_best = z_init.copy()
    D_best = -np.inf

    radius = float(radius_init)
    z_center = z_init.copy()

    hist = {
        "method": method,
        "dual_values": [],
        "model_upper_bounds": [],
        "estimated_gaps": [],
        "oracle_calls": 0,
        "z_norms": [],
        "residual_norms": [],
        "accepted_steps": [],
        "radius_or_rho": [],
        "runtime_sec": 0.0,
    }
    t0 = time.perf_counter()
    z_cur = z_init.copy()

    for k in range(1, max_iter + 1):
        D_k, s_k, diag = oracle(z_cur)
        hist["oracle_calls"] += 1
        cuts.append((float(D_k), np.asarray(s_k, dtype=float).copy(),
                     np.asarray(z_cur, dtype=float).copy()))

        # Limit bundle size: drop oldest cuts but always keep the best one.
        if len(cuts) > bundle_size:
            best_idx = max(range(len(cuts)), key=lambda i: cuts[i][0])
            keep = cuts[-bundle_size:]
            if all(id(c) != id(cuts[best_idx]) for c in keep):
                keep[0] = cuts[best_idx]
            cuts = keep

        if D_k > D_best:
            D_best = float(D_k)
            z_best = z_cur.copy()

        # Solve subproblem to get z_next.
        if method == "kelley_bounded":
            z_next, theta = _solve_kelley_lp(cuts, z_max, n_dim)
        elif method == "trust_region_kelley":
            z_next, theta = _solve_trust_region_lp(
                cuts, z_max, n_dim, z_center, radius,
            )
        elif method == "proximal_bundle":
            z_next, theta = _solve_proximal_bundle(
                cuts, z_max, n_dim, z_center, rho,
            )
        else:
            raise ValueError(f"Unknown cutting-plane method {method!r}")

        # Acceptance / radius update.
        accepted = True
        if method == "trust_region_kelley":
            # Estimate predicted vs actual improvement.
            if theta is None or theta <= D_best + 1e-9:
                accepted = False
                radius = max(radius * 0.5, 1e-6)
            else:
                # Quick acceptance: query the oracle at z_next (will happen
                # next iter), but we update centre tentatively if model says
                # there is upside.
                z_center = z_next.copy()
                radius = min(radius * 1.5, float(z_max.max()))
        elif method == "proximal_bundle":
            # Move centre to best-so-far z and keep rho.
            z_center = z_best.copy()
        # kelley_bounded: nothing to update.

        if history:
            hist["dual_values"].append(float(D_k))
            hist["model_upper_bounds"].append(
                float(theta) if theta is not None else float("nan"))
            hist["estimated_gaps"].append(
                float(theta - D_best) if theta is not None else float("nan"))
            hist["z_norms"].append(float(np.linalg.norm(z_cur)))
            hist["residual_norms"].append(float(np.linalg.norm(s_k)))
            hist["accepted_steps"].append(bool(accepted))
            hist["radius_or_rho"].append(
                float(radius) if method == "trust_region_kelley" else float(rho))

        if verbose and (k % max(1, max_iter // 20) == 0 or k == 1):
            theta_print = f"{theta:.4g}" if theta is not None else "n/a"
            print(f"  cp[{method}] iter {k}/{max_iter}  "
                  f"D={D_k:.4g}  D*={D_best:.4g}  "
                  f"theta={theta_print}  "
                  f"||s||={np.linalg.norm(s_k):.3e}")

        z_cur = z_next

    hist["runtime_sec"] = float(time.perf_counter() - t0)
    return z_best, hist
