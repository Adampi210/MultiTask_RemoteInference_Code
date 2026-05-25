"""
probability_profiles_probabilistic.py - q_{m, j} link-reliability profiles
used by the probabilistic ErrorVsSources experiment.

Each profile returns a (M, km) array of per-link success probabilities, clipped
to [Q_MIN, Q_MAX], plus a metadata dict. Profiles are deterministic given the
seed.
"""
import numpy as np

Q_MIN = 0.05
Q_MAX = 0.99


_PROFILE_NAMES = [
    "constant_high",
    "constant_low",
    "two_cluster_60_95",
    "uniform_wide",
    "uniform_high",
    "uniform_low",
    "source_gradient",
    "task_gradient",
    "bimodal_extreme",
    "adversarial_by_penalty_type",
]


def list_q_profiles():
    """Return the names of all available q profiles, in canonical order."""
    return list(_PROFILE_NAMES)


def _clip(q):
    return np.clip(q, Q_MIN, Q_MAX)


def _summary(q):
    return {
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "q_mean": float(q.mean()),
        "q_std": float(q.std()),
    }


def make_q_profile(profile_name, M, km, seed=0):
    """Build a (M, km) q matrix for the named profile.

    Parameters
    ----------
    profile_name : one of list_q_profiles().
    M, km        : problem dimensions.
    seed         : controls profiles with randomness.

    Returns
    -------
    q        : (M, km) float in [Q_MIN, Q_MAX].
    metadata : dict with 'profile', 'description', and summary stats.
    """
    M = int(M)
    km = int(km)
    rng = np.random.default_rng(seed)

    if profile_name == "constant_high":
        q = np.full((M, km), 0.95)
        desc = "All links q = 0.95"

    elif profile_name == "constant_low":
        q = np.full((M, km), 0.60)
        desc = "All links q = 0.60"

    elif profile_name == "two_cluster_60_95":
        n_total = M * km
        half = n_total // 2
        values = np.concatenate([np.full(half, 0.60),
                                 np.full(n_total - half, 0.95)])
        rng.shuffle(values)
        q = values.reshape(M, km)
        desc = "Half of links 0.60, half 0.95 (shuffled by seed)"

    elif profile_name == "uniform_wide":
        q = rng.uniform(0.35, 0.98, size=(M, km))
        desc = "Uniform(0.35, 0.98)"

    elif profile_name == "uniform_high":
        q = rng.uniform(0.85, 0.99, size=(M, km))
        desc = "Uniform(0.85, 0.99)"

    elif profile_name == "uniform_low":
        q = rng.uniform(0.35, 0.70, size=(M, km))
        desc = "Uniform(0.35, 0.70)"

    elif profile_name == "source_gradient":
        if M == 1:
            base = np.array([0.95])
        else:
            base = np.linspace(0.95, 0.55, M)
        q = np.broadcast_to(base[:, None], (M, km)).copy()
        desc = "Per-source linspace(0.95, 0.55), replicated over tasks"

    elif profile_name == "task_gradient":
        if km == 1:
            base = np.array([0.95])
        else:
            base = np.linspace(0.95, 0.55, km)
        q = np.broadcast_to(base[None, :], (M, km)).copy()
        desc = "Per-task linspace(0.95, 0.55), replicated over sources"

    elif profile_name == "bimodal_extreme":
        n_total = M * km
        mask_high = rng.random(n_total) < 0.70
        high_vals = rng.normal(0.95, 0.02, size=n_total)
        low_vals = rng.normal(0.30, 0.05, size=n_total)
        values = np.where(mask_high, high_vals, low_vals)
        q = values.reshape(M, km)
        desc = "Bimodal: ~70% around 0.95 (std 0.02), ~30% around 0.30 (std 0.05)"

    elif profile_name == "adversarial_by_penalty_type":
        # Task j is "exponential-penalty" when j % 3 == 2 (matches ErrorVsSources.py),
        # so give those tasks low reliability (~0.55) and the rest high (~0.90).
        q = np.zeros((M, km))
        j_arr = np.arange(km)
        is_exp = (j_arr % 3) == 2
        # Add a small per-link jitter so it's not literally piecewise constant.
        jitter = rng.normal(0.0, 0.02, size=(M, km))
        base = np.where(is_exp[None, :], 0.55, 0.90)
        q = base + jitter
        desc = ("Adversarial: exponential-penalty tasks (j%3==2) get q~0.55; "
                "linear/log tasks get q~0.90")

    else:
        raise ValueError(
            f"Unknown profile {profile_name!r}. "
            f"Choose from {list_q_profiles()}."
        )

    q = _clip(np.asarray(q, dtype=float))
    metadata = {"profile": profile_name, "description": desc, **_summary(q)}
    return q, metadata


if __name__ == "__main__":
    for name in list_q_profiles():
        q, meta = make_q_profile(name, M=10, km=9, seed=0)
        print(f"{name:>32}  min={meta['q_min']:.3f}  "
              f"mean={meta['q_mean']:.3f}  max={meta['q_max']:.3f}  "
              f"std={meta['q_std']:.3f}")
