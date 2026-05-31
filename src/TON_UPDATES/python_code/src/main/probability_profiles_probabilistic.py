"""
probability_profiles_probabilistic.py - q_{m, j} link-reliability profiles
used by the probabilistic ErrorVsSources experiment.

Each profile returns a (M, km) array of per-link success probabilities, clipped
to [Q_MIN, Q_MAX], plus a metadata dict. Profiles are deterministic given the
seed.

The 12 profiles are organized in three families that mirror the three settings
which empirically discriminated policies the most in earlier sweeps
(uniform_wide, uniform_low, bimodal_extreme):

  * Uniform-style (wide / low / mid / very_wide / with_perfect_outliers)
  * Bimodal-style with one cluster at q=1 (balanced / 30:70 / 70:30)
  * Mixed structures with perfect (q=1) links interleaved with lossy ones
    (trimodal / source_split / adversarial)

Many profiles include some links with q = 1.0 (no channel drops) alongside
links with significantly lower reliability (e.g. ~0.35-0.60) to stress
reliability-aware vs. reliability-blind scheduling.
"""
import numpy as np

# Q_MAX is 1.0 so we can express "perfect" links exactly.  Q_MIN keeps the
# lower tail away from 0 because the value function would otherwise see a
# scheduled action that can NEVER reduce AoI (no algorithmic problem; just
# uninformative as a stress test).
Q_MIN = 0.05
Q_MAX = 1.0


_PROFILE_NAMES = [
    # Uniform-style
    "uniform_wide",
    "uniform_low",
    "uniform_mid",
    "uniform_very_wide",
    "uniform_with_perfect_outliers",
    # Bimodal-style
    "bimodal_extreme",
    "bimodal_balanced",
    "bimodal_q1_vs_lossy_30_70",
    "bimodal_q1_vs_lossy_70_30",
    # Mixed perfect / lossy structures
    "trimodal_perfect_mid_low",
    "source_split_perfect_or_lossy",
    "adversarial_perfect_with_critical_lossy",
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

    # ------------------------------------------------------------------
    # Uniform-style
    # ------------------------------------------------------------------
    if profile_name == "uniform_wide":
        q = rng.uniform(0.35, 0.98, size=(M, km))
        desc = "Uniform(0.35, 0.98) -- wide heterogeneity baseline"

    elif profile_name == "uniform_low":
        q = rng.uniform(0.35, 0.70, size=(M, km))
        desc = "Uniform(0.35, 0.70) -- all moderately-low reliability"

    elif profile_name == "uniform_mid":
        q = rng.uniform(0.55, 0.85, size=(M, km))
        desc = "Uniform(0.55, 0.85) -- moderate band, different mean than wide/low"

    elif profile_name == "uniform_very_wide":
        q = rng.uniform(0.20, 0.99, size=(M, km))
        desc = "Uniform(0.20, 0.99) -- wider than uniform_wide, more extremes"

    elif profile_name == "uniform_with_perfect_outliers":
        # 85% draws from Uniform(0.40, 0.80); 15% upgraded to exact q=1.
        n_total = M * km
        base = rng.uniform(0.40, 0.80, size=n_total)
        is_perfect = rng.random(n_total) < 0.15
        values = np.where(is_perfect, 1.0, base)
        q = values.reshape(M, km)
        desc = ("85% Uniform(0.40, 0.80) and 15% perfect links q=1 "
                "(shuffled by seed)")

    # ------------------------------------------------------------------
    # Bimodal-style
    # ------------------------------------------------------------------
    elif profile_name == "bimodal_extreme":
        n_total = M * km
        mask_high = rng.random(n_total) < 0.70
        high_vals = rng.normal(0.95, 0.02, size=n_total)
        low_vals = rng.normal(0.30, 0.05, size=n_total)
        values = np.where(mask_high, high_vals, low_vals)
        q = values.reshape(M, km)
        desc = "Bimodal: ~70% around 0.95 (std 0.02), ~30% around 0.30 (std 0.05)"

    elif profile_name == "bimodal_balanced":
        n_total = M * km
        mask_high = rng.random(n_total) < 0.50
        high_vals = rng.normal(0.95, 0.02, size=n_total)
        low_vals = rng.normal(0.40, 0.04, size=n_total)
        values = np.where(mask_high, high_vals, low_vals)
        q = values.reshape(M, km)
        desc = "Balanced bimodal: 50% around 0.95, 50% around 0.40"

    elif profile_name == "bimodal_q1_vs_lossy_30_70":
        # 30% of links are exactly q=1 (perfect channel); 70% are lossy.
        n_total = M * km
        is_perfect = rng.random(n_total) < 0.30
        lossy = rng.uniform(0.30, 0.50, size=n_total)
        values = np.where(is_perfect, 1.0, lossy)
        q = values.reshape(M, km)
        desc = ("30% perfect links q=1, 70% lossy Uniform(0.30, 0.50)")

    elif profile_name == "bimodal_q1_vs_lossy_70_30":
        # 70% perfect, 30% lossy.
        n_total = M * km
        is_perfect = rng.random(n_total) < 0.70
        lossy = rng.uniform(0.30, 0.55, size=n_total)
        values = np.where(is_perfect, 1.0, lossy)
        q = values.reshape(M, km)
        desc = ("70% perfect links q=1, 30% lossy Uniform(0.30, 0.55)")

    # ------------------------------------------------------------------
    # Mixed perfect/lossy structures
    # ------------------------------------------------------------------
    elif profile_name == "trimodal_perfect_mid_low":
        # Three equal-size clusters: perfect q=1, mid (~0.65), low (~0.35).
        n_total = M * km
        cluster = rng.integers(0, 3, size=n_total)   # 0,1,2 uniform
        mid_vals = rng.normal(0.65, 0.03, size=n_total)
        low_vals = rng.normal(0.35, 0.04, size=n_total)
        values = np.where(cluster == 0, 1.0,
                          np.where(cluster == 1, mid_vals, low_vals))
        q = values.reshape(M, km)
        desc = ("Trimodal equal thirds: q=1, ~0.65, ~0.35")

    elif profile_name == "source_split_perfect_or_lossy":
        # Each source is *entirely* perfect (all q=1) or *entirely* lossy
        # (Uniform(0.35, 0.60) per link). About half/half by source.
        is_perfect = rng.random(M) < 0.5
        lossy = rng.uniform(0.35, 0.60, size=(M, km))
        q = np.where(is_perfect[:, None], 1.0, lossy)
        n_perfect = int(is_perfect.sum())
        desc = (f"Per-source split: {n_perfect}/{M} sources have all links q=1, "
                f"the rest have all links ~ Uniform(0.35, 0.60)")

    elif profile_name == "adversarial_perfect_with_critical_lossy":
        # Most links are perfect (q=1), but the high-growth (exponential)
        # tasks j with j%3 == 2 are lossy around 0.35. This puts unreliable
        # channels exactly on the tasks where AoI matters most.
        j_arr = np.arange(km)
        is_critical = (j_arr % 3) == 2
        lossy = rng.normal(0.35, 0.03, size=(M, km))
        q = np.where(is_critical[None, :], lossy, 1.0)
        desc = ("Adversarial: all links q=1 EXCEPT exponential-penalty tasks "
                "(j%3==2), which get q ~ 0.35 (worst-case for AoI growth)")

    else:
        raise ValueError(
            f"Unknown profile {profile_name!r}. "
            f"Choose from {list_q_profiles()}."
        )

    q = _clip(np.asarray(q, dtype=float))
    metadata = {"profile": profile_name, "description": desc, **_summary(q)}
    return q, metadata


if __name__ == "__main__":
    print(f"{'profile':>42}  {'min':>5}  {'mean':>5}  {'max':>5}  {'std':>5}")
    for name in list_q_profiles():
        q, meta = make_q_profile(name, M=10, km=9, seed=0)
        print(f"{name:>42}  {meta['q_min']:5.3f}  "
              f"{meta['q_mean']:5.3f}  {meta['q_max']:5.3f}  "
              f"{meta['q_std']:5.3f}")
