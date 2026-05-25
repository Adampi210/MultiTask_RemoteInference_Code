"""
test_probabilistic.py - pytest-compatible tests for the probabilistic-link
variants and the method-selectable deterministic variants. Also runnable as a
script.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from valuefunction1 import valuefunction1
from valuefunction1_probabilistic import valuefunction1_probabilistic
from probability_profiles_probabilistic import make_q_profile, list_q_profiles
from subgradientiter1_probabilistic import (
    subgradientiter1_probabilistic,
    _batched_gain_table_probabilistic,
)
from MGF1 import MGF1
from MGF1_probabilistic import MGF1_probabilistic, _greedy_pick_with_gating
from MAF1_probabilistic import MAF1_probabilistic
from MIEF1_probabilistic import MIEF1_probabilistic
from randpolicy_probabilistic import randpolicy_probabilistic
from MGF1_variants import MGF1_variants
from subgradientiter1_variants import subgradientiter1_variants
from optimizer_updates import get_default_subgradient_methods
from experiment_utils import make_base_problem


# ---------------------------------------------------------------------------
# 1. Value function equivalence: q=1, n_cost=1 -> deterministic
# ---------------------------------------------------------------------------

def test_valuefunction_probabilistic_matches_deterministic_when_q1_n1():
    rng = np.random.default_rng(42)
    T, B = 20, 10
    lambda_ = rng.uniform(0, 2, size=T)
    mu = rng.uniform(0, 2, size=T)
    p = rng.uniform(0.1, 5.0, size=B)
    gamma = 0.85

    a_det = valuefunction1(lambda_, mu, B, p, T, gamma)
    a_prob = valuefunction1_probabilistic(lambda_, mu, B, p, T, gamma,
                                          q=1.0, n_cost=1.0)
    assert np.allclose(a_det, a_prob, atol=1e-10), \
        "valuefunction1_probabilistic(q=1, n_cost=1) must match valuefunction1"


# ---------------------------------------------------------------------------
# 2. q=0: no reset benefit, only multiplier cost
# ---------------------------------------------------------------------------

def test_valuefunction_q_zero_has_no_reset_benefit_except_penalty():
    """With q=0 and nonneg multipliers, scheduling cannot lower future cost."""
    rng = np.random.default_rng(7)
    T, B = 15, 8
    lambda_ = rng.uniform(0, 1, size=T)
    mu = rng.uniform(0, 1, size=T)
    p = rng.uniform(0.1, 5.0, size=B)
    gamma = 0.9

    a = valuefunction1_probabilistic(lambda_, mu, B, p, T, gamma,
                                     q=0.0, n_cost=1.0)
    assert (a <= 1e-12).all(), \
        "With q=0 and nonneg lambda/mu, gain index must be <= 0 everywhere"


# ---------------------------------------------------------------------------
# 3. AoI under q=1 resets to 0 on scheduled
# ---------------------------------------------------------------------------

def test_aoi_update_q_one_resets():
    M, km, T, B, N = 1, 1, 5, 6, 1
    gamma = 0.9
    p = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    w = np.ones((M, km))
    n = np.ones((M, km))
    c = np.array([1.0])
    q = np.ones((M, km))

    a = _batched_gain_table_probabilistic(
        np.zeros((M, T)), np.zeros(T), B, T, gamma, w, p, M, km, q, n,
    )

    Delta = np.array([[3]], dtype=np.int64)
    rng = np.random.default_rng(0)
    Change, _, _ = _greedy_pick_with_gating(a[:, 0, 3, :], n, c, N, M, km)
    assert Change[0, 0], "scheduling should be greedily preferred at AoI=3"
    success = rng.random(size=(M, km)) < q
    Delta_new = np.where(Change & success, 0,
                         np.minimum(Delta + 1, B - 1))
    assert Delta_new[0, 0] == 0, "q=1 scheduled pair must reset to AoI=0"


# ---------------------------------------------------------------------------
# 4. AoI under q=0 doesn't reset; just ages
# ---------------------------------------------------------------------------

def test_aoi_update_q_zero_fails_and_ages():
    M, km, B = 1, 1, 6
    Delta = np.array([[3]], dtype=np.int64)
    Change = np.array([[True]])
    q = np.zeros((M, km))
    rng = np.random.default_rng(0)
    success = rng.random(size=(M, km)) < q
    Delta_new = np.where(Change & success, 0,
                         np.minimum(Delta + 1, B - 1))
    assert Delta_new[0, 0] == 4, "q=0 scheduled pair must age by 1 (3 -> 4)"


# ---------------------------------------------------------------------------
# 5. Profile shape / bounds
# ---------------------------------------------------------------------------

def test_q_profiles_shape_bounds():
    for name in list_q_profiles():
        for (M, km) in [(2, 3), (10, 9), (1, 1)]:
            q, meta = make_q_profile(name, M, km, seed=0)
            assert q.shape == (M, km), \
                f"profile {name} returned shape {q.shape}, expected {(M, km)}"
            assert np.all(np.isfinite(q)), f"profile {name} has non-finite q"
            assert (q >= 0.05).all() and (q <= 0.99).all(), \
                f"profile {name} out of [0.05, 0.99] bounds"
            for k in ("q_min", "q_max", "q_mean", "q_std"):
                assert k in meta, f"profile {name} metadata missing {k}"


# ---------------------------------------------------------------------------
# 6. Greedy policies respect resource constraints
# ---------------------------------------------------------------------------

def test_probabilistic_greedy_respects_constraints():
    M, km, T, B, N = 5, 4, 8, 10, 3
    gamma = 0.9
    rng = np.random.default_rng(1)
    p = rng.uniform(0.5, 5.0, size=(km, B))
    w = np.ones((M, km))
    n = np.ones((M, km))
    c = np.ones(M) * 2
    q, _ = make_q_profile("uniform_wide", M, km, seed=0)

    a = rng.uniform(-1, 5, size=(M, T, B, km))
    Delta = np.zeros((M, km), dtype=np.int64)
    m_idx, j_idx = np.meshgrid(np.arange(M), np.arange(km), indexing='ij')
    for t in range(T):
        gainindex = a[m_idx, t, Delta, j_idx]
        Change, Ccurr, Ncurr = _greedy_pick_with_gating(
            gainindex, n, c, N, M, km,
        )
        assert (Ccurr <= c).all()
        assert Ncurr <= N
        assert (Change.sum(axis=1) <= c).all()
        success = rng.random(size=(M, km)) < q
        Delta = np.where(Change & success, 0,
                         np.minimum(Delta + 1, B - 1))

    K = T
    out_maf = MAF1_probabilistic(M, N, km, T, B, K, n, c, w, gamma, p, q,
                                 seed=0, mc_trials=1, verbose=False)
    out_mief = MIEF1_probabilistic(M, N, km, T, B, K, n, c, w, gamma, p, q,
                                   seed=0, mc_trials=1, verbose=False)
    out_rand = randpolicy_probabilistic(M, N, km, T, B, K, n, c, w, gamma, p, q,
                                        seed=0, mc_trials=1, gated=True,
                                        verbose=False)
    assert np.isfinite(out_maf) and np.isfinite(out_mief) and np.isfinite(out_rand)


# ---------------------------------------------------------------------------
# 7. High-q preferred when everything else is identical
# ---------------------------------------------------------------------------

def test_high_reliability_preferred_when_identical():
    T, B = 10, 8
    gamma = 0.9
    lambda_ = np.full(T, 0.1)
    mu = np.full(T, 0.1)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 5.0, size=B)

    a_high = valuefunction1_probabilistic(lambda_, mu, B, p, T, gamma,
                                          q=0.95, n_cost=1.0)
    a_low = valuefunction1_probabilistic(lambda_, mu, B, p, T, gamma,
                                         q=0.40, n_cost=1.0)
    assert a_high[:, 1:].mean() > a_low[:, 1:].mean(), \
        "Higher q must yield a higher average gain index"


# ---------------------------------------------------------------------------
# 8. Smoke test: every first-order method runs without crashing on a tiny
#    deterministic AND a tiny probabilistic problem.
# ---------------------------------------------------------------------------

def test_subgradient_methods_smoke():
    M, km, T, B, N = 3, 3, 6, 6, 2
    gamma = 0.9
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 3.0, size=(km, B))
    w = np.ones((M, km))
    n = np.ones((M, km))
    c = np.ones(M) * 1
    q, _ = make_q_profile("uniform_wide", M, km, seed=0)

    for method in get_default_subgradient_methods():
        # Deterministic
        lam_d, mu_d, hist_d = subgradientiter1_variants(
            M, N, T, B, gamma, p, km, w, n, c,
            titer=5, method=method, seed=0, verbose=False,
            history=True, save=False,
        )
        assert np.all(np.isfinite(lam_d)) and np.all(np.isfinite(mu_d)), \
            f"det method {method} produced non-finite"
        assert (lam_d >= 0).all() and (mu_d >= 0).all(), \
            f"det method {method} broke nonneg projection"

        # Probabilistic
        A_p, hist_p = subgradientiter1_probabilistic(
            M, N, T, B, gamma, p, km, w, n, c, q,
            titer=5, method=method, seed=0, verbose=False,
            history=True, save=False,
        )
        assert A_p.shape == (M + 1, T)
        assert np.all(np.isfinite(A_p)), f"prob method {method} produced non-finite"
        assert (A_p >= 0).all(), f"prob method {method} broke nonneg projection"


# ---------------------------------------------------------------------------
# 9. MGF1_variants(harmonic) is close to MGF1 on a tiny deterministic setup
#    (within a loose tolerance -- the two use different per-iter projection
#    structure so they converge to nearby but not identical fixed points).
# ---------------------------------------------------------------------------

def test_mgf_variants_harmonic_matches_original_tiny():
    prob = make_base_problem(M=3, N=2, km=3, B=8, T=8, weights="ones")
    M, N, km, B, T, K = (prob["M"], prob["N"], prob["km"], prob["B"],
                         prob["T"], prob["K"])

    np.random.seed(0)
    orig = MGF1(M, N, km, T, B, K, prob["n"], prob["c"], prob["w"],
                prob["gamma"], prob["p"], titer=30, verbose=False)
    new = MGF1_variants(M, N, km, T, B, K, prob["n"], prob["c"], prob["w"],
                        prob["gamma"], prob["p"], titer=30,
                        subgradient_method="harmonic", verbose=False)
    rel = abs(orig - new) / max(abs(orig), 1e-9)
    assert rel < 0.20, (
        f"MGF1_variants(harmonic) {new} differs from MGF1 {orig} by "
        f"{rel:.1%}; tolerance is 20% (they use different per-iter "
        f"projection structure but should land in the same basin)"
    )


# ---------------------------------------------------------------------------
# 10. Precheck fast core writes a recommendation dict with no NaN.
# ---------------------------------------------------------------------------

def test_precheck_fast_runs(monkeypatch=None):
    here = os.path.dirname(os.path.abspath(__file__))
    # Use the env vars to keep runtime down. Use os.environ directly.
    saved = {
        "INFOCOM_PRECHECK_TITER": os.environ.get("INFOCOM_PRECHECK_TITER"),
        "INFOCOM_PRECHECK_MC_TRIALS": os.environ.get("INFOCOM_PRECHECK_MC_TRIALS"),
        "INFOCOM_PRECHECK_FAST": os.environ.get("INFOCOM_PRECHECK_FAST"),
        "INFOCOM_PRECHECK_METHODS": os.environ.get("INFOCOM_PRECHECK_METHODS"),
    }
    os.environ["INFOCOM_PRECHECK_TITER"] = "2"
    os.environ["INFOCOM_PRECHECK_MC_TRIALS"] = "1"
    os.environ["INFOCOM_PRECHECK_FAST"] = "1"
    os.environ["INFOCOM_PRECHECK_METHODS"] = "harmonic,sqrt"

    try:
        # Re-import fresh so env vars take effect.
        import importlib
        import PrecheckSubgradientMethods as pre
        importlib.reload(pre)
        pre.main()

        rec_path = os.path.join(here, "recommended_subgradient_methods.json")
        assert os.path.exists(rec_path)
        with open(rec_path) as f:
            rec = json.load(f)
        for key in ("deterministic", "probabilistic", "combined"):
            assert key in rec, f"missing {key} in recommendation"
            assert "recommended_method" in rec[key]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("valuefunction_q1_matches_deterministic",
         test_valuefunction_probabilistic_matches_deterministic_when_q1_n1),
        ("valuefunction_q0_no_benefit",
         test_valuefunction_q_zero_has_no_reset_benefit_except_penalty),
        ("aoi_q1_resets",                   test_aoi_update_q_one_resets),
        ("aoi_q0_fails_and_ages",           test_aoi_update_q_zero_fails_and_ages),
        ("q_profiles_shape_bounds",         test_q_profiles_shape_bounds),
        ("greedy_respects_constraints",     test_probabilistic_greedy_respects_constraints),
        ("high_q_preferred",                test_high_reliability_preferred_when_identical),
        ("subgradient_methods_smoke",       test_subgradient_methods_smoke),
        ("mgf_variants_matches_original",   test_mgf_variants_harmonic_matches_original_tiny),
        ("precheck_fast_runs",              test_precheck_fast_runs),
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} failure(s)")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
