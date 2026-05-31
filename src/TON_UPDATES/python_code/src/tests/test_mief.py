"""
test_mief.py - Unit / sanity tests for the MIEF baseline.

Run either with pytest:
    pytest test_mief.py -v
or directly:
    python test_mief.py
"""
import numpy as np

from _bootstrap import paths   # noqa: F401  (bootstraps sys.path)

from greedy_scheduler import greedy_select, as_1d_c
from MIEF1 import MIEF1, mief_select, mief_priority


def _maf_select(Delta, M, km, n, c_use, N):
    """Reference MAF priority (Delta + 1) -> shared greedy. Used to compare
    MIEF's ordering against MAF's under the trivial penalty / equal weights."""
    priority = Delta.astype(float) + 1.0
    return greedy_select(priority, n, c_use, N, M, km)


def test_mief_matches_maf_when_p_is_identity_and_weights_equal():
    """If p[j, d] = d + 1 (i.e. "p(delta) = delta" using 1-based AoI) and all
    weights are equal, MIEF's priority is proportional to MAF's, so the two
    policies must produce identical schedules."""
    rng = np.random.default_rng(0)
    M, km, B, N = 6, 3, 20, 4
    c_use = np.full(M, 2.0)
    n = np.ones((M, km))
    w = np.full((M, km), 0.7)
    p = np.tile(np.arange(1, B + 1, dtype=float), (km, 1))  # p[j, d] = d + 1

    for trial in range(20):
        Delta = rng.integers(0, B, size=(M, km))
        a_maf, _, _, _ = _maf_select(Delta, M, km, n, c_use, N)
        a_mief, _, _, _ = mief_select(Delta, M, km, n, c_use, N, w, p)
        assert np.array_equal(a_maf, a_mief), (
            f"MIEF != MAF on trial {trial}\nDelta=\n{Delta}\n"
            f"MAF=\n{a_maf}\nMIEF=\n{a_mief}"
        )


def test_mief_prefers_high_penalty_over_high_age():
    """A lower-age task with a much larger penalty must be scheduled before
    a higher-age task with a tiny penalty (a behaviour MAF cannot produce)."""
    M, km, B, N = 2, 1, 20, 1
    c_use = np.array([1.0, 1.0])
    n = np.ones((M, km))
    w = np.ones((M, km))
    # Task 0: penalty grows fast (exp); task 0 has small age.
    # Task 1: same penalty function but tiny weight; task 1 has big age.
    p = np.zeros((km, B))
    p[0, :] = np.arange(1, B + 1)  # p(delta) = delta
    Delta = np.array([[1], [10]])  # row 0 age 2, row 1 age 11 (1-based)

    # Heavily upweight the younger source.
    w_biased = np.array([[100.0], [0.001]])
    a_mief, _, _, _ = mief_select(Delta, M, km, n, c_use, N, w_biased, p)
    a_maf, _, _, _ = _maf_select(Delta, M, km, n, c_use, N)

    assert a_maf[1, 0] == 1 and a_maf[0, 0] == 0, "MAF must pick the older task"
    assert a_mief[0, 0] == 1 and a_mief[1, 0] == 0, (
        "MIEF must pick the younger high-penalty task"
    )


def test_mief_respects_compute_and_channel_constraints():
    """Over many random states the MIEF action must satisfy:
       sum_j action[m, j]                <= c_use[m]   for every m,
       sum_{m, j} action[m, j] * n[m, j] <= N.
    """
    rng = np.random.default_rng(42)
    M, km, B = 8, 4, 15
    p = np.zeros((km, B))
    for j in range(km):
        p[j] = np.exp(0.3 * np.arange(1, B + 1))

    for trial in range(50):
        N = int(rng.integers(1, 2 * M))
        c_use = rng.integers(1, km + 1, size=M).astype(float)
        n = rng.integers(1, 4, size=(M, km)).astype(float)
        w = rng.random((M, km)) + 1e-3
        Delta = rng.integers(0, B, size=(M, km))

        action, _, _, _ = mief_select(Delta, M, km, n, c_use, N, w, p)

        assert action.dtype.kind in ("i", "u"), "action must be integer-valued"
        assert set(np.unique(action)).issubset({0, 1}), "action entries must be 0/1"
        comp_used = action.sum(axis=1)
        assert np.all(comp_used <= c_use + 1e-9), (
            f"trial {trial}: compute cap violated. "
            f"used={comp_used}, cap={c_use}"
        )
        chan_used = float((action * n).sum())
        assert chan_used <= N + 1e-9, (
            f"trial {trial}: channel cap violated. used={chan_used}, cap={N}"
        )


def test_mief_priority_is_w_times_p_of_delta():
    """Direct check of the priority-score formula score = w * p(Delta)."""
    M, km, B = 3, 2, 5
    Delta = np.array([[0, 4], [2, 1], [3, 0]])
    w = np.array([[1.0, 2.0], [0.5, 1.5], [0.1, 3.0]])
    p = np.array([
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [0.5, 1.5, 2.5, 3.5, 4.5],
    ])
    pri = mief_priority(Delta, w, p, km)
    expected = np.array([
        [w[0, 0] * p[0, 0], w[0, 1] * p[1, 4]],
        [w[1, 0] * p[0, 2], w[1, 1] * p[1, 1]],
        [w[2, 0] * p[0, 3], w[2, 1] * p[1, 0]],
    ])
    np.testing.assert_allclose(pri, expected)


def test_mief_end_to_end_runs_and_satisfies_constraints():
    """Full-rollout smoke test: MIEF1 finishes, returns a finite scalar, and
    every per-step decision respects both resource budgets."""
    M, N, km, T, B = 5, 3, 3, 6, 12
    K, gamma = T, 0.9
    n = np.ones((M, km))
    c = np.full((M, km), 2)  # passed as (M, km); as_1d_c picks column 0
    w = np.full((M, km), 0.5)
    w[0, 0] = 5.0
    p = np.zeros((km, B))
    p[0, :] = np.arange(1, B + 1)
    p[1, :] = 10 * np.log(np.arange(1, B + 1))
    p[2, :] = np.exp(0.4 * np.arange(1, B + 1))

    result = MIEF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=False)
    assert np.isfinite(result)

    # Re-simulate so we can assert per-step feasibility.
    c_use = as_1d_c(c, M)
    Delta = np.zeros((M, km), dtype=np.int64)
    for t in range(K):
        action, Change, _, _ = mief_select(Delta, M, km, n, c_use, N, w, p)
        assert np.all(action.sum(axis=1) <= c_use + 1e-9)
        assert (action * n).sum() <= N + 1e-9
        Delta = np.where(Change, 0, np.minimum(Delta + 1, B - 1))


if __name__ == "__main__":
    tests = [
        test_mief_matches_maf_when_p_is_identity_and_weights_equal,
        test_mief_prefers_high_penalty_over_high_age,
        test_mief_respects_compute_and_channel_constraints,
        test_mief_priority_is_w_times_p_of_delta,
        test_mief_end_to_end_runs_and_satisfies_constraints,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)
