"""
ErrorVsSources_probabilistic.py - probabilistic-link ErrorVsSources experiment.

For each q profile, sweep M (sources) and compare:
    MGF probabilistic, MAF probabilistic (pure), MIEF probabilistic
    (reliability-aware), Random probabilistic (gated).

Optional extra rows: MAF reliability-aware and pure MIEF, controlled by
INFOCOM_EXTRA_POLICIES=1.

Configuration via env vars:
    INFOCOM_TITER              (default 1000)
    INFOCOM_MC_TRIALS          (default 10)
    INFOCOM_SEED               (default 0)
    INFOCOM_SOURCES            (default '2,4,6,8,10,12,14,16,18,20')
    INFOCOM_PROFILES           (default = list_q_profiles())
    INFOCOM_EXTRA_POLICIES     (default 0)
    INFOCOM_SUBGRADIENT_METHOD (optional override; default = read recommended
                                 method from recommended_subgradient_methods.json)

Writes:
    data/probabilistic/ErrorVsSources_probabilistic_data.csv
    data/probabilistic/ErrorVsSources_probabilistic_summary.csv
    data/probabilistic/ErrorVsSources_probabilistic_q_profiles.npz
    plots/probabilistic/ErrorVsSources_probabilistic.png
"""
import os
import numpy as np

from _bootstrap import paths   # noqa: F401  (bootstraps sys.path)

from _probabilistic_sweep_helpers import (
    run_sweep,
    resolve_subgradient_method,
    make_synthetic_penalty,
)


def _parse_int_list(s, default):
    if not s:
        return default
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    km = 9
    B = 20
    N = 10
    T = 100
    K = T
    gamma = 0.9
    p = make_synthetic_penalty(km, B)

    sources = _parse_int_list(os.environ.get("INFOCOM_SOURCES"),
                              list(range(2, 21, 2)))

    def build_problem(M):
        n = np.ones((M, km))
        c = np.ones((M, km)) * 2
        w = np.ones((M, km))   # probabilistic experiment: w=1 everywhere
        return {
            "M": M, "N": N, "km": km, "T": T, "B": B, "K": K,
            "n": n, "c": c, "w": w, "p": p, "gamma": gamma,
        }

    method, source = resolve_subgradient_method()
    run_sweep(
        output_stem="ErrorVsSources_probabilistic",
        sweep_name="M",
        sweep_values=sources,
        build_problem=build_problem,
        sweep_label="M (sources)",
        title_prefix="ErrorVsSources (synthetic model)",
        subgradient_method=method,
        subgradient_source=source,
        use_log_y=True,
    )


if __name__ == "__main__":
    main()
