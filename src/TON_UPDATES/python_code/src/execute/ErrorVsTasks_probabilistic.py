"""
ErrorVsTasks_probabilistic.py - probabilistic-link variant of ErrorVsTasks.

Sweeps over the number of tasks per source km (3..15 step 3) using the same
synthetic penalty model as ErrorVsChannelsmodel (j%3==0 linear, j%3==1
10*log, j%3==2 exp(0.5 d), 1-indexed). Compares MGF / MAF / MIEF / Random
under each of the 12 q profiles.

Run in two weight modes (see experiment_configs.py): deterministic (paper
half-weights: 1 for m+1<=M/2 else 0.01) and ones (w=1, ``_weights_1`` suffix).

Writes (once per weight mode):
    data/probabilistic/ErrorVsTasks_probabilistic[_weights_1]_data.csv
    data/probabilistic/ErrorVsTasks_probabilistic[_weights_1]_summary.csv
    data/probabilistic/ErrorVsTasks_probabilistic[_weights_1]_q_profiles.npz
    plots/probabilistic/ErrorVsTasks_probabilistic[_weights_1].png
"""
import os
import numpy as np

from _bootstrap import paths   # noqa: F401  (bootstraps sys.path)

from _probabilistic_sweep_helpers import (
    run_sweep_both_modes,
    resolve_subgradient_method,
    make_synthetic_penalty,
)


def _parse_int_list(s, default):
    if not s:
        return default
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    M = 20
    N = 10
    B = 20
    T = 100
    K = T
    gamma = 0.9

    p_full = make_synthetic_penalty(15, B)

    tasks = _parse_int_list(os.environ.get("INFOCOM_TASKS"),
                            list(range(3, 16, 3)))

    def build_problem(km):
        # Weights are supplied by run_sweep_both_modes per weight mode.
        n = np.ones((M, km))
        c = np.ones((M, km)) * 2
        p = p_full[:km, :]
        return {
            "M": M, "N": N, "km": km, "T": T, "B": B, "K": K,
            "n": n, "c": c, "p": p, "gamma": gamma,
        }

    method, source = resolve_subgradient_method()
    run_sweep_both_modes(
        base_stem="ErrorVsTasks_probabilistic",
        experiment_name="ErrorVsTasks",
        sweep_name="km",
        sweep_values=tasks,
        build_problem=build_problem,
        sweep_label="Number of Tasks (km)",
        title_prefix="ErrorVsTasks (synthetic model)",
        subgradient_method=method,
        subgradient_source=source,
        use_log_y=True,
    )


if __name__ == "__main__":
    main()
