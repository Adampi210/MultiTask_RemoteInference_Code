"""
ErrorVsChannelsmodel_probabilistic.py - probabilistic-link variant of
ErrorVsChannelsmodel.

Sweeps over the number of channels N (2..20 step 2) using the km=9 synthetic
penalty model (j%3==0 linear, j%3==1 10*log, j%3==2 exp(0.5 d), 1-indexed).
Compares MGF / MAF / MIEF / Random under each of the 12 q profiles.

Run in two weight modes (see experiment_configs.py): deterministic (paper
half-weights: 1 for m+1<=M/2 else 0.01) and ones (w=1, ``_weights_1`` suffix).

Writes (once per weight mode):
    data/probabilistic/ErrorVsChannelsmodel_probabilistic[_weights_1]_data.csv
    data/probabilistic/ErrorVsChannelsmodel_probabilistic[_weights_1]_summary.csv
    data/probabilistic/ErrorVsChannelsmodel_probabilistic[_weights_1]_q_profiles.npz
    plots/probabilistic/ErrorVsChannelsmodel_probabilistic[_weights_1].png
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
    km = 9
    B = 20
    T = 100
    K = T
    gamma = 0.9
    p = make_synthetic_penalty(km, B)
    n = np.ones((M, km))
    c = np.ones((M, km)) * 2

    channels = _parse_int_list(os.environ.get("INFOCOM_CHANNELS"),
                               list(range(2, 21, 2)))

    def build_problem(N):
        # Weights are supplied by run_sweep_both_modes per weight mode.
        return {
            "M": M, "N": N, "km": km, "T": T, "B": B, "K": K,
            "n": n, "c": c, "p": p, "gamma": gamma,
        }

    method, source = resolve_subgradient_method()
    run_sweep_both_modes(
        base_stem="ErrorVsChannelsmodel_probabilistic",
        experiment_name="ErrorVsChannelsmodel",
        sweep_name="N",
        sweep_values=channels,
        build_problem=build_problem,
        sweep_label="Number of Channels (N)",
        title_prefix="ErrorVsChannelsmodel (synthetic 9-task model)",
        subgradient_method=method,
        subgradient_source=source,
        use_log_y=True,
    )


if __name__ == "__main__":
    main()
