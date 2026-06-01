"""
ErrorVsChannel_probabilistic.py - probabilistic-link variant of ErrorVsChannel.

Sweeps over the number of channels N (2..20 step 2) with km=2 empirical
penalties loaded from data/deterministic/loss.mat (sub-sampled p1, p2 from
inference-error data). Compares MGF / MAF (pure) / MIEF (reliability-aware)
/ Random (gated) under each of the 12 q profiles.

This sweep is run in TWO weight modes (see experiment_configs.py):
  * deterministic -- the heterogeneous weight matrix of the deterministic
    ErrorVsChannel.py (w=0.01 everywhere with two priority pairs = 1). Outputs
    keep the bare ``ErrorVsChannel_probabilistic`` stem.
  * ones (``weights_1``) -- w = 1 everywhere so reliability heterogeneity is the
    sole driver. Outputs carry the ``_weights_1`` suffix.

Configuration via env vars (defaults from _probabilistic_sweep_helpers):
    INFOCOM_TITER, INFOCOM_MC_TRIALS, INFOCOM_SEED, INFOCOM_PROFILES,
    INFOCOM_EXTRA_POLICIES, INFOCOM_SUBGRADIENT_METHOD
    INFOCOM_CHANNELS (e.g. "2,4,6,8,10,12,14,16,18,20")
    INFOCOM_WEIGHT_MODES (e.g. "deterministic" or "ones" to run a single mode)

Writes (under the reorganized layout), once per weight mode:
    data/probabilistic/ErrorVsChannel_probabilistic[_weights_1]_data.csv
    data/probabilistic/ErrorVsChannel_probabilistic[_weights_1]_summary.csv
    data/probabilistic/ErrorVsChannel_probabilistic[_weights_1]_q_profiles.npz
    plots/probabilistic/ErrorVsChannel_probabilistic[_weights_1].png
"""
import os
import numpy as np
import scipy.io as sio

from _bootstrap import paths

from _probabilistic_sweep_helpers import (
    run_sweep_both_modes,
    resolve_subgradient_method,
)


def _load_empirical_penalty(km, B):
    """Subsample p1, p2 every 5 indices to length B (matches MATLAB)."""
    loss = sio.loadmat(paths.data_path('loss.mat', probabilistic=False))
    p1 = loss['p1'].flatten()
    p2 = loss['p2'].flatten()
    p = np.zeros((km, B))
    i = 0
    for j_mat in range(1, 101, 5):
        if i >= B:
            break
        j_py = j_mat - 1
        if j_py < p1.size:
            p[0, i] = p1[j_py]
        if km >= 2 and j_py < p2.size:
            p[1, i] = p2[j_py]
        i += 1
    return p


def _parse_int_list(s, default):
    if not s:
        return default
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    M = 20
    km = 2
    B = 20
    T = 100
    K = T
    gamma = 0.9
    p = _load_empirical_penalty(km, B)
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
        base_stem="ErrorVsChannel_probabilistic",
        experiment_name="ErrorVsChannel",
        sweep_name="N",
        sweep_values=channels,
        build_problem=build_problem,
        sweep_label="Number of Channels (N)",
        title_prefix="ErrorVsChannel (empirical p1, p2)",
        subgradient_method=method,
        subgradient_source=source,
        use_log_y=False,
    )


if __name__ == "__main__":
    main()
