"""
ErrorVsNRobot_horizon_probabilistic.py - robot ErrorVsN under a *single*
q-profile, swept over a range of horizons T (small to large).

Same faithful robot problem as ErrorVsNRobot_probabilistic.py (M=4,
km=[2,1,1,1], B=40, gamma=0.1, c[m]=1, n=1, per-(source,task) robot-car
penalties, Bernoulli(q) delivery), with the q-profile fixed (default
``uniform_very_wide``) and the horizon T swept. T is both the value-function
depth and the number of simulated steps. With the robot's small gamma the
discounted objective saturates quickly, so this sweep makes the
"make T small" guidance from TODO.md explicit.

Layout: a GRID with one panel per T; inside each panel channels N = 1..6 are
swept with the four policies:

    Random -> red circle    ('ro-')  "Random Policy"
    MAF    -> blue star      ('b*-')  "MAF Policy"
    MEF    -> green triangle ('g^-')  "MEF Policy"   (reliability-aware q*w*p)
    MGF    -> black diamond  ('kd-')  "Reoptimized MGF Policy"

A complementary single-axis summary (discounted error vs T at a fixed N) is
also written.

Outputs (plots/probabilistic/robot/ + data/probabilistic/):
    plots/.../ErrorVsNRobot_horizon_<wmode>.png            (grid, one panel/T)
    plots/.../ErrorVsNRobot_horizon_summary_<wmode>.png    (error vs T, N fixed)
    data/.../ErrorVsNRobot_horizon_<wmode>_data.csv

Config (env vars):
    INFOCOM_PROFILE (uniform_very_wide), INFOCOM_WEIGHT_MODE (ones | priority),
    INFOCOM_GAMMA (0.1), INFOCOM_TS (default 12 values), INFOCOM_CHANNELS (1..6),
    INFOCOM_SUMMARY_N (2),
    INFOCOM_MC_TRIALS (20), INFOCOM_RAND_TRIALS (100), INFOCOM_SEED (0),
    INFOCOM_SUBG_TITER (1000).
"""
import os
import csv

import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths  # noqa: F401  (bootstraps sys.path -> src/main)

from robot_data import load_robot_problem
from robot_probabilistic import (
    MGF_robot_probabilistic,
    MAF_robot_probabilistic,
    MEF_robot_probabilistic,
    randpolicy_robot_probabilistic,
)
from probability_profiles_probabilistic import make_q_profile


PROFILE = os.environ.get("INFOCOM_PROFILE", "uniform_very_wide")
WEIGHT_MODE = os.environ.get("INFOCOM_WEIGHT_MODE", "ones")
GAMMA = float(os.environ.get("INFOCOM_GAMMA", "0.1"))
SUMMARY_N = int(os.environ.get("INFOCOM_SUMMARY_N", "2"))

MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "20"))
RAND_TRIALS = int(os.environ.get("INFOCOM_RAND_TRIALS", "100"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
SUBG_TITER = int(os.environ.get("INFOCOM_SUBG_TITER", "1000"))

DEFAULT_TS = [2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

POLICY_STYLE = [
    ("Random", "ro-", "Random Policy"),
    ("MAF",    "b*-", "MAF Policy"),
    ("MEF",    "g^-", "MEF Policy"),
    ("MGF",    "kd-", "Reoptimized MGF Policy"),
]


def _parse_int_list(s, default):
    if not s:
        return list(default)
    return [int(x) for x in s.split(",") if x.strip()]


def make_robot_weights(prob, mode):
    M, km_max = prob["M"], prob["km_max"]
    if mode == "ones":
        w = np.ones((M, km_max))
    elif mode == "priority":
        w = np.full((M, km_max), 0.01)
        w[0, 1] = 1.0
        w[1, 0] = 1.0
    else:
        raise ValueError(f"Unknown weight mode {mode!r} (use ones|priority)")
    return w * prob["valid"]


def _run_policies(prob, N, w, q, T):
    return {
        "MGF": MGF_robot_probabilistic(prob, N, w, q, GAMMA, T, seed=SEED,
                                       mc_trials=MC_TRIALS, titer=SUBG_TITER),
        "MAF": MAF_robot_probabilistic(prob, N, w, q, GAMMA, T, seed=SEED,
                                       mc_trials=MC_TRIALS),
        "MEF": MEF_robot_probabilistic(prob, N, w, q, GAMMA, T, seed=SEED,
                                       mc_trials=MC_TRIALS,
                                       reliability_aware=True),
        "Random": randpolicy_robot_probabilistic(prob, N, w, q, GAMMA, T,
                                                  seed=SEED,
                                                  mc_trials=RAND_TRIALS),
    }


def _plot_grid(channels, ts, data, wmode, png_path):
    n_p = len(ts)
    n_cols = min(3, n_p)
    n_rows = (n_p + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.2 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    for idx, T in enumerate(ts):
        ax = axes[idx // n_cols][idx % n_cols]
        for name, fmt, label in POLICY_STYLE:
            ax.plot(channels, data[T][name], fmt, linewidth=2, markersize=8,
                    label=label)
        ax.set_yscale("log")
        ax.set_title(f"T = {T}", fontsize=12)
        ax.set_xlabel(r"Number of Channels ($N$)", fontsize=13)
        ax.set_ylabel("Dis. Sum of Inference Errors", fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=10)
    for idx in range(n_p, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(f"Robot ErrorVsN vs horizon T "
                 f"(profile={PROFILE}, w={wmode}, gamma={GAMMA:g})",
                 fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(ts, data, wmode, png_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = []
    for name, fmt, label in POLICY_STYLE:
        ys = [data[T][name] for T in ts]
        ax.plot(ts, ys, fmt, linewidth=2, markersize=10)
        labels.append(label)
    ax.set_yscale("log")
    ax.set_xlabel(r"Horizon $T$", fontsize=18)
    ax.set_ylabel("Dis. Sum of Inference Errors", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_title(f"profile={PROFILE}, w={wmode}, N={SUMMARY_N}, "
                 f"gamma={GAMMA:g}", fontsize=12)
    ax.legend(labels, loc="best", fontsize=12)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(channels, ts, data, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "N"] + [n for n, _, _ in POLICY_STYLE])
        for T in ts:
            for i, N in enumerate(channels):
                writer.writerow([T, N]
                                + [data[T][n][i] for n, _, _ in POLICY_STYLE])


def main():
    plot_dir = os.path.join(paths.PLOTS_PROBABILISTIC_DIR, "robot")
    os.makedirs(plot_dir, exist_ok=True)

    prob = load_robot_problem()
    channels = _parse_int_list(os.environ.get("INFOCOM_CHANNELS"), range(1, 7))
    ts = _parse_int_list(os.environ.get("INFOCOM_TS"), DEFAULT_TS)
    w = make_robot_weights(prob, WEIGHT_MODE)
    q = np.where(prob["valid"],
                 make_q_profile(PROFILE, prob["M"], prob["km_max"],
                                seed=SEED)[0], 1.0)

    print("Robot ErrorVsN vs horizon T")
    print(f"  profile={PROFILE}  weight_mode={WEIGHT_MODE}  gamma={GAMMA:g}")
    print(f"  channels={channels}")
    print(f"  Ts ({len(ts)}): {ts}")
    print(f"  MC_TRIALS={MC_TRIALS} RAND_TRIALS={RAND_TRIALS} "
          f"SEED={SEED} SUBG_TITER={SUBG_TITER}")

    data = {}
    for T in ts:
        print(f"  --- T = {T} ---")
        data[T] = {n: [] for n, _, _ in POLICY_STYLE}
        for N in channels:
            means = _run_policies(prob, N, w, q, T)
            for n in data[T]:
                data[T][n].append(means[n])
            print(f"    N={N}: " + ", ".join(
                f"{n}={means[n]:.4g}" for n, _, _ in POLICY_STYLE))

    if SUMMARY_N in channels:
        sidx = channels.index(SUMMARY_N)
    else:
        sidx = 0
        print(f"  [warn] SUMMARY_N={SUMMARY_N} not in channels; "
              f"using N={channels[0]} for the summary plot.")
    data_summary = {T: {n: data[T][n][sidx] for n, _, _ in POLICY_STYLE}
                    for T in ts}

    grid_png = os.path.join(plot_dir,
                            f"ErrorVsNRobot_horizon_{WEIGHT_MODE}.png")
    summ_png = os.path.join(plot_dir,
                            f"ErrorVsNRobot_horizon_summary_{WEIGHT_MODE}.png")
    csv_path = paths.data_path(
        f"ErrorVsNRobot_horizon_{WEIGHT_MODE}_data.csv", probabilistic=True)

    _plot_grid(channels, ts, data, WEIGHT_MODE, grid_png)
    _plot_summary(ts, data_summary, WEIGHT_MODE, summ_png)
    _write_csv(channels, ts, data, csv_path)
    print(f"  saved grid    {grid_png}")
    print(f"  saved summary {summ_png}")
    print(f"  saved data    {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
