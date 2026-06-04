"""
ErrorVsNRobot_discount_probabilistic.py - robot ErrorVsN under a *single*
q-profile, swept over 12 discount factors gamma.

Same faithful robot problem as ErrorVsNRobot_probabilistic.py (M=4,
km=[2,1,1,1], B=40, T=100, c[m]=1, n=1, per-(source,task) robot-car penalties,
Bernoulli(q) delivery), but here the q-profile is fixed (default
``uniform_very_wide``) and gamma is swept over 12 values. This shows how the
policy separation vs channels N changes as the planner weighs the future more
heavily.

Layout: a GRID with one panel per discount factor; inside each panel the
channels N = 1..6 are swept (the natural robot control from main.m) with the
four policies:

    Random -> red circle    ('ro-')  "Random Policy"
    MAF    -> blue star      ('b*-')  "MAF Policy"
    MEF    -> green triangle ('g^-')  "MEF Policy"   (reliability-aware q*w*p)
    MGF    -> black diamond  ('kd-')  "Reoptimized MGF Policy"

A complementary single-axis summary (discounted error vs gamma at a fixed N) is
also written.

Outputs (plots/probabilistic/robot/ + data/probabilistic/):
    plots/.../ErrorVsNRobot_discount_<wmode>.png            (grid, one panel/gamma)
    plots/.../ErrorVsNRobot_discount_summary_<wmode>.png    (error vs gamma, N fixed)
    data/.../ErrorVsNRobot_discount_<wmode>_data.csv

Config (env vars):
    INFOCOM_PROFILE (uniform_very_wide), INFOCOM_WEIGHT_MODE (ones | priority),
    INFOCOM_GAMMAS (default 12 values), INFOCOM_CHANNELS (default 1..6),
    INFOCOM_T (100), INFOCOM_SUMMARY_N (2),
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
T = int(os.environ.get("INFOCOM_T", "100"))
SUMMARY_N = int(os.environ.get("INFOCOM_SUMMARY_N", "2"))

MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "20"))
RAND_TRIALS = int(os.environ.get("INFOCOM_RAND_TRIALS", "100"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
SUBG_TITER = int(os.environ.get("INFOCOM_SUBG_TITER", "1000"))

DEFAULT_GAMMAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

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


def _parse_float_list(s, default):
    if not s:
        return list(default)
    return [float(x) for x in s.split(",") if x.strip()]


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


def _run_policies(prob, N, w, q, gamma):
    return {
        "MGF": MGF_robot_probabilistic(prob, N, w, q, gamma, T, seed=SEED,
                                       mc_trials=MC_TRIALS, titer=SUBG_TITER),
        "MAF": MAF_robot_probabilistic(prob, N, w, q, gamma, T, seed=SEED,
                                       mc_trials=MC_TRIALS),
        "MEF": MEF_robot_probabilistic(prob, N, w, q, gamma, T, seed=SEED,
                                       mc_trials=MC_TRIALS,
                                       reliability_aware=True),
        "Random": randpolicy_robot_probabilistic(prob, N, w, q, gamma, T,
                                                  seed=SEED,
                                                  mc_trials=RAND_TRIALS),
    }


def _plot_grid(channels, gammas, data, wmode, png_path):
    """data[gamma][policy] = [vals over channels]; one panel per gamma."""
    n_p = len(gammas)
    n_cols = min(3, n_p)
    n_rows = (n_p + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.2 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    for idx, g in enumerate(gammas):
        ax = axes[idx // n_cols][idx % n_cols]
        for name, fmt, label in POLICY_STYLE:
            ax.plot(channels, data[g][name], fmt, linewidth=2, markersize=8,
                    label=label)
        ax.set_yscale("log")
        ax.set_title(rf"$\gamma$ = {g:g}", fontsize=12)
        ax.set_xlabel(r"Number of Channels ($N$)", fontsize=13)
        ax.set_ylabel("Dis. Sum of Inference Errors", fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=10)
    for idx in range(n_p, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(f"Robot ErrorVsN vs discount factor "
                 f"(profile={PROFILE}, w={wmode}, T={T})", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(gammas, data, wmode, png_path):
    """Discounted error vs gamma at N = SUMMARY_N, one line per policy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = []
    for name, fmt, label in POLICY_STYLE:
        ys = [data[g][name] for g in gammas]
        ax.plot(gammas, ys, fmt, linewidth=2, markersize=10)
        labels.append(label)
    ax.set_yscale("log")
    ax.set_xlabel(r"Discount factor $\gamma$", fontsize=18)
    ax.set_ylabel("Dis. Sum of Inference Errors", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_title(f"profile={PROFILE}, w={wmode}, N={SUMMARY_N}, T={T}",
                 fontsize=12)
    ax.legend(labels, loc="best", fontsize=12)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(channels, gammas, data, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "N"] + [n for n, _, _ in POLICY_STYLE])
        for g in gammas:
            for i, N in enumerate(channels):
                writer.writerow([g, N]
                                + [data[g][n][i] for n, _, _ in POLICY_STYLE])


def main():
    plot_dir = os.path.join(paths.PLOTS_PROBABILISTIC_DIR, "robot")
    os.makedirs(plot_dir, exist_ok=True)

    prob = load_robot_problem()
    channels = _parse_int_list(os.environ.get("INFOCOM_CHANNELS"), range(1, 7))
    gammas = _parse_float_list(os.environ.get("INFOCOM_GAMMAS"),
                               DEFAULT_GAMMAS)
    w = make_robot_weights(prob, WEIGHT_MODE)
    q = np.where(prob["valid"],
                 make_q_profile(PROFILE, prob["M"], prob["km_max"],
                                seed=SEED)[0], 1.0)

    print("Robot ErrorVsN vs discount factor")
    print(f"  profile={PROFILE}  weight_mode={WEIGHT_MODE}  T={T}")
    print(f"  channels={channels}")
    print(f"  gammas ({len(gammas)}): {gammas}")
    print(f"  MC_TRIALS={MC_TRIALS} RAND_TRIALS={RAND_TRIALS} "
          f"SEED={SEED} SUBG_TITER={SUBG_TITER}")

    data = {}
    for g in gammas:
        print(f"  --- gamma = {g:g} ---")
        data[g] = {n: [] for n, _, _ in POLICY_STYLE}
        for N in channels:
            means = _run_policies(prob, N, w, q, g)
            for n in data[g]:
                data[g][n].append(means[n])
            print(f"    N={N}: " + ", ".join(
                f"{n}={means[n]:.4g}" for n, _, _ in POLICY_STYLE))
        # Pin down the summary value (N = SUMMARY_N) -> per-policy scalar.
        data[g] = {n: vals for n, vals in data[g].items()}

    # Build summary dict: data_summary[gamma][policy] = value at SUMMARY_N.
    if SUMMARY_N in channels:
        sidx = channels.index(SUMMARY_N)
    else:
        sidx = 0
        print(f"  [warn] SUMMARY_N={SUMMARY_N} not in channels; "
              f"using N={channels[0]} for the summary plot.")
    data_summary = {g: {n: data[g][n][sidx] for n, _, _ in POLICY_STYLE}
                    for g in gammas}

    grid_png = os.path.join(plot_dir,
                            f"ErrorVsNRobot_discount_{WEIGHT_MODE}.png")
    summ_png = os.path.join(plot_dir,
                            f"ErrorVsNRobot_discount_summary_{WEIGHT_MODE}.png")
    csv_path = paths.data_path(
        f"ErrorVsNRobot_discount_{WEIGHT_MODE}_data.csv", probabilistic=True)

    _plot_grid(channels, gammas, data, WEIGHT_MODE, grid_png)
    _plot_summary(gammas, data_summary, WEIGHT_MODE, summ_png)
    _write_csv(channels, gammas, data, csv_path)
    print(f"  saved grid    {grid_png}")
    print(f"  saved summary {summ_png}")
    print(f"  saved data    {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
