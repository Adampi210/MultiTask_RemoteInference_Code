"""
ErrorVsNRobot_probabilistic.py - probabilistic-link reimplementation of the
MATLAB robot experiment ``matlab_code/robot_code/main.m`` (ErrorVsNRobot).

Faithful problem setup (see robot_probabilistic.py / robot_data.py):
    M = 4 robot cars, km = [2, 1, 1, 1] tasks/source, B = 40, T = 100,
    gamma = 0.1, compute cap c[m] = 1, channel cost n = 1, per-(source, task)
    empirical 1-IoU penalties from the detection_results_robot_*.csv files.

Extension: each scheduled pair (m, j) is delivered only with probability
q[m, j] (Bernoulli). At q = 1 the policies reduce exactly to the deterministic
MATLAB robot policies.

This sweeps channels N = 1..6 (main.m) and produces, for every q-profile from
probability_profiles_probabilistic (plus a q=1 deterministic reference panel),
the MATLAB robot policies (plus MEF, the robot MIEF):

    Random -> red circle    ('ro-')  "Random Policy"
    MAF    -> blue star      ('b*-')  "MAF Policy"
    MEF    -> green triangle ('g^-')  "MEF Policy"   (reliability-aware q*w*p)
    MGF    -> black diamond  ('kd-')  "Reoptimized MGF Policy"

It runs BOTH weight modes the user asked for:
    * "priority" (orig_weights): w = 0.01 everywhere except two priority pairs
      = 1  (pair (1,2) -- source 1's second task, emphasized by mainweight.m --
      and pair (2,1)).
    * "ones" (weights_1): w = 1 everywhere.

Outputs (plots/probabilistic/robot/ + data/probabilistic/):
    plots/.../ErrorVsNRobot_probabilistic_<wtag>.png        (grid, one panel/profile)
    plots/.../ErrorVsNRobot_<profile>_<wtag>.png            (single-axis, MATLAB look)
    data/.../ErrorVsNRobot_probabilistic_<wtag>_data.csv
    data/.../ErrorVsNRobot_probabilistic_q_profiles.npz

Config (env vars):
    INFOCOM_MC_TRIALS (20), INFOCOM_RAND_TRIALS (100), INFOCOM_SEED (0),
    INFOCOM_SUBG_TITER (2000), INFOCOM_PROFILES (default all 12),
    INFOCOM_CHANNELS (default 1..6), INFOCOM_WEIGHT_MODES (default both),
    INFOCOM_Q1_REF (1 -> prepend a q=1 reference panel),
    INFOCOM_SINGLEPLOT_PROFILES (default uniform_very_wide,bimodal_extreme).
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
from probability_profiles_probabilistic import make_q_profile, list_q_profiles
from _probabilistic_sweep_helpers import parse_profiles


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMMA = 0.1
T = 100

MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "20"))
RAND_TRIALS = int(os.environ.get("INFOCOM_RAND_TRIALS", "100"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
SUBG_TITER = int(os.environ.get("INFOCOM_SUBG_TITER", "2000"))

INCLUDE_Q1_REF = os.environ.get("INFOCOM_Q1_REF", "1") != "0"
Q1_REF_NAME = "q=1 (deterministic ref)"

# (filename tag, human label) for the two weight modes.
ALL_WEIGHT_MODES = ["priority", "ones"]
WEIGHT_LABEL = {"priority": "priority weights (0.01 except 2 priority pairs=1)",
                "ones": "w = 1"}

# Plot/legend order + MATLAB-style formats (main.m + MEF, the robot MIEF).
POLICY_STYLE = [
    ("Random", "ro-", "Random Policy"),
    ("MAF",    "b*-", "MAF Policy"),
    ("MEF",    "g^-", "MEF Policy"),
    ("MGF",    "kd-", "Reoptimized MGF Policy"),
]

SINGLEPLOT_PROFILES = [
    p.strip() for p in os.environ.get(
        "INFOCOM_SINGLEPLOT_PROFILES",
        "uniform_very_wide,bimodal_extreme").split(",") if p.strip()
]


def _parse_int_list(s, default):
    if not s:
        return list(default)
    return [int(x) for x in s.split(",") if x.strip()]


def _resolve_weight_modes():
    env = os.environ.get("INFOCOM_WEIGHT_MODES")
    if not env:
        return list(ALL_WEIGHT_MODES)
    requested = [m.strip() for m in env.split(",") if m.strip()]
    chosen = [m for m in ALL_WEIGHT_MODES if m in requested]
    if not chosen:
        raise ValueError(
            f"INFOCOM_WEIGHT_MODES={env!r} has no valid modes; choose from "
            f"{ALL_WEIGHT_MODES}")
    return chosen


def make_robot_weights(prob, mode):
    """(M, km_max) weights for a mode; invalid pairs forced to 0.

    priority: 0.01 everywhere, with priority pairs (0,1) and (1,0) = 1
              (MATLAB 1-indexed (1,2) and (2,1)).
    ones    : 1 everywhere.
    """
    M, km_max = prob["M"], prob["km_max"]
    valid = prob["valid"]
    if mode == "ones":
        w = np.ones((M, km_max))
    elif mode == "priority":
        w = np.full((M, km_max), 0.01)
        w[0, 1] = 1.0   # source 1, task 2 (emphasized by mainweight.m)
        w[1, 0] = 1.0   # source 2, task 1
    else:
        raise ValueError(f"Unknown weight mode {mode!r}")
    return w * valid    # zero out padded pairs


def _q_for_profile(profile, prob):
    if profile == Q1_REF_NAME:
        return np.ones((prob["M"], prob["km_max"]))
    q, _ = make_q_profile(profile, prob["M"], prob["km_max"], seed=SEED)
    # Padded pairs never matter, but keep them in [0,1] tidy.
    return np.where(prob["valid"], q, 1.0)


def _run_policies(prob, N, w, q):
    mgf = MGF_robot_probabilistic(prob, N, w, q, GAMMA, T, seed=SEED,
                                  mc_trials=MC_TRIALS, titer=SUBG_TITER)
    maf = MAF_robot_probabilistic(prob, N, w, q, GAMMA, T, seed=SEED,
                                  mc_trials=MC_TRIALS)
    mef = MEF_robot_probabilistic(prob, N, w, q, GAMMA, T, seed=SEED,
                                  mc_trials=MC_TRIALS, reliability_aware=True)
    rnd = randpolicy_robot_probabilistic(prob, N, w, q, GAMMA, T, seed=SEED,
                                         mc_trials=RAND_TRIALS)
    return {"MGF": mgf, "MAF": maf, "MEF": mef, "Random": rnd}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_csv(channels, profile_series, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["profile", "N"] + [n for n, _, _ in POLICY_STYLE])
        for profile, series in profile_series.items():
            for i, N in enumerate(channels):
                writer.writerow([profile, N]
                                + [series[n][i] for n, _, _ in POLICY_STYLE])


def _plot_grid(channels, profiles, profile_series, wlabel, png_path):
    n_p = len(profiles)
    n_cols = min(3, n_p)
    n_rows = (n_p + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.2 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    for idx, profile in enumerate(profiles):
        ax = axes[idx // n_cols][idx % n_cols]
        series = profile_series[profile]
        for name, fmt, label in POLICY_STYLE:
            ax.plot(channels, series[name], fmt, linewidth=2, markersize=8,
                    label=label)
        ax.set_yscale("log")
        ax.set_title(profile, fontsize=12)
        ax.set_xlabel(r"Number of Channels ($N$)", fontsize=13)
        ax.set_ylabel("Dis. Sum of Inference Errors", fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=10)
    for idx in range(n_p, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(f"Robot ErrorVsN (probabilistic, {wlabel})", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_single(channels, series, png_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = []
    for name, fmt, label in POLICY_STYLE:
        ax.plot(channels, series[name], fmt, linewidth=2, markersize=10)
        labels.append(label)
    ax.set_xlabel(r"Number of Channels ($N$)", fontsize=18)
    ax.set_ylabel("Dis. Sum of Inference Errors", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_yscale("log")
    ax.legend(labels, loc="best", fontsize=12)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_dir = os.path.join(paths.PLOTS_PROBABILISTIC_DIR, "robot")
    os.makedirs(plot_dir, exist_ok=True)

    prob = load_robot_problem()
    channels = _parse_int_list(os.environ.get("INFOCOM_CHANNELS"),
                               range(1, 7))
    profiles = parse_profiles(default=list_q_profiles())
    weight_modes = _resolve_weight_modes()
    panel_profiles = ([Q1_REF_NAME] + profiles if INCLUDE_Q1_REF
                      else list(profiles))

    print("Robot ErrorVsN (probabilistic)")
    print(f"  M={prob['M']} km={prob['km_vec']} B={prob['B']} "
          f"T={T} gamma={GAMMA}")
    print(f"  MC_TRIALS={MC_TRIALS} RAND_TRIALS={RAND_TRIALS} "
          f"SEED={SEED} SUBG_TITER={SUBG_TITER}")
    print(f"  channels={channels}")
    print(f"  profiles ({len(panel_profiles)}): {panel_profiles}")
    print(f"  weight modes: {weight_modes}")

    # Save q profiles once for reproducibility.
    npz_path = paths.data_path("ErrorVsNRobot_probabilistic_q_profiles.npz",
                               probabilistic=True)
    np.savez(npz_path, **{
        prof.replace(" ", "_").replace("=", ""):
            _q_for_profile(prof, prob) for prof in panel_profiles})

    for wmode in weight_modes:
        w = make_robot_weights(prob, wmode)
        wlabel = WEIGHT_LABEL[wmode]
        print(f"\n========== weight mode: {wmode} ({wlabel}) ==========")
        profile_series = {}
        for profile in panel_profiles:
            print(f"  --- profile: {profile} ---")
            q = _q_for_profile(profile, prob)
            series = {n: [] for n, _, _ in POLICY_STYLE}
            for N in channels:
                means = _run_policies(prob, N, w, q)
                for n in series:
                    series[n].append(means[n])
                print(f"    N={N}: " + ", ".join(
                    f"{n}={means[n]:.4g}" for n, _, _ in POLICY_STYLE))
            profile_series[profile] = series

        png_path = os.path.join(
            plot_dir, f"ErrorVsNRobot_probabilistic_{wmode}.png")
        csv_path = paths.data_path(
            f"ErrorVsNRobot_probabilistic_{wmode}_data.csv", probabilistic=True)
        _plot_grid(channels, panel_profiles, profile_series, wlabel, png_path)
        _write_csv(channels, profile_series, csv_path)
        print(f"  saved grid {png_path}")
        print(f"  saved data {csv_path}")

        for profile in SINGLEPLOT_PROFILES:
            if profile not in profile_series:
                continue
            sp_path = os.path.join(
                plot_dir, f"ErrorVsNRobot_{profile}_{wmode}.png")
            _plot_single(channels, profile_series[profile], sp_path)
            print(f"  saved single {sp_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
