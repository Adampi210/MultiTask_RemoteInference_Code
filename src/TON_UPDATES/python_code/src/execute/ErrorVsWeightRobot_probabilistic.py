"""
ErrorVsWeightRobot_probabilistic.py - probabilistic-link reimplementation of the
MATLAB robot experiment ``matlab_code/robot_code/mainweight.m`` (ErrorVsWeight).

Same faithful robot problem as ErrorVsNRobot_probabilistic.py (M=4,
km=[2,1,1,1], B=40, T=100, gamma=0.1, c[m]=1, n=1, per-(source,task) robot-car
penalties), with per-link delivery reliability q[m, j].

mainweight.m fixes N = 2 and sweeps the weight w_{1,2} on source 1's second
task (pair (1,2) in MATLAB, (0,1) here) over 1..10 while every other pair keeps
weight 1. The weight scales that pair's penalty in the objective, exactly as in
the deterministic ``w * p`` convention. We reproduce that sweep for every
q-profile (plus a q=1 deterministic reference panel).

Policies / styling match main.m / mainweight.m (plus MEF, the robot MIEF):
    Random -> red circle    ('ro-')  "Random Policy"
    MAF    -> blue star      ('b*-')  "MAF Policy"
    MEF    -> green triangle ('g^-')  "MEF Policy"   (reliability-aware q*w*p)
    MGF    -> black diamond  ('kd-')  "Reoptimized MGF Policy"

Outputs (plots/probabilistic/robot/ + data/probabilistic/):
    plots/.../ErrorVsWeightRobot_probabilistic.png       (grid, one panel/profile)
    plots/.../ErrorVsWeightRobot_<profile>.png           (single-axis, MATLAB look)
    data/.../ErrorVsWeightRobot_probabilistic_data.csv

Config (env vars):
    INFOCOM_MC_TRIALS (20), INFOCOM_RAND_TRIALS (100), INFOCOM_SEED (0),
    INFOCOM_SUBG_TITER (2000), INFOCOM_PROFILES (default all 12),
    INFOCOM_WEIGHTS (default 1..10), INFOCOM_N (default 2),
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


GAMMA = 0.1
T = 100

MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "20"))
RAND_TRIALS = int(os.environ.get("INFOCOM_RAND_TRIALS", "100"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
SUBG_TITER = int(os.environ.get("INFOCOM_SUBG_TITER", "2000"))
N_FIXED = int(os.environ.get("INFOCOM_N", "2"))

INCLUDE_Q1_REF = os.environ.get("INFOCOM_Q1_REF", "1") != "0"
Q1_REF_NAME = "q=1 (deterministic ref)"

# The swept weight is applied to source 1's second task (MATLAB pair (1,2)).
WEIGHT_PAIR = (0, 1)

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


def _parse_float_list(s, default):
    if not s:
        return list(default)
    return [float(x) for x in s.split(",") if x.strip()]


def _weights_for(prob, weight):
    """w = 1 everywhere (valid pairs) except the swept pair gets `weight`."""
    M, km_max = prob["M"], prob["km_max"]
    w = np.ones((M, km_max))
    w[WEIGHT_PAIR] = float(weight)
    return w * prob["valid"]


def _q_for_profile(profile, prob):
    if profile == Q1_REF_NAME:
        return np.ones((prob["M"], prob["km_max"]))
    q, _ = make_q_profile(profile, prob["M"], prob["km_max"], seed=SEED)
    return np.where(prob["valid"], q, 1.0)


def _run_policies(prob, w, q):
    mgf = MGF_robot_probabilistic(prob, N_FIXED, w, q, GAMMA, T, seed=SEED,
                                  mc_trials=MC_TRIALS, titer=SUBG_TITER)
    maf = MAF_robot_probabilistic(prob, N_FIXED, w, q, GAMMA, T, seed=SEED,
                                  mc_trials=MC_TRIALS)
    mef = MEF_robot_probabilistic(prob, N_FIXED, w, q, GAMMA, T, seed=SEED,
                                  mc_trials=MC_TRIALS, reliability_aware=True)
    rnd = randpolicy_robot_probabilistic(prob, N_FIXED, w, q, GAMMA, T,
                                         seed=SEED, mc_trials=RAND_TRIALS)
    return {"MGF": mgf, "MAF": maf, "MEF": mef, "Random": rnd}


def _write_csv(weights, profile_series, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["profile", "weight"] + [n for n, _, _ in POLICY_STYLE])
        for profile, series in profile_series.items():
            for i, wv in enumerate(weights):
                writer.writerow([profile, wv]
                                + [series[n][i] for n, _, _ in POLICY_STYLE])


def _plot_grid(weights, profiles, profile_series, png_path):
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
            ax.plot(weights, series[name], fmt, linewidth=2, markersize=8,
                    label=label)
        ax.set_title(profile, fontsize=12)
        ax.set_xlabel(r"weight $w_{1,2}$", fontsize=13)
        ax.set_ylabel("Dis. Sum of Inference Errors", fontsize=13)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=10)
    for idx in range(n_p, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(f"Robot ErrorVsWeight (probabilistic, N={N_FIXED})",
                 fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_single(weights, series, png_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = []
    for name, fmt, label in POLICY_STYLE:
        ax.plot(weights, series[name], fmt, linewidth=2, markersize=10)
        labels.append(label)
    ax.set_xlabel(r"weight $w_{1,2}$", fontsize=18)
    ax.set_ylabel("Dis. Sum of Inference Errors", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend(labels, loc="best", fontsize=12)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_dir = os.path.join(paths.PLOTS_PROBABILISTIC_DIR, "robot")
    os.makedirs(plot_dir, exist_ok=True)

    prob = load_robot_problem()
    weights = _parse_float_list(os.environ.get("INFOCOM_WEIGHTS"),
                                range(1, 11))
    profiles = parse_profiles(default=list_q_profiles())
    panel_profiles = ([Q1_REF_NAME] + profiles if INCLUDE_Q1_REF
                      else list(profiles))

    print("Robot ErrorVsWeight (probabilistic)")
    print(f"  M={prob['M']} km={prob['km_vec']} B={prob['B']} "
          f"T={T} gamma={GAMMA} N={N_FIXED}")
    print(f"  MC_TRIALS={MC_TRIALS} RAND_TRIALS={RAND_TRIALS} "
          f"SEED={SEED} SUBG_TITER={SUBG_TITER}")
    print(f"  weights={weights}  (swept on pair {WEIGHT_PAIR})")
    print(f"  profiles ({len(panel_profiles)}): {panel_profiles}")

    profile_series = {}
    for profile in panel_profiles:
        print(f"  --- profile: {profile} ---")
        q = _q_for_profile(profile, prob)
        series = {n: [] for n, _, _ in POLICY_STYLE}
        for wv in weights:
            w = _weights_for(prob, wv)
            means = _run_policies(prob, w, q)
            for n in series:
                series[n].append(means[n])
            print(f"    w={wv:g}: " + ", ".join(
                f"{n}={means[n]:.4g}" for n, _, _ in POLICY_STYLE))
        profile_series[profile] = series

    png_path = os.path.join(plot_dir, "ErrorVsWeightRobot_probabilistic.png")
    csv_path = paths.data_path("ErrorVsWeightRobot_probabilistic_data.csv",
                               probabilistic=True)
    _plot_grid(weights, panel_profiles, profile_series, png_path)
    _write_csv(weights, profile_series, csv_path)
    print(f"  saved grid {png_path}")
    print(f"  saved data {csv_path}")

    for profile in SINGLEPLOT_PROFILES:
        if profile not in profile_series:
            continue
        sp_path = os.path.join(plot_dir, f"ErrorVsWeightRobot_{profile}.png")
        _plot_single(weights, profile_series[profile], sp_path)
        print(f"  saved single {sp_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
