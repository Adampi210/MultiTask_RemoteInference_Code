"""
ErrorVsChannel_robot_probabilistic.py - probabilistic version of the robot-loss
ErrorVsChannel experiment, laid out as a per-q-profile grid for picking the best
schedule.

This is the probabilistic counterpart of the robot-loss ErrorVsChannel
(`ErrorVsChannel_steps.py` / `ErrorVsChannel.py`): same problem (M=20, km=2,
penalties from the robot-collected loss.mat, channels N=2..20), but each
scheduled pair (m,j) is delivered only with probability q_{m,j}. It

  * uses the BEST subgradient method (resolved from
    recommended_subgradient_methods.json -> "adam"; override with
    INFOCOM_SUBGRADIENT_METHOD),
  * sweeps ALL q-profiles ("schedules") from probability_profiles_probabilistic,
  * runs BOTH weight modes -- the original empirical weights (w=0.01 except the
    two priority cells = 1) and w=1 everywhere,
  * and saves every profile as one subplot of a single GRID figure per weight
    mode, so the best-looking schedule can be chosen by eye.

Policies / styling match the agreed MATLAB look:
    Random -> red circle  ('ro-')  "Random Policy"
    MAF    -> blue star    ('b*-')  "MAF Policy"
    MGF    -> black diamond('kd-')  "MGF Policy"
    MEF    -> green tri.    ('g^-')  "MEF Policy"   (= reliability-aware MIEF)

Configuration via env vars:
    INFOCOM_TITER (1000), INFOCOM_MC_TRIALS (20), INFOCOM_SEED (0),
    INFOCOM_PROFILES (default: all 12), INFOCOM_CHANNELS (default 2..20 step 2),
    INFOCOM_SUBGRADIENT_METHOD (default: JSON recommendation),
    INFOCOM_WEIGHT_MODES (default: both; e.g. "ones" or "deterministic").

Writes (under plots/probabilistic/robot/ and data/probabilistic/):
    plots/probabilistic/robot/ErrorVsChannel_robot_probabilistic_<wtag>.png
    data/probabilistic/ErrorVsChannel_robot_probabilistic_<wtag>_data.csv
"""
import os
import csv
import numpy as np
import scipy.io as sio
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths

from MGF1_probabilistic import MGF1_probabilistic
from MAF1_probabilistic import MAF1_probabilistic
from MIEF1_probabilistic import MIEF1_probabilistic
from randpolicy_probabilistic import randpolicy_probabilistic
from probability_profiles_probabilistic import make_q_profile, list_q_profiles
from experiment_configs import make_weights, WEIGHTS_ONES, WEIGHTS_DETERMINISTIC
from _probabilistic_sweep_helpers import (
    stats,
    resolve_subgradient_method,
    parse_profiles,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TITER = int(os.environ.get("INFOCOM_TITER", "1000"))
MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "20"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
SUBGRADIENT_METHOD, SUBGRADIENT_SOURCE = resolve_subgradient_method()

# Prepend a q=1 (deterministic) reference panel to every grid. At q=1 the
# probabilistic policies reduce exactly to the deterministic robot ErrorVsChannel
# case, which is where the policy separation is clearest -- it anchors what the
# lossy schedules are being compared against. Disable with INFOCOM_Q1_REF=0.
INCLUDE_Q1_REF = os.environ.get("INFOCOM_Q1_REF", "1") != "0"
Q1_REF_NAME = "q=1 (deterministic ref)"

# Robot-loss problem (identical to ErrorVsChannel_steps._build_problem).
M = 20
KM = 2
B = 20
T = 100
GAMMA = 0.9

# (weight mode, filename tag, human label)
ALL_WEIGHT_MODES = [
    (WEIGHTS_DETERMINISTIC, "orig_weights", "original weights"),
    (WEIGHTS_ONES, "weights_1", "w = 1"),
]

# Plot/legend order + MATLAB-style formats.
POLICY_STYLE = [
    ("Random", "ro-", "Random Policy"),
    ("MAF",    "b*-", "MAF Policy"),
    ("MGF",    "kd-", "MGF Policy"),
    ("MEF",    "g^-", "MEF Policy"),
]


def _parse_int_list(s, default):
    if not s:
        return list(default)
    return [int(x) for x in s.split(",") if x.strip()]


def _resolve_weight_modes():
    env = os.environ.get("INFOCOM_WEIGHT_MODES")
    if not env:
        return ALL_WEIGHT_MODES
    requested = [m.strip() for m in env.split(",") if m.strip()]
    chosen = [wm for wm in ALL_WEIGHT_MODES if wm[0] in requested]
    if not chosen:
        raise ValueError(
            f"INFOCOM_WEIGHT_MODES={env!r} has no valid modes; choose from "
            f"{[wm[0] for wm in ALL_WEIGHT_MODES]}")
    return chosen


def _load_robot_penalty():
    """Robot-collected loss.mat penalties (km=2), subsampled every 5 (MATLAB)."""
    loss = sio.loadmat(paths.data_path("loss.mat", probabilistic=False))
    p1 = loss["p1"].flatten()
    p2 = loss["p2"].flatten()
    p = np.zeros((KM, B))
    i = 0
    for j_mat in range(1, 101, 5):
        if i >= B:
            break
        j_py = j_mat - 1
        if j_py < p1.size:
            p[0, i] = p1[j_py]
        if j_py < p2.size:
            p[1, i] = p2[j_py]
        i += 1
    return p


def _run_policies(N, w, p, q):
    """Return {policy: mean_error} for the four plotted policies at channel N."""
    n = np.ones((M, KM))
    c = np.ones((M, KM)) * 2.0
    K = T
    out = {}
    out["MGF"] = MGF1_probabilistic(
        M, N, KM, T, B, K, n, c, w, GAMMA, p, q,
        titer=TITER, subgradient_method=SUBGRADIENT_METHOD,
        seed=SEED, mc_trials=MC_TRIALS, verbose=False,
    )
    out["MAF"] = MAF1_probabilistic(
        M, N, KM, T, B, K, n, c, w, GAMMA, p, q,
        seed=SEED, mc_trials=MC_TRIALS, reliability_aware=False, verbose=False,
    )
    out["MEF"] = MIEF1_probabilistic(
        M, N, KM, T, B, K, n, c, w, GAMMA, p, q,
        seed=SEED, mc_trials=MC_TRIALS, reliability_aware=True, verbose=False,
    )
    out["Random"] = randpolicy_probabilistic(
        M, N, KM, T, B, K, n, c, w, GAMMA, p, q,
        seed=SEED, mc_trials=MC_TRIALS, gated=True, verbose=False,
    )
    return {name: stats(res)[0] for name, res in out.items()}


def _write_csv(channels, profile_series, csv_path):
    """profile_series: {profile: {policy: [vals over channels]}}."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["profile", "N"] + [name for name, _, _ in POLICY_STYLE])
        for profile, series in profile_series.items():
            for i, N in enumerate(channels):
                writer.writerow([profile, N]
                                + [series[name][i] for name, _, _ in POLICY_STYLE])


def _plot_grid(channels, profiles, profile_series, wlabel, png_path):
    """One grid figure: one subplot per q-profile, four policy lines each."""
    n_profiles = len(profiles)
    n_cols = min(3, n_profiles)
    n_rows = (n_profiles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.0 * n_rows),
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
        ax.set_ylabel("Dis. Sum of Errors", fontsize=13)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=10)
    for idx in range(n_profiles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(
        f"Robot-loss ErrorVsChannel (probabilistic, {wlabel}, "
        f"{SUBGRADIENT_METHOD} subgradient)", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_dir = os.path.join(paths.PLOTS_PROBABILISTIC_DIR, "robot")
    os.makedirs(plot_dir, exist_ok=True)

    channels = _parse_int_list(os.environ.get("INFOCOM_CHANNELS"),
                               range(2, 21, 2))
    profiles = parse_profiles(default=list_q_profiles())
    weight_modes = _resolve_weight_modes()
    p = _load_robot_penalty()

    print("Robot-loss probabilistic ErrorVsChannel (grid per weight mode)")
    print(f"  TITER={TITER}  MC_TRIALS={MC_TRIALS}  SEED={SEED}")
    print(f"  subgradient method: {SUBGRADIENT_METHOD}  ({SUBGRADIENT_SOURCE})")
    print(f"  channels: {channels}")
    print(f"  profiles ({len(profiles)}): {profiles}")
    print(f"  weight modes: {[wm[0] for wm in weight_modes]}")

    for wmode, wtag, wlabel in weight_modes:
        w = make_weights("ErrorVsChannel", M, KM, wmode)
        print(f"\n========== weight mode: {wmode} ({wlabel}) ==========")
        panel_profiles = ([Q1_REF_NAME] + profiles if INCLUDE_Q1_REF
                          else list(profiles))
        profile_series = {}
        for profile in panel_profiles:
            print(f"  --- profile: {profile} ---")
            series = {name: [] for name, _, _ in POLICY_STYLE}
            for N in channels:
                # q keyed by (M, km); fixed seed -> reproducible. The reference
                # panel uses q=1 (perfect channel) -> reduces to deterministic.
                if profile == Q1_REF_NAME:
                    q = np.ones((M, KM))
                else:
                    q, _ = make_q_profile(profile, M, KM, seed=SEED)
                means = _run_policies(N, w, p, q)
                for name in series:
                    series[name].append(means[name])
                print(f"    N={N:>3d}: " + ", ".join(
                    f"{name}={means[name]:.3g}" for name, _, _ in POLICY_STYLE))
            profile_series[profile] = series

        png_path = os.path.join(
            plot_dir, f"ErrorVsChannel_robot_probabilistic_{wtag}.png")
        csv_path = paths.data_path(
            f"ErrorVsChannel_robot_probabilistic_{wtag}_data.csv",
            probabilistic=True)
        _plot_grid(channels, panel_profiles, profile_series, wlabel, png_path)
        _write_csv(channels, profile_series, csv_path)
        print(f"  saved grid {png_path}")
        print(f"  saved data {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
