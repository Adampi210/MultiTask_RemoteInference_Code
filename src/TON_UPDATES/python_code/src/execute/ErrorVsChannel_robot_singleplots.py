"""
ErrorVsChannel_robot_singleplots.py - single-axis MATLAB-style figures for two
chosen robot-loss q-profiles.

Reads the per-profile data produced by ErrorVsChannel_robot_probabilistic.py
(data/probabilistic/ErrorVsChannel_robot_probabilistic_<wtag>_data.csv) and emits
one standalone figure per (profile x weight-mode), styled exactly like
MatlabStylePlots_probabilistic.py (log y-axis, MATLAB markers, large fonts, no
title):

    Random -> red circle  ('ro-')  "Random Policy"
    MAF    -> blue star    ('b*-')  "MAF Policy"
    MGF    -> black diamond('kd-')  "MGF Policy"
    MEF    -> green tri.    ('g^-')  "MEF Policy"

Profiles: uniform_very_wide, bimodal_q1_vs_lossy_30_70.
Weight modes: orig_weights, weights_1.  -> 4 figures.

This is a pure re-plot of existing data (no policy recomputation). Run
ErrorVsChannel_robot_probabilistic.py first if the CSVs are missing.

Writes:
    plots/probabilistic/robot/ErrorVsChannel_robot_<profile>_<wtag>.png
"""
import os
import csv
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths


PROFILES = ["uniform_very_wide", "bimodal_q1_vs_lossy_30_70"]
WEIGHT_TAGS = ["orig_weights", "weights_1"]

# Plot/legend order + MATLAB-style formats (matches MatlabStylePlots).
POLICY_STYLE = [
    ("Random", "ro-", "Random Policy"),
    ("MAF",    "b*-", "MAF Policy"),
    ("MGF",    "kd-", "MGF Policy"),
    ("MEF",    "g^-", "MEF Policy"),
]


def _read_profile(csv_path, profile):
    """Return (channels, {policy: [vals]}) for one profile from a robot CSV."""
    channels = []
    series = {name: [] for name, _, _ in POLICY_STYLE}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["profile"] != profile:
                continue
            channels.append(int(row["N"]))
            for name, _, _ in POLICY_STYLE:
                series[name].append(float(row[name]))
    if not channels:
        raise ValueError(f"profile {profile!r} not found in {csv_path}")
    return channels, series


def _plot_one(channels, series, png_path):
    """One MATLAB-styled figure: four lines, log y, LineWidth 2, MS 10, no title."""
    fig, ax = plt.subplots(figsize=(7, 5))
    legend_labels = []
    for name, fmt, label in POLICY_STYLE:
        ax.plot(channels, series[name], fmt, linewidth=2, markersize=10)
        legend_labels.append(label)
    ax.set_xlabel(r"Number of Channels ($N$)", fontsize=18)
    ax.set_ylabel("Dis. Sum of Errors", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_yscale("log")
    ax.legend(legend_labels, loc="best", fontsize=12)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_dir = os.path.join(paths.PLOTS_PROBABILISTIC_DIR, "robot")
    os.makedirs(plot_dir, exist_ok=True)

    n = 0
    for wtag in WEIGHT_TAGS:
        csv_path = paths.data_path(
            f"ErrorVsChannel_robot_probabilistic_{wtag}_data.csv",
            probabilistic=True)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found -- run ErrorVsChannel_robot_probabilistic.py first")
        for profile in PROFILES:
            channels, series = _read_profile(csv_path, profile)
            png_path = os.path.join(
                plot_dir, f"ErrorVsChannel_robot_{profile}_{wtag}.png")
            _plot_one(channels, series, png_path)
            n += 1
            print(f"saved {png_path}")
    print(f"\nDone. Wrote {n} figures to {plot_dir}")


if __name__ == "__main__":
    main()
