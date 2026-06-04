"""
InferenceErrorRobot.py - reimplementation of the MATLAB robot figure
``matlab_code/robot_code/ErrorFunction.m`` (InferenceErrorRobot).

Plots the empirical 1-IoU inference error vs AoI for the five robot cars used
as penalties in the robot scheduling experiments. This is a deterministic
data plot (no scheduling, no q), kept here so the robot figure set is complete.

Curves and legend order match ErrorFunction.m exactly:
    robotCar1 = robot_9   'bo--'
    robotCar2 = robot_1   'rx--'
    robotCar3 = robot_8   'gd--'
    robotCar4 = robot_2   'k-'
    robotCar5 = robot_4   'kx-'

Reads the per-car detection_results_robot_*.csv files (full Error(1:end),
AoI = 0..40) via robot_data.load_inference_curves().

Writes:
    plots/deterministic/InferenceErrorRobot.png
    data/deterministic/InferenceErrorRobot_data.csv
"""
import os
import csv

import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths  # noqa: F401  (bootstraps sys.path -> src/main)

from robot_data import load_inference_curves


# ErrorFunction.m line styles, in legend order.
STYLES = ["bo--", "rx--", "gd--", "k-", "kx-"]


def main():
    aoi, curves = load_inference_curves()

    fig, ax = plt.subplots(figsize=(7, 5))
    for (label, err), fmt in zip(curves, STYLES):
        ax.plot(aoi, err, fmt, linewidth=2, markersize=7, label=label)
    ax.set_xlabel("AoI", fontsize=18)
    ax.set_ylabel("1-IoU", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend([label for label, _ in curves], loc="best", fontsize=12)
    plt.tight_layout()

    png_path = paths.plot_path("InferenceErrorRobot.png", probabilistic=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = paths.data_path("InferenceErrorRobot_data.csv",
                               probabilistic=False)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AoI"] + [label for label, _ in curves])
        for i, a in enumerate(aoi):
            writer.writerow([int(a)] + [f"{err[i]:.10f}" for _, err in curves])

    print(f"saved {png_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
