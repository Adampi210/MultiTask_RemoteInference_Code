"""
robot_data.py - loads the robot-car inference-error penalties used by the
MATLAB ``matlab_code/robot_code`` experiments (main.m, mainweight.m,
ErrorFunction.m) and exposes them in the heterogeneous-``km`` layout the Python
robot policies expect.

The MATLAB robot setting (main.m) is:

    M  = 4 sources (robot cars)
    km = [2, 1, 1, 1]            (source 1 runs two tasks, the rest one)
    B  = 40                      (AoI bound; AoI in 1..B in MATLAB, 0..B-1 here)

with one *empirical* 1-IoU(AoI) penalty curve per (source, task) pair, read from
the per-car ``detection_results_robot_<k>_error_function.csv`` files. The pairing
reproduces main.m / mainweight.m exactly:

    p(1,1) = robot_9     ->  p[0, 0]
    p(1,2) = robot_1     ->  p[0, 1]
    p(2,1) = robot_8     ->  p[1, 0]
    p(3,1) = robot_2     ->  p[2, 0]
    p(4,1) = robot_4     ->  p[3, 0]

Each CSV has columns ``AoI,Error`` for AoI = 0..40 (41 rows). MATLAB builds the
penalty from ``Error(2:end)`` (AoI = 1..40, i.e. B=40 values); the AoI=0 row
(Error=0) is dropped. Our 0-indexed convention therefore stores

    p[m, j, d] = Error(AoI = d + 1),   d = 0 .. B-1

so that a freshly delivered pair (reset -> Delta = 0) incurs the AoI=1 penalty,
matching MATLAB's reset-to-Delta=1 convention.

The ErrorFunction.m figure (InferenceErrorRobot) instead uses the *full* curve
``Error(1:end)`` (AoI = 0..40) for the five cars, so :func:`load_inference_curves`
returns those untruncated.

Heterogeneous km is represented by padding to ``km_max = max(km) = 2`` and a
boolean ``valid`` mask; padded (invalid) pairs carry zero penalty and zero weight
so they can never be scheduled and never contribute to the objective.
"""
import csv

import numpy as np

from paths import matlab_path


# Folder (under matlab_code/) holding the robot CSVs + MATLAB sources.
ROBOT_SUBDIR = "robot_code"

# MATLAB main.m / mainweight.m (source, task) -> robot-car CSV index.
#   key   = (m, j) 0-indexed
#   value = robot index k in detection_results_robot_<k>_error_function.csv
_PAIR_TO_ROBOT = {
    (0, 0): 9,
    (0, 1): 1,
    (1, 0): 8,
    (2, 0): 2,
    (3, 0): 4,
}

# ErrorFunction.m legend order (robotCar1..5) -> robot-car CSV index.
_INFERENCE_CARS = [
    ("robotCar1", 9),
    ("robotCar2", 1),
    ("robotCar3", 8),
    ("robotCar4", 2),
    ("robotCar5", 4),
]

M = 4
KM_VEC = (2, 1, 1, 1)
KM_MAX = max(KM_VEC)
B = 40


def _csv_path(robot_idx):
    return matlab_path(f"{ROBOT_SUBDIR}/"
                       f"detection_results_robot_{robot_idx}_error_function.csv")


def _read_error_column(robot_idx):
    """Return the full Error column (AoI = 0..40 -> 41 values) for one car."""
    rows = []
    with open(_csv_path(robot_idx), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(float(row["AoI"])), float(row["Error"])))
    rows.sort(key=lambda r: r[0])
    return np.array([err for _, err in rows], dtype=float)


def load_robot_problem():
    """Build the robot penalty tensor in heterogeneous-km layout.

    Returns a dict with:
        M       : int (= 4)
        km_vec  : tuple, tasks per source (= (2, 1, 1, 1))
        km_max  : int (= 2)
        B       : int (= 40)
        p       : (M, km_max, B) per-(source, task) penalty; p[m, j, d] is the
                  1-IoU error at AoI = d+1. Invalid (padded) pairs are 0.
        valid   : (M, km_max) bool mask of real (schedulable) pairs.
        n_pairs : int, number of valid pairs (= sum(km_vec) = 5).
    """
    p = np.zeros((M, KM_MAX, B), dtype=float)
    valid = np.zeros((M, KM_MAX), dtype=bool)

    # Cache each car's truncated penalty (Error(2:end) -> AoI 1..40).
    cache = {}
    for (m, j), robot_idx in _PAIR_TO_ROBOT.items():
        if robot_idx not in cache:
            full = _read_error_column(robot_idx)        # AoI 0..40 (41 values)
            cache[robot_idx] = full[1:1 + B]            # AoI 1..40 (B values)
        pen = cache[robot_idx]
        if pen.size < B:
            raise ValueError(
                f"robot {robot_idx} CSV has {pen.size + 1} AoI rows; need "
                f"at least {B + 1} (AoI 0..{B}).")
        p[m, j, :] = pen
        valid[m, j] = True

    return dict(M=M, km_vec=KM_VEC, km_max=KM_MAX, B=B,
                p=p, valid=valid, n_pairs=int(sum(KM_VEC)))


def load_inference_curves():
    """Return (aoi, curves) for the InferenceErrorRobot (ErrorFunction.m) plot.

    aoi    : (41,) array, AoI = 0..40.
    curves : list of (label, error_curve) tuples in ErrorFunction.m legend order
             (robotCar1..robotCar5), each error_curve the full Error(1:end).
    """
    curves = []
    aoi = None
    for label, robot_idx in _INFERENCE_CARS:
        err = _read_error_column(robot_idx)
        if aoi is None:
            aoi = np.arange(err.size)
        curves.append((label, err))
    return aoi, curves


if __name__ == "__main__":
    prob = load_robot_problem()
    print(f"M={prob['M']}  km_vec={prob['km_vec']}  km_max={prob['km_max']}  "
          f"B={prob['B']}  n_pairs={prob['n_pairs']}")
    print("valid mask:\n", prob["valid"].astype(int))
    for m in range(prob["M"]):
        for j in range(prob["km_max"]):
            if prob["valid"][m, j]:
                pen = prob["p"][m, j]
                print(f"  p[{m},{j}]: AoI1={pen[0]:.4f} ... AoI{B}={pen[-1]:.4f}")
    aoi, curves = load_inference_curves()
    print(f"inference curves: {[lbl for lbl, _ in curves]}  (AoI 0..{aoi[-1]})")
