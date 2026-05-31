"""
paths.py - centralized filesystem layout for the reorganized python_code tree.

Project layout (with PROJECT_ROOT = python_code/ and REPO_ROOT one level up):

    INFOCOM Code/                <- REPO_ROOT
      matlab_code/               <- MATLAB sources + .fig / .jpg references + raw CSVs
        loss.mat, multipliers.mat
        smooth_*.csv (inference-error data)
        *.m, *.fig, *.jpg
      python_code/               <- PROJECT_ROOT
        data/
          deterministic/         <- CSV / MAT / JSON output for deterministic runs
          probabilistic/         <- CSV / MAT / NPZ output for probabilistic runs
        plots/
          deterministic/         <- PNG output for deterministic runs
          probabilistic/         <- PNG output for probabilistic runs
        src/
          main/                  <- library code (this file lives here)
          execute/               <- experiment runner scripts
          tests/                 <- unit / sanity tests

Every executable in src/execute/ pulls its output paths from this module so
plots and data land in the right subdirectory without each script duplicating
the logic. MATLAB-side references (raw inference CSVs, .fig files for the
comparison script) are resolved via MATLAB_DIR / matlab_path().
"""
import os


# This file lives at python_code/src/main/paths.py
# - dirname(__file__)              -> python_code/src/main
# - dirname(dirname(__file__))     -> python_code/src
# - dirname(dirname(dirname(...))) -> python_code/
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MAIN_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)            # python_code/
REPO_ROOT = os.path.dirname(PROJECT_ROOT)          # INFOCOM Code/  (one level up)

# MATLAB-side directory (raw inference CSVs, .fig / .jpg references, loss.mat).
MATLAB_DIR = os.path.join(REPO_ROOT, "matlab_code")

EXECUTE_DIR = os.path.join(SRC_DIR, "execute")
TESTS_DIR = os.path.join(SRC_DIR, "tests")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_DETERMINISTIC_DIR = os.path.join(DATA_DIR, "deterministic")
DATA_PROBABILISTIC_DIR = os.path.join(DATA_DIR, "probabilistic")

PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
PLOTS_DETERMINISTIC_DIR = os.path.join(PLOTS_DIR, "deterministic")
PLOTS_PROBABILISTIC_DIR = os.path.join(PLOTS_DIR, "probabilistic")


def ensure_dirs():
    """Create the standard output directories if they don't already exist."""
    for d in (
        DATA_DETERMINISTIC_DIR,
        DATA_PROBABILISTIC_DIR,
        PLOTS_DETERMINISTIC_DIR,
        PLOTS_PROBABILISTIC_DIR,
    ):
        os.makedirs(d, exist_ok=True)


def data_path(name, probabilistic=False):
    """Resolve a data file (CSV / MAT / NPZ / JSON) under data/<flavor>/."""
    base = DATA_PROBABILISTIC_DIR if probabilistic else DATA_DETERMINISTIC_DIR
    return os.path.join(base, name)


def plot_path(name, probabilistic=False):
    """Resolve a plot file (PNG) under plots/<flavor>/."""
    base = PLOTS_PROBABILISTIC_DIR if probabilistic else PLOTS_DETERMINISTIC_DIR
    return os.path.join(base, name)


def matlab_path(name):
    """Resolve a file under matlab_code/ (raw CSVs, .fig references, loss.mat)."""
    return os.path.join(MATLAB_DIR, name)
