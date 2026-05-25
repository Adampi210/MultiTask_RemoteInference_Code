"""
run_all.py - Run every Python plot script (matches all .m plot scripts) and
then produce a side-by-side comparison against the original MATLAB .fig data.

Sets a fixed numpy seed before each script so the random-policy output is
reproducible run-to-run on Python.

Usage:
    python run_all.py
    INFOCOM_TITER=10000 python run_all.py        # default
    INFOCOM_TITER=100   python run_all.py        # fast smoke test
    INFOCOM_SEED=42 INFOCOM_TITER=2000 python run_all.py

The scripts also generate the per-figure CSVs (*_data.csv) and PNGs.
"""
import os
import sys
import time

# Run headless by default so the GUI window doesn't try to pop up.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import numpy as np                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PLOT_SCRIPTS = [
    'ErrorVsChannel',
    'ErrorVsChannelsmodel',
    'ErrorVsSources',
    'ErrorVsSourcesM',
    'ErrorVsTasks',
]

UTIL_SCRIPTS = ['lossread']


def main():
    seed = int(os.environ.get("INFOCOM_SEED", "0"))
    titer = int(os.environ.get("INFOCOM_TITER", "10000"))
    print(f"=== run_all  (TITER={titer}, SEED={seed}) ===\n")

    for name in UTIL_SCRIPTS + PLOT_SCRIPTS:
        print(f"\n---- {name} ----")
        np.random.seed(seed)
        t0 = time.time()
        mod = __import__(name)
        mod.main()
        print(f"---- {name} done in {time.time() - t0:.0f}s ----")

    print("\n=== running comparison ===")
    import compare_with_matlab
    compare_with_matlab.main()


if __name__ == "__main__":
    main()
