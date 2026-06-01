"""
run_all.py - Run every Python experiment (deterministic + probabilistic) and
produce side-by-side comparisons against the original MATLAB .fig data.

Sets a fixed numpy seed before each script so the random-policy output is
reproducible run-to-run on Python.

Usage:
    python src/execute/run_all.py
    INFOCOM_TITER=10000 python src/execute/run_all.py        # MATLAB default
    INFOCOM_TITER=100   python src/execute/run_all.py        # fast smoke test
    INFOCOM_SEED=42 INFOCOM_TITER=2000 python src/execute/run_all.py

    # Restrict to a subset:
    python src/execute/run_all.py --deterministic-only
    python src/execute/run_all.py --probabilistic-only
    python src/execute/run_all.py --skip-precheck

Each experiment writes:
    - data CSV/MAT/NPZ outputs to python_code/data/{deterministic,probabilistic}/
    - PNG plots to python_code/plots/{deterministic,probabilistic}/
"""
import argparse
import os
import time

# Headless plotting by default so no GUI window pops up during long runs.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import numpy as np                      # noqa: E402

from _bootstrap import paths            # noqa: E402,F401  (bootstraps sys.path)


DETERMINISTIC_UTIL = ['lossread']
DETERMINISTIC_SWEEPS = [
    'ErrorVsChannel',
    'ErrorVsChannelsmodel',
    'ErrorVsSources',
    'ErrorVsSourcesM',
    'ErrorVsTasks',
]
DETERMINISTIC_VARIANTS = ['ErrorVsSources_variants']

PROBABILISTIC_SWEEPS = [
    'ErrorVsChannel_probabilistic',
    'ErrorVsChannelsmodel_probabilistic',
    'ErrorVsSources_probabilistic',
    'ErrorVsTasks_probabilistic',
]
# MGF-vs-iterations + q=1 sanity check (probabilistic-flavored outputs but uses
# the deterministic settings; reduces to MGF1 at q=1).
PROBABILISTIC_EXTRA = [
    'ErrorVsIterations_probabilistic',
    'CompareSubgradient_probabilistic',
    'CompareSubgradientIterations_probabilistic',
]
# Parameter dump runs last (documents every experiment) regardless of flavor.
PARAMETER_EXPORT = ['ExportExperimentParameters']

PRECHECK = ['PrecheckSubgradientMethods']


def _import_and_run(name, seed, label):
    """Import a script in src/execute/ by name and call its main()."""
    print(f"\n---- {label} ----")
    np.random.seed(seed)
    t0 = time.time()
    mod = __import__(name)
    mod.main()
    print(f"---- {label} done in {time.time() - t0:.0f}s ----")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deterministic-only", action="store_true",
                        help="Skip probabilistic sweeps and CompareSubgradient.")
    parser.add_argument("--probabilistic-only", action="store_true",
                        help="Skip deterministic sweeps (still runs lossread).")
    parser.add_argument("--skip-precheck", action="store_true",
                        help="Don't run PrecheckSubgradientMethods first.")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Don't run compare_with_matlab at the end.")
    args = parser.parse_args()

    seed = int(os.environ.get("INFOCOM_SEED", "0"))
    titer = int(os.environ.get("INFOCOM_TITER", "10000"))
    print(f"=== run_all  (TITER={titer}, SEED={seed}) ===\n")
    print(f"  data outputs -> {paths.DATA_DIR}/[deterministic|probabilistic]/")
    print(f"  plot outputs -> {paths.PLOTS_DIR}/[deterministic|probabilistic]/")

    run_det = not args.probabilistic_only
    run_prob = not args.deterministic_only

    if not args.skip_precheck:
        # Precheck picks the recommended subgradient method used by both the
        # deterministic variants script and the probabilistic sweeps.
        for name in PRECHECK:
            _import_and_run(name, seed, name)

    if run_det:
        for name in DETERMINISTIC_UTIL + DETERMINISTIC_SWEEPS + DETERMINISTIC_VARIANTS:
            _import_and_run(name, seed, name)

        if not args.skip_compare:
            print("\n=== running comparison ===")
            np.random.seed(seed)
            import compare_with_matlab
            compare_with_matlab.main()

    if run_prob:
        for name in PROBABILISTIC_SWEEPS + PROBABILISTIC_EXTRA:
            _import_and_run(name, seed, name)

    # Always export the parameter summary for every experiment at the end.
    for name in PARAMETER_EXPORT:
        _import_and_run(name, seed, name)


if __name__ == "__main__":
    main()
