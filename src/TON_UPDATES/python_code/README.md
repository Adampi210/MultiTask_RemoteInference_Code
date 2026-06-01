# INFOCOM / ToN Python Port

Python implementation of the AoI-aware scheduling experiments from the
INFOCOM / ToN paper. The deterministic flavor is a 1:1 port of the original
MATLAB code; the probabilistic flavor extends the model with unreliable
links (Bernoulli(q_{m,j}) deliveries).

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the conceptual model and
file-by-file rundown, and [PROBABILISTIC_IMPLEMENTATION_NOTES.md](PROBABILISTIC_IMPLEMENTATION_NOTES.md)
for the differences between the deterministic and probabilistic variants.

**New to running these experiments? Start with
[EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)** — a copy-paste guide to running
every experiment and changing its hyper-parameters (weight modes, the
MGF-vs-iterations experiment, and the parameter-export script are all covered
there).

---

## Directory Layout

```
INFOCOM Code/                 <- repo root
├── matlab_code/              <- MATLAB sources + .fig/.jpg references + raw inference CSVs
│   ├── *.m                   (Episode1.m, MGF1.m, subgradientiter1.m, ...)
│   ├── *.fig, *.jpg          (ErrorVsChannel1.fig, ErrorVsTasks.fig, MLexp.fig, ...)
│   ├── loss.mat, multipliers.mat
│   └── smooth_*.csv          (raw segmentation / detection inference-error data)
│
└── python_code/              <- Python port (this README)
    ├── README.md             <- you are here
    ├── PROJECT_OVERVIEW.md   <- conceptual model + per-file documentation
    ├── PROBABILISTIC_IMPLEMENTATION_NOTES.md
    ├── SUBGRADIENT_METHODS.md
    ├── TODO.md
    │
    ├── data/                 <- CSV / MAT / NPZ / JSON outputs
    │   ├── deterministic/
    │   └── probabilistic/
    │
    ├── plots/                <- PNG figures
    │   ├── deterministic/
    │   └── probabilistic/
    │
    └── src/
        ├── main/             <- library code (importable modules)
        │   ├── paths.py      <- centralized output-path resolution
        │   ├── valuefunction1.py, Episode1.py, subgradientiter1.py
        │   ├── MGF1.py, MAF1.py, MIEF1.py, randpolicy.py, lowerbound.py
        │   ├── *_probabilistic.py <- probabilistic variants of the above
        │   ├── *_variants.py <- method-selectable deterministic variants
        │   ├── greedy_scheduler.py, experiment_utils.py
        │   ├── optimizer_updates.py, probability_profiles_probabilistic.py
        │   ├── dual_oracle_*.py, cuttingplaneiter_*.py, _cutting_plane_core.py
        │   ├── _probabilistic_sweep_helpers.py  <- shared driver for prob. sweeps
        │   └── fig_reader.py
    │
    ├── execute/               <- experiment runner scripts (each has main())
    │   ├── _bootstrap.py      <- adds src/main to sys.path; exposes paths
    │   ├── run_all.py         <- master runner: every experiment + comparison
    │   ├── lossread.py        <- ingests raw CSVs -> data/deterministic/loss.mat
    │   ├── compare_with_matlab.py
    │   ├── ErrorVsChannel.py, ErrorVsChannelsmodel.py
    │   ├── ErrorVsSources.py, ErrorVsSourcesM.py, ErrorVsTasks.py
    │   ├── ErrorVs*_probabilistic.py
    │   ├── ErrorVsSources_variants.py
    │   ├── PrecheckSubgradientMethods.py
    │   └── CompareSubgradient_probabilistic.py
    │
    └── tests/                 <- pytest tests + standalone sanity checks
        ├── _bootstrap.py
        ├── test_mief.py, test_probabilistic.py
        └── checkmaf.py, checkmgf.py, checkmief.py, checkrand.py
```

### Where each kind of file lands

| Output kind         | Deterministic                                | Probabilistic                                  |
| ------------------- | -------------------------------------------- | ---------------------------------------------- |
| CSV / NPZ / MAT     | `data/deterministic/`                        | `data/probabilistic/`                          |
| PNG plots           | `plots/deterministic/`                       | `plots/probabilistic/`                         |
| Recommendation JSON | `data/deterministic/recommended_subgradient_methods.json` (shared) ||

`src/main/paths.py` is the single source of truth for these locations; every
runner script imports `paths.data_path()` / `paths.plot_path()` instead of
hardcoding `os.path.join(here, ...)`. MATLAB references (raw inference CSVs,
`.fig` files) are resolved via `paths.matlab_path()` -> `../matlab_code/`.

---

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `matplotlib`

Install with `pip install numpy scipy matplotlib`.

---

## Running everything (the easy path)

```bash
# From python_code/

# Full reproduction with MATLAB defaults (slow - 10000 subgradient iters/script)
python src/execute/run_all.py

# Fast smoke test
INFOCOM_TITER=100 INFOCOM_SEED=42 python src/execute/run_all.py

# Only deterministic / only probabilistic
python src/execute/run_all.py --deterministic-only
python src/execute/run_all.py --probabilistic-only

# Skip the precheck (use the existing recommendation JSON)
python src/execute/run_all.py --skip-precheck
```

`run_all.py` runs, in order:

1. **PrecheckSubgradientMethods** -> writes
   `data/deterministic/recommended_subgradient_methods.json`. This file is
   read by `ErrorVsSources_variants.py` and the probabilistic sweeps to
   choose their subgradient method.
2. **Deterministic util**: `lossread` (reads raw CSVs from `../matlab_code/`,
   produces `data/deterministic/loss.mat` + lossread plots).
3. **Deterministic sweeps**: `ErrorVsChannel`, `ErrorVsChannelsmodel`,
   `ErrorVsSources`, `ErrorVsSourcesM`, `ErrorVsTasks`,
   `ErrorVsSources_variants`.
4. **compare_with_matlab**: builds side-by-side `*_compare.png` comparing
   Python output against MATLAB `.fig` references in `../matlab_code/`.
5. **Probabilistic sweeps**: `ErrorVsChannel_probabilistic`,
   `ErrorVsChannelsmodel_probabilistic`, `ErrorVsSources_probabilistic`,
   `ErrorVsTasks_probabilistic`, `CompareSubgradient_probabilistic`.

---

## Running an individual experiment

Each script in `src/execute/` has its own `main()` and reads/writes the
standard data and plot directories. Run any one in isolation:

```bash
# Deterministic
python src/execute/ErrorVsChannel.py             # needs data/deterministic/loss.mat
python src/execute/ErrorVsChannelsmodel.py
python src/execute/ErrorVsSources.py
python src/execute/ErrorVsSourcesM.py
python src/execute/ErrorVsTasks.py
python src/execute/ErrorVsSources_variants.py    # method-selectable MGF

# Probabilistic (each emits BOTH a deterministic-weights and a _weights_1 variant)
python src/execute/ErrorVsChannel_probabilistic.py
python src/execute/ErrorVsChannelsmodel_probabilistic.py
python src/execute/ErrorVsSources_probabilistic.py
python src/execute/ErrorVsTasks_probabilistic.py
python src/execute/CompareSubgradient_probabilistic.py
python src/execute/ErrorVsIterations_probabilistic.py  # MGF vs #iterations (q=1) + MATLAB sanity

# Auxiliary
python src/execute/lossread.py                   # rebuild loss.mat
python src/execute/PrecheckSubgradientMethods.py # refresh recommended_subgradient_methods.json
python src/execute/compare_with_matlab.py        # rebuild side-by-side compare PNGs
python src/execute/ExportExperimentParameters.py # dump every experiment's parameters
```

> Each probabilistic sweep is run in two **weight modes**: the heterogeneous
> deterministic/paper weights (bare stem) and a uniform `w=1` variant
> (`_weights_1` suffix). Restrict to one with
> `INFOCOM_WEIGHT_MODES=deterministic` or `INFOCOM_WEIGHT_MODES=ones`. See
> [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md) for full details.

Each script writes:

- `data/<flavor>/<stem>_data.csv` (raw numerical results)
- (probabilistic only) `data/probabilistic/<stem>_summary.csv` and
  `<stem>_q_profiles.npz`
- `plots/<flavor>/<stem>.png`

`<stem>` is the script's filename without `.py`; `<flavor>` is
`deterministic` for `ErrorVs*.py` and `probabilistic` for `ErrorVs*_probabilistic.py`.

---

## Subgradient methods

The Lagrange multipliers are learned by projected ascent on the dual. Two
families of update rules exist:

- **`episode1_mstep`** (default for probabilistic MGF): MATLAB-equivalent
  closed-form M-step projected ascent (the same logic as `Episode1.m`). When
  `q = 1` and `n = 1`, the probabilistic MGF reduces to the deterministic
  MGF exactly (byte-identical to `MGF1.py`). This is the preferred default.
- **First-order alternatives**: `harmonic` (step `beta/j`), `sqrt`,
  `normalized_global`, `normalized_blocks`, `adagrad`, `rmsprop`, `adam`,
  `deflected_sqrt`. These use a single projected ascent step per outer
  iteration; they converge to the same dual saddle point but at different
  rates and slightly different fixed points.

`PrecheckSubgradientMethods.py` ranks these methods on small scenarios and
writes `data/deterministic/recommended_subgradient_methods.json`. The
probabilistic sweep scripts read this file to pick their default method;
override with `INFOCOM_SUBGRADIENT_METHOD=<name>`.

---

## Configuration via environment variables

All scripts read configuration from environment variables (defaults shown):

### Deterministic

| Variable                     | Default | Used by                              |
|------------------------------|---------|--------------------------------------|
| `INFOCOM_TITER`              | `10000` | every `ErrorVs*.py`                  |
| `INFOCOM_SEED`               | `0`     | every script (randpolicy reproducibility) |
| `INFOCOM_SOURCES`            | `2..20` step 2 | `ErrorVsSources_variants.py` |
| `INFOCOM_SUBGRADIENT_METHOD` | (none)  | `ErrorVsSources_variants.py`         |

### Probabilistic

| Variable                     | Default | Used by                              |
|------------------------------|---------|--------------------------------------|
| `INFOCOM_TITER`              | `1000`  | every `ErrorVs*_probabilistic.py`    |
| `INFOCOM_MC_TRIALS`          | `10`    | every probabilistic script           |
| `INFOCOM_SEED`               | `0`     | every probabilistic script           |
| `INFOCOM_PROFILES`           | all 12  | every probabilistic script (`uniform_wide,bimodal_extreme,...`) |
| `INFOCOM_WEIGHT_MODES`       | `deterministic,ones` | which weight modes a probabilistic sweep runs |
| `INFOCOM_EXTRA_POLICIES`     | `0`     | adds MAF_relaware + MIEF_pure rows   |
| `INFOCOM_SUBGRADIENT_METHOD` | from JSON | every probabilistic script         |
| `INFOCOM_SOURCES`            | `2..20` step 2 | `ErrorVsSources_probabilistic.py` |
| `INFOCOM_CHANNELS`           | `2..20` step 2 | `ErrorVs*Channel*_probabilistic.py` |
| `INFOCOM_TASKS`              | `3..15` step 3 | `ErrorVsTasks_probabilistic.py` |

### MGF-vs-iterations experiment (`ErrorVsIterations_probabilistic.py`)

| Variable                     | Default | Used by                              |
|------------------------------|---------|--------------------------------------|
| `INFOCOM_ITERATIONS`         | `1,2,4,8,16,32` | subgradient-iteration counts swept |
| `INFOCOM_ITER_REFERENCE`     | `10000` | converged reference MGF1 iteration count |
| `INFOCOM_ITER_SETTINGS`      | all 3   | subset of `ErrorVsChannel,ErrorVsChannelsmodel,ErrorVsSources` |

### Precheck

| Variable                                  | Default |
|-------------------------------------------|---------|
| `INFOCOM_PRECHECK_TITER`                  | `100`   |
| `INFOCOM_PRECHECK_MC_TRIALS`              | `5`     |
| `INFOCOM_PRECHECK_SEED`                   | `0`     |
| `INFOCOM_PRECHECK_FAST`                   | `1`     |
| `INFOCOM_PRECHECK_INCLUDE_CUTTING_PLANES` | `0`     |
| `INFOCOM_PRECHECK_METHODS`                | (defaults to all first-order methods) |

`MPLBACKEND=Agg` is set automatically by `run_all.py` for headless runs.

---

## How a runner script finds the library and the output directories

Every script in `src/execute/` starts with:

```python
from _bootstrap import paths
```

`src/execute/_bootstrap.py`:
1. Adds `src/main/` to `sys.path` so `from MGF1 import MGF1` works.
2. Re-exports the `paths` module from `src/main/paths.py`.
3. Calls `paths.ensure_dirs()` to create `data/{deterministic,probabilistic}/`
   and `plots/{deterministic,probabilistic}/` if they do not already exist.

Output paths inside the script then look like:

```python
csv_out = paths.data_path('ErrorVsChannel_data.csv', probabilistic=False)
png_out = paths.plot_path('ErrorVsChannel.png',     probabilistic=False)
```

There is also `src/tests/_bootstrap.py` for the same purpose, so tests can
import library modules without a setup file.

---

## Tests and sanity checks

```bash
# Pytest tests
pytest src/tests/test_mief.py -v
pytest src/tests/test_probabilistic.py -v

# Run a single test file as a script
python src/tests/test_mief.py
python src/tests/test_probabilistic.py

# Standalone smoke checks per policy on a tiny problem
python src/tests/checkmaf.py
python src/tests/checkmgf.py
python src/tests/checkmief.py
python src/tests/checkrand.py
```

---

## Common workflows

### Refresh the deterministic plots end-to-end

```bash
python src/execute/lossread.py
python src/execute/run_all.py --deterministic-only
```

### Refresh only the probabilistic results

```bash
# Prerequisite: data/deterministic/recommended_subgradient_methods.json exists.
# If it doesn't, run PrecheckSubgradientMethods first.
python src/execute/PrecheckSubgradientMethods.py
python src/execute/run_all.py --probabilistic-only --skip-precheck
```

### Quick smoke test (under a minute)

```bash
INFOCOM_TITER=50 INFOCOM_MC_TRIALS=2 INFOCOM_PROFILES=uniform_wide,bimodal_extreme \
    INFOCOM_SOURCES=4,8 INFOCOM_CHANNELS=4,8 INFOCOM_TASKS=3,6 \
    python src/execute/run_all.py
```

### Use a specific subgradient method

```bash
INFOCOM_SUBGRADIENT_METHOD=adam python src/execute/ErrorVsSources_probabilistic.py
```

---

## Adding a new experiment script

1. Create `src/execute/<my_experiment>.py`.
2. Start with the standard bootstrap and imports:

   ```python
   import numpy as np
   from _bootstrap import paths       # bootstraps sys.path

   from MGF1 import MGF1              # or whatever you need from src/main/
   from MAF1 import MAF1
   ```

3. Save data via `paths.data_path('<stem>_data.csv', probabilistic=<bool>)`
   and plots via `paths.plot_path('<stem>.png', probabilistic=<bool>)`.
4. Wrap the body in a `main()` and add a
   `if __name__ == "__main__": main()` guard so `run_all.py` can pick it up
   by name.
5. (Optional) add the stem to one of the lists in `src/execute/run_all.py`
   so it gets exercised by the master runner.

---

## Notes

- The deterministic sweep scripts (`ErrorVsChannel.py` etc.) are intentional
  1:1 ports of the MATLAB code. See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
  for MATLAB-fidelity quirks (e.g. `ErrorVsSourcesM` reproduces a leftover-`j`
  bug from the original `.m`).
- Probabilistic outputs are random (Bernoulli deliveries); average across MC
  trials (`INFOCOM_MC_TRIALS`) before drawing conclusions. Production runs
  typically use 30+ trials.
- `compare_with_matlab.py` requires the MATLAB `.fig` files in the repo
  root (`INFOCOM Code/`); missing references are skipped with a warning.
- The `recommended_subgradient_methods.json` shipped in
  `data/deterministic/` is a default fallback. Re-run
  `PrecheckSubgradientMethods.py` with the full TITER for an authoritative
  recommendation.
