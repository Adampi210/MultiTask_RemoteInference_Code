# How to Run the Experiments

A practical, copy-paste guide to running every experiment in `python_code/` and
changing its hyper-parameters. For the conceptual model see
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md); for deterministic-vs-probabilistic
modelling details see
[PROBABILISTIC_IMPLEMENTATION_NOTES.md](PROBABILISTIC_IMPLEMENTATION_NOTES.md);
for the dual-learning math see [SUBGRADIENT_METHODS.md](SUBGRADIENT_METHODS.md).

> **All commands are run from `python_code/`.** Every runner lives in
> `src/execute/` and writes to `data/{deterministic,probabilistic}/` and
> `plots/{deterministic,probabilistic}/`.

---

## 0. TL;DR

```bash
# Everything (slow at production fidelity)
python src/execute/run_all.py

# Fast smoke test of the entire pipeline (well under a minute)
INFOCOM_TITER=50 INFOCOM_MC_TRIALS=2 \
  INFOCOM_PROFILES=uniform_wide,bimodal_extreme \
  INFOCOM_SOURCES=4,8 INFOCOM_CHANNELS=4,8 INFOCOM_TASKS=3,6 \
  INFOCOM_ITER_REFERENCE=100 \
  python src/execute/run_all.py --skip-precheck

# The three new things this guide adds:
python src/execute/ErrorVsIterations_probabilistic.py   # MGF vs #iterations (q=1) + MATLAB sanity
python src/execute/ExportExperimentParameters.py        # dump every experiment's parameters
# ...and every probabilistic sweep now emits a `_weights_1` (w=1) variant too.
```

---

## 1. The experiment catalogue

### Deterministic sweeps (1:1 MATLAB ports — the paper figures)

| Script | Sweep | Penalty | Weights |
|---|---|---|---|
| `ErrorVsChannel.py` | channels `N` | empirical `loss.mat` (`p1`,`p2`), `km=2` | two priority pairs = 1, rest 0.01 |
| `ErrorVsChannelsmodel.py` | channels `N` | synthetic 9-task model | 1 for `m+1<=M/2`, else 0.01 |
| `ErrorVsSources.py` | sources `M` | synthetic 9-task model | 1 for `m+1<=M/2`, else 0.01 |
| `ErrorVsSourcesM.py` | sources `M` (step 3) | synthetic; reproduces a MATLAB bug | (see PROJECT_OVERVIEW) |
| `ErrorVsTasks.py` | tasks `km` | synthetic model | 1 for `m+1<=M/2`, else 0.01 |
| `ErrorVsSources_variants.py` | sources `M` | synthetic | method-selectable subgradient |

These compare **MGF / MAF / MIEF / Random** and write one PNG + one CSV each.

### Probabilistic sweeps (unreliable links, `q_{m,j}` Bernoulli)

| Script | Sweep |
|---|---|
| `ErrorVsChannel_probabilistic.py` | channels `N`, empirical penalties (`km=2`) |
| `ErrorVsChannelsmodel_probabilistic.py` | channels `N`, synthetic 9-task model |
| `ErrorVsSources_probabilistic.py` | sources `M`, synthetic model |
| `ErrorVsTasks_probabilistic.py` | tasks `km`, synthetic model |
| `CompareSubgradient_probabilistic.py` | dual-update method benchmark |

Each probabilistic sweep produces **one subplot per reliability (`q`) profile**
and is run in **two weight modes** (see §3):

- *deterministic* weights → bare stem, e.g. `ErrorVsChannelsmodel_probabilistic`,
- *uniform* `w=1` → `_weights_1` suffix, e.g.
  `ErrorVsChannelsmodel_probabilistic_weights_1`.

### New experiments added in this round

| Script | What it does |
|---|---|
| `ErrorVsIterations_probabilistic.py` | MGF objective vs. the **number of subgradient iterations** (local episodes) `1,2,4,8,16,32`, run probabilistically with **`q=1`** on the three deterministic settings. Doubles as the **`q=1` ⇒ MATLAB sanity check**. |
| `CompareSubgradientIterations_probabilistic.py` | MGF objective vs. **number of episodes** `1,2,4,8,16,32,64,128`, **one curve per subgradient method** (all the rules we have), one subplot per `q`-profile. The convergence-rate companion to `CompareSubgradient_probabilistic.py`. |
| `ExportExperimentParameters.py` | Dumps **all parameters** of every experiment (weights `w`, reliabilities `q`, `M`, `N`, `km`, `B`, `T`, `gamma`, `n`, `c`, solver settings, profiles) to readable `.txt` + `.json` plus a master `EXPERIMENT_PARAMETERS.md`. |

---

## 2. The master runner: `run_all.py`

```bash
python src/execute/run_all.py                 # precheck → deterministic → compare → probabilistic → params
python src/execute/run_all.py --deterministic-only
python src/execute/run_all.py --probabilistic-only --skip-precheck
python src/execute/run_all.py --skip-precheck  # reuse existing recommended_subgradient_methods.json
python src/execute/run_all.py --skip-compare   # don't build the side-by-side MATLAB comparison PNGs
```

Order of operations:

1. `PrecheckSubgradientMethods` → `recommended_subgradient_methods.json`
2. `lossread` → `data/deterministic/loss.mat`
3. deterministic sweeps
4. `compare_with_matlab` (side-by-side vs `matlab_code/*.fig`)
5. probabilistic sweeps (**both** weight modes each)
6. `ErrorVsIterations_probabilistic` (+ q=1 sanity check)
7. `CompareSubgradient_probabilistic`
8. `ExportExperimentParameters` (always, last)

`MPLBACKEND=Agg` is set automatically by `run_all.py`. For a single headless
script, set it yourself: `MPLBACKEND=Agg python src/execute/<script>.py`.

---

## 3. Weight modes (deterministic vs `weights_1`)

Every probabilistic sweep is run in two weight modes, controlled centrally by
`src/main/experiment_configs.py`:

| Mode | `w` | Output stem |
|---|---|---|
| `deterministic` (default) | the heterogeneous weights of the matching paper figure (e.g. 1 for the first half of the sources, else 0.01) | `<stem>` |
| `ones` (`weights_1`) | `w = 1` everywhere | `<stem>_weights_1` |

Run a single mode with `INFOCOM_WEIGHT_MODES`:

```bash
INFOCOM_WEIGHT_MODES=ones          python src/execute/ErrorVsSources_probabilistic.py  # only weights_1
INFOCOM_WEIGHT_MODES=deterministic python src/execute/ErrorVsSources_probabilistic.py  # only paper weights
INFOCOM_WEIGHT_MODES=deterministic,ones python src/execute/ErrorVsSources_probabilistic.py  # both (default)
```

The weight matrices themselves are defined once in `experiment_configs.make_weights`;
to change a pattern, edit that function (the parameter dump and every sweep will
pick the change up automatically).

---

## 4. Hyper-parameters (environment variables)

Everything is configured via environment variables — no code edits needed for
the common knobs. Defaults are shown.

### Common

| Variable | Default | Meaning |
|---|---|---|
| `INFOCOM_SEED` | `0` | RNG seed (random policy, `q` draws, MC rollouts) |
| `INFOCOM_TITER` | `10000` det / `1000` prob | subgradient iterations (dual-ascent steps / local episodes) |

### Deterministic

| Variable | Default | Used by |
|---|---|---|
| `INFOCOM_TITER` | `10000` | every `ErrorVs*.py` |
| `INFOCOM_SOURCES` | `2..20` step 2 | `ErrorVsSources_variants.py` |
| `INFOCOM_SUBGRADIENT_METHOD` | (none) | `ErrorVsSources_variants.py` |

### Probabilistic sweeps

| Variable | Default | Meaning |
|---|---|---|
| `INFOCOM_TITER` | `1000` | dual iterations |
| `INFOCOM_MC_TRIALS` | `10` | Monte-Carlo rollouts averaged per point (production: 30+) |
| `INFOCOM_PROFILES` | all 12 | comma-separated `q` profiles (see §7) |
| `INFOCOM_WEIGHT_MODES` | `deterministic,ones` | which weight modes to run |
| `INFOCOM_EXTRA_POLICIES` | `0` | `1` adds `MAF_relaware` + `MIEF_pure` rows |
| `INFOCOM_SUBGRADIENT_METHOD` | from JSON (`episode1_mstep`) | dual update rule |
| `INFOCOM_SOURCES` | `2..20` step 2 | `ErrorVsSources_probabilistic.py` |
| `INFOCOM_CHANNELS` | `2..20` step 2 | `ErrorVs*Channel*_probabilistic.py` |
| `INFOCOM_TASKS` | `3..15` step 3 | `ErrorVsTasks_probabilistic.py` |

### MGF-vs-iterations experiment

| Variable | Default | Meaning |
|---|---|---|
| `INFOCOM_ITERATIONS` | `1,2,4,8,16,32` | subgradient-iteration counts to sweep |
| `INFOCOM_ITER_REFERENCE` | `10000` | reference (converged) MGF1 iteration count for the dashed line / sanity check |
| `INFOCOM_ITER_SETTINGS` | all | subset of `ErrorVsChannel,ErrorVsChannelsmodel,ErrorVsSources` |
| `INFOCOM_SEED` | `0` | baseline (random-policy) seed |

### Subgradient-method-vs-episodes comparison (`CompareSubgradientIterations_probabilistic.py`)

| Variable | Default | Meaning |
|---|---|---|
| `INFOCOM_COMPARE_ITERATIONS` | `1,2,4,8,16,32,64,128` | episode (iteration) counts to sweep |
| `INFOCOM_COMPARE_M` | `10` | number of sources for the comparison setting |
| `INFOCOM_PROFILES` | `uniform_wide,uniform_low,bimodal_extreme` | `q`-profiles (one subplot each) |
| `INFOCOM_COMPARE_METHODS` | all 9 first-order + `episode1_mstep` | comma-separated method override |
| `INFOCOM_COMPARE_INCLUDE_CUTTING_PLANES` | `0` | `1` also runs `kelley_bounded`, `trust_region_kelley`, `proximal_bundle` |
| `INFOCOM_MC_TRIALS` | `10` | MC rollouts averaged per (method, iters) |
| `INFOCOM_SEED` | `0` | seed |

```bash
# Compare every subgradient method's convergence vs episodes, 3 profiles
python src/execute/CompareSubgradientIterations_probabilistic.py

# ...including the (slower) cutting-plane methods, finer episode grid
INFOCOM_COMPARE_INCLUDE_CUTTING_PLANES=1 \
  INFOCOM_COMPARE_ITERATIONS=1,2,4,8,16,32,64,128,256 \
  python src/execute/CompareSubgradientIterations_probabilistic.py
```

### Precheck

| Variable | Default |
|---|---|
| `INFOCOM_PRECHECK_TITER` | `100` |
| `INFOCOM_PRECHECK_MC_TRIALS` | `5` |
| `INFOCOM_PRECHECK_SEED` | `0` |
| `INFOCOM_PRECHECK_FAST` | `1` |
| `INFOCOM_PRECHECK_INCLUDE_CUTTING_PLANES` | `0` |
| `INFOCOM_PRECHECK_METHODS` | all first-order methods |

### Examples

```bash
# Probabilistic ErrorVsSources, only the weights_1 variant, Adam dual, 30 MC trials
INFOCOM_WEIGHT_MODES=ones INFOCOM_SUBGRADIENT_METHOD=adam INFOCOM_MC_TRIALS=30 \
  python src/execute/ErrorVsSources_probabilistic.py

# Channelsmodel sweep, custom channel grid, three profiles, both weight modes
INFOCOM_CHANNELS=2,6,10,14,18 INFOCOM_PROFILES=uniform_wide,uniform_low,bimodal_extreme \
  python src/execute/ErrorVsChannelsmodel_probabilistic.py

# Iteration experiment with a finer iteration grid and a cheaper reference
INFOCOM_ITERATIONS=1,2,4,8,16,32,64,128 INFOCOM_ITER_REFERENCE=2000 \
  python src/execute/ErrorVsIterations_probabilistic.py
```

---

## 5. The MGF-vs-iterations experiment (and the q=1 sanity check)

```bash
python src/execute/ErrorVsIterations_probabilistic.py
```

For each of the three deterministic settings (`ErrorVsChannel`,
`ErrorVsChannelsmodel`, `ErrorVsSources` at fixed `M=10`), it runs the
**probabilistic** MGF with **`q = 1` on every link** for an increasing number of
subgradient iterations (`INFOCOM_ITERATIONS`), and plots the discounted
weighted-error objective against that iteration count (log2 x-axis), with
`MAF` / `MIEF` / `Random` and the converged `MGF1 @ INFOCOM_ITER_REFERENCE` drawn
as reference lines.

Because `q = 1` and `n = 1`, the probabilistic MGF with the `episode1_mstep`
subgradient is algebraically identical to the deterministic `MGF1` (the 1:1
MATLAB port). The script verifies this at **every** iteration count and prints:

```
SANITY CHECK PASSED: probabilistic MGF (q=1) matches deterministic MGF1 ...
  max relative difference = 6.13e-16 (tol 1e-06)
```

Outputs:

- `data/probabilistic/ErrorVsIterations_probabilistic_data.csv`
  (columns include `mgf_prob_q1`, `mgf_det`, `rel_diff`, `mgf_reference`, baselines)
- `plots/probabilistic/ErrorVsIterations_probabilistic.png` (combined)
- `plots/probabilistic/ErrorVsIterations_<setting>_probabilistic.png` (per setting)

---

## 6. Exporting every experiment's parameters

```bash
python src/execute/ExportExperimentParameters.py
```

Writes, for every experiment (every generated plot):

- `data/<flavor>/params/<stem>_parameters.txt` — human-readable: dimensions,
  sweep variable + values, `n`, `c`, penalty model, the full **weight matrix**
  (+ value summary), policies, solver (method / iterations / `mc_trials` / seed),
  and — for probabilistic experiments — the **`q` profile statistics** at a
  representative sweep point.
- `data/<flavor>/params/<stem>_parameters.json` — the same, machine-readable
  (full `w` matrix and `q` stats as arrays).
- `EXPERIMENT_PARAMETERS.md` — a master index aggregating all of the above.

It reconstructs everything from the same builders the runners use, reading the
**same environment variables**, so set them the way you set them for your runs
(e.g. run it with the same `INFOCOM_PROFILES` / `INFOCOM_CHANNELS` you used). The
full per-(profile, sweep-value) `q` matrices are saved by the sweeps themselves
in `data/probabilistic/<stem>_q_profiles.npz`.

---

## 7. Reliability (`q`) profiles

12 named profiles (in `probability_profiles_probabilistic.py`), selected via
`INFOCOM_PROFILES`:

- **Uniform**: `uniform_wide`, `uniform_low`, `uniform_mid`, `uniform_very_wide`,
  `uniform_with_perfect_outliers`
- **Bimodal**: `bimodal_extreme`, `bimodal_balanced`, `bimodal_q1_vs_lossy_30_70`,
  `bimodal_q1_vs_lossy_70_30`
- **Mixed**: `trimodal_perfect_mid_low`, `source_split_perfect_or_lossy`,
  `adversarial_perfect_with_critical_lossy`

```bash
INFOCOM_PROFILES=uniform_wide,bimodal_extreme,adversarial_perfect_with_critical_lossy \
  python src/execute/ErrorVsSources_probabilistic.py
```

---

## 8. Tests & sanity checks

```bash
# Standalone (these run as scripts)
python src/tests/test_probabilistic.py   # includes the q=1 reduction invariant
python src/tests/test_mief.py
python src/tests/checkmgf.py
python src/tests/checkmaf.py
python src/tests/checkmief.py
python src/tests/checkrand.py

# The q=1 ⇒ MATLAB sanity check (end to end, three settings)
python src/execute/ErrorVsIterations_probabilistic.py
```

---

## 9. Reproducing at full production fidelity

The default grid (12 profiles × 2 weight modes × 4 sweeps × ~10 points × MC
trials) is large. For publication-quality numbers:

```bash
# Full deterministic figures (MATLAB-matching; ~30 min)
python src/execute/run_all.py --deterministic-only

# Full probabilistic grid, both weight modes, all 12 profiles, 30 MC trials
INFOCOM_TITER=1000 INFOCOM_MC_TRIALS=30 \
  python src/execute/run_all.py --probabilistic-only --skip-precheck

# Iteration experiment at the MATLAB reference (10000 iters)
INFOCOM_ITER_REFERENCE=10000 python src/execute/ErrorVsIterations_probabilistic.py

# Refresh all parameter dumps to match
python src/execute/ExportExperimentParameters.py
```

To cut runtime while iterating, lower `INFOCOM_TITER` / `INFOCOM_MC_TRIALS`,
narrow `INFOCOM_PROFILES`, shorten the sweep grids (`INFOCOM_CHANNELS` /
`INFOCOM_SOURCES` / `INFOCOM_TASKS`), or restrict `INFOCOM_WEIGHT_MODES`.
```
