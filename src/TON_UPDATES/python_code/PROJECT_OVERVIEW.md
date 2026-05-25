# INFOCOM Code — Python Port Overview

This directory contains a Python port of the MATLAB code from an INFOCOM/ToN paper on
**Age-of-Information (AoI) aware scheduling of multiple sources/tasks over a shared
channel**. The codebase implements several scheduling policies (MGF, MAF, MIEF,
Random, Lower Bound), sweeps over problem parameters (channels, sources, tasks),
loads empirical inference-error data, and compares the Python output against the
original MATLAB `.fig` references.

The whole port is intentionally **1:1 with the MATLAB sources** (one `.py` per `.m`,
plus a few extras for the new MIEF baseline and the comparison tooling). MATLAB
1-based indexing has been carefully translated, including quirks (e.g.
`ErrorVsSourcesM.m`'s leftover-`j` bug) that are reproduced exactly so results match.

---

## Conceptual Model

| Symbol | Meaning |
|--------|---------|
| `M`    | number of sources |
| `N`    | total channel/bandwidth capacity at each time step |
| `km`   | number of tasks per source |
| `T`    | horizon (Bellman value-function depth) |
| `K`    | number of simulated time steps (usually `K == T`) |
| `B`    | AoI bound — `Delta[m,j]` is clipped to `0..B-1` |
| `gamma`| discount factor (e.g. `0.9`) |
| `Delta[m,j]` | current Age-of-Information for source `m`, task `j` (reset to 0 when scheduled, +1 otherwise) |
| `n[m,j]` | channel cost of scheduling pair `(m,j)` |
| `c[m]`   | per-source compute cap (max simultaneous tasks per source) |
| `w[m,j]` | task weight in the objective |
| `p[j,d]` | penalty for task `j` at AoI level `d` |

The objective being minimized is the discounted sum

$$\sum_{t=0}^{K-2} \gamma^t \cdot \tfrac{1}{Mk_m} \sum_{m,j} w_{m,j} \, p_j(\Delta_{m,j}(t))$$

subject to per-source compute caps and a total channel budget.

---

## File-by-File Reference

### Core algorithms

#### [valuefunction1.py](valuefunction1.py)
1:1 port of `valuefunction1.m`.

- **`valuefunction1(lambda_, mu, B, p, T, gamma) -> a`**
  Computes the Bellman value function `V` and the **gain index** `a[i,d] = Q1 - Q2`
  for a single (source, task) by backward induction.
  - `Q1 = p(d) + gamma * V[i+1, d+1]` — cost of *not* scheduling (AoI ages by 1).
  - `Q2 = p(d) + lambda(i) + mu(i) + gamma * V[i+1, 0]` — cost of scheduling
    (AoI resets, pay multipliers).
  - Returns `(T, B)` gain matrix later used as the index policy.
  - Mirrors the MATLAB cap `V(i, B+1) = V(i, B)`.

#### [Episode1.py](Episode1.py)
1:1 port of `Episode1.m`, vectorized over the inner `M` loop using a
closed-form for the iterated `max(0, ·)` projection.

- **`Episode1(asource1, M, T, B, gamma, N, beta, lambdasource, mu, km, w, n, c) -> A`**
  Runs one outer subgradient episode. Returns `A` of shape `(M+1, T)` where
  `A[:M]` is the updated `lambdasource` (per-source multipliers) and `A[M]` is
  the updated channel multiplier `mu`.
- **`_project_M_steps(A_init, s_vec, dp_vec, m_vec, M)`** — closed-form for
  the M sequential projected updates per source (the standard
  `A_M = max(0, S_M - min_j S_j, A_init + S_M)` identity).
- **`_project_channel(A_init_mu, delta_mu)`** — same identity for the scalar
  channel multiplier.
- **`_as_1d_c(c, M)`** — accept `c` shaped either `(M,)` or `(M, km)`.

#### [subgradientiter1.py](subgradientiter1.py)
1:1 port of `subgradientiter1.m`. Runs the projected subgradient method to
learn Lagrange multipliers `(lambdasource, mu)`. Speed-optimized by batching the
`M*km` value-function recursions into one tensorized Bellman backward pass.

- **`subgradientiter1(M, N, T, B, gamma, p, km, w, n, c, titer=10000, verbose=False) -> A`**
  Runs `titer` outer iterations with shrinking step size `beta / j`. Saves the
  result to `multipliers.mat`.
- **`_batched_gain_table(lambdasource, mu, B, T, gamma, w, p, M, km) -> asource1`**
  Returns `asource1[m, t, d, task]` produced by a single batched Bellman pass
  — identical algebra to looping `valuefunction1` over `(m, task)`.
- **`load_multipliers() -> (lambdasource, mu)`** — reload from `multipliers.mat`.

#### [greedy_scheduler.py](greedy_scheduler.py)
Shared resource-feasibility loop reused by MAF, MIEF and (a variant of) MGF.

- **`greedy_select(priority, n, c_use, N, M, km) -> (action, Change, Ccurr, Ncurr)`**
  Repeatedly pick the `(m, j)` with the largest positive priority and schedule
  it iff (a) per-source compute cap `sum_j action[m,j] <= c_use[m]` and
  (b) total channel cap `sum action*n <= N` are respected. Tie-breaking
  matches the original MAF/MGF MATLAB loops (column-wise argmax first, then
  argmax across columns).
- **`as_1d_c(c, M)`** — coerce per-source compute cap to `(M,)`.

---

### Scheduling policies

#### [MGF1.py](MGF1.py)
**Max-Gain-First / index policy** (1:1 port of `MGF1.m`).

- **`MGF1(M, N, km, T, B, K, n, c, w, gamma, p, titer=10000, verbose=True) -> presult`**
  1. Calls `subgradientiter1` to compute multipliers, then loads them.
  2. Builds the gain-index table `asource1[m, t, d, task]` from
     `valuefunction1`.
  3. At each time step, greedily picks pairs with the largest *positive*
     gain index, gated by capacity (`c[m]`) and channel (`N`).
  4. Returns the discounted sum of weighted errors.

#### [MAF1.py](MAF1.py)
**Max-Age-First policy** (1:1 port of `MAF1.m`). Priority = current AoI.

- **`MAF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=True) -> presult`**
  Schedules in descending order of `Delta[m,j] + 1`, gated by resources
  (`greedy_select`).

#### [MIEF1.py](MIEF1.py)
**Maximum Instantaneous Error First** — a Python-only addition (no MATLAB
counterpart). Priority = `w[m,j] * p[j, Delta[m,j]]`.

- **`mief_priority(Delta, w, p, km) -> priority`** — `score[m,j] = w*p(Delta)`.
- **`mief_select(Delta, M, km, n, c_use, N, w, p)`** — build priority and run
  the shared greedy selection.
- **`MIEF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=True) -> presult`** —
  full rollout returning the discounted sum of weighted errors.

#### [randpolicy.py](randpolicy.py)
**Randomized scheduling baseline** (1:1 port of `randpolicy.m`).

- **`randpolicy(M, N, km, T, B, K, n, c, w, gamma, p, titer=100, verbose=True) -> presult`**
  Each step: pick `min(N, M)` sources uniformly without replacement, each
  schedules a uniformly random task. No capacity gating. Averages over
  `titer=100` Monte-Carlo trials.

#### [lowerbound.py](lowerbound.py)
**Relaxed Lagrangian lower bound** (1:1 port of `lowerbound.m`).

- **`lowerbound(M, N, km, T, B, K, n, c, w, gamma, p, titer=10000, verbose=True) -> presult`**
  Calls `subgradientiter1`, then makes greedy per-source decisions *without*
  joint resource gating, adding the multiplier contributions to the objective.
  Provides a theoretical lower bound for comparison.

---

### Top-level sweep scripts (each generates a figure + CSV)

All of these read `TITER` and `SEED` from the environment variables
`INFOCOM_TITER` and `INFOCOM_SEED` (defaults `10000` and `0`). Each writes
`<name>_data.csv` (raw numerics) and `<name>.png` (plot), and is a 1:1 port of
the corresponding `.m`.

| Script | Sweep variable | Penalty source |
|--------|----------------|----------------|
| [ErrorVsChannel.py](ErrorVsChannel.py)               | `N = 2..20 step 2` (channels)   | `loss.mat` (empirical p1, p2) |
| [ErrorVsChannelsmodel.py](ErrorVsChannelsmodel.py)   | `N = 2..20 step 2` (channels)   | Synthetic 9-task model (linear / log / exp by `j%3`) |
| [ErrorVsSources.py](ErrorVsSources.py)               | `M = 2..20 step 2` (sources)    | Same synthetic model |
| [ErrorVsSourcesM.py](ErrorVsSourcesM.py)             | `M = 3..21 step 3` (sources)    | Same model — reproduces a MATLAB **bug** where only the last column of `w` is non-zero |
| [ErrorVsTasks.py](ErrorVsTasks.py)                   | `km = 3..15 step 3` (tasks)     | Same model |

Each compares **MGF, MAF, MIEF, Random** on the same axes (MIEF was added in
the port). All have a single `main()` entry point that:
1. Sets up `n, c, w, p` for the swept variable.
2. Calls each policy.
3. Saves CSV + PNG.

---

### Data ingestion

#### [lossread.py](lossread.py)
1:1 port of `lossread.m`. Reads the smoothed segmentation/detection
loss CSVs (`smooth_segmentation_*` and `smooth_averaged_detection_*` from the
project root), plots three figures (segmentation errorbar, detection errorbar,
combined `yyaxis`), and writes `loss.mat` with arrays `p1` and `p2` that the
`ErrorVsChannel` script later reads back in.

- **`_read_csv(path)`** — mirrors MATLAB `readmatrix` (skips header row).
- **`main()`** — produces `lossread_fig1.png`, `lossread_fig2.png`,
  `lossread_fig3.png`, and `loss.mat`.

#### [fig_reader.py](fig_reader.py)
Helper for reading MATLAB `.fig` files (which are MAT-format containers with
an `hgS_070000` graphics tree). Lets `compare_with_matlab.py` lift numerical
series out of the original MATLAB figures.

- **`extract_all(fig_path) -> list[dict]`** — every node with `XData`/`YData`
  becomes a dict `{type, name, x, y, L, U}`.
- **`extract_series(fig_path) -> {name: (x, y)}`** — compat helper.
- **`extract_named_lines(fig_path) -> {name: (x, y)}`** — accepts any series
  with a non-empty `DisplayName`.
- **`extract_errorbars(fig_path) -> list[dict]`** — only `errorbarseries`.
- Has a CLI when run directly: dumps a summary of every series found.

---

### Verification, smoke tests, comparison

#### [run_all.py](run_all.py)
Driver that:
1. Sets `MPLBACKEND=Agg` (headless plotting).
2. Re-seeds NumPy before each script for reproducibility.
3. Runs `lossread` then every `ErrorVs*` script.
4. Calls `compare_with_matlab.main()` to build side-by-side images.

#### [compare_with_matlab.py](compare_with_matlab.py)
Builds side-by-side `*_compare.png` images plus a per-policy numerical
diff table (printed to stdout) for every plot the port produces.

- **`_load_csv(path)`** — load `*_data.csv`, accepts both 4-column and
  5-column (with MIEF) layouts.
- **`_print_diff_table(...)`** — abs/rel-error table for the 3 MATLAB policies.
- **`_compare_policy_plot(...)`** — side-by-side Python vs MATLAB on a
  policy plot.
- **`_emit_python_only_compare(...)`** — when MATLAB has no reference
  (e.g. `ErrorVsSourcesM`), emit an explanatory note panel.
- **`_compare_with_jpg(...)`** — side-by-side using the MATLAB JPG when
  there's no raw figure data.
- **`_compare_errorbar(...)`** — errorbar series comparison vs a `.fig`.
- **`main()`** — orchestrates all the comparisons.

#### [checkmgf.py](checkmgf.py), [checkmaf.py](checkmaf.py), [checkrand.py](checkrand.py), [checkmief.py](checkmief.py)
Stand-alone sanity-check scripts for each policy. Each ports its
corresponding `.m` (`checkmgf.m` etc.) and runs the policy on a small
synthetic problem (`M=10, N=10, km=2, B=20, T=10, gamma=0.9`, two penalty
functions: `log` and linear). Useful for quick smoke-testing in isolation
from the sweep scripts.

`checkmief.py` additionally asserts that the resource constraints hold
on every step and that the step-by-step rollout matches the `MIEF1` wrapper.

#### [test_mief.py](test_mief.py)
Pytest-style (also runnable as a script) tests for the MIEF baseline:

- **`test_mief_matches_maf_when_p_is_identity_and_weights_equal`** — when
  `p(d) = d` and all `w` are equal, MIEF should produce the exact same
  schedule as MAF.
- **`test_mief_prefers_high_penalty_over_high_age`** — a low-age,
  high-penalty pair must beat a high-age, low-penalty pair (something
  pure MAF cannot do).
- **`test_mief_respects_compute_and_channel_constraints`** — 50 random
  states; the action must satisfy both budgets.
- **`test_mief_priority_is_w_times_p_of_delta`** — direct check of the
  priority-score formula.
- **`test_mief_end_to_end_runs_and_satisfies_constraints`** — full
  rollout returns a finite scalar and every step is feasible.

---

## Data Flow at a Glance

```
                CSV inference-error data        (smooth_*_pk_data.csv)
                         |
                         v
                   lossread.py  ----------------> loss.mat (p1, p2)
                                                      |
                                                      v
   subgradientiter1.py  ---->  multipliers.mat   ErrorVsChannel.py
        ^                            ^                 |
        |                            |                 v
   Episode1.py             load_multipliers()      MGF1 / MAF1
        |                                          MIEF1 / randpolicy
   valuefunction1.py                                 |
                                                     v
                                              *_data.csv + *.png
                                                     |
                                                     v
                                          compare_with_matlab.py
                                                     |
                                                     v
                                              *_compare.png
```

---

## Quick Start

```bash
# Full reproduction with MATLAB defaults (slow — 10000 subgradient iters/script)
python run_all.py

# Fast smoke test
INFOCOM_TITER=100 INFOCOM_SEED=42 python run_all.py

# Run a single sweep
python ErrorVsChannel.py

# Run the MIEF unit tests
python test_mief.py            # or: pytest test_mief.py -v

# Stand-alone policy sanity checks on a tiny problem
python checkmief.py
python checkmaf.py
python checkmgf.py
python checkrand.py
```

---

## Notes on MATLAB Fidelity

- **`ErrorVsSourcesM.py`** intentionally reproduces a MATLAB bug where, after
  the penalty-initialization loop, the leftover loop variable `j` (equal to
  `km`) is reused as the only column index when filling `w`, leaving every
  other column zero. Removing this would change the numerical output.
- **`ErrorVsSources.py`** notes that the original `ErrorVsSources.m` references
  `M` *before* defining it; the Python port initializes `n` and `c` inside
  the sweep loop so the script runs on a clean interpreter.
- **MIEF** is a port-only addition with no MATLAB counterpart, hence the
  side-by-side comparison shows it only on the Python panel.
- The subgradient method's per-iteration projected update is mathematically
  equivalent in the port — the MATLAB inner `for m=1:M` loop is closed-formed
  via the standard `max(0, S_M - min_j S_j, A_init + S_M)` identity.
