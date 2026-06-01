# Probabilistic-Transmission Implementation Notes

This document records the differences between the original deterministic Python port (`valuefunction1.py`, `Episode1.py`, `subgradientiter1.py`, `MGF1.py`, `MAF1.py`, `MIEF1.py`, `randpolicy.py`, `lowerbound.py`) and the new probabilistic-transmission variants (`*_probabilistic.py`). The probabilistic variants implement the paper's unreliable-channel AoI model exactly; the originals were a simplified deterministic case.

## 1. The paper's communication model vs. what the port implemented

**Paper.** Each scheduled pair `(m, j)` succeeds with probability `q_{m,j}`:

```
u_{m,j}(t) ~ Bernoulli(q_{m,j})
Delta_{m,j}(t+1) = 1            if pi_{m,j}(t) = 1 and u_{m,j}(t) = 1   (success)
                   Delta_{m,j}(t) + 1                                    (otherwise)
```

The reset value `1` is the paper's 1-indexed convention. In the code we use 0-indexed AoI throughout, so the reset becomes `0` and the "age" cap becomes `B-1`:

```
on success:   Delta -> 0
on failure:   Delta -> min(Delta + 1, B - 1)
on no-schedule: Delta -> min(Delta + 1, B - 1)
```

**Original port.** All `q_{m,j}` were implicitly 1: scheduling a pair always reset AoI to 0. This is the deterministic special case.

## 2. The paper's Bellman/action-value equation

For the relaxed per-pair value function (after Lagrangian relaxation), the paper's scheduled-action value is:

```
Q_schedule(d) = p(d) + lambda + mu * n_{m,j}
              + gamma * ( q_{m,j} * V(d_reset)
                          + (1 - q_{m,j}) * V(d_aged) )
```

where `d_aged = min(d+1, B-1)` and `d_reset = 0`. The no-schedule action value is:

```
Q_no(d) = p(d) + gamma * V(d_aged)
```

The **gain index** is `gain(d) = Q_no(d) - Q_schedule(d)`, and the relaxed optimal action picks the pair iff `gain > 0`.

**Original port (`valuefunction1.py`).** Implemented:

```
Q1 = p(d) + gamma * V[i+1, d+1]
Q2 = p(d) + lambda(i) + mu(i) + gamma * V[i+1, 0]
```

Two issues vs. the paper:

1. `mu(i)` lacked the `n_{m,j}` multiplier. In `ErrorVsSources.py` the original `n` is always 1, so this didn't bite numerically, but the formula is wrong in general.
2. Reset to `V[i+1, 0]` was unconditional — equivalent to `q_{m,j} = 1`.

The probabilistic variant `valuefunction1_probabilistic.py` implements the paper's Q2 verbatim. With `q=1` and `n_cost=1` it reduces algebraically to the original — this is asserted by `test_valuefunction_probabilistic_matches_deterministic_when_q1_n1`.

## 3. Subgradient / Lagrangian updates

The subgradient update uses the **attempted** schedule `pi`, not the **delivered** result `u`. This is correct: the compute and channel resources are consumed at the moment of attempted transmission, regardless of success. The probabilistic variants preserve this — `subgradientiter1_probabilistic.py` computes the supergradient from `pi`, and the relaxed (per-pair) Bellman recursion uses *expected* probabilistic transitions (`q * V_reset + (1-q) * V_age`) so the dual learning is deterministic in `q`.

The current `subgradientiter1_probabilistic.py` exposes `method=` from a single centralised registry in `optimizer_updates.py`:

| method              | rule                                                  |
| ------------------- | ----------------------------------------------------- |
| `harmonic`          | `step = beta / j`                                     |
| `sqrt`              | `step = beta / sqrt(j)`                               |
| `normalized_global` | `step = beta / sqrt(j)`, direction = `g / ||g||`      |
| `normalized_blocks` | per-block normalization (lambda and mu separately)    |
| `adagrad`           | coordinatewise scaling by accumulated `g^2`           |
| `rmsprop`           | EMA of `g^2` instead of cumulative                    |
| `adam`              | Adam moments with bias correction                     |
| `deflected_sqrt`    | momentum direction `alpha*d_prev + (1-alpha)*g`        |

Legacy method aliases preserved for backward compatibility: `constant` -> harmonic with held step `beta`, `normalized` / `polyak_like` -> `normalized_global`.

The same registry drives the deterministic `subgradientiter1_variants.py` and `MGF1_variants.py` so deterministic and probabilistic experiments can be compared head-to-head.

Cutting-plane / bundle methods are also wired in (and selectable via `subgradient_method=`):

- `kelley_bounded` — Kelley's outer-approximation LP with box bounds.
- `trust_region_kelley` — Kelley + shrinking inf-norm trust region.
- `proximal_bundle` — quadratic-prox bundle with limited bundle size.

These call `dual_oracle_probabilistic.py` / `dual_oracle_deterministic.py`, which return a consistent `(D(z), supergradient, diagnostics)`. The dual is solved in **simplified time-invariant form** by default: `z = [lambda_1, ..., lambda_M, mu]`, then expanded to per-`t` multipliers for the MGF rollout. The dual value is in raw units (sum over pairs, not mean) and the supergradient is the discounted resource residual `sum_t gamma^t * (sum_j pi^* - c_m)` and `sum_t gamma^t * (sum_{mj} pi^* n_mj - N)`, so both LP- and prox-based cuts are unit-consistent.

## 4. Online (greedy) rollout

The online policies need a stochastic AoI update. For each scheduled pair `(m, j)`:

```python
success = rng.random() < q[m, j]
if success:
    Delta[m, j] = 0
else:
    Delta[m, j] = min(Delta[m, j] + 1, B - 1)
```

The penalty `w * p(j, Delta_new)` is then evaluated on the realized `Delta_new`. Resource feasibility (per-source compute `c[m]`, total channel `N`) is checked against attempted `pi`, never against realized `u`.

## 5. Weights `w` and per-pair channel cost `n_{m,j}`

- The existing `ErrorVsSources.py` uses heterogeneous `w` (`1` for `m <= M/2`, `0.01` otherwise) and `n = 1`. As of the latest revision, each probabilistic sweep is run in **two weight modes** (centralised in `experiment_configs.make_weights`): a `deterministic` mode that mirrors the matching paper figure's heterogeneous weights (the default, written to the bare stem) and an `ones` mode with `w = 1` everywhere (written to a `_weights_1` stem) so reliability heterogeneity alone drives the result. Restrict modes with `INFOCOM_WEIGHT_MODES`. Earlier versions of these scripts hard-coded `w = ones`; that case is now the `_weights_1` variant.
- The probabilistic Bellman includes `mu * n_{m,j}`, so the value table now depends on `n_{m,j}` (not just on aggregate `N`). `subgradientiter1_probabilistic._batched_gain_table_probabilistic` passes per-pair `n_{m,j}` into the recursion.

## 6. Baselines and reliability awareness

- **MAF** — unchanged priority (`Delta + 1`) by default, but with an opt-in `reliability_aware=True` that scales by `q`. Default is pure MAF for baseline comparability.
- **MIEF** — original priority `w * p(j, Delta)`. The probabilistic version defaults to `reliability_aware=True` (`q * w * p(j, Delta)`), which approximates expected immediate usefulness; `reliability_aware=False` preserves the pure baseline.
- **Random** — the original `randpolicy` lacks capacity gating. The probabilistic variant defaults to a **gated** random policy (respects per-source compute and total channel) but exposes `gated=False` for ungated comparison.

## 7. Outputs

Probabilistic code never writes to `multipliers.mat`. With `save=True` it writes `multipliers_probabilistic_{method}.mat`; the default is `save=False` so the deterministic `multipliers.mat` is undisturbed. Deterministic variants similarly write `multipliers_deterministic_{method}.mat` only when `save=True`.

Top-level outputs:

- `ErrorVsSources_probabilistic_data.csv` / `_summary.csv` / `.png` — `M` sweep
- `ErrorVsChannel_probabilistic_data.csv` / `_summary.csv` / `.png` — `N` sweep with empirical loss.mat penalties (km=2)
- `ErrorVsChannelsmodel_probabilistic_data.csv` / `_summary.csv` / `.png` — `N` sweep with synthetic 9-task model
- `ErrorVsTasks_probabilistic_data.csv` / `_summary.csv` / `.png` — `km` sweep with synthetic model
- `ErrorVsSources_variants_data.csv` / `.png` (deterministic, method-selectable)
- `PrecheckSubgradientMethods_data.csv` / `_summary.csv` / `.png`
- `CompareSubgradient_probabilistic_data.csv` / `.png`
- `recommended_subgradient_methods.json`
- `q_profiles_probabilistic.npz` — raw q matrices used per profile / per sweep value (for reproducibility)

Each probabilistic sweep also emits a `_weights_1` companion (uniform `w=1`):

- `ErrorVs*_probabilistic_weights_1_data.csv` / `_summary.csv` / `.png` — same sweep with `w = 1` everywhere.
- `ErrorVsIterations_probabilistic_data.csv` / `.png` — MGF objective vs. subgradient-iteration count at `q=1` (also the `q=1` ⇒ MATLAB sanity check).
- `data/<flavor>/params/<stem>_parameters.txt` / `.json` and `EXPERIMENT_PARAMETERS.md` — full parameter dump for every experiment, from `ExportExperimentParameters.py`.

All probabilistic sweep scripts share the same conventions:

- Weights come from `experiment_configs.make_weights` in one of two modes — `deterministic` (heterogeneous, matches the paper figure) or `ones` (`w=1`, the `_weights_1` variant). The CSV records the mode in a `weights_mode` column.
- `c = 2 * ones((M, km))`, `n = ones((M, km))` by default
- Read `INFOCOM_TITER`, `INFOCOM_MC_TRIALS`, `INFOCOM_SEED`, and `INFOCOM_PROFILES` from env
- Default sweep variable, fixed problem dims, and 1-indexed synthetic penalty (`j%3==0` linear, `j%3==1` 10·log, `j%3==2` exp(0.5·d)) all match the deterministic scripts
- Read `recommended_subgradient_methods.json` for the MGF subgradient method unless `INFOCOM_SUBGRADIENT_METHOD` overrides it
- One PNG per script with one subplot per q profile; CSV rows have `(profile, sweep_var, policy, mean_error, std_error, ...)`

## 8. Probability profiles

`probability_profiles_probabilistic.py` exposes 12 named profiles in three families. They focus on settings that empirically discriminate policies the most (uniform variants and bimodal variants), and many of them mix exact `q = 1.0` (perfect) links with significantly lower-reliability links to stress reliability-aware vs. reliability-blind scheduling.

| Family | Profile | Description |
|---|---|---|
| Uniform | `uniform_wide` | Uniform(0.35, 0.98) wide baseline |
| Uniform | `uniform_low` | Uniform(0.35, 0.70) moderate-low |
| Uniform | `uniform_mid` | Uniform(0.55, 0.85) moderate band |
| Uniform | `uniform_very_wide` | Uniform(0.20, 0.99) extreme spread |
| Uniform | `uniform_with_perfect_outliers` | 85% Uniform(0.40, 0.80) + 15% q=1 |
| Bimodal | `bimodal_extreme` | 70% ~0.95 / 30% ~0.30 |
| Bimodal | `bimodal_balanced` | 50% ~0.95 / 50% ~0.40 |
| Bimodal | `bimodal_q1_vs_lossy_30_70` | 30% q=1 / 70% Uniform(0.30, 0.50) |
| Bimodal | `bimodal_q1_vs_lossy_70_30` | 70% q=1 / 30% Uniform(0.30, 0.55) |
| Mixed | `trimodal_perfect_mid_low` | Equal thirds: q=1 / ~0.65 / ~0.35 |
| Mixed | `source_split_perfect_or_lossy` | Per source: all q=1 OR all Uniform(0.35, 0.60) |
| Mixed | `adversarial_perfect_with_critical_lossy` | All q=1 except exponential-penalty tasks (j%3==2) get q ~ 0.35 |

`Q_MAX = 1.0` (perfect links are allowed) and `Q_MIN = 0.05` (lower tail bounded away from 0).

## 9. Caveats

- All probabilistic results are random. Reported numbers should be averaged over multiple MC trials (`mc_trials`). The compare scripts use a small default for speed; production runs should use 30+ trials.
- The "expected" gain index uses `q_{m,j}` only and does not model the variance of returns — this matches the paper's relaxation.
- The lower bound is an *approximate* relaxed lower-bound (probabilistic variant) — exact lower bound for the unreliable case is more involved and is not derived here.
- Cutting-plane methods are implemented in simplified time-invariant form (`z = [lambda_m, mu]`, length M+1). Full time-indexed cutting-plane variants are not implemented; they would require an LP with M*T + T variables and per-cut residuals at the (m, t) level. In our precheck cutting-plane methods consistently produced **worse** MGF rollout objectives than the first-order methods (because the simplified form gives up time-varying multipliers); the precheck therefore does not recommend them by default.
- `MGF1_variants(harmonic)` is **not byte-identical** to `MGF1`: the legacy `subgradientiter1.py` uses the MATLAB `Episode1` M-step closed-form (M projected ascent steps per outer iter), while the variants do one projected ascent step per outer iter. The fixed points differ by a few percent on tiny problems; both converge to the same basin on production sizes.

## 10. Method recommendation flow

```
PrecheckSubgradientMethods.py
   |
   v
recommended_subgradient_methods.json    (deterministic / probabilistic / combined)
   |
   v
ErrorVsSources_probabilistic.py  reads the "probabilistic" key
ErrorVsSources_variants.py       reads the "deterministic" key
```

If the JSON does not exist, both scripts fall back to `harmonic`. Use `INFOCOM_SUBGRADIENT_METHOD` to override at the env-var level.
