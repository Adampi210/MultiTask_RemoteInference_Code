# Subgradient Methods Reference

This document describes every multiplier-learning method exposed by the
project: how it works mathematically, where it lives in the code, and the
contract it implements. All methods solve the same outer problem

$$\max_{z \ge 0}\; D(z)$$

where $D$ is a concave Lagrangian dual and $z$ is a flattened nonneg
multiplier vector. In our setting

$$z = [\,\lambda_{1,1},\ldots,\lambda_{M,T},\; \mu_1,\ldots,\mu_T\,] \in \mathbb{R}_{\ge 0}^{MT+T}$$

for the full time-indexed formulation, or

$$z = [\,\lambda_1,\ldots,\lambda_M,\; \mu\,] \in \mathbb{R}_{\ge 0}^{M+1}$$

for the simplified time-invariant formulation used by the cutting-plane
methods.

The Lagrangian comes from relaxing the per-source compute caps and the total
channel cap of the AoI scheduling problem:

$$L(\pi, z) \;=\; \sum_{t=0}^{T-1} \gamma^t \!\left[\;
\frac{1}{Mk_m}\sum_{m,j} w_{m,j}\, p_j(\Delta_{m,j}(t))
\;+\; \sum_m \lambda_{m,t} \!\!\left(\sum_j \pi_{m,j,t} - c_m\!\right)
\;+\; \mu_t\!\!\left(\sum_{m,j} \pi_{m,j,t}\, n_{m,j} - N\!\right)
\!\right].$$

The dual function $D(z) = \min_{\pi} L(\pi, z)$ is concave (it is the
pointwise infimum of affine functions of $z$) and its supergradient at $z$ is
simply the constraint residual evaluated at the minimizing $\pi^*$.

In the probabilistic case, the AoI transitions inside the inner $\pi$
minimization use the expected Bellman operator
$\mathbb{E}[V_{\text{next}}] = q V_{\text{reset}} + (1-q) V_{\text{age}}$,
so $D(z)$ is still the relaxed expected DP value.

---

## Shared infrastructure

### `optimizer_updates.py`

Every first-order method is implemented as a single call to
`projected_update(z, g, k, method, state, beta, eps, block_slices, config)`:

```python
def projected_update(z, g, k, method, state,
                     beta=0.9, eps=1e-8,
                     block_slices=None, config=None):
    ...
    return z_new, state, diagnostics
```

| arg              | meaning                                                  |
| ---------------- | -------------------------------------------------------- |
| `z`              | current iterate, flattened, nonneg                       |
| `g`              | supergradient direction (same shape as `z`)              |
| `k`              | 1-indexed outer iteration                                |
| `method`         | string in `get_default_subgradient_methods()`            |
| `state`          | `OptimizerState` carrying adaptive accumulators          |
| `beta`           | base step size (default 0.9 — matches `subgradientiter1.py`) |
| `eps`            | safety constant for divisions                            |
| `block_slices`   | list of `(start, stop)` tuples for `normalized_blocks`    |
| `config`         | optional dict overriding hyperparameters (e.g. `rho`, `b1`) |

`OptimizerState(shape, method)` allocates four arrays of size `shape`:
`G` (Adagrad accumulator), `v` (RMSprop / Adam second moment), `m` (Adam first
moment), and `d_prev` (deflected_sqrt previous direction). Unused fields stay
at zero. Each `projected_update` call returns updated `z` and a `state` that
the driver passes back next iteration.

### Drivers

Two thin wrappers package the inner loop:

- `subgradientiter1_variants.py` (deterministic dual)
- `subgradientiter1_probabilistic.py` (probabilistic dual, expected Bellman)

Both follow the same template:

```python
z = np.concatenate([lambdasource.ravel(), mu.ravel()])
state = init_optimizer_state(z.shape, method)
block_slices = [(0, M*T), (M*T, M*T + T)]   # lambda block, mu block
for j in range(1, titer + 1):
    asource1 = batched_gain_table(lambdasource, mu, ...)
    g_lambda, g_mu, diag = compute_relaxed_actions_and_residuals(...)
    g = np.concatenate([g_lambda.ravel(), g_mu.ravel()])
    z, state, upd_diag = projected_update(z, g, j, method, state,
                                          beta=beta,
                                          block_slices=block_slices)
    lambdasource = z[:M*T].reshape(M, T)
    mu = z[M*T:].reshape(T)
```

The supergradient is computed by a deterministic *relaxed-action* rollout
starting from $\Delta = 0$: at each $t$ we read the gain index, set
$\pi_{m,j,t} = 1\{\text{gain} > 0\}$, and aggregate
`gamma**t * (compute_use_m - c_m)` and `gamma**t * (channel_use - N)`.

---

## First-order methods

### 1. `harmonic`

**Update:**

$$z_{k+1} = \max\!\big(0,\; z_k + \tfrac{\beta}{k}\, g_k\big)$$

**Why this works.** The step $\beta/k$ is the canonical diminishing step for
subgradient methods on a convex/concave objective. The harmonic series
$\sum 1/k = \infty$ and $\sum 1/k^2 < \infty$ are the textbook
Robbins–Monro–style conditions that guarantee convergence to a $D^*$-attaining
point in expectation for arbitrary bounded $g$.

**Implementation:**

```python
step = beta / k
z_new = np.maximum(0.0, z + step * g)
```

**State used:** none.

**Strengths.** Provable convergence with no tuning; matches the original
MATLAB `subgradientiter1.m` step schedule.
**Weaknesses.** Slow once `k` grows; flat regions take many iterations.

### 2. `sqrt`

**Update:**

$$z_{k+1} = \max\!\big(0,\; z_k + \tfrac{\beta}{\sqrt{k}}\, g_k\big)$$

**Why this works.** $\beta/\sqrt{k}$ is the *optimal* diminishing step for
non-strongly-convex subgradient ascent — it gives an $O(1/\sqrt{k})$
suboptimality bound versus $O(\log k / \sqrt{k})$ for $\beta/k$ in the worst
case. In practice it stays more responsive than harmonic in mid iterations.

**Implementation:** identical to harmonic with `step = beta / sqrt(k)`.

**State used:** none.

### 3. `normalized_global`

**Update:**

$$z_{k+1} = \max\!\big(0,\; z_k + \tfrac{\beta}{\sqrt{k}}\cdot \tfrac{g_k}{\|g_k\|_2 + \varepsilon}\big)$$

**Why this works.** Plain subgradient steps shrink whenever the gradient is
small, even when far from the optimum. Normalizing $g$ to unit norm
*decouples step size from gradient scale*, so the iterate moves by exactly
$\beta/\sqrt{k}$ every iteration. This is the Polyak-style "normalized
subgradient" rule; it is robust when $\|g\|$ varies by orders of magnitude
between coordinates or iterations.

**Implementation:**

```python
step = beta / np.sqrt(k)
direction = g / (np.linalg.norm(g) + eps)
z_new = np.maximum(0.0, z + step * direction)
```

**State used:** none.
**Strengths.** No need to know $\|g\|$ in advance; uniform progress.
**Weaknesses.** A single global normalizer can wash out scale differences
*between* the $\lambda$ and $\mu$ blocks, which is why we ship the next
method.

### 4. `normalized_blocks` ✦ recommended by precheck

**Update.** Split $z$ into blocks $B_1, B_2, \ldots$ (in our case
$B_1 = \lambda$, $B_2 = \mu$). For each block $B_i$, normalize independently:

$$z_{k+1}[B_i] = \max\!\big(0,\; z_k[B_i] + \tfrac{\beta}{\sqrt{k}} \cdot \tfrac{g_k[B_i]}{\|g_k[B_i]\|_2 + \varepsilon}\big)$$

**Why this works.** The compute-residual block (per-source $\lambda$) and the
channel-residual block ($\mu$) have **very different scales** in our problem.
The $\mu$ block is a single scalar per time step that aggregates over all
$(m,j)$ pairs, while each $\lambda_{m,t}$ aggregates only over tasks of a
single source. Global normalization punishes $\mu$ for having a large absolute
gradient and slows learning of channel multipliers. Per-block normalization
gives each block its own "stride", which the precheck consistently picks as
the best method (see `recommended_subgradient_methods.json`).

**Implementation:**

```python
step = beta / np.sqrt(k)
direction = np.zeros_like(g)
for lo, hi in block_slices:
    block = g[lo:hi]
    direction[lo:hi] = block / (np.linalg.norm(block) + eps)
z_new = np.maximum(0.0, z + step * direction)
```

If `block_slices` is `None`, the method falls back to `normalized_global`.

**State used:** none.

### 5. `adagrad`

**Update:**

$$G_k = G_{k-1} + g_k \odot g_k, \qquad z_{k+1} = \max\!\big(0,\; z_k + \beta \cdot \tfrac{g_k}{\sqrt{G_k} + \varepsilon}\big)$$

(division and square-root are coordinatewise; $\odot$ is the Hadamard product.)

**Why this works.** Coordinates with consistently large gradients get a
*smaller* effective step (because $\sqrt{G_k}$ grows); coordinates that rarely
fire keep a large step. This is the original Adagrad idea — well-suited to
sparse subgradients, which our residual vector tends to produce (many time
steps see zero or near-zero compute violations once multipliers are warm).

**Implementation:**

```python
state.G += g * g
z_new = np.maximum(0.0, z + beta * g / (np.sqrt(state.G) + eps))
```

**State used:** `G` (sum of squared gradients).
**Strengths.** No step-size schedule needed; great early progress.
**Weaknesses.** $G$ grows monotonically, so the effective step decays to zero
— Adagrad can stall on long runs.

### 6. `rmsprop`

**Update:**

$$v_k = \rho\, v_{k-1} + (1-\rho)\, g_k \odot g_k, \qquad z_{k+1} = \max\!\big(0,\; z_k + \beta \cdot \tfrac{g_k}{\sqrt{v_k} + \varepsilon}\big)$$

with default $\rho = 0.9$. Pass `config={"rho": ...}` to override.

**Why this works.** RMSprop replaces Adagrad's cumulative $G$ with an
exponential moving average $v$. The effective step doesn't decay to zero, so
RMSprop keeps learning indefinitely. It is essentially "Adagrad that forgets".

**Implementation:**

```python
rho = float(config.get("rho", 0.9))
state.v = rho * state.v + (1.0 - rho) * g * g
z_new = np.maximum(0.0, z + beta * g / (np.sqrt(state.v) + eps))
```

**State used:** `v` (EMA of squared gradient).

### 7. `adam`

**Update:**

$$
\begin{aligned}
m_k &= \beta_1\, m_{k-1} + (1-\beta_1)\, g_k \\
v_k &= \beta_2\, v_{k-1} + (1-\beta_2)\, g_k \odot g_k \\
\hat{m}_k &= m_k / (1 - \beta_1^k), \quad \hat{v}_k = v_k / (1 - \beta_2^k) \\
z_{k+1} &= \max\!\big(0,\; z_k + \beta \cdot \hat{m}_k / (\sqrt{\hat{v}_k} + \varepsilon)\big)
\end{aligned}
$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$.

**Why this works.** Adam combines RMSprop's second-moment scaling with a
*first*-moment momentum term. The bias correction $1/(1-\beta_1^k)$ removes
the initialization bias of EMAs in early iterations — without it $\hat{m}_1$
would be tiny because $m_1 = (1-\beta_1) g_1$. Adam is widely the most robust
"works out of the box" first-order method, and it does well here too.

**Implementation:**

```python
b1 = float(config.get("b1", 0.9))
b2 = float(config.get("b2", 0.999))
state.m = b1 * state.m + (1.0 - b1) * g
state.v = b2 * state.v + (1.0 - b2) * g * g
bc1 = 1.0 - b1 ** k
bc2 = 1.0 - b2 ** k
m_hat = state.m / bc1
v_hat = state.v / bc2
z_new = np.maximum(0.0, z + beta * m_hat / (np.sqrt(v_hat) + eps))
```

**State used:** `m` (first moment), `v` (second moment).
**Strengths.** Momentum lets it cross flat regions; bias correction stabilizes
early steps.
**Weaknesses.** Two hyperparameters ($\beta_1$, $\beta_2$); momentum can
overshoot on highly non-smooth duals.

### 8. `deflected_sqrt`

**Update:**

$$d_k = \alpha\, d_{k-1} + (1-\alpha)\, g_k, \qquad z_{k+1} = \max\!\big(0,\; z_k + \tfrac{\beta}{\sqrt{k}}\, d_k\big)$$

with default $\alpha = 0.5$ (override via `config={"alpha": ...}`).

**Why this works.** A "deflected" subgradient method low-pass-filters the raw
$g$ before stepping. This dampens the zig-zag pattern that vanilla subgradient
methods exhibit on non-smooth problems (when consecutive $g_k$'s point in
opposite directions, $d_k$ smooths them out). It is conceptually similar to
heavy-ball momentum but without the Polyak parameter.

**Implementation:**

```python
alpha = float(config.get("alpha", 0.5))
d_new = alpha * state.d_prev + (1.0 - alpha) * g
state.d_prev = d_new
step = beta / np.sqrt(k)
z_new = np.maximum(0.0, z + step * d_new)
```

**State used:** `d_prev` (previous direction).

---

## Cutting-plane / bundle methods

These three live in `_cutting_plane_core.py` (the LP/QP machinery) and are
exposed through `cuttingplaneiter_probabilistic.py` and
`cuttingplaneiter_variants.py`. They work on the **simplified** form
$z \in \mathbb{R}^{M+1}_{\ge 0}$ and expand the result to per-time multipliers
before the MGF rollout.

### Common machinery: cuts and the model

After querying the dual oracle at $z_k$, we get $(D_k, s_k)$ with
$D(z) \le D_k + s_k^\top (z - z_k)$ for all $z$ (this is the standard
super-gradient inequality for concave $D$). The *cutting-plane model* is the
piecewise-linear upper envelope of the cuts:

$$\hat{D}_K(z) = \min_{i \le K}\; D_i + s_i^\top (z - z_i).$$

In `_solve_kelley_lp` and friends each cut is written in standard form
$\theta - s_i^\top z \le D_i - s_i^\top z_i$ for a slack variable $\theta$ to
be maximized.

The supergradient and dual value returned by `dual_oracle_*` are in
**unit-consistent raw units**: the dual is
$D(z) = \sum_{m,j} V_{m,j}(0,0; z) - \sum_t \gamma^t [\sum_m \lambda_m c_m + \mu N]$
and the supergradient is
$\partial_{\lambda_m} D = \sum_t \gamma^t (\sum_j \pi^*_{m,j,t} - c_m)$.

### 1. `kelley_bounded`

**Subproblem (LP):**

$$
\max_{\theta, z}\; \theta \quad\text{s.t.}\quad \theta - s_i^\top z \le D_i - s_i^\top z_i \;\;\forall i,\; 0 \le z \le z_{\max}.
$$

**Why this works.** Kelley's method is the prototypical outer-approximation
algorithm: each new cut tightens $\hat{D}_K$, the LP picks the next $z$ at
the highest point of the model, and the model converges to $D$ from above.
Convergence requires the dual to be polyhedral or strictly concave; on a
general concave $D$ Kelley can stall (zig-zag), motivating the next two
methods.

**Implementation:** `_solve_kelley_lp(cuts, z_max, n_dim)` builds the LP
matrices and calls `scipy.optimize.linprog(method='highs')`. Variables are
$[\theta, z_1, \ldots, z_{n_{\rm dim}}]$; to maximize $\theta$ we minimize
$-\theta$.

**Strengths.** Simple; exact for polyhedral duals.
**Weaknesses.** No control on step size; the LP can pick a $z$ far from the
last centre, causing oscillation.

### 2. `trust_region_kelley`

**Subproblem (LP):** same cuts plus an inf-norm trust region around the
current centre $z_c$:

$$\|z - z_c\|_\infty \le \rho_{\text{TR}}$$

(translated to per-coordinate box constraints
$\max(0, z_c - \rho_{\text{TR}}) \le z \le \min(z_{\max}, z_c + \rho_{\text{TR}})$).

**Why this works.** The trust region prevents Kelley from jumping into a
region where the model is unreliable. We update $\rho_{\text{TR}}$ by a
simple rule: shrink to $\rho_{\text{TR}}/2$ when the LP model fails to
improve on $D_{\text{best}}$, expand by $1.5\times$ on a good acceptance
(capped at `z_max.max()`). The centre moves with each accepted step.

**Implementation:** `_solve_trust_region_lp(cuts, z_max, n_dim, z_center, radius)`.
The radius and acceptance logic are in `cutting_plane_loop` in
`_cutting_plane_core.py`.

**Strengths.** Robust to non-polyhedral $D$; tames Kelley's zig-zag.
**Weaknesses.** Need to manage the radius schedule; a too-small radius
freezes progress.

### 3. `proximal_bundle`

**Subproblem (QP, solved with SLSQP):**

$$
\max_{\theta, z}\; \theta - \tfrac{\rho_{\text{prox}}}{2} \|z - z_c\|_2^2 \quad\text{s.t.}\quad \theta - s_i^\top z \le D_i - s_i^\top z_i \;\;\forall i,\; 0 \le z \le z_{\max}.
$$

The bundle is limited to the most recent 50 cuts plus the cut achieving the
best $D_i$ so far (so the iterate cannot lose the best lower-bound it has
seen).

**Why this works.** Bundle methods replace Kelley's hard trust region with a
*quadratic prox* term: the iterate is "pulled" toward $z_c$ proportionally to
the squared distance, which gives a smooth-and-stable update. The QP always
has a unique optimum (because the objective is strongly concave), and the
limited bundle keeps the QP size bounded over long runs. Reference:
Lemaréchal's bundle method literature.

**Implementation:** `_solve_proximal_bundle(cuts, z_max, n_dim, z_center, rho)`
uses `scipy.optimize.minimize(method='SLSQP')` with a `LinearConstraint` for
the cuts and a box for $z$. The objective gradient is provided analytically.

**Strengths.** Smooth iterate trajectory; theoretically the most reliable of
the three for general concave $D$.
**Weaknesses.** QP per iteration is more expensive than an LP; SLSQP is not
a state-of-the-art QP solver but suffices at our problem sizes.

### Generic loop: `cutting_plane_loop`

All three methods share `cutting_plane_loop(oracle, z_init, z_max, ...)`:

1. Query oracle at $z_k$: get $(D_k, s_k)$.
2. Append cut $(D_k, s_k, z_k)$; trim bundle to size 50 if needed (always
   keep the best cut by $D$).
3. Update $z_{\text{best}}$ and $D_{\text{best}}$ if $D_k > D_{\text{best}}$.
4. Solve the method-specific subproblem to get $z_{k+1}$.
5. (Trust region only) shrink/expand radius based on whether the model
   improved.
6. (Proximal bundle only) move centre to $z_{\text{best}}$.
7. Record history.

Returns `(z_best, history)` where history contains `dual_values`,
`model_upper_bounds`, `estimated_gaps`, `oracle_calls`, `runtime_sec`,
`z_norms`, `residual_norms`, `accepted_steps`, `radius_or_rho`.

### Default `z_max`

`cuttingplaneiter_*` builds a generous per-coordinate upper bound:

```python
z_max = 10.0 * (p.max() * w.max() + 1.0)
```

broadcast to length $M+1$. This is a loose primal-value bound (any dual
optimum $z^*$ satisfies $\|z^*\|_\infty \le D^*/\text{min constraint slack}$,
which is below the primal optimum). You can override with `z_max=...`.

---

## Legacy aliases (probabilistic only)

For backward compatibility with the previous `subgradientiter1_probabilistic`
API, three legacy aliases are accepted and mapped to canonical names inside
`_resolve_method`:

| legacy        | canonical            |
| ------------- | -------------------- |
| `constant`    | (single-step ascent with held step `beta` — bypasses the registry) |
| `normalized`  | `normalized_global`  |
| `polyak_like` | `normalized_global`  |

These never write any history other than what their canonical equivalents
produce.

---

## When to use which

| situation                              | recommended method      |
| -------------------------------------- | ----------------------- |
| Default / safe choice                   | `normalized_blocks` (precheck winner) |
| Reproducing the MATLAB port             | `harmonic`              |
| Heavy sparsity in residual              | `adagrad`               |
| Long run with non-decaying step needed  | `rmsprop` or `adam`     |
| Heavily non-smooth (zig-zag)            | `deflected_sqrt`        |
| Polyhedral dual + want global model     | `kelley_bounded`        |
| Non-polyhedral dual                     | `trust_region_kelley` or `proximal_bundle` |
| You want bounded QP per iter            | `proximal_bundle`       |

In practice run `PrecheckSubgradientMethods.py` on the actual problem and
read `recommended_subgradient_methods.json`. Override at runtime with
`INFOCOM_SUBGRADIENT_METHOD=...`.

---

## Cross-references

- Math + paper-vs-port differences: [PROBABILISTIC_IMPLEMENTATION_NOTES.md](PROBABILISTIC_IMPLEMENTATION_NOTES.md)
- Project map: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- Method registry: [optimizer_updates.py](optimizer_updates.py)
- Deterministic driver: [subgradientiter1_variants.py](subgradientiter1_variants.py) + [MGF1_variants.py](MGF1_variants.py)
- Probabilistic driver: [subgradientiter1_probabilistic.py](subgradientiter1_probabilistic.py) + [MGF1_probabilistic.py](MGF1_probabilistic.py)
- Dual oracle (det / prob): [dual_oracle_deterministic.py](dual_oracle_deterministic.py), [dual_oracle_probabilistic.py](dual_oracle_probabilistic.py)
- Cutting plane core + LP/QP solvers: [_cutting_plane_core.py](_cutting_plane_core.py)
- Cutting-plane drivers: [cuttingplaneiter_variants.py](cuttingplaneiter_variants.py), [cuttingplaneiter_probabilistic.py](cuttingplaneiter_probabilistic.py)
- Method comparison: [PrecheckSubgradientMethods.py](PrecheckSubgradientMethods.py)
- Tests: [test_probabilistic.py](test_probabilistic.py) (`test_subgradient_methods_smoke` covers all 8 first-order methods)
