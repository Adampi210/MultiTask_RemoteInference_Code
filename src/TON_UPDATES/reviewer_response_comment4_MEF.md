# Response to Comment 4 — Maximum (Instantaneous) Error First as a baseline

> **Comment 4.** *The Introduction states that inference error is not always monotone in AoI.
> Consequently, the Maximum Age First (MAF) policy seems unreasonable and an unfair benchmark.
> A more direct and arguably stronger baseline is to prioritize by the instantaneous value
> $w_{m,j}\,p_{m,j}(\Delta_{m,j}(t))$. Can the authors comment on using this as a benchmark and
> how well it might perform?*

## Summary of our response

We thank the reviewer for this excellent and well-motivated suggestion. The reviewer is correct on
both counts: (i) because the inference-error functions $p_{m,j}(\cdot)$ are **not necessarily
monotone** in AoI, a pure age-based rule such as MAF can prioritize a task whose error is already
small (or even decreasing) simply because its AoI is large; and (ii) prioritizing by the
**instantaneous weighted error** $w_{m,j}\,p_{m,j}(\Delta_{m,j}(t))$ is a strictly more sensible,
error-aware greedy rule that directly attacks the objective rather than a proxy for it.

We have therefore **added exactly this policy as a new baseline**, which we call **Maximum Error
First (MEF)**. At each slot it greedily schedules the tasks with the largest current weighted
error $w_{m,j}\,p_{m,j}(\Delta_{m,j}(t))$, subject to the same per-source compute caps $C_m$ and
total channel budget $N$ as every other policy. MEF is, by construction, the "one-step-greedy on
the true objective" policy the reviewer describes.

Our findings, summarized below and now reported in the revision, are:

1. **MEF is indeed a far stronger baseline than MAF.** Across our sweeps MEF reduces the
   discounted error to roughly **4%–13% of MAF's** in the operating regime (i.e. MEF is on the
   order of **8×–25× better than MAF**), and to **~3% of the Random policy's** error. This
   confirms the reviewer's intuition and we agree MAF alone is a weak benchmark when errors are
   non-monotone.
2. **MGF (our policy) still outperforms MEF** wherever resources are not pathologically scarce: in
   the interior of all sweeps MGF's error is **~73%–87% of MEF's** (i.e. MGF is **~13%–27% lower**
   than the reviewer's greedy baseline). The reason is structural: MEF is *myopic* — it spends
   the budget on whatever has the largest error *right now*, whereas MGF schedules by the
   **discounted gain index** derived from the Lagrangian/value-function, which accounts for how
   each transmission reduces *future* error over the horizon (and, in the unreliable-channel
   extension below, the probability that the transmission actually succeeds). MGF will, for
   example, *not* burn a scarce channel on a high-error task whose penalty is near-flat, and will
   instead refresh a task whose error is climbing steeply.
3. **The one regime where MEF can edge out MGF is extreme channel scarcity** (e.g. $N=2$ with
   $M=20$, $rk_m=9$), where almost nothing can be scheduled and "fix the single worst thing now"
   is close to optimal; there MEF is ~15%–19% below MGF. As soon as the budget is large enough to
   make scheduling decisions non-trivial ($N \ge 4$), MGF's lookahead advantage reappears and
   grows.

Net effect for the paper: adding MEF *strengthens* our claims. The reviewer's stronger baseline
beats MAF and Random handily, yet MGF still dominates it across the practically relevant range —
precisely because MGF optimizes the discounted *future* error rather than the instantaneous one.

---

## 1. The new MEF baseline (the reviewer's suggested policy)

**Priority.** At slot $t$, for every source–task pair $(m,j)$,

$$
\text{score}_{m,j}(t) \;=\; w_{m,j}\,p_{m,j}\!\bigl(\Delta_{m,j}(t)\bigr).
$$

**Scheduling.** Tasks are scheduled greedily in descending order of $\text{score}_{m,j}(t)$,
subject to the identical resource constraints used by MGF/MAF/Random:

$$
\sum_{j} a_{m,j}(t) \le C_m \quad \forall m,
\qquad
\sum_{m,j} a_{m,j}(t)\,n_{m,j} \le N .
$$

This is implemented in `MIEF1.py` (deterministic) and `MIEF1_probabilistic.py` (unreliable-channel
extension), reusing the **same** greedy resource-feasibility loop (`greedy_scheduler.greedy_select`)
as MAF, so the only difference between MAF and MEF is the priority key (age $\Delta_{m,j}+1$ for
MAF vs. weighted error $w_{m,j}\,p_{m,j}(\Delta_{m,j})$ for MEF). This isolates exactly the effect
the reviewer asks about.

**Relationship to MAF.** When $p_{m,j}(\delta)=\delta$ and weights are equal, MEF and MAF coincide.
They diverge precisely when the error is non-monotone or task-heterogeneous — the case our paper is
about — which is why MEF is the more faithful benchmark.

**Relationship to MGF.** MGF replaces the *instantaneous* score with the *discounted gain index*
$a_{m,j}(\Delta)=Q_{\text{no}}(\Delta)-Q_{\text{sched}}(\Delta)$ obtained from the relaxed Bellman
recursion, i.e. the expected reduction in **future** discounted error from scheduling now. MEF is
the special case of "score by the current penalty only," with no lookahead.

---

## 2. The new (probabilistic / unreliable-channel) experimental setup

To make the comparison more realistic and more discriminating between policies, we also evaluate
all baselines under an **unreliable-channel extension** of the model. (In the deterministic limit
this reduces algebraically to the model in the paper.)

**Model.** Each scheduled pair $(m,j)$ is delivered only with probability $q_{m,j}$ (an independent
Bernoulli per slot); with probability $1-q_{m,j}$ the transmission is attempted but **fails**.
Resource feasibility is always checked against the *attempted* schedule $\pi$ (the budget is spent
on the attempt, not on the realized delivery). The AoI update per pair is

$$
\Delta_{m,j}(t+1)=
\begin{cases}
0, & \text{scheduled and delivered (prob. } q_{m,j}\text{),}\\[2pt]
\min(\Delta_{m,j}(t)+1,\;B-1), & \text{not scheduled, or scheduled but dropped.}
\end{cases}
$$

**Policies compared (four).**

| Policy | Priority at slot $t$ | Lookahead? | Reliability-aware? |
|---|---|---|---|
| Random | uniform pick (gated by caps) | no | no |
| MAF | $\Delta_{m,j}(t)+1$ | no | no |
| **MEF** (new — reviewer's suggestion) | $q_{m,j}\,w_{m,j}\,p_{m,j}(\Delta_{m,j}(t))$ | no | yes (expected one-step error) |
| MGF (ours) | discounted gain index from the relaxed Bellman value function | **yes** | yes |

In the unreliable setting MEF uses the **expected** instantaneous error $q_{m,j}\,w_{m,j}\,p_{m,j}(\Delta)$,
i.e. it already gets the benefit of being reliability-aware — this is the strongest reasonable
form of the reviewer's baseline. MGF's gain index uses the paper's relaxed Bellman value with the
$q\,V(0)+(1-q)\,V(\min(\Delta+1,B-1))$ transition, so it accounts for *both* future error growth
*and* the chance the transmission fails.

**Multiplier learning.** The Lagrange multipliers $(\lambda_m,\mu)$ for MGF are learned by the
projected dual-ascent / value-function procedure of the paper (`subgradientiter1_probabilistic.py`),
using the expected Bellman recursion so the dual is deterministic in $q$.

**Common simulation parameters** (matching the synthetic experiments in the paper):
AoI penalties $p_{m,j}(\delta)\in\{\delta,\;10\log\delta,\;\exp(0.5\delta)\}$ assigned to one-third
of the tasks each ($j \bmod 3$); AoI bound $B=20$; horizon $T=100$; discount $\gamma=0.9$; unit
channel cost $n_{m,j}=1$; per-source compute cap $C_m=2$. Base configuration $M=20$, $N=10$,
$rk_m=9$, with one dimension swept at a time. Each plotted point is averaged over **Monte-Carlo
trials** (the channel realizations are random), and we report two weight regimes:

- **original (heterogeneous) weights** — the paper's pattern: $w_{m,j}=1$ for the first half of the
  sources, $0.01$ for the rest; and
- **uniform weights** ($w_{m,j}=1$ everywhere), so that reliability — not weighting — drives the
  ordering.

The conclusions below hold in both weight regimes.

---

## 3. The two channel-reliability profiles we report

The unreliable channel is specified by the matrix $q=\{q_{m,j}\}$. We deliberately chose **two**
contrasting profiles that stress the difference between a myopic error-greedy rule (MEF) and a
lookahead policy (MGF) in complementary ways:

### (a) `uniform_very_wide` — $q_{m,j}\sim \text{Uniform}(0.20,\,0.99)$

Every link has a different reliability, spread across almost the entire range. There are no
"perfect" links to hide behind: a high-error task may sit on a flaky channel, so simply chasing the
largest instantaneous error (MEF) repeatedly wastes attempts on links that are unlikely to deliver,
whereas the gain index discounts those attempts by $q_{m,j}$ *and* weighs the future. This profile
tests **graceful degradation under broad heterogeneity**.

### (b) `bimodal_q1_vs_lossy_30_70` — 30% perfect links ($q=1$), 70% lossy ($q\sim\text{Uniform}(0.30,0.50)$)

A structured, "good vs. bad channel" world: 30% of links are perfect, the remaining 70% are
unreliable. This is the more adversarial case for a myopic rule, because the largest *instantaneous*
error often lives on a *lossy* link; MEF will keep selecting it (it does scale by $q$, but only
one-step), while MGF's value-function reasoning more aggressively reallocates the budget toward
links where a transmission both succeeds and durably suppresses future error. This profile tests
**behavior when reliability is sharply clustered**.

Together these two profiles bracket the realistic middle ground (a continuum of link qualities vs.
a clean good/bad split), and the policy ordering is consistent across both.

---

## 4. Results

Representative numbers below are from the **`ErrorVsChannel`** sweep ($M=20$, $rk_m=9$, original
heterogeneous weights), discounted sum of errors (lower is better). Full curves for all three
sweeps (`ErrorVsChannel`, `ErrorVsSources`, `ErrorVsTasks`), both profiles, and both weight modes
are in the revision; the script is `MatlabStylePlots_probabilistic.py` and the raw values are in
`data/probabilistic/matlab_style_*.csv`.

**Profile (a) `uniform_very_wide`:**

| $N$ | Random | MAF | MGF (ours) | MEF (reviewer) | MEF/MAF | MGF/MEF |
|----:|-------:|------:|-----------:|---------------:|--------:|--------:|
| 2  | 5989 | 6074 | 2731 | **2214** | 0.36 | 1.23 |
| 4  | 5336 | 4851 | **363**  | 422  | 0.09 | 0.86 |
| 10 | 3793 | 3242 | **97**   | 127  | 0.04 | 0.76 |
| 20 | 2169 | 573  | **58**   | 77   | 0.13 | 0.76 |

**Profile (b) `bimodal_q1_vs_lossy_30_70`:**

| $N$ | Random | MAF | MGF (ours) | MEF (reviewer) | MEF/MAF | MGF/MEF |
|----:|-------:|------:|-----------:|---------------:|--------:|--------:|
| 2  | 6070 | 6543 | 3071 | **2660** | 0.41 | 1.15 |
| 4  | 5451 | 4942 | **554**  | 633  | 0.13 | 0.87 |
| 10 | 4123 | 3245 | **108**  | 136  | 0.04 | 0.80 |
| 20 | 2523 | 819  | **60**   | 79   | 0.10 | 0.76 |

**Reading the tables.**

- **MEF $\gg$ MAF and Random.** Outside the saturated $N=2$ corner, MEF's error is ~4%–13% of
  MAF's and ~3% of Random's — i.e. the reviewer's instantaneous-error baseline is **8×–25×
  better than MAF**. We agree this is the more honest benchmark and we now report it as such.
- **MGF $<$ MEF in the operating regime.** For $N\ge 4$, MGF/MEF $\approx 0.73$–$0.87$: MGF is
  consistently **~13%–27% lower error** than MEF. The gap is the value of lookahead — scheduling
  by *expected discounted future* error reduction rather than *current* error. The same ordering
  holds in the `ErrorVsSources` and `ErrorVsTasks` sweeps and under uniform weights.
- **Extreme scarcity ($N=2$) is MEF's only win.** When the channel is so tight that essentially one
  task is served per slot, the greedy "fix the single worst error now" rule is near-optimal and MEF
  is ~15%–19% below MGF. This corner is degenerate (all policies are within a small factor of each
  other and far above their unconstrained values); it disappears by $N=4$.

---

## 5. Proposed revision text

We will (i) replace the discussion of MAF as the primary intelligent baseline with the
acknowledgement that, because $p_{m,j}$ is non-monotone, an instantaneous-error rule is more
appropriate; (ii) add **MEF — Maximum Error First, prioritizing $w_{m,j}\,p_{m,j}(\Delta_{m,j}(t))$**
— as a baseline in Section VII alongside MAF and Random; and (iii) add the two reliability profiles
above and the corresponding figures. The accompanying text will make the point that MEF is a strong,
error-aware greedy baseline that dominates MAF/Random, and that MGF's remaining advantage over MEF
is precisely attributable to its use of the discounted *gain index* (lookahead over the horizon and,
under unreliable channels, over delivery success) rather than the instantaneous error.

We thank the reviewer again; this comment led to a stronger and more convincing benchmark suite.
