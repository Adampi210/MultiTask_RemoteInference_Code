"""
CompareSubgradient_probabilistic.py - benchmark of dual-update methods for
the probabilistic-link Lagrangian.

Methods compared: harmonic, sqrt, normalized, adagrad, adam (+constant if
INFOCOM_INCLUDE_CONSTANT=1).

For each (method, q_profile) pair we:
  1. Run subgradientiter1_probabilistic with history=True.
  2. Build the gain table from the learned multipliers.
  3. Roll out MGF probabilistic with mc_trials trials and record mean +/- std.

Metrics recorded per (method, profile):
  - final mean MGF objective and its std over MC trials
  - subgradient runtime (seconds)
  - final max|lambda| / max|mu|
  - tail stability of multipliers (mean abs change over last 20% iters)
  - average positive compute / channel constraint residual during training

Configuration via env vars:
  INFOCOM_TITER             (default 100)
  INFOCOM_MC_TRIALS         (default 3)
  INFOCOM_SEED              (default 0)
  INFOCOM_INCLUDE_CONSTANT  (default 0)
  INFOCOM_M                 (default '6,10')
  INFOCOM_PROFILES          (default 'two_cluster_60_95,uniform_wide,bimodal_extreme')

Outputs (next to this file):
  CompareSubgradient_probabilistic_data.csv
  CompareSubgradient_probabilistic.png
"""
import os
import sys
import csv
import time
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subgradientiter1_probabilistic import (
    subgradientiter1_probabilistic,
    load_multipliers_probabilistic,
    _batched_gain_table_probabilistic,
)
from MGF1_probabilistic import _single_rollout as _mgf_rollout
from probability_profiles_probabilistic import make_q_profile


TITER = int(os.environ.get("INFOCOM_TITER", "100"))
MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "3"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
INCLUDE_CONST = int(os.environ.get("INFOCOM_INCLUDE_CONSTANT", "0"))


def _parse_list(s, default):
    if not s:
        return default
    return [x.strip() for x in s.split(",") if x.strip()]


M_VALS = [int(x) for x in _parse_list(os.environ.get("INFOCOM_M"), ["6", "10"])]
PROFILES = _parse_list(
    os.environ.get("INFOCOM_PROFILES"),
    ["two_cluster_60_95", "uniform_wide", "bimodal_extreme"],
)


def _make_penalty(km, B):
    p = np.zeros((km, B))
    for j in range(1, km + 1):
        if j % 3 == 0:
            p[j - 1, :B] = np.arange(1, B + 1)
        elif j % 3 == 1:
            p[j - 1, :B] = 10 * np.log(np.arange(1, B + 1))
        else:
            p[j - 1, :B] = np.exp(0.5 * np.arange(1, B + 1))
    return p


def _as_1d_c(c, M):
    c = np.asarray(c)
    if c.ndim == 1:
        return c.astype(float)
    return c[:, 0].astype(float)


def _tail_stability(values, frac=0.2):
    """Mean absolute consecutive change over the last `frac` of `values`."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 4:
        return float("nan")
    k = max(2, int(len(arr) * frac))
    tail = arr[-k:]
    return float(np.mean(np.abs(np.diff(tail))))


def _evaluate_objective(asource1, M, T, K, B, km, n, c_use, N, w, p,
                        gamma, q, base_rng, mc_trials):
    """Run MGF rollouts with given gain table, return (mean, std, trials)."""
    Delta_init = np.zeros((M, km), dtype=np.int64)
    seeds = base_rng.integers(0, 2**31 - 1, size=mc_trials)
    out = np.zeros(mc_trials)
    for trial in range(mc_trials):
        rng = np.random.default_rng(int(seeds[trial]))
        out[trial] = _mgf_rollout(
            asource1, Delta_init, M, T, K, B, km, n, c_use, N,
            w, p, gamma, q, rng,
        )
    return float(out.mean()), float(out.std(ddof=1)) if mc_trials > 1 else 0.0


def main():
    methods = ["harmonic", "sqrt", "normalized", "adagrad", "adam"]
    if INCLUDE_CONST:
        methods.append("constant")

    print(f"Comparing subgradient methods: {methods}")
    print(f"  TITER={TITER}  MC_TRIALS={MC_TRIALS}  SEED={SEED}")
    print(f"  M_VALS={M_VALS}  PROFILES={PROFILES}")

    km = 9
    B = 20
    T = 100
    K = T
    N = 10
    gamma = 0.9
    p = _make_penalty(km, B)

    rows = []
    here = os.path.dirname(os.path.abspath(__file__))

    for M in M_VALS:
        n_arr = np.ones((M, km))
        c = np.ones((M, km)) * 2
        c_use = _as_1d_c(c, M)
        w = np.ones((M, km))

        for profile in PROFILES:
            q, qmeta = make_q_profile(profile, M, km, seed=SEED)
            print(f"\n--- M={M}, profile={profile} ---")

            for method in methods:
                t0 = time.perf_counter()
                _, hist = subgradientiter1_probabilistic(
                    M, N, T, B, gamma, p, km, w, n_arr, c, q,
                    titer=TITER, method=method,
                    seed=SEED, verbose=False, history=True,
                    save_path=os.path.join(
                        here,
                        f"multipliers_probabilistic_{method}_{profile}_M{M}.mat",
                    ),
                )
                training_runtime = time.perf_counter() - t0

                lambdasource, mu = load_multipliers_probabilistic(
                    os.path.join(
                        here,
                        f"multipliers_probabilistic_{method}_{profile}_M{M}.mat",
                    )
                )
                asource1 = _batched_gain_table_probabilistic(
                    lambdasource, mu, B, T, gamma, w, p, M, km, q, n_arr,
                )

                eval_rng = np.random.default_rng(SEED + 17)
                mean_obj, std_obj = _evaluate_objective(
                    asource1, M, T, K, B, km, n_arr, c_use, N, w, p,
                    gamma, q, eval_rng, MC_TRIALS,
                )

                tail_lam = _tail_stability(hist['lambda_max'])
                tail_mu = _tail_stability(hist['mu_max'])
                avg_comp = float(np.mean(hist['compute_violation']))
                avg_chan = float(np.mean(hist['channel_violation']))

                print(f"  {method:>11s}: obj={mean_obj:.5g}+-{std_obj:.3g}  "
                      f"runtime={training_runtime:.2f}s  "
                      f"|lam|max={hist['lambda_max'][-1]:.3f}  "
                      f"|mu|max={hist['mu_max'][-1]:.3f}  "
                      f"tail_lam={tail_lam:.3e}  "
                      f"viol(c,N)=({avg_comp:.3e},{avg_chan:.3e})")

                rows.append({
                    "M": M,
                    "profile": profile,
                    "method": method,
                    "titer": TITER,
                    "mc_trials": MC_TRIALS,
                    "seed": SEED,
                    "mean_objective": mean_obj,
                    "std_objective": std_obj,
                    "training_runtime_s": training_runtime,
                    "final_lambda_max": hist['lambda_max'][-1],
                    "final_mu_max": hist['mu_max'][-1],
                    "tail_stability_lambda": tail_lam,
                    "tail_stability_mu": tail_mu,
                    "avg_compute_violation": avg_comp,
                    "avg_channel_violation": avg_chan,
                })

    # --- CSV ---
    csv_path = os.path.join(here, "CompareSubgradient_probabilistic_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # --- Ranking ---
    print("\nRanking by mean objective (lower = better):")
    avg_by_method = {}
    for method in methods:
        objs = [r["mean_objective"] for r in rows if r["method"] == method]
        avg_by_method[method] = float(np.mean(objs))
    for method, val in sorted(avg_by_method.items(), key=lambda kv: kv[1]):
        print(f"  {method:>11s}: avg obj = {val:.5g}")

    print("\nRanking by training runtime (lower = better):")
    rt_by_method = {}
    for method in methods:
        rts = [r["training_runtime_s"] for r in rows if r["method"] == method]
        rt_by_method[method] = float(np.mean(rts))
    for method, val in sorted(rt_by_method.items(), key=lambda kv: kv[1]):
        print(f"  {method:>11s}: avg runtime = {val:.2f}s")

    # --- Plot ---
    n_profiles = len(PROFILES)
    n_M = len(M_VALS)
    fig, axes = plt.subplots(n_M, n_profiles,
                             figsize=(4 * n_profiles, 3.5 * n_M),
                             squeeze=False)
    x = np.arange(len(methods))
    for i, M in enumerate(M_VALS):
        for j, profile in enumerate(PROFILES):
            ax = axes[i][j]
            ys = []
            yerr = []
            for method in methods:
                r = [r for r in rows
                     if r["M"] == M and r["profile"] == profile
                     and r["method"] == method][0]
                ys.append(r["mean_objective"])
                yerr.append(r["std_objective"])
            ax.bar(x, ys, yerr=yerr, capsize=4,
                   color=["C0", "C1", "C2", "C3", "C4", "C5"][:len(methods)])
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right")
            ax.set_title(f"M={M}, {profile}")
            ax.set_ylabel("MGF probabilistic obj")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(here, "CompareSubgradient_probabilistic.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
