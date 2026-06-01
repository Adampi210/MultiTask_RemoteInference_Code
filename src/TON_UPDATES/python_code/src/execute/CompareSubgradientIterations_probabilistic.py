"""
CompareSubgradientIterations_probabilistic.py - convergence of the probabilistic
MGF objective vs. the number of subgradient episodes (iterations), for EVERY
subgradient method.

Where CompareSubgradient_probabilistic.py benchmarks the methods at a single
fixed iteration count, this script sweeps the number of episodes
    iters in {1, 2, 4, 8, 16, 32, 64, 128}
and, for each (method, iters, q-profile), learns the multipliers from scratch to
that iteration count, builds the gain table, rolls out probabilistic MGF
(mc_trials trials) and records the discounted-error objective. The result is one
convergence curve per method (one subplot per q-profile).

Methods compared (all the first-order rules we have + the MATLAB M-step):
    harmonic, sqrt, normalized_global, normalized_blocks, adagrad, rmsprop,
    adam, deflected_sqrt, episode1_mstep
Cutting-plane / bundle methods (kelley_bounded, trust_region_kelley,
proximal_bundle) are slower and OFF by default; enable with
INFOCOM_COMPARE_INCLUDE_CUTTING_PLANES=1.

Weights are uniform (w = 1) so the curves reflect the optimizer, not weight
heterogeneity -- consistent with CompareSubgradient_probabilistic.py.

Configuration via env vars:
    INFOCOM_COMPARE_ITERATIONS  (default '1,2,4,8,16,32,64,128')
    INFOCOM_PROFILES            (default 'uniform_wide,uniform_low,bimodal_extreme')
    INFOCOM_COMPARE_M           (default 10)   - number of sources
    INFOCOM_MC_TRIALS           (default 10)
    INFOCOM_SEED                (default 0)
    INFOCOM_COMPARE_INCLUDE_CUTTING_PLANES (default 0)
    INFOCOM_COMPARE_METHODS     (optional override; comma-separated method list)

Writes:
    data/probabilistic/CompareSubgradientIterations_probabilistic_data.csv
    plots/probabilistic/CompareSubgradientIterations_probabilistic.png
"""
import os
import csv
import time
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths

from MGF1_probabilistic import MGF1_probabilistic
from optimizer_updates import get_default_subgradient_methods
from probability_profiles_probabilistic import make_q_profile
from experiment_utils import make_synthetic_p


SEED = int(os.environ.get("INFOCOM_SEED", "0"))
MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "10"))
COMPARE_M = int(os.environ.get("INFOCOM_COMPARE_M", "10"))
INCLUDE_CP = int(os.environ.get("INFOCOM_COMPARE_INCLUDE_CUTTING_PLANES", "0"))

_CUTTING_PLANE_METHODS = ["kelley_bounded", "trust_region_kelley",
                          "proximal_bundle"]


def _parse_int_list(s, default):
    if not s:
        return list(default)
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str_list(s, default):
    if not s:
        return list(default)
    return [x.strip() for x in s.split(",") if x.strip()]


def methods_list():
    """All subgradient methods to compare (env-overridable)."""
    override = os.environ.get("INFOCOM_COMPARE_METHODS")
    if override:
        return _parse_str_list(override, [])
    methods = list(get_default_subgradient_methods()) + ["episode1_mstep"]
    if INCLUDE_CP:
        methods += _CUTTING_PLANE_METHODS
    return methods


def compare_config():
    """Config for this experiment (also consumed by ExportExperimentParameters)."""
    return dict(
        stem="CompareSubgradientIterations_probabilistic",
        M=COMPARE_M, N=10, km=9, B=20, T=100, gamma=0.9,
        n_value=1.0, c_value=2.0, weights="ones (uniform)",
        penalty="synthetic 9-task (j%3: linear / 10*log / exp(0.5 d))",
        iterations=_parse_int_list(os.environ.get("INFOCOM_COMPARE_ITERATIONS"),
                                   [1, 2, 4, 8, 16, 32, 64, 128]),
        methods=methods_list(),
        profiles=_parse_str_list(
            os.environ.get("INFOCOM_PROFILES"),
            ["uniform_wide", "uniform_low", "bimodal_extreme"]),
        mc_trials=MC_TRIALS, seed=SEED,
    )


def _stats(result):
    if isinstance(result, dict):
        return float(result["mean"]), float(result["std"])
    return float(result), 0.0


def main():
    cfg = compare_config()
    M, N, km = cfg["M"], cfg["N"], cfg["km"]
    B, T, gamma = cfg["B"], cfg["T"], cfg["gamma"]
    K = T
    iterations = cfg["iterations"]
    methods = cfg["methods"]
    profiles = cfg["profiles"]

    print("CompareSubgradientIterations (probabilistic MGF objective vs #episodes)")
    print(f"  M={M}, N={N}, km={km}, T={T}, B={B}, gamma={gamma}")
    print(f"  iterations: {iterations}")
    print(f"  methods   : {methods}")
    print(f"  profiles  : {profiles}")
    print(f"  MC_TRIALS={MC_TRIALS}, SEED={SEED}")

    p = make_synthetic_p(km, B)
    n = np.ones((M, km))
    c = np.ones((M, km)) * 2
    w = np.ones((M, km))   # uniform weights: isolate the optimizer

    rows = []
    # curves[profile][method] = (xs, ys, yerr)
    curves = {prof: {} for prof in profiles}
    t_global = time.perf_counter()

    for profile in profiles:
        q, qmeta = make_q_profile(profile, M, km, seed=SEED)
        print(f"\n--- profile: {profile} "
              f"(q in [{qmeta['q_min']:.2f}, {qmeta['q_max']:.2f}]) ---")
        for method in methods:
            ys, yerr = [], []
            for it in iterations:
                t0 = time.perf_counter()
                try:
                    res = MGF1_probabilistic(
                        M, N, km, T, B, K, n, c, w, gamma, p, q,
                        titer=it, subgradient_method=method,
                        seed=SEED, mc_trials=MC_TRIALS, verbose=False,
                    )
                    mean_obj, std_obj = _stats(res)
                    ok = True
                except Exception as exc:   # keep one bad method from killing all
                    mean_obj, std_obj, ok = float("nan"), float("nan"), False
                    print(f"    {method} @ {it}: ERROR {exc}")
                runtime = time.perf_counter() - t0
                ys.append(mean_obj)
                yerr.append(std_obj)
                rows.append(dict(
                    profile=profile, method=method, iterations=it,
                    mean_objective=mean_obj, std_objective=std_obj,
                    runtime_sec=runtime, ok=ok,
                    M=M, N=N, km=km, B=B, T=T, gamma=gamma,
                    mc_trials=MC_TRIALS, seed=SEED,
                    q_min=qmeta["q_min"], q_max=qmeta["q_max"],
                    q_mean=qmeta["q_mean"], q_std=qmeta["q_std"],
                ))
            curves[profile][method] = (list(iterations), ys, yerr)
            best = np.nanmin(ys) if len(ys) else float("nan")
            print(f"  {method:>18s}: final={ys[-1]:.4g}  best={best:.4g}")

    print(f"\nTotal runtime: {time.perf_counter() - t_global:.1f}s")

    # ---- CSV ----
    csv_out = paths.data_path(
        "CompareSubgradientIterations_probabilistic_data.csv", probabilistic=True)
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_out}")

    # ---- Plot: one subplot per profile, one line per method ----
    n_prof = len(profiles)
    n_cols = min(3, n_prof)
    n_rows = (n_prof + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "d"]
    for idx, profile in enumerate(profiles):
        ax = axes[idx // n_cols][idx % n_cols]
        for mi, method in enumerate(methods):
            xs, ys, yerr = curves[profile][method]
            ax.errorbar(xs, ys, yerr=yerr, label=method,
                        color=cmap(mi % 10),
                        marker=markers[mi % len(markers)],
                        markersize=5, linewidth=1.5, capsize=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(iterations)
        ax.set_xticklabels([str(i) for i in iterations])
        try:
            ax.set_yscale("log")
        except Exception:
            pass
        ax.set_xlabel("Subgradient episodes (iterations)")
        ax.set_ylabel("MGF discounted error")
        ax.set_title(profile)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="best", ncol=2)
    for idx in range(n_prof, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"Subgradient method comparison vs #episodes "
                 f"(probabilistic MGF, M={M}, km={km}, w=1)", fontsize=13)
    plt.tight_layout()
    out = paths.plot_path("CompareSubgradientIterations_probabilistic.png",
                          probabilistic=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
