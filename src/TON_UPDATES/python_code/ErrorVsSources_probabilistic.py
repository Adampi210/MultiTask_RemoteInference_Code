"""
ErrorVsSources_probabilistic.py - probabilistic-link ErrorVsSources experiment.

For each q profile, sweep M (sources) and compare:
    MGF probabilistic, MAF probabilistic (pure), MIEF probabilistic
    (reliability-aware), Random probabilistic (gated).

Optional extra rows: MAF reliability-aware and pure MIEF, controlled by
INFOCOM_EXTRA_POLICIES=1.

Configuration via env vars:
    INFOCOM_TITER              (default 1000)
    INFOCOM_MC_TRIALS          (default 10)
    INFOCOM_SEED               (default 0)
    INFOCOM_SOURCES            (default '2,4,6,8,10,12,14,16,18,20')
    INFOCOM_PROFILES           (default = list_q_profiles())
    INFOCOM_EXTRA_POLICIES     (default 0)
    INFOCOM_SUBGRADIENT_METHOD (optional override; default = read recommended
                                 method from recommended_subgradient_methods.json
                                 for the "probabilistic" key)

Outputs (next to this file):
    ErrorVsSources_probabilistic_data.csv
    ErrorVsSources_probabilistic_summary.csv
    ErrorVsSources_probabilistic.png
    q_profiles_probabilistic.npz
"""
import os
import sys
import csv
import json
import time
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MGF1_probabilistic import MGF1_probabilistic
from MAF1_probabilistic import MAF1_probabilistic
from MIEF1_probabilistic import MIEF1_probabilistic
from randpolicy_probabilistic import randpolicy_probabilistic
from probability_profiles_probabilistic import (
    make_q_profile,
    list_q_profiles,
)


TITER = int(os.environ.get("INFOCOM_TITER", "1000"))
MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "10"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
EXTRA_POLICIES = int(os.environ.get("INFOCOM_EXTRA_POLICIES", "0"))


def _resolve_subgradient_method():
    """Read recommended_subgradient_methods.json unless overridden by env."""
    override = os.environ.get("INFOCOM_SUBGRADIENT_METHOD")
    if override:
        return override.strip(), "env override"
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "recommended_subgradient_methods.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                rec = json.load(f)
            method = rec.get("probabilistic", {}).get(
                "recommended_method", "harmonic"
            )
            return method, f"from {os.path.basename(path)}"
        except Exception:
            pass
    return "harmonic", "default fallback (no precheck recommendation found)"


SUBGRAD_METHOD, SUBGRAD_SOURCE = _resolve_subgradient_method()


def _parse_int_list(s, default):
    if not s:
        return default
    return [int(x) for x in s.split(",") if x.strip()]


SOURCES = _parse_int_list(os.environ.get("INFOCOM_SOURCES"),
                          list(range(2, 21, 2)))
PROFILES_ENV = os.environ.get("INFOCOM_PROFILES")
PROFILES = ([p.strip() for p in PROFILES_ENV.split(",") if p.strip()]
            if PROFILES_ENV else list_q_profiles())


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


def _stats(result):
    """Coerce policy output (scalar or dict) to (mean, std)."""
    if isinstance(result, dict):
        return result['mean'], result['std']
    return float(result), 0.0


def main():
    print(f"Running probabilistic ErrorVsSources sweep")
    print(f"  TITER={TITER}, MC_TRIALS={MC_TRIALS}, SEED={SEED}")
    print(f"  SOURCES={SOURCES}")
    print(f"  PROFILES={PROFILES}")
    print(f"  EXTRA_POLICIES={EXTRA_POLICIES}")
    print(f"  SUBGRAD_METHOD={SUBGRAD_METHOD}  ({SUBGRAD_SOURCE})")

    here = os.path.dirname(os.path.abspath(__file__))

    B = 20
    km = 9
    p = _make_penalty(km, B)
    N = 10
    T = 100
    K = T
    gamma = 0.9

    policies = ["MGF", "MAF", "MIEF", "Random"]
    if EXTRA_POLICIES:
        policies += ["MAF_relaware", "MIEF_pure"]

    rows = []
    summary_rows = []
    q_archive = {}   # for npz dump
    t_global = time.perf_counter()

    for profile in PROFILES:
        print(f"\n--- profile: {profile} ---")
        for M in SOURCES:
            n = np.ones((M, km))
            c = np.ones((M, km)) * 2
            w = np.ones((M, km))   # probabilistic experiment: w=1 everywhere
            q, meta = make_q_profile(profile, M, km, seed=SEED)
            q_archive[f"{profile}_M{M}"] = q

            t_pair = time.perf_counter()

            res = {}
            res["MGF"] = MGF1_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                titer=TITER, subgradient_method=SUBGRAD_METHOD,
                seed=SEED, mc_trials=MC_TRIALS, verbose=False,
            )
            res["MAF"] = MAF1_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                seed=SEED, mc_trials=MC_TRIALS,
                reliability_aware=False, verbose=False,
            )
            res["MIEF"] = MIEF1_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                seed=SEED, mc_trials=MC_TRIALS,
                reliability_aware=True, verbose=False,
            )
            res["Random"] = randpolicy_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                seed=SEED, mc_trials=MC_TRIALS,
                gated=True, verbose=False,
            )
            if EXTRA_POLICIES:
                res["MAF_relaware"] = MAF1_probabilistic(
                    M, N, km, T, B, K, n, c, w, gamma, p, q,
                    seed=SEED, mc_trials=MC_TRIALS,
                    reliability_aware=True, verbose=False,
                )
                res["MIEF_pure"] = MIEF1_probabilistic(
                    M, N, km, T, B, K, n, c, w, gamma, p, q,
                    seed=SEED, mc_trials=MC_TRIALS,
                    reliability_aware=False, verbose=False,
                )

            pair_runtime = time.perf_counter() - t_pair
            print(f"  M={M:>3d}: " + ", ".join(
                f"{name}={_stats(res[name])[0]:.3g}"
                for name in policies
            ) + f"   ({pair_runtime:.1f}s)")

            for policy in policies:
                mean_e, std_e = _stats(res[policy])
                rows.append({
                    "profile": profile,
                    "M": M,
                    "policy": policy,
                    "mean_error": mean_e,
                    "std_error": std_e,
                    "seed": SEED,
                    "titer": TITER,
                    "mc_trials": MC_TRIALS,
                    "subgradient_method": SUBGRAD_METHOD,
                    "q_min": meta["q_min"],
                    "q_max": meta["q_max"],
                    "q_mean": meta["q_mean"],
                    "q_std": meta["q_std"],
                })

        for policy in policies:
            errors = [r["mean_error"] for r in rows
                      if r["profile"] == profile and r["policy"] == policy]
            summary_rows.append({
                "profile": profile,
                "policy": policy,
                "mean_over_M": float(np.mean(errors)),
                "median_over_M": float(np.median(errors)),
                "max_over_M": float(np.max(errors)),
                "min_over_M": float(np.min(errors)),
            })

    total_runtime = time.perf_counter() - t_global
    print(f"\nTotal runtime: {total_runtime:.1f}s")

    # --- write CSVs ---
    data_csv = os.path.join(here, "ErrorVsSources_probabilistic_data.csv")
    with open(data_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {data_csv}")

    summary_csv = os.path.join(here, "ErrorVsSources_probabilistic_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_csv}")

    # --- q matrices ---
    np.savez(os.path.join(here, "q_profiles_probabilistic.npz"), **q_archive)

    # --- plot: one subplot per profile ---
    n_profiles = len(PROFILES)
    n_cols = min(3, n_profiles)
    n_rows = (n_profiles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    style = {
        "MGF":          ("g-", "MGF (prob.)"),
        "MAF":          ("b*-", "MAF (pure)"),
        "MIEF":         ("ms--", "MIEF (rel-aware)"),
        "Random":       ("ro-", "Random (gated)"),
        "MAF_relaware": ("c^-", "MAF (rel-aware)"),
        "MIEF_pure":    ("yP--", "MIEF (pure)"),
    }
    for idx, profile in enumerate(PROFILES):
        ax = axes[idx // n_cols][idx % n_cols]
        for policy in policies:
            ys = [r["mean_error"] for r in rows
                  if r["profile"] == profile and r["policy"] == policy]
            stds = [r["std_error"] for r in rows
                    if r["profile"] == profile and r["policy"] == policy]
            fmt, label = style.get(policy, ("k-", policy))
            ax.errorbar(SOURCES, ys, yerr=stds, fmt=fmt,
                        label=label, linewidth=1.5, markersize=6, capsize=3)
        ax.set_title(profile)
        ax.set_xlabel("M (sources)")
        ax.set_ylabel("Discounted error")
        try:
            ax.set_yscale("log")
        except Exception:
            pass
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    # Hide unused axes.
    for idx in range(n_profiles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    out = os.path.join(here, "ErrorVsSources_probabilistic.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
