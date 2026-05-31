"""
_probabilistic_sweep_helpers.py - shared scaffolding for the probabilistic
experiment scripts (ErrorVsSources / ErrorVsChannel / ErrorVsChannelsmodel /
ErrorVsTasks).

Each sweep script defines (a) what to sweep over, (b) how to build the per-
sweep-value problem (n, c, w, p, km, M, N, ...), and (c) the output filename
stem. Everything else (q profile loop, policy calls, CSV + plot output, env
configuration, recommended-method resolution) lives here.
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
from probability_profiles_probabilistic import make_q_profile, list_q_profiles
from paths import (
    data_path,
    plot_path,
    DATA_DETERMINISTIC_DIR,
    DATA_PROBABILISTIC_DIR,
    PLOTS_PROBABILISTIC_DIR,
)


# ---------------------------------------------------------------------------
# Env-var configuration (single source of truth across all sweep scripts)
# ---------------------------------------------------------------------------

TITER = int(os.environ.get("INFOCOM_TITER", "1000"))
MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "10"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
EXTRA_POLICIES = int(os.environ.get("INFOCOM_EXTRA_POLICIES", "0"))


def resolve_subgradient_method():
    """Read recommended_subgradient_methods.json unless overridden by env."""
    override = os.environ.get("INFOCOM_SUBGRADIENT_METHOD")
    if override:
        return override.strip(), "env override"
    path = data_path("recommended_subgradient_methods.json", probabilistic=False)
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


def parse_profiles(default=None):
    env = os.environ.get("INFOCOM_PROFILES")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    return list(default) if default is not None else list_q_profiles()


def stats(result):
    """Coerce policy output (scalar or dict) to (mean, std)."""
    if isinstance(result, dict):
        return result["mean"], result["std"]
    return float(result), 0.0


# ---------------------------------------------------------------------------
# Synthetic penalty (matches MATLAB 1-indexed convention)
# ---------------------------------------------------------------------------

def make_synthetic_penalty(km, B):
    p = np.zeros((km, B))
    for j in range(1, km + 1):
        if j % 3 == 0:
            p[j - 1, :B] = np.arange(1, B + 1)
        elif j % 3 == 1:
            p[j - 1, :B] = 10 * np.log(np.arange(1, B + 1))
        else:
            p[j - 1, :B] = np.exp(0.5 * np.arange(1, B + 1))
    return p


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_sweep(
    *,
    output_stem,
    sweep_name,       # e.g. "M", "N", "km" -- used as CSV column header
    sweep_values,     # iterable of ints
    build_problem,    # callable(sweep_value) -> dict with keys:
                      #   M, N, km, T, B, K, n, c, w, p, gamma
    sweep_label,      # x-axis label, e.g. "Number of Channels"
    title_prefix,     # plot title prefix
    subgradient_method=None,
    subgradient_source=None,
    profiles=None,
    titer=None,
    mc_trials=None,
    seed=None,
    extra_policies=None,
    use_log_y=True,
):
    """Run the full {profile x sweep x policy} grid and emit CSV + PNG."""
    data_dir = DATA_PROBABILISTIC_DIR
    plot_dir = PLOTS_PROBABILISTIC_DIR
    if profiles is None:
        profiles = parse_profiles()  # honors INFOCOM_PROFILES env var
    titer = titer if titer is not None else TITER
    mc_trials = mc_trials if mc_trials is not None else MC_TRIALS
    seed = seed if seed is not None else SEED
    extra_policies = (extra_policies if extra_policies is not None
                      else EXTRA_POLICIES)
    if subgradient_method is None:
        subgradient_method, subgradient_source = resolve_subgradient_method()

    print(f"Running probabilistic {output_stem} sweep")
    print(f"  TITER={titer}, MC_TRIALS={mc_trials}, SEED={seed}")
    print(f"  {sweep_name} values: {list(sweep_values)}")
    print(f"  profiles: {profiles}")
    print(f"  extra policies: {extra_policies}")
    print(f"  subgradient method: {subgradient_method}  ({subgradient_source})")

    policies = ["MGF", "MAF", "MIEF", "Random"]
    if extra_policies:
        policies += ["MAF_relaware", "MIEF_pure"]

    rows = []
    summary_rows = []
    q_archive = {}
    t_global = time.perf_counter()

    for profile in profiles:
        print(f"\n--- profile: {profile} ---")
        for v in sweep_values:
            prob = build_problem(int(v))
            M, N = prob["M"], prob["N"]
            km, T, B, K = prob["km"], prob["T"], prob["B"], prob["K"]
            n, c, w = prob["n"], prob["c"], prob["w"]
            p, gamma = prob["p"], prob["gamma"]

            # q is keyed by (M, km) so it must be re-drawn per sweep point.
            q, meta = make_q_profile(profile, M, km, seed=seed)
            q_archive[f"{profile}_{sweep_name}{int(v)}"] = q

            t_pair = time.perf_counter()
            res = {}
            res["MGF"] = MGF1_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                titer=titer, subgradient_method=subgradient_method,
                seed=seed, mc_trials=mc_trials, verbose=False,
            )
            res["MAF"] = MAF1_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                seed=seed, mc_trials=mc_trials,
                reliability_aware=False, verbose=False,
            )
            res["MIEF"] = MIEF1_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                seed=seed, mc_trials=mc_trials,
                reliability_aware=True, verbose=False,
            )
            res["Random"] = randpolicy_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                seed=seed, mc_trials=mc_trials,
                gated=True, verbose=False,
            )
            if extra_policies:
                res["MAF_relaware"] = MAF1_probabilistic(
                    M, N, km, T, B, K, n, c, w, gamma, p, q,
                    seed=seed, mc_trials=mc_trials,
                    reliability_aware=True, verbose=False,
                )
                res["MIEF_pure"] = MIEF1_probabilistic(
                    M, N, km, T, B, K, n, c, w, gamma, p, q,
                    seed=seed, mc_trials=mc_trials,
                    reliability_aware=False, verbose=False,
                )

            pair_runtime = time.perf_counter() - t_pair
            print(f"  {sweep_name}={int(v):>3d}: " + ", ".join(
                f"{name}={stats(res[name])[0]:.3g}" for name in policies
            ) + f"   ({pair_runtime:.1f}s)")

            for policy in policies:
                mean_e, std_e = stats(res[policy])
                rows.append({
                    "profile": profile,
                    sweep_name: int(v),
                    "policy": policy,
                    "mean_error": mean_e,
                    "std_error": std_e,
                    "seed": seed,
                    "titer": titer,
                    "mc_trials": mc_trials,
                    "subgradient_method": subgradient_method,
                    "q_min": meta["q_min"],
                    "q_max": meta["q_max"],
                    "q_mean": meta["q_mean"],
                    "q_std": meta["q_std"],
                })

        for policy in policies:
            errs = [r["mean_error"] for r in rows
                    if r["profile"] == profile and r["policy"] == policy]
            summary_rows.append({
                "profile": profile,
                "policy": policy,
                "mean_over_sweep": float(np.mean(errs)),
                "median_over_sweep": float(np.median(errs)),
                "max_over_sweep": float(np.max(errs)),
                "min_over_sweep": float(np.min(errs)),
            })

    print(f"\nTotal runtime: {time.perf_counter() - t_global:.1f}s")

    # CSV outputs -> data/probabilistic/
    data_csv = os.path.join(data_dir, f"{output_stem}_data.csv")
    with open(data_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {data_csv}")

    summary_csv = os.path.join(data_dir, f"{output_stem}_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_csv}")

    np.savez(os.path.join(data_dir, f"{output_stem}_q_profiles.npz"), **q_archive)

    # Per-profile subplot grid
    n_profiles = len(profiles)
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
    xs = list(sweep_values)
    for idx, profile in enumerate(profiles):
        ax = axes[idx // n_cols][idx % n_cols]
        for policy in policies:
            ys = [r["mean_error"] for r in rows
                  if r["profile"] == profile and r["policy"] == policy]
            ystd = [r["std_error"] for r in rows
                    if r["profile"] == profile and r["policy"] == policy]
            fmt, label = style.get(policy, ("k-", policy))
            ax.errorbar(xs, ys, yerr=ystd, fmt=fmt,
                        label=label, linewidth=1.5, markersize=6, capsize=3)
        ax.set_title(profile)
        ax.set_xlabel(sweep_label)
        ax.set_ylabel("Discounted error")
        if use_log_y:
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    for idx in range(n_profiles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"{title_prefix} (probabilistic, "
                 f"{subgradient_method} subgradient)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(plot_dir, f"{output_stem}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")
    return rows, summary_rows
