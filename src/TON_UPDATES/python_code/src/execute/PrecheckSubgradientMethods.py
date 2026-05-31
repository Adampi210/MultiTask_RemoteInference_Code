"""
PrecheckSubgradientMethods.py - compare multiplier-learning methods before
running expensive full experiments. Tests both deterministic MGF and
probabilistic MGF over a small set of scenarios. Writes:

    PrecheckSubgradientMethods_data.csv
    PrecheckSubgradientMethods_summary.csv
    PrecheckSubgradientMethods.png
    recommended_subgradient_methods.json

Environment variables:
    INFOCOM_PRECHECK_TITER             (default 100)
    INFOCOM_PRECHECK_MC_TRIALS         (default 5)
    INFOCOM_PRECHECK_SEED              (default 0)
    INFOCOM_PRECHECK_FAST              (default 1)
    INFOCOM_PRECHECK_INCLUDE_CUTTING_PLANES  (default 0)
    INFOCOM_PRECHECK_METHODS           (optional comma-separated)

Usage:
    python PrecheckSubgradientMethods.py
"""
import os
import csv
import json
import time
import datetime as dt
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths

from experiment_utils import make_base_problem
from probability_profiles_probabilistic import make_q_profile
from MGF1_variants import MGF1_variants
from MGF1_probabilistic import MGF1_probabilistic
from optimizer_updates import get_default_subgradient_methods


# Precheck spans deterministic AND probabilistic scenarios but writes one
# recommendation JSON consumed by both, so the outputs live in the
# deterministic data/plot directories.
DATA_DIR = paths.DATA_DETERMINISTIC_DIR
PLOT_DIR = paths.PLOTS_DETERMINISTIC_DIR

TITER = int(os.environ.get("INFOCOM_PRECHECK_TITER", "100"))
MC_TRIALS = int(os.environ.get("INFOCOM_PRECHECK_MC_TRIALS", "5"))
SEED = int(os.environ.get("INFOCOM_PRECHECK_SEED", "0"))
FAST = int(os.environ.get("INFOCOM_PRECHECK_FAST", "1"))
INCLUDE_CP = int(os.environ.get("INFOCOM_PRECHECK_INCLUDE_CUTTING_PLANES", "0"))
METHODS_ENV = os.environ.get("INFOCOM_PRECHECK_METHODS")


_DEFAULT_FIRST_ORDER = [
    "harmonic",
    "sqrt",
    "normalized_blocks",
    "adagrad",
    "rmsprop",
    "adam",
    "deflected_sqrt",
    "episode1_mstep",  # MATLAB-equivalent Episode1 closed-form M-step (probabilistic only)
]
_DEFAULT_CUTTING_PLANE = [
    "kelley_bounded",
    "trust_region_kelley",
    "proximal_bundle",
]


def resolve_methods():
    if METHODS_ENV:
        return [m.strip() for m in METHODS_ENV.split(",") if m.strip()]
    methods = list(_DEFAULT_FIRST_ORDER)
    if INCLUDE_CP:
        methods += _DEFAULT_CUTTING_PLANE
    return methods


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def build_scenarios():
    """Return a list of dicts: {name, mode, q_profile, problem}."""
    small = make_base_problem(M=6, N=4, km=6, B=20, T=50, gamma=0.9,
                              weights="ones", c_value=2, n_value=1)
    medium = make_base_problem(M=10, N=6, km=9, B=20, T=75, gamma=0.9,
                               weights="ones", c_value=2, n_value=1)

    scenarios = [
        {"name": "deterministic_synthetic_small",
         "mode": "deterministic", "q_profile": "", "problem": small},
        {"name": "deterministic_synthetic_medium",
         "mode": "deterministic", "q_profile": "", "problem": medium},
        {"name": "probabilistic_bimodal_balanced",
         "mode": "probabilistic", "q_profile": "bimodal_balanced",
         "problem": small},
        {"name": "probabilistic_uniform_wide",
         "mode": "probabilistic", "q_profile": "uniform_wide",
         "problem": small},
        {"name": "probabilistic_adversarial",
         "mode": "probabilistic",
         "q_profile": "adversarial_perfect_with_critical_lossy",
         "problem": small},
    ]

    if FAST:
        keep = {"deterministic_synthetic_small",
                "probabilistic_bimodal_balanced",
                "probabilistic_adversarial"}
        scenarios = [s for s in scenarios if s["name"] in keep]
    return scenarios


# ---------------------------------------------------------------------------
# Single-scenario runner
# ---------------------------------------------------------------------------

def run_method_on_scenario(scenario, method, titer, mc_trials, seed):
    """Run one method on one scenario. Returns a dict (always with status)."""
    prob = scenario["problem"]
    M, N, km, B, T, K = (prob["M"], prob["N"], prob["km"], prob["B"],
                         prob["T"], prob["K"])
    p, n, c, w, gamma = prob["p"], prob["n"], prob["c"], prob["w"], prob["gamma"]

    row = {
        "scenario_name": scenario["name"],
        "mode": scenario["mode"],
        "method": method,
        "M": M, "N": N, "km": km, "B": B, "T": T, "K": K, "gamma": gamma,
        "titer": titer, "beta": 0.9, "seed": seed, "mc_trials": mc_trials,
        "q_profile": scenario["q_profile"],
        "objective_mean": float("nan"),
        "objective_std": float("nan"),
        "objective_trials": "",
        "runtime_sec": float("nan"),
        "final_max_lambda": float("nan"),
        "final_mean_lambda": float("nan"),
        "final_max_mu": float("nan"),
        "final_mean_mu": float("nan"),
        "multiplier_variation_last_20pct": float("nan"),
        "mean_compute_positive_residual": float("nan"),
        "mean_channel_positive_residual": float("nan"),
        "final_residual_norm": float("nan"),
        "best_dual_value": float("nan"),
        "final_dual_value": float("nan"),
        "cutting_plane_gap": float("nan"),
        "oracle_calls": -1,
        "status": "failed",
        "error_message": "",
    }

    try:
        t0 = time.perf_counter()
        if scenario["mode"] == "deterministic":
            res = MGF1_variants(
                M, N, km, T, B, K, n, c, w, gamma, p,
                titer=titer, subgradient_method=method,
                seed=seed, verbose=False, return_diagnostics=True,
            )
            obj_trials = [res["objective"]]
            row["objective_mean"] = float(res["objective"])
            row["objective_std"] = 0.0
            hist = res["subgradient_history"]
            lam, mu = res["lambdasource"], res["mu"]
        else:
            q, _ = make_q_profile(scenario["q_profile"], M, km, seed=seed)
            res = MGF1_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                titer=titer, subgradient_method=method,
                seed=seed, mc_trials=mc_trials,
                verbose=False, return_diagnostics=True,
            )
            row["objective_mean"] = float(res["objective_mean"])
            row["objective_std"] = float(res["objective_std"])
            obj_trials = list(res["objective_trials"])
            hist = res["subgradient_history"]
            lam, mu = res["lambdasource"], res["mu"]

        row["runtime_sec"] = float(time.perf_counter() - t0)
        row["objective_trials"] = json.dumps([float(x) for x in obj_trials])

        if lam is not None:
            row["final_max_lambda"] = float(np.max(np.abs(lam)))
            row["final_mean_lambda"] = float(np.mean(lam))
        if mu is not None:
            row["final_max_mu"] = float(np.max(np.abs(mu)))
            row["final_mean_mu"] = float(np.mean(mu))

        # First-order: hist has max_lambda time series; cutting-plane: dual_values.
        if "max_lambda" in hist and hist["max_lambda"]:
            arr = np.asarray(hist["max_lambda"], dtype=float)
            tail_n = max(1, int(0.2 * len(arr)))
            tail = arr[-tail_n:]
            row["multiplier_variation_last_20pct"] = float(tail.std())
        if "mean_compute_positive_residual" in hist and hist["mean_compute_positive_residual"]:
            row["mean_compute_positive_residual"] = float(np.mean(
                hist["mean_compute_positive_residual"][-tail_n:]
            ))
        if "mean_channel_positive_residual" in hist and hist["mean_channel_positive_residual"]:
            row["mean_channel_positive_residual"] = float(np.mean(
                hist["mean_channel_positive_residual"][-tail_n:]
            ))
        if "residual_norm" in hist and hist["residual_norm"]:
            row["final_residual_norm"] = float(hist["residual_norm"][-1])

        if "dual_values" in hist and hist["dual_values"]:
            duals = np.asarray(hist["dual_values"], dtype=float)
            row["best_dual_value"] = float(np.max(duals))
            row["final_dual_value"] = float(duals[-1])
            row["oracle_calls"] = int(hist.get("oracle_calls", len(duals)))
            if "estimated_gaps" in hist and hist["estimated_gaps"]:
                gaps = np.asarray(hist["estimated_gaps"], dtype=float)
                gaps = gaps[np.isfinite(gaps)]
                if len(gaps) > 0:
                    row["cutting_plane_gap"] = float(gaps[-1])

        ok = (np.isfinite(row["objective_mean"])
              and row["objective_mean"] < 1e6)
        # Penalty checks
        if row["final_max_lambda"] > 1e4 or row["final_max_mu"] > 1e4:
            row["status"] = "warn_huge_multipliers"
        elif not ok:
            row["status"] = "failed"
            row["error_message"] = "non-finite or absurd objective"
        else:
            row["status"] = "ok"

    except Exception as e:
        row["status"] = "failed"
        row["error_message"] = f"{type(e).__name__}: {e}"
    return row


# ---------------------------------------------------------------------------
# Ranking + recommendation
# ---------------------------------------------------------------------------

def _rank_within_scenario(rows_in_scenario):
    """Assign objective_rank, stability_rank, runtime_rank per scenario.

    Lower is better in all three. Failed/NaN rows get a heavy penalty rank.
    """
    n = len(rows_in_scenario)
    if n == 0:
        return rows_in_scenario

    # Build sortable keys; NaN/failure -> +inf so they rank last.
    def _safe(x, default=float("inf")):
        try:
            xv = float(x)
            if not np.isfinite(xv):
                return default
            return xv
        except (TypeError, ValueError):
            return default

    obj_keys = [(_safe(r["objective_mean"]), i) for i, r in enumerate(rows_in_scenario)]
    stab_keys = [
        (_safe(r["multiplier_variation_last_20pct"]) + _safe(r["final_residual_norm"]) * 1e-3,
         i)
        for i, r in enumerate(rows_in_scenario)
    ]
    run_keys = [(_safe(r["runtime_sec"]), i) for i, r in enumerate(rows_in_scenario)]

    for keys, name in [(obj_keys, "objective_rank"),
                       (stab_keys, "stability_rank"),
                       (run_keys, "runtime_rank")]:
        keys_sorted = sorted(keys, key=lambda kv: kv[0])
        for rank, (_, i) in enumerate(keys_sorted):
            rows_in_scenario[i][name] = rank + 1   # 1-indexed
    return rows_in_scenario


def _aggregate(rows, mode_filter=None):
    """Compute average total_score per method.

    total_score = 0.70 * objective + 0.15 * stability + 0.15 * runtime, averaged
    over scenarios. Lower is better.
    """
    by_method = {}
    for r in rows:
        if mode_filter is not None and r["mode"] != mode_filter:
            continue
        m = r["method"]
        score = (0.70 * r.get("objective_rank", 1e6)
                 + 0.15 * r.get("stability_rank", 1e6)
                 + 0.15 * r.get("runtime_rank", 1e6))
        # Penalize failures
        if r["status"] == "failed":
            score += 1e6
        elif r["status"] == "warn_huge_multipliers":
            score += 100.0
        by_method.setdefault(m, []).append(score)

    if not by_method:
        return None, {}
    method_scores = {m: float(np.mean(s)) for m, s in by_method.items()}
    best = min(method_scores, key=method_scores.get)
    return best, method_scores


def _make_recommendation(rows):
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    det_best, det_scores = _aggregate(rows, mode_filter="deterministic")
    prob_best, prob_scores = _aggregate(rows, mode_filter="probabilistic")
    comb_best, comb_scores = _aggregate(rows, mode_filter=None)

    def _reason(scores, best):
        if best is None:
            return "no successful runs; fallback to harmonic"
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        top3 = ", ".join(f"{m}={s:.2f}" for m, s in ranked[:3])
        return f"lowest mean total_score ({scores[best]:.2f}); top3: {top3}"

    # Fallback if all failed
    if det_best is None:
        det_best = "harmonic"
    if prob_best is None:
        prob_best = "harmonic"
    if comb_best is None:
        comb_best = "harmonic"

    rec = {
        "deterministic": {
            "recommended_method": det_best,
            "reason": _reason(det_scores, det_best),
            "timestamp": timestamp,
            "precheck_file": "PrecheckSubgradientMethods_data.csv",
        },
        "probabilistic": {
            "recommended_method": prob_best,
            "reason": _reason(prob_scores, prob_best),
            "timestamp": timestamp,
            "precheck_file": "PrecheckSubgradientMethods_data.csv",
        },
        "combined": {
            "recommended_method": comb_best,
            "reason": _reason(comb_scores, comb_best),
            "timestamp": timestamp,
            "precheck_file": "PrecheckSubgradientMethods_data.csv",
        },
    }
    return rec


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(rows, out_path):
    scenarios = sorted({r["scenario_name"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    if not scenarios or not methods:
        return

    n = len(scenarios)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    for idx, scen in enumerate(scenarios):
        ax = axes[idx // n_cols][idx % n_cols]
        scen_rows = [r for r in rows if r["scenario_name"] == scen]
        labels = [r["method"] for r in scen_rows]
        vals = [r["objective_mean"] for r in scen_rows]
        stds = [r["objective_std"] for r in scen_rows]
        xpos = np.arange(len(labels))
        ax.bar(xpos, vals, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(scen, fontsize=10)
        ax.set_ylabel("objective (lower is better)")
        ax.grid(True, axis="y", alpha=0.3)
    for idx in range(len(scenarios), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    methods = resolve_methods()
    scenarios = build_scenarios()
    print(f"PrecheckSubgradientMethods")
    print(f"  TITER={TITER}, MC_TRIALS={MC_TRIALS}, SEED={SEED}, "
          f"FAST={FAST}, INCLUDE_CP={INCLUDE_CP}")
    print(f"  Methods: {methods}")
    print(f"  Scenarios: {[s['name'] for s in scenarios]}")

    rows = []
    t_global = time.perf_counter()
    for scen in scenarios:
        print(f"\n--- scenario: {scen['name']} ({scen['mode']}) ---")
        scen_rows = []
        for method in methods:
            t0 = time.perf_counter()
            row = run_method_on_scenario(
                scen, method, TITER, MC_TRIALS, SEED,
            )
            scen_rows.append(row)
            obj = row["objective_mean"]
            print(f"  {method:25s} obj={obj:.5g}  "
                  f"({time.perf_counter()-t0:.1f}s)  status={row['status']}")
        scen_rows = _rank_within_scenario(scen_rows)
        rows.extend(scen_rows)
    total = time.perf_counter() - t_global
    print(f"\nTotal runtime: {total:.1f}s")

    # Write CSVs.
    data_csv = os.path.join(DATA_DIR, "PrecheckSubgradientMethods_data.csv")
    fieldnames = list(rows[0].keys())
    # Ensure rank columns are present.
    for col in ("objective_rank", "stability_rank", "runtime_rank"):
        if col not in fieldnames:
            fieldnames.append(col)
    with open(data_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {data_csv}")

    # Summary CSV.
    summary_csv = os.path.join(DATA_DIR, "PrecheckSubgradientMethods_summary.csv")
    method_set = sorted({r["method"] for r in rows})
    scen_set = sorted({r["scenario_name"] for r in rows})
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "objective_mean", "objective_std",
                    "runtime_sec", "objective_rank", "stability_rank",
                    "runtime_rank", "status"])
        for r in rows:
            w.writerow([r["scenario_name"], r["method"],
                        r["objective_mean"], r["objective_std"],
                        r["runtime_sec"], r.get("objective_rank", ""),
                        r.get("stability_rank", ""), r.get("runtime_rank", ""),
                        r["status"]])
    print(f"Wrote {summary_csv}")

    # Plot.
    plot_out_path = os.path.join(PLOT_DIR, "PrecheckSubgradientMethods.png")
    try:
        make_plot(rows, plot_out_path)
        print(f"Wrote {plot_out_path}")
    except Exception as e:
        print(f"(plot failed: {e})")

    # Recommendation.
    rec = _make_recommendation(rows)
    rec_path = os.path.join(DATA_DIR, "recommended_subgradient_methods.json")
    with open(rec_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Wrote {rec_path}")

    print("\n=== Recommendations ===")
    print(f"  Best deterministic method : {rec['deterministic']['recommended_method']}")
    print(f"    {rec['deterministic']['reason']}")
    print(f"  Best probabilistic method : {rec['probabilistic']['recommended_method']}")
    print(f"    {rec['probabilistic']['reason']}")
    print(f"  Best combined method      : {rec['combined']['recommended_method']}")
    print(f"    {rec['combined']['reason']}")
    print(f"\n  -> Full deterministic experiments will use: "
          f"{rec['deterministic']['recommended_method']}")
    print(f"  -> Full probabilistic experiments will use: "
          f"{rec['probabilistic']['recommended_method']}")


if __name__ == "__main__":
    main()
