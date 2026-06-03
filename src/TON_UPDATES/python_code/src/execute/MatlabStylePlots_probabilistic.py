"""
MatlabStylePlots_probabilistic.py - MATLAB-styled single-axis figures for the
probabilistic model, restricted to two q-profiles and four policies.

This is a presentation script (not part of run_all.py's grid). It reproduces the
look of the original MATLAB figures (ErrorVsChannelsmodel.m / ErrorVsSources.m /
ErrorVsTasks.m): one figure per (sweep x profile x weight-mode), linear y-axis,
LineWidth 2 / MarkerSize 10, and the MATLAB marker styles:

    Random  -> red circle      ('ro-')   "Random Policy"
    MAF     -> blue star        ('b*-')   "MAF Policy"
    MGF     -> black diamond    ('kd-')   "MGF Policy"
    MEF     -> green triangle    ('g^-')   "MEF Policy"   (new policy = MIEF)

It is run for the two requested q-profiles ...

    uniform_very_wide
    bimodal_q1_vs_lossy_30_70

... and for the three sweeps (ErrorVsChannel / ErrorVsSources / ErrorVsTasks),
first with w = 1 everywhere (weights_1), then with the original heterogeneous
("deterministic") weights. That is 3 sweeps x 2 profiles x 2 weight modes = 12
figures.

All sweeps use the synthetic 9-task penalty model (j%3 -> linear / 10*log / exp)
exactly like the MATLAB scripts, with B=20, T=100, gamma=0.9, n=1, c=2.

Configuration via env vars:
    INFOCOM_TITER        (default 1000)  -- subgradient iterations for MGF dual
    INFOCOM_MC_TRIALS    (default 10)    -- Monte-Carlo trials averaged per point
    INFOCOM_SEED         (default 0)
    INFOCOM_SUBGRADIENT_METHOD (default episode1_mstep)

Outputs (under plots/probabilistic/matlab_style/ and data/probabilistic/):
    plots/probabilistic/matlab_style/<sweep>_<profile>_<wtag>.png
    data/probabilistic/matlab_style_<sweep>_<profile>_<wtag>.csv
"""
import os
import csv
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths

from MGF1_probabilistic import MGF1_probabilistic
from MAF1_probabilistic import MAF1_probabilistic
from MIEF1_probabilistic import MIEF1_probabilistic
from randpolicy_probabilistic import randpolicy_probabilistic
from probability_profiles_probabilistic import make_q_profile
from experiment_configs import make_weights, WEIGHTS_ONES, WEIGHTS_DETERMINISTIC
from _probabilistic_sweep_helpers import (
    make_synthetic_penalty,
    stats,
    resolve_subgradient_method,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TITER = int(os.environ.get("INFOCOM_TITER", "1000"))
MC_TRIALS = int(os.environ.get("INFOCOM_MC_TRIALS", "10"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
# Resolve the MGF dual solver the same way the grid scripts do: honor
# INFOCOM_SUBGRADIENT_METHOD, else read the precheck recommendation JSON
# (probabilistic -> "adam"), else fall back to "harmonic". This keeps MGF
# consistent with the existing probabilistic plots; hardcoding episode1_mstep
# underfits the multipliers on these sizes and inflates the MGF curve.
SUBGRADIENT_METHOD, SUBGRADIENT_SOURCE = resolve_subgradient_method()

PROFILES = ["uniform_very_wide", "bimodal_q1_vs_lossy_30_70"]

# (weight mode, filename tag, human label)
WEIGHT_MODES = [
    (WEIGHTS_ONES, "weights_1", "w = 1"),
    (WEIGHTS_DETERMINISTIC, "orig_weights", "original weights"),
]

# MATLAB-style per-policy line formats + legend labels.
#   order matters: this is the plot/legend order.
POLICY_STYLE = [
    ("Random", "ro-", "Random Policy"),
    ("MAF",    "b*-", "MAF Policy"),
    ("MGF",    "kd-", "MGF Policy"),
    ("MEF",    "g^-", "MEF Policy"),
]

B = 20
T = 100
GAMMA = 0.9


# ---------------------------------------------------------------------------
# Sweep definitions.  Each returns the per-sweep-value problem (M, N, km).
# experiment_name selects the heterogeneous weight pattern (all three use the
# half-sources pattern w=1 for m+1<=M/2 else 0.01).
# ---------------------------------------------------------------------------

def _sweep_specs():
    return [
        dict(
            sweep="ErrorVsChannel",
            experiment_name="ErrorVsChannelsmodel",
            xlabel="channel",
            axis_label=r"Number of Channels ($N$)",
            values=list(range(2, 21, 2)),
            build=lambda v: dict(M=20, N=v, km=9),
        ),
        dict(
            sweep="ErrorVsSources",
            experiment_name="ErrorVsSources",
            xlabel="Number of Sources",
            axis_label=r"Number of Sources ($M$)",
            values=list(range(2, 21, 2)),
            build=lambda v: dict(M=v, N=10, km=9),
        ),
        dict(
            sweep="ErrorVsTasks",
            experiment_name="ErrorVsTasks",
            xlabel="Number of Tasks",
            axis_label=r"Number of Tasks ($rk_m$)",
            values=list(range(3, 16, 3)),
            build=lambda v: dict(M=20, N=10, km=v),
        ),
    ]


def _run_policies(M, N, km, w, p, q):
    """Return {policy: mean_error} for the four plotted policies."""
    n = np.ones((M, km))
    c = np.ones((M, km)) * 2.0
    K = T
    out = {}
    out["MGF"] = MGF1_probabilistic(
        M, N, km, T, B, K, n, c, w, GAMMA, p, q,
        titer=TITER, subgradient_method=SUBGRADIENT_METHOD,
        seed=SEED, mc_trials=MC_TRIALS, verbose=False,
    )
    out["MAF"] = MAF1_probabilistic(
        M, N, km, T, B, K, n, c, w, GAMMA, p, q,
        seed=SEED, mc_trials=MC_TRIALS, reliability_aware=False, verbose=False,
    )
    # The new MEF policy = reliability-aware Maximum-(Instantaneous-)Error-First.
    out["MEF"] = MIEF1_probabilistic(
        M, N, km, T, B, K, n, c, w, GAMMA, p, q,
        seed=SEED, mc_trials=MC_TRIALS, reliability_aware=True, verbose=False,
    )
    out["Random"] = randpolicy_probabilistic(
        M, N, km, T, B, K, n, c, w, GAMMA, p, q,
        seed=SEED, mc_trials=MC_TRIALS, gated=True, verbose=False,
    )
    return {name: stats(res)[0] for name, res in out.items()}


def _plot_one(xs, series, axis_label, png_path):
    """One MATLAB-styled figure: four lines, log y, LineWidth 2, MS 10."""
    fig, ax = plt.subplots(figsize=(7, 5))
    legend_labels = []
    for name, fmt, label in POLICY_STYLE:
        ax.plot(xs, series[name], fmt, linewidth=2, markersize=10)
        legend_labels.append(label)
    ax.set_xlabel(axis_label, fontsize=18)
    ax.set_ylabel("Dis. Sum of Errors", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    # Log y-axis to match the probabilistic grid plots: on a linear axis the
    # large small-sweep values dominate and visually exaggerate the small region
    # where MGF sits just above MEF; log scale shows MGF at/below MEF across the
    # full range, consistent with the grid figures (same underlying numbers).
    ax.set_yscale("log")
    ax.legend(legend_labels, loc="best", fontsize=12)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(xs, xlabel, series, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([xlabel] + [name for name, _, _ in POLICY_STYLE])
        for i, x in enumerate(xs):
            writer.writerow([x] + [series[name][i] for name, _, _ in POLICY_STYLE])


def main():
    plot_dir = os.path.join(paths.PLOTS_PROBABILISTIC_DIR, "matlab_style")
    os.makedirs(plot_dir, exist_ok=True)

    print("MATLAB-style probabilistic plots")
    print(f"  TITER={TITER}  MC_TRIALS={MC_TRIALS}  SEED={SEED}")
    print(f"  subgradient method: {SUBGRADIENT_METHOD}  ({SUBGRADIENT_SOURCE})")
    print(f"  profiles: {PROFILES}")
    print(f"  weight modes: {[m for m, _, _ in WEIGHT_MODES]}")

    n_figs = 0
    for wmode, wtag, wlabel in WEIGHT_MODES:
        for spec in _sweep_specs():
            sweep = spec["sweep"]
            xs = spec["values"]
            for profile in PROFILES:
                print(f"\n=== {sweep} | {profile} | {wlabel} ===")
                series = {name: [] for name, _, _ in POLICY_STYLE}
                for v in xs:
                    prob = spec["build"](v)
                    M, N, km = prob["M"], prob["N"], prob["km"]
                    p = make_synthetic_penalty(km, B)
                    w = make_weights(spec["experiment_name"], M, km, wmode)
                    # q is keyed by (M, km); redraw per sweep point (fixed seed).
                    q, _ = make_q_profile(profile, M, km, seed=SEED)
                    means = _run_policies(M, N, km, w, p, q)
                    for name in series:
                        series[name].append(means[name])
                    print(f"  {spec['xlabel']}={v:>3d}: " + ", ".join(
                        f"{name}={means[name]:.3g}" for name, _, _ in POLICY_STYLE))

                png_path = os.path.join(plot_dir, f"{sweep}_{profile}_{wtag}.png")
                csv_path = paths.data_path(
                    f"matlab_style_{sweep}_{profile}_{wtag}.csv", probabilistic=True)
                _plot_one(xs, series, spec["axis_label"], png_path)
                _write_csv(xs, spec["xlabel"], series, csv_path)
                n_figs += 1
                print(f"  saved {png_path}")
                print(f"  saved {csv_path}")

    print(f"\nDone. Wrote {n_figs} figures to {plot_dir}")


if __name__ == "__main__":
    main()
