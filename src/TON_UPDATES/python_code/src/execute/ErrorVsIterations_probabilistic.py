"""
ErrorVsIterations_probabilistic.py - MGF objective vs. number of subgradient
iterations ("local episodes" of the dual-ascent method).

For each of three problem settings (taken verbatim from the deterministic
experiments but driven through the *probabilistic* MGF with q = 1 everywhere),
we run MGF for an increasing number of subgradient outer iterations
    iters in {1, 2, 4, 8, 16, 32}
and record the resulting discounted weighted-error objective. This shows how the
learned Lagrange multipliers (and hence the MGF schedule) converge as more dual
iterations are spent.

Because q = 1 and the channel cost n = 1, the probabilistic MGF with the
``episode1_mstep`` subgradient reduces *exactly* to the deterministic MGF1 (the
1:1 MATLAB port). The script therefore doubles as a sanity check: at every
iteration count it also runs the deterministic MGF1 and asserts the two agree to
floating-point tolerance. A reference line is drawn at the deterministic MGF1
objective after the full ``INFOCOM_ITER_REFERENCE`` (default 10000) iterations,
i.e. the value the original MATLAB code produces.

Settings (q = 1 everywhere, deterministic/paper weights):
  * ErrorVsChannel       - M=20, N=10, km=2, empirical loss.mat penalties.
  * ErrorVsChannelsmodel - M=20, N=10, km=9, synthetic penalty model.
  * ErrorVsSources       - M=10, N=10, km=9, synthetic penalty model.

Configuration via env vars:
  INFOCOM_ITERATIONS       (default '1,2,4,8,16,32')   - iteration counts swept
  INFOCOM_ITER_REFERENCE   (default 10000)             - reference MGF1 iters
  INFOCOM_SEED             (default 0)                  - randpolicy baseline seed
  INFOCOM_ITER_SETTINGS    (default all)               - subset of setting names

Writes:
  data/probabilistic/ErrorVsIterations_probabilistic_data.csv
  plots/probabilistic/ErrorVsIterations_probabilistic.png            (combined)
  plots/probabilistic/ErrorVsIterations_<setting>_probabilistic.png  (per setting)
"""
import os
import csv
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths

from MGF1 import MGF1
from MGF1_probabilistic import MGF1_probabilistic
from MAF1 import MAF1
from MIEF1 import MIEF1
from randpolicy import randpolicy
from experiment_configs import make_weights, WEIGHTS_DETERMINISTIC
import scipy.io as sio


SEED = int(os.environ.get("INFOCOM_SEED", "0"))
REFERENCE_ITERS = int(os.environ.get("INFOCOM_ITER_REFERENCE", "10000"))
SANITY_TOL = 1e-6   # relative tolerance for the q=1 reduction to deterministic


def _parse_int_list(s, default):
    if not s:
        return list(default)
    return [int(x) for x in s.split(",") if x.strip()]


def _synthetic_penalty(km, B):
    p = np.zeros((km, B))
    for j in range(1, km + 1):
        if j % 3 == 0:
            p[j - 1, :B] = np.arange(1, B + 1)
        elif j % 3 == 1:
            p[j - 1, :B] = 10 * np.log(np.arange(1, B + 1))
        else:
            p[j - 1, :B] = np.exp(0.5 * np.arange(1, B + 1))
    return p


def _empirical_penalty(km, B):
    """Subsample loss.mat p1, p2 every 5 indices to length B (matches MATLAB)."""
    loss = sio.loadmat(paths.data_path('loss.mat', probabilistic=False))
    p1 = loss['p1'].flatten()
    p2 = loss['p2'].flatten()
    p = np.zeros((km, B))
    i = 0
    for j_mat in range(1, 101, 5):
        if i >= B:
            break
        j_py = j_mat - 1
        if j_py < p1.size:
            p[0, i] = p1[j_py]
        if km >= 2 and j_py < p2.size:
            p[1, i] = p2[j_py]
        i += 1
    return p


def iteration_settings():
    """Return the list of single-point settings used by the iteration sweep.

    Each setting is a dict with name / experiment / M / N / km / B / T / gamma /
    penalty_kind. ``experiment`` selects the deterministic weight pattern via
    experiment_configs.make_weights.
    """
    return [
        dict(name="ErrorVsChannel", experiment="ErrorVsChannel",
             M=20, N=10, km=2, B=20, T=100, gamma=0.9,
             penalty_kind="empirical loss.mat (p1, p2)"),
        dict(name="ErrorVsChannelsmodel", experiment="ErrorVsChannelsmodel",
             M=20, N=10, km=9, B=20, T=100, gamma=0.9,
             penalty_kind="synthetic 9-task model"),
        dict(name="ErrorVsSources", experiment="ErrorVsSources",
             M=10, N=10, km=9, B=20, T=100, gamma=0.9,
             penalty_kind="synthetic 9-task model (M fixed at 10)"),
    ]


def _build_setting(setting):
    """Materialize (n, c, w, p, q) arrays for a setting dict."""
    M, km, B = setting["M"], setting["km"], setting["B"]
    n = np.ones((M, km))
    c = np.ones((M, km)) * 2
    w = make_weights(setting["experiment"], M, km, WEIGHTS_DETERMINISTIC)
    if setting["penalty_kind"].startswith("empirical"):
        p = _empirical_penalty(km, B)
    else:
        p = _synthetic_penalty(km, B)
    q = np.ones((M, km))   # q = 1 for every link
    return n, c, w, p, q


def main():
    iterations = _parse_int_list(os.environ.get("INFOCOM_ITERATIONS"),
                                 [1, 2, 4, 8, 16, 32])
    settings = iteration_settings()
    only = os.environ.get("INFOCOM_ITER_SETTINGS")
    if only:
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        settings = [s for s in settings if s["name"] in wanted]

    print("ErrorVsIterations (probabilistic MGF, q=1) vs deterministic MGF1")
    print(f"  iterations: {iterations}")
    print(f"  reference iters: {REFERENCE_ITERS}")
    print(f"  settings: {[s['name'] for s in settings]}")

    rows = []
    per_setting = {}        # name -> dict of curves for plotting
    sanity_max_rel = 0.0
    sanity_fail = False

    for setting in settings:
        name = setting["name"]
        M, N, km = setting["M"], setting["N"], setting["km"]
        B, T, gamma = setting["B"], setting["T"], setting["gamma"]
        K = T
        n, c, w, p, q = _build_setting(setting)
        print(f"\n=== setting {name}  (M={M}, N={N}, km={km}) ===")

        # Titer-independent baselines (deterministic; q=1 makes them exact).
        np.random.seed(SEED)
        maf = MAF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=False)
        mief = MIEF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=False)
        rand = randpolicy(M, N, km, T, B, K, n, c, w, gamma, p, verbose=False)

        mgf_prob = []
        mgf_det = []
        for it in iterations:
            vp = MGF1_probabilistic(
                M, N, km, T, B, K, n, c, w, gamma, p, q,
                titer=it, subgradient_method="episode1_mstep",
                seed=SEED, mc_trials=1, verbose=False,
            )
            vd = MGF1(M, N, km, T, B, K, n, c, w, gamma, p,
                      titer=it, verbose=False)
            mgf_prob.append(float(vp))
            mgf_det.append(float(vd))
            rel = abs(vp - vd) / (abs(vd) + 1e-12)
            sanity_max_rel = max(sanity_max_rel, rel)
            if rel > SANITY_TOL:
                sanity_fail = True
            print(f"  iters={it:>4d}: MGF(prob,q=1)={vp:.6f}  "
                  f"MGF1(det)={vd:.6f}  rel.diff={rel:.2e}")
            rows.append(dict(
                setting=name, iterations=it,
                mgf_prob_q1=vp, mgf_det=vd, rel_diff=rel,
                maf=maf, mief=mief, random=rand,
                M=M, N=N, km=km, B=B, T=T, gamma=gamma, seed=SEED,
            ))

        # Reference: deterministic MGF1 at the full reference iteration count.
        ref = float(MGF1(M, N, km, T, B, K, n, c, w, gamma, p,
                         titer=REFERENCE_ITERS, verbose=False))
        print(f"  reference MGF1 @ {REFERENCE_ITERS} iters = {ref:.6f}")
        for r in rows:
            if r["setting"] == name:
                r["mgf_reference"] = ref
                r["reference_iters"] = REFERENCE_ITERS

        per_setting[name] = dict(
            iters=list(iterations), mgf_prob=mgf_prob, mgf_det=mgf_det,
            maf=maf, mief=mief, random=rand, reference=ref,
        )

    # ---- CSV ----
    csv_out = paths.data_path("ErrorVsIterations_probabilistic_data.csv",
                              probabilistic=True)
    fieldnames = ["setting", "iterations", "mgf_prob_q1", "mgf_det", "rel_diff",
                  "mgf_reference", "reference_iters", "maf", "mief", "random",
                  "M", "N", "km", "B", "T", "gamma", "seed"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWrote {csv_out}")

    # ---- Plots ----
    def _plot_one(ax, name, d):
        its = d["iters"]
        ax.plot(its, d["mgf_prob"], "g-o", label="MGF (prob, q=1)",
                linewidth=2, markersize=8)
        ax.plot(its, d["mgf_det"], "kx", label="MGF1 (det, same iters)",
                markersize=9, markeredgewidth=2)
        ax.axhline(d["reference"], color="g", linestyle=":",
                   label=f"MGF1 @ {REFERENCE_ITERS} iters")
        ax.axhline(d["maf"], color="b", linestyle="--", label="MAF")
        ax.axhline(d["mief"], color="m", linestyle="--", label="MIEF")
        ax.axhline(d["random"], color="r", linestyle="--", label="Random")
        ax.set_xscale("log", base=2)
        ax.set_xticks(its)
        ax.set_xticklabels([str(i) for i in its])
        ax.set_xlabel("Subgradient iterations (local episodes)")
        ax.set_ylabel("Discounted weighted error")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Per-setting figures
    for name, d in per_setting.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_one(ax, name, d)
        fig.suptitle("MGF objective vs subgradient iterations (q=1)",
                     fontsize=12)
        plt.tight_layout()
        out = paths.plot_path(f"ErrorVsIterations_{name}_probabilistic.png",
                              probabilistic=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out}")

    # Combined figure
    n_set = len(per_setting)
    if n_set:
        fig, axes = plt.subplots(1, n_set, figsize=(6 * n_set, 5),
                                 squeeze=False)
        for idx, (name, d) in enumerate(per_setting.items()):
            _plot_one(axes[0][idx], name, d)
        fig.suptitle("MGF objective vs subgradient iterations "
                     "(probabilistic q=1 == deterministic MGF1)", fontsize=13)
        plt.tight_layout()
        out = paths.plot_path("ErrorVsIterations_probabilistic.png",
                              probabilistic=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out}")

    # ---- Sanity verdict ----
    print("\n" + "=" * 60)
    if sanity_fail:
        print(f"SANITY CHECK FAILED: max relative MGF(prob,q=1) vs MGF1(det) "
              f"diff = {sanity_max_rel:.3e} > {SANITY_TOL:.0e}")
    else:
        print(f"SANITY CHECK PASSED: probabilistic MGF (q=1) matches "
              f"deterministic MGF1 (the MATLAB 1:1 port) at every iteration "
              f"count.\n  max relative difference = {sanity_max_rel:.3e} "
              f"(tol {SANITY_TOL:.0e})")
    print("=" * 60)
    return not sanity_fail


if __name__ == "__main__":
    main()
