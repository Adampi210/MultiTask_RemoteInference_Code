"""
ErrorVsSources_variants.py - deterministic ErrorVsSources sweep using
MGF1_variants (method-selectable subgradient driver). Does NOT replace the
original ErrorVsSources.py.

Reads the recommended deterministic method from
data/deterministic/recommended_subgradient_methods.json unless
INFOCOM_SUBGRADIENT_METHOD is set.

Configuration via env vars:
    INFOCOM_TITER              (default 1000)
    INFOCOM_SEED               (default 0)
    INFOCOM_SOURCES            (default '2,4,6,8,10,12,14,16,18,20')
    INFOCOM_SUBGRADIENT_METHOD (optional override)

Writes:
    data/deterministic/ErrorVsSources_variants_data.csv
    plots/deterministic/ErrorVsSources_variants.png
"""
import os
import csv
import json
import time
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from _bootstrap import paths   # bootstraps sys.path

from experiment_utils import make_synthetic_p
from MGF1_variants import MGF1_variants
from MAF1 import MAF1
from MIEF1 import MIEF1
from randpolicy import randpolicy


TITER = int(os.environ.get("INFOCOM_TITER", "1000"))
SEED = int(os.environ.get("INFOCOM_SEED", "0"))


def _parse_int_list(s, default):
    if not s:
        return default
    return [int(x) for x in s.split(",") if x.strip()]


SOURCES = _parse_int_list(os.environ.get("INFOCOM_SOURCES"),
                          list(range(2, 21, 2)))


def _resolve_subgradient_method():
    override = os.environ.get("INFOCOM_SUBGRADIENT_METHOD")
    if override:
        return override.strip(), "env override"
    path = paths.data_path("recommended_subgradient_methods.json",
                           probabilistic=False)
    if os.path.exists(path):
        try:
            with open(path) as f:
                rec = json.load(f)
            method = rec.get("deterministic", {}).get(
                "recommended_method", "harmonic"
            )
            return method, f"from {os.path.basename(path)}"
        except Exception:
            pass
    return "harmonic", "default fallback (no precheck recommendation found)"


SUBGRAD_METHOD, SUBGRAD_SOURCE = _resolve_subgradient_method()


def main():
    print(f"ErrorVsSources_variants  (deterministic, method-selectable MGF)")
    print(f"  TITER={TITER}, SEED={SEED}, SOURCES={SOURCES}")
    print(f"  SUBGRAD_METHOD={SUBGRAD_METHOD}  ({SUBGRAD_SOURCE})")

    np.random.seed(SEED)
    B = 20
    km = 9
    N = 10
    T = 100
    K = T
    gamma = 0.9
    p = make_synthetic_p(km, B)

    pmgf = np.zeros(len(SOURCES))
    pmaf = np.zeros(len(SOURCES))
    pmief = np.zeros(len(SOURCES))
    prand = np.zeros(len(SOURCES))
    t_global = time.perf_counter()

    for ch, M in enumerate(SOURCES):
        print(f"\n=== M = {M} ===")
        n = np.ones((M, km))
        c = np.ones((M, km)) * 2

        # Heterogeneous w like the original deterministic experiment.
        w = np.zeros((M, km))
        for m in range(M):
            w[m, :] = 1.0 if (m + 1) <= M / 2 else 0.01

        pmgf[ch] = MGF1_variants(M, N, km, T, B, K, n, c, w, gamma, p,
                                 titer=TITER, subgradient_method=SUBGRAD_METHOD,
                                 seed=SEED, verbose=False)
        pmaf[ch] = MAF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=False)
        pmief[ch] = MIEF1(M, N, km, T, B, K, n, c, w, gamma, p, verbose=False)
        prand[ch] = randpolicy(M, N, km, T, B, K, n, c, w, gamma, p,
                               verbose=False)
        print(f"  MGF_{SUBGRAD_METHOD}={pmgf[ch]:.4g} "
              f"MAF={pmaf[ch]:.4g} MIEF={pmief[ch]:.4g} "
              f"Rand={prand[ch]:.4g}")

    total = time.perf_counter() - t_global
    print(f"\nTotal runtime: {total:.1f}s")

    csv_path = paths.data_path("ErrorVsSources_variants_data.csv",
                               probabilistic=False)
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["M", "MGF", "MAF", "MIEF", "Random",
                     "subgradient_method"])
        for i, M in enumerate(SOURCES):
            wr.writerow([M, pmgf[i], pmaf[i], pmief[i], prand[i],
                         SUBGRAD_METHOD])
    print(f"Wrote {csv_path}")

    plt.figure()
    plt.plot(SOURCES, prand, "ro-", label="Random", linewidth=2, markersize=8)
    plt.plot(SOURCES, pmaf, "b*-", label="MAF", linewidth=2, markersize=8)
    plt.plot(SOURCES, pmgf, "g-", label=f"MGF ({SUBGRAD_METHOD})",
             linewidth=2, markersize=8)
    plt.plot(SOURCES, pmief, "ms--", label="MIEF", linewidth=2, markersize=8)
    plt.xlabel("Number of Sources (M)")
    plt.ylabel("Discounted Sum of Errors")
    plt.yscale("log")
    plt.legend()
    plt.title(f"ErrorVsSources_variants  (subgradient = {SUBGRAD_METHOD})")
    out = paths.plot_path("ErrorVsSources_variants.png", probabilistic=False)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
