"""
ExportExperimentParameters.py - dump the full parameter set of every experiment
(every generated plot) in an easy-to-read form.

For each experiment it writes, under data/<flavor>/params/:
    <stem>_parameters.txt   - human-readable summary
    <stem>_parameters.json  - machine-readable (full weight matrix, q-profile
                              stats, dimensions, sweep, solver settings)

It also writes a single master index at:
    python_code/EXPERIMENT_PARAMETERS.md

Covered experiments:
  * the deterministic 1:1-port sweeps (ErrorVsChannel / Channelsmodel / Sources
    / Tasks),
  * the probabilistic sweeps in BOTH weight modes (deterministic and the
    ``_weights_1`` uniform-weight variant),
  * the MGF-vs-iterations experiment (ErrorVsIterations_probabilistic).

Everything is reconstructed from the same builders the runner scripts use
(experiment_configs.make_weights, the synthetic/empirical penalty builders, and
probability_profiles_probabilistic.make_q_profile), so the dump documents
exactly the configuration a run with the *current* environment variables would
use. Run after (or before) the experiments; it does not execute any policy.

Configuration: reads the same env vars the experiments read (INFOCOM_TITER,
INFOCOM_MC_TRIALS, INFOCOM_SEED, INFOCOM_PROFILES, INFOCOM_CHANNELS,
INFOCOM_SOURCES, INFOCOM_TASKS, INFOCOM_SUBGRADIENT_METHOD), so set them the same
way you set them for the runs you want documented.
"""
import os
import json
import numpy as np

from _bootstrap import paths

from experiment_configs import registry, make_weights, describe_weights
from probability_profiles_probabilistic import make_q_profile
from ErrorVsIterations_probabilistic import iteration_settings, REFERENCE_ITERS
from CompareSubgradientIterations_probabilistic import compare_config


SEED = int(os.environ.get("INFOCOM_SEED", "0"))


def _params_dir(flavor):
    base = (paths.DATA_PROBABILISTIC_DIR if flavor == "probabilistic"
            else paths.DATA_DETERMINISTIC_DIR)
    d = os.path.join(base, "params")
    os.makedirs(d, exist_ok=True)
    return d


def _matrix_block(mat, indent="    "):
    """Pretty-print a 2D array with row indices, capped for readability."""
    mat = np.asarray(mat)
    with np.printoptions(precision=4, suppress=True, linewidth=200,
                         threshold=10_000):
        text = np.array2string(mat, separator=", ")
    return "\n".join(indent + line for line in text.splitlines())


def _value_summary(mat):
    """Return {value: count} for the distinct entries of a matrix."""
    mat = np.asarray(mat, dtype=float)
    vals, counts = np.unique(np.round(mat, 6), return_counts=True)
    return {float(v): int(c) for v, c in zip(vals, counts)}


def _representative_dims(entry):
    """Pick a representative (M, km) for weights/q given the swept variable."""
    fixed = entry["fixed"]
    sweep = entry["sweep"]
    vals = list(sweep["values"]) or [None]
    M = fixed.get("M")
    km = fixed.get("km")
    N = fixed.get("N")
    if sweep["name"] == "M":
        M = max(vals)
    elif sweep["name"] == "km":
        km = max(vals)
    elif sweep["name"] == "N":
        N = "swept"
    return M, km, N


def _q_profile_stats(profiles, M, km, seed):
    """Summary stats per q profile at a representative (M, km)."""
    out = []
    for prof in profiles or []:
        try:
            q, meta = make_q_profile(prof, M, km, seed=seed)
            out.append(dict(profile=prof, q_min=meta["q_min"],
                            q_mean=meta["q_mean"], q_max=meta["q_max"],
                            q_std=meta["q_std"],
                            description=meta.get("description", "")))
        except Exception as exc:  # pragma: no cover - defensive
            out.append(dict(profile=prof, error=str(exc)))
    return out


def _write_sweep_experiment(entry):
    """Write txt + json for one registry (sweep) experiment. Returns json dict."""
    stem = entry["stem"]
    flavor = entry["flavor"]
    fixed = entry["fixed"]
    sweep = entry["sweep"]
    M_rep, km_rep, N_rep = _representative_dims(entry)
    w = make_weights(entry["family"], int(M_rep), int(km_rep),
                     entry["weights_mode"])
    sg = entry["subgradient"]
    is_prob = flavor == "probabilistic"

    q_stats = (_q_profile_stats(entry.get("profiles"), int(M_rep),
                                int(km_rep), SEED)
               if is_prob else None)

    # ---- JSON ----
    j = dict(
        stem=stem, family=entry["family"], flavor=flavor,
        description=entry["description"],
        dimensions=dict(
            M=fixed.get("M"), N=fixed.get("N"), km=fixed.get("km"),
            B=fixed.get("B"), T=fixed.get("T"),
            K=fixed.get("T"), gamma=fixed.get("gamma"),
        ),
        sweep=dict(variable=sweep["name"], values=list(sweep["values"]),
                   env_var=sweep["env"]),
        resources=dict(n_channel_cost=entry["n_value"],
                       c_compute_cap=entry["c_value"]),
        penalty=entry["penalty"],
        weights=dict(mode=entry["weights_mode"],
                     pattern=describe_weights(entry["family"],
                                              entry["weights_mode"]),
                     representative_dims=dict(M=int(M_rep), km=int(km_rep)),
                     value_summary=_value_summary(w),
                     matrix=np.asarray(w).tolist()),
        policies=entry["policies"],
        solver=dict(method=sg.get("method"), iterations=sg.get("titer"),
                    iterations_env=sg.get("titer_env"),
                    mc_trials=sg.get("mc_trials"), seed=sg.get("seed")),
        q_profiles=q_stats,
        outputs=dict(
            data_csv=f"data/{flavor}/{stem}_data.csv",
            plot_png=f"plots/{flavor}/{stem}.png",
            summary_csv=(f"data/{flavor}/{stem}_summary.csv"
                         if is_prob else None),
            q_profiles_npz=(f"data/{flavor}/{stem}_q_profiles.npz"
                            if is_prob else None),
        ),
    )

    # ---- TXT ----
    L = []
    L.append("=" * 70)
    L.append(f"EXPERIMENT: {stem}")
    L.append("=" * 70)
    L.append(f"flavor      : {flavor}")
    L.append(f"family      : {entry['family']}")
    L.append(f"description : {entry['description']}")
    L.append("")
    L.append("PROBLEM DIMENSIONS")
    for k in ("M", "N", "km", "B", "T"):
        v = fixed.get(k)
        if v is None and sweep["name"] == k:
            v = f"swept ({sweep['name']})"
        L.append(f"  {k:<6}= {v}")
    L.append(f"  {'K':<6}= {fixed.get('T')}  (K = T)")
    L.append(f"  {'gamma':<6}= {fixed.get('gamma')}")
    L.append("")
    L.append("SWEEP")
    L.append(f"  variable : {sweep['name']}")
    L.append(f"  values   : {list(sweep['values'])}")
    L.append(f"  env var  : {sweep['env']}")
    L.append("")
    L.append("RESOURCES")
    L.append(f"  n (channel cost per pair)  : {entry['n_value']} "
             f"(uniform over all (m, j))")
    L.append(f"  c (compute cap per source) : {entry['c_value']} "
             f"(uniform over all m)")
    L.append("")
    L.append("PENALTY MODEL")
    L.append(f"  {entry['penalty']}")
    L.append("")
    L.append(f"WEIGHTS  (mode: {entry['weights_mode']})")
    L.append(f"  pattern : {describe_weights(entry['family'], entry['weights_mode'])}")
    L.append(f"  value summary (value: count): {_value_summary(w)}")
    L.append(f"  representative matrix (M={int(M_rep)} x km={int(km_rep)}):")
    L.append(_matrix_block(w))
    L.append("")
    L.append("POLICIES")
    L.append(f"  {', '.join(entry['policies'])}")
    L.append("")
    L.append("SUBGRADIENT / SOLVER")
    L.append(f"  method     : {sg.get('method')}")
    L.append(f"  iterations : {sg.get('titer')}  (env {sg.get('titer_env')})")
    if is_prob:
        L.append(f"  mc_trials  : {sg.get('mc_trials')}  (env INFOCOM_MC_TRIALS)")
    L.append(f"  seed       : {sg.get('seed')}  (env INFOCOM_SEED)")
    L.append("")
    if is_prob:
        L.append("RELIABILITY q PROFILES")
        L.append(f"  representative point: M={int(M_rep)}, km={int(km_rep)}, "
                 f"seed={SEED}")
        L.append(f"  {'profile':<44}{'min':>7}{'mean':>8}{'max':>7}{'std':>8}")
        for s in q_stats:
            if "error" in s:
                L.append(f"  {s['profile']:<44}  ERROR: {s['error']}")
            else:
                L.append(f"  {s['profile']:<44}{s['q_min']:7.3f}"
                         f"{s['q_mean']:8.3f}{s['q_max']:7.3f}{s['q_std']:8.3f}")
        L.append(f"  full q matrices (per profile, per sweep value) stored in:")
        L.append(f"    data/{flavor}/{stem}_q_profiles.npz "
                 f"(keys: <profile>_{sweep['name']}<value>)")
        L.append("")
    L.append("OUTPUTS")
    L.append(f"  data : data/{flavor}/{stem}_data.csv")
    if is_prob:
        L.append(f"         data/{flavor}/{stem}_summary.csv")
        L.append(f"         data/{flavor}/{stem}_q_profiles.npz")
    L.append(f"  plot : plots/{flavor}/{stem}.png")
    L.append("")

    pdir = _params_dir(flavor)
    with open(os.path.join(pdir, f"{stem}_parameters.txt"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(pdir, f"{stem}_parameters.json"), "w") as f:
        json.dump(j, f, indent=2)
    return j


def _write_iterations_experiment():
    """Write txt + json for the MGF-vs-iterations experiment."""
    stem = "ErrorVsIterations_probabilistic"
    flavor = "probabilistic"
    iters = os.environ.get("INFOCOM_ITERATIONS", "1,2,4,8,16,32")
    iter_list = [int(x) for x in iters.split(",") if x.strip()]
    settings = iteration_settings()

    settings_json = []
    L = []
    L.append("=" * 70)
    L.append(f"EXPERIMENT: {stem}")
    L.append("=" * 70)
    L.append("flavor      : probabilistic (q = 1 everywhere)")
    L.append("description : MGF discounted-error objective vs. number of "
             "subgradient")
    L.append("              iterations (local episodes). q=1 makes the "
             "probabilistic MGF")
    L.append("              reduce exactly to the deterministic MGF1 (MATLAB "
             "1:1 port),")
    L.append("              so the curve is also a q=1 sanity check.")
    L.append("")
    L.append("ITERATION COUNTS SWEPT (local episodes)")
    L.append(f"  {iter_list}   (env INFOCOM_ITERATIONS)")
    L.append(f"  reference iterations : {REFERENCE_ITERS}  "
             f"(env INFOCOM_ITER_REFERENCE)")
    L.append(f"  q : 1.0 for every (source, task) pair")
    L.append(f"  subgradient method : episode1_mstep")
    L.append(f"  seed : {SEED}")
    L.append("")
    L.append("SETTINGS (each a single fixed problem point)")
    for s in settings:
        M, km = s["M"], s["km"]
        w = make_weights(s["experiment"], M, km, "deterministic")
        L.append("-" * 60)
        L.append(f"  {s['name']}")
        L.append(f"    M = {M}, N = {s['N']}, km = {km}, B = {s['B']}, "
                 f"T = {s['T']}, gamma = {s['gamma']}")
        L.append(f"    penalty : {s['penalty_kind']}")
        L.append(f"    n = 1.0 (uniform), c = 2.0 (uniform)")
        L.append(f"    weights : {describe_weights(s['experiment'], 'deterministic')}")
        L.append(f"    weight value summary: {_value_summary(w)}")
        L.append(f"    weight matrix (M={M} x km={km}):")
        L.append(_matrix_block(w, indent="      "))
        settings_json.append(dict(
            name=s["name"], M=M, N=s["N"], km=km, B=s["B"], T=s["T"],
            gamma=s["gamma"], penalty=s["penalty_kind"],
            n_channel_cost=1.0, c_compute_cap=2.0,
            q="1.0 everywhere",
            weights=dict(mode="deterministic",
                         pattern=describe_weights(s["experiment"],
                                                  "deterministic"),
                         value_summary=_value_summary(w),
                         matrix=np.asarray(w).tolist()),
        ))
    L.append("")
    L.append("OUTPUTS")
    L.append(f"  data : data/{flavor}/{stem}_data.csv")
    L.append(f"  plot : plots/{flavor}/{stem}.png  (combined)")
    L.append(f"         plots/{flavor}/ErrorVsIterations_<setting>_"
             f"probabilistic.png  (per setting)")
    L.append("")

    j = dict(stem=stem, flavor=flavor,
             description="MGF objective vs number of subgradient iterations "
                         "(q=1).",
             iterations=iter_list, reference_iterations=REFERENCE_ITERS,
             q="1.0 everywhere", subgradient_method="episode1_mstep",
             seed=SEED, settings=settings_json,
             outputs=dict(data_csv=f"data/{flavor}/{stem}_data.csv",
                          plot_png=f"plots/{flavor}/{stem}.png"))

    pdir = _params_dir(flavor)
    with open(os.path.join(pdir, f"{stem}_parameters.txt"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(pdir, f"{stem}_parameters.json"), "w") as f:
        json.dump(j, f, indent=2)
    return j


def _write_compare_iterations_experiment():
    """Write txt + json for the subgradient-method-vs-iterations comparison."""
    cfg = compare_config()
    stem = cfg["stem"]
    flavor = "probabilistic"

    q_stats = _q_profile_stats(cfg["profiles"], cfg["M"], cfg["km"], SEED)

    L = []
    L.append("=" * 70)
    L.append(f"EXPERIMENT: {stem}")
    L.append("=" * 70)
    L.append("flavor      : probabilistic")
    L.append("description : probabilistic MGF discounted-error objective vs. the")
    L.append("              number of subgradient episodes (iterations), one curve")
    L.append("              per subgradient method, one subplot per q-profile.")
    L.append("")
    L.append("PROBLEM DIMENSIONS")
    for k in ("M", "N", "km", "B", "T"):
        L.append(f"  {k:<6}= {cfg[k]}")
    L.append(f"  {'K':<6}= {cfg['T']}  (K = T)")
    L.append(f"  {'gamma':<6}= {cfg['gamma']}")
    L.append("")
    L.append("RESOURCES")
    L.append(f"  n (channel cost per pair)  : {cfg['n_value']} (uniform)")
    L.append(f"  c (compute cap per source) : {cfg['c_value']} (uniform)")
    L.append("")
    L.append("PENALTY MODEL")
    L.append(f"  {cfg['penalty']}")
    L.append("")
    L.append("WEIGHTS")
    L.append(f"  {cfg['weights']} (isolates the optimizer)")
    L.append("")
    L.append("EPISODES (subgradient iteration counts swept)")
    L.append(f"  {cfg['iterations']}   (env INFOCOM_COMPARE_ITERATIONS)")
    L.append("")
    L.append("SUBGRADIENT METHODS COMPARED")
    for m in cfg["methods"]:
        L.append(f"  - {m}")
    L.append("")
    L.append("SOLVER / MC")
    L.append(f"  mc_trials : {cfg['mc_trials']}  (env INFOCOM_MC_TRIALS)")
    L.append(f"  seed      : {cfg['seed']}  (env INFOCOM_SEED)")
    L.append("")
    L.append("RELIABILITY q PROFILES")
    L.append(f"  representative point: M={cfg['M']}, km={cfg['km']}, seed={SEED}")
    L.append(f"  {'profile':<44}{'min':>7}{'mean':>8}{'max':>7}{'std':>8}")
    for s in q_stats:
        if "error" not in s:
            L.append(f"  {s['profile']:<44}{s['q_min']:7.3f}"
                     f"{s['q_mean']:8.3f}{s['q_max']:7.3f}{s['q_std']:8.3f}")
    L.append("")
    L.append("OUTPUTS")
    L.append(f"  data : data/{flavor}/{stem}_data.csv")
    L.append(f"  plot : plots/{flavor}/{stem}.png")
    L.append("")

    j = dict(stem=stem, flavor=flavor,
             description="Probabilistic MGF objective vs number of subgradient "
                         "episodes, per method.",
             dimensions=dict(M=cfg["M"], N=cfg["N"], km=cfg["km"], B=cfg["B"],
                             T=cfg["T"], K=cfg["T"], gamma=cfg["gamma"]),
             resources=dict(n_channel_cost=cfg["n_value"],
                            c_compute_cap=cfg["c_value"]),
             penalty=cfg["penalty"], weights=cfg["weights"],
             episodes=cfg["iterations"], methods=cfg["methods"],
             mc_trials=cfg["mc_trials"], seed=cfg["seed"],
             q_profiles=q_stats,
             outputs=dict(data_csv=f"data/{flavor}/{stem}_data.csv",
                          plot_png=f"plots/{flavor}/{stem}.png"))

    pdir = _params_dir(flavor)
    with open(os.path.join(pdir, f"{stem}_parameters.txt"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(pdir, f"{stem}_parameters.json"), "w") as f:
        json.dump(j, f, indent=2)
    return j


def _write_master_index(all_json):
    """Write python_code/EXPERIMENT_PARAMETERS.md aggregating everything."""
    lines = []
    lines.append("# Experiment Parameters")
    lines.append("")
    lines.append("Auto-generated by `src/execute/ExportExperimentParameters.py`. "
                 "It records the full parameter set of every experiment (every "
                 "generated plot). Per-experiment detail (including the full "
                 "weight matrix and q-profile statistics) lives next to the data "
                 "in `data/<flavor>/params/<stem>_parameters.txt` (human "
                 "readable) and `.json` (machine readable).")
    lines.append("")
    lines.append("Values reflect the environment variables in effect when this "
                 "script was run (`INFOCOM_TITER`, `INFOCOM_MC_TRIALS`, "
                 "`INFOCOM_SEED`, `INFOCOM_PROFILES`, sweep ranges, ...).")
    lines.append("")
    lines.append("## Index")
    lines.append("")
    lines.append("| Experiment | Flavor | Weights | Sweep | Dimensions | Solver |")
    lines.append("|---|---|---|---|---|---|")
    for j in all_json:
        if j.get("sweep"):
            dims = j["dimensions"]
            dim_str = ", ".join(
                f"{k}={v}" for k, v in
                [("M", dims["M"]), ("N", dims["N"]), ("km", dims["km"]),
                 ("B", dims["B"]), ("T", dims["T"]), ("gamma", dims["gamma"])]
                if v is not None)
            sweep = j["sweep"]
            sweep_str = f"{sweep['variable']} in {sweep['values']}"
            solver = j["solver"]
            solver_str = f"{solver['method']}, iters={solver['iterations']}"
            if solver.get("mc_trials") is not None:
                solver_str += f", mc={solver['mc_trials']}"
            wmode = j["weights"]["mode"]
            lines.append(f"| `{j['stem']}` | {j['flavor']} | {wmode} | "
                         f"{sweep_str} | {dim_str} | {solver_str} |")
        elif j.get("settings") is not None:
            # MGF-vs-iterations experiment
            lines.append(f"| `{j['stem']}` | {j['flavor']} | deterministic | "
                         f"iters in {j['iterations']} | 3 settings (q=1) | "
                         f"{j['subgradient_method']} |")
        else:
            # subgradient-method-vs-iterations comparison
            dims = j["dimensions"]
            dim_str = f"M={dims['M']}, km={dims['km']}, N={dims['N']}"
            lines.append(f"| `{j['stem']}` | {j['flavor']} | ones | "
                         f"episodes in {j['episodes']} | {dim_str} | "
                         f"{len(j['methods'])} methods |")
    lines.append("")

    # Per-experiment detail blocks
    for j in all_json:
        lines.append(f"## {j['stem']}")
        lines.append("")
        lines.append(f"- **flavor**: {j['flavor']}")
        lines.append(f"- **description**: {j['description']}")
        if j.get("sweep"):
            dims = j["dimensions"]
            lines.append(f"- **dimensions**: " + ", ".join(
                f"{k} = {v}" for k, v in dims.items() if v is not None))
            lines.append(f"- **sweep**: `{j['sweep']['variable']}` over "
                         f"{j['sweep']['values']} (env "
                         f"`{j['sweep']['env_var']}`)")
            lines.append(f"- **resources**: n (channel cost) = "
                         f"{j['resources']['n_channel_cost']}, c (compute cap) "
                         f"= {j['resources']['c_compute_cap']}")
            lines.append(f"- **penalty**: {j['penalty']}")
            lines.append(f"- **weights** ({j['weights']['mode']}): "
                         f"{j['weights']['pattern']}; value counts "
                         f"{j['weights']['value_summary']}")
            lines.append(f"- **policies**: {', '.join(j['policies'])}")
            s = j["solver"]
            solver_bits = [f"method = {s['method']}",
                           f"iterations = {s['iterations']}"]
            if s.get("mc_trials") is not None:
                solver_bits.append(f"mc_trials = {s['mc_trials']}")
            solver_bits.append(f"seed = {s['seed']}")
            lines.append(f"- **solver**: " + ", ".join(solver_bits))
            if j.get("q_profiles"):
                lines.append(f"- **q profiles** ({len(j['q_profiles'])}): " +
                             ", ".join(p["profile"] for p in j["q_profiles"]))
            outs = j["outputs"]
            lines.append(f"- **outputs**: `{outs['data_csv']}`, "
                         f"`{outs['plot_png']}`")
        elif j.get("settings") is not None:
            lines.append(f"- **iterations**: {j['iterations']} "
                         f"(reference {j['reference_iterations']})")
            lines.append(f"- **q**: {j['q']}")
            lines.append(f"- **subgradient method**: {j['subgradient_method']}")
            lines.append("- **settings**:")
            for st in j["settings"]:
                lines.append(f"  - `{st['name']}`: M={st['M']}, N={st['N']}, "
                             f"km={st['km']}, B={st['B']}, T={st['T']}, "
                             f"gamma={st['gamma']}; penalty: {st['penalty']}; "
                             f"weights {st['weights']['value_summary']}")
            outs = j["outputs"]
            lines.append(f"- **outputs**: `{outs['data_csv']}`, "
                         f"`{outs['plot_png']}`")
        else:
            dims = j["dimensions"]
            lines.append(f"- **dimensions**: " + ", ".join(
                f"{k} = {v}" for k, v in dims.items() if v is not None))
            lines.append(f"- **episodes**: {j['episodes']} (env "
                         f"INFOCOM_COMPARE_ITERATIONS)")
            lines.append(f"- **weights**: {j['weights']}")
            lines.append(f"- **penalty**: {j['penalty']}")
            lines.append(f"- **methods** ({len(j['methods'])}): "
                         f"{', '.join(j['methods'])}")
            lines.append(f"- **mc_trials**: {j['mc_trials']}, **seed**: "
                         f"{j['seed']}")
            if j.get("q_profiles"):
                lines.append(f"- **q profiles** ({len(j['q_profiles'])}): " +
                             ", ".join(p["profile"] for p in j["q_profiles"]))
            outs = j["outputs"]
            lines.append(f"- **outputs**: `{outs['data_csv']}`, "
                         f"`{outs['plot_png']}`")
        lines.append("")
        lines.append(f"_Full detail_: `data/{j['flavor']}/params/"
                     f"{j['stem']}_parameters.txt`")
        lines.append("")

    out = os.path.join(paths.PROJECT_ROOT, "EXPERIMENT_PARAMETERS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    return out


def main():
    print("Exporting experiment parameters...")
    all_json = []
    for entry in registry():
        j = _write_sweep_experiment(entry)
        all_json.append(j)
        print(f"  wrote params for {entry['stem']} ({entry['flavor']})")
    j_iter = _write_iterations_experiment()
    all_json.append(j_iter)
    print(f"  wrote params for {j_iter['stem']} (probabilistic)")

    try:
        j_cmp = _write_compare_iterations_experiment()
        all_json.append(j_cmp)
        print(f"  wrote params for {j_cmp['stem']} (probabilistic)")
    except Exception as exc:   # never let this break the rest of the export
        print(f"  WARNING: could not export CompareSubgradientIterations "
              f"params: {exc}")

    md = _write_master_index(all_json)
    print(f"\nWrote master index: {md}")
    print(f"Per-experiment files under: "
          f"data/deterministic/params/ and data/probabilistic/params/")


if __name__ == "__main__":
    main()
