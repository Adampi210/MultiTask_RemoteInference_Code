"""
experiment_configs.py - shared experiment configuration: the single source of
truth for (a) the weight matrices used by every sweep and (b) a machine-readable
registry of every experiment's parameters (consumed by
ExportExperimentParameters.py).

Two weight modes exist for the probabilistic experiments:

  * ``deterministic`` (the default) -- the heterogeneous weights of the matching
    deterministic / paper figure, so the probabilistic sweep is a true
    probabilistic counterpart of the deterministic experiment.
  * ``ones`` -- ``w = 1`` everywhere.  Outputs from this mode are tagged with the
    ``_weights_1`` suffix and live alongside the default outputs.

The deterministic weight patterns reproduced here match the original 1:1-port
scripts exactly:

  * ErrorVsChannel (empirical loss):  w = 0.01 everywhere, with w[0,1] = 1 and
    w[4,0] = 1 (MATLAB w(1,2)=1, w(5,1)=1).
  * ErrorVsChannelsmodel / ErrorVsSources / ErrorVsTasks:  w[m,j] = 1 for the
    first half of the sources (m+1 <= M/2), else 0.01.

The registry at the bottom (``registry()``) intentionally mirrors the dimensions
hard-coded in the runner scripts; it is the documentation/inspection surface used
by the parameter-export script and is kept in sync with the scripts by hand.
"""
import os
import numpy as np


# ---------------------------------------------------------------------------
# Weight modes
# ---------------------------------------------------------------------------

WEIGHTS_DETERMINISTIC = "deterministic"
WEIGHTS_ONES = "ones"
WEIGHT_MODES = (WEIGHTS_DETERMINISTIC, WEIGHTS_ONES)


def weights_suffix(mode):
    """Output-stem suffix for a weight mode ('' for deterministic, '_weights_1')."""
    if mode == WEIGHTS_DETERMINISTIC:
        return ""
    if mode == WEIGHTS_ONES:
        return "_weights_1"
    raise ValueError(f"Unknown weight mode {mode!r}; choose from {WEIGHT_MODES}")


def _half_weights(M, km, high=1.0, low=0.01):
    """w[m,j] = high for the first half of the sources (m+1 <= M/2), else low.

    Matches ErrorVsChannelsmodel.m / ErrorVsSources.m / ErrorVsTasks.m exactly:
        if (m+1) > M/2 -> 0.01   else -> 1
    """
    M = int(M)
    km = int(km)
    w = np.full((M, km), float(low))
    for m in range(M):
        if (m + 1) <= M / 2:
            w[m, :] = float(high)
    return w


def _channel_empirical_weights(M, km):
    """ErrorVsChannel.m weight pattern: 0.01 everywhere, two priority cells = 1.

    MATLAB: w(:)=0.01; w(1,2)=1; w(5,1)=1  (1-indexed) ->  w[0,1]=1; w[4,0]=1.
    """
    M = int(M)
    km = int(km)
    w = np.full((M, km), 0.01)
    if M >= 1 and km >= 2:
        w[0, 1] = 1.0
    if M >= 5 and km >= 1:
        w[4, 0] = 1.0
    return w


# Which deterministic weight pattern each experiment family uses.
_DETERMINISTIC_WEIGHT_BUILDER = {
    "ErrorVsChannel": _channel_empirical_weights,
    "ErrorVsChannelsmodel": _half_weights,
    "ErrorVsSources": _half_weights,
    "ErrorVsTasks": _half_weights,
}


def make_weights(experiment, M, km, mode=WEIGHTS_DETERMINISTIC):
    """Build the (M, km) weight matrix for ``experiment`` in the given mode.

    Parameters
    ----------
    experiment : one of ErrorVsChannel / ErrorVsChannelsmodel / ErrorVsSources /
                 ErrorVsTasks.
    M, km      : problem dimensions.
    mode       : 'deterministic' (heterogeneous, matches the paper figure) or
                 'ones' (w = 1 everywhere).
    """
    M = int(M)
    km = int(km)
    if mode == WEIGHTS_ONES:
        return np.ones((M, km))
    if mode == WEIGHTS_DETERMINISTIC:
        builder = _DETERMINISTIC_WEIGHT_BUILDER.get(experiment)
        if builder is None:
            raise ValueError(
                f"No deterministic weight pattern for experiment {experiment!r}; "
                f"known: {sorted(_DETERMINISTIC_WEIGHT_BUILDER)}"
            )
        return builder(M, km)
    raise ValueError(f"Unknown weight mode {mode!r}; choose from {WEIGHT_MODES}")


def describe_weights(experiment, mode):
    """One-line human-readable description of the weight matrix for a mode."""
    if mode == WEIGHTS_ONES:
        return "w = 1 for every (source, task) pair  [weights_1]"
    if experiment == "ErrorVsChannel":
        return ("w = 0.01 everywhere except w[0,1]=1 and w[4,0]=1 "
                "(two priority pairs; MATLAB w(1,2)=1, w(5,1)=1)")
    return ("w = 1 for the first half of the sources (m+1 <= M/2), "
            "else w = 0.01")


# ---------------------------------------------------------------------------
# Registry of every experiment's parameters (for ExportExperimentParameters.py)
# ---------------------------------------------------------------------------

def _default_int_list(env_name, default):
    """Parse a comma-separated int list from env, falling back to ``default``."""
    s = os.environ.get(env_name)
    if not s:
        return list(default)
    return [int(x) for x in s.split(",") if x.strip()]


def registry():
    """Return a list of experiment-parameter dicts (mirrors the runner scripts).

    Each entry has keys:
        stem, family, flavor, description, sweep (name/values/env), fixed
        (M/N/km/B/T/gamma as available), n_value, c_value, penalty, weights_mode,
        policies, subgradient (method/titer/mc_trials/seed env + defaults),
        profiles (list or None).

    Values reflect the *current environment* (same env vars the scripts read), so
    the export documents exactly what a run with the current settings would use.
    """
    det_titer = int(os.environ.get("INFOCOM_TITER", "10000"))
    prob_titer = int(os.environ.get("INFOCOM_TITER", "1000"))
    mc_trials = int(os.environ.get("INFOCOM_MC_TRIALS", "10"))
    seed = int(os.environ.get("INFOCOM_SEED", "0"))

    # Probabilistic default profile list comes from the profile module.
    try:
        from probability_profiles_probabilistic import list_q_profiles
        all_profiles = list_q_profiles()
    except Exception:
        all_profiles = None
    prof_env = os.environ.get("INFOCOM_PROFILES")
    profiles = ([p.strip() for p in prof_env.split(",") if p.strip()]
                if prof_env else all_profiles)

    det_policies = ["Random", "MAF", "MGF", "MIEF"]
    prob_policies = ["MGF", "MAF", "MIEF", "Random"]

    entries = []

    # ---- Deterministic 1:1-port sweeps -----------------------------------
    entries.append(dict(
        stem="ErrorVsChannel", family="ErrorVsChannel", flavor="deterministic",
        description="Sweep channels N; empirical penalties from loss.mat (km=2).",
        sweep=dict(name="N", values=_default_int_list("INFOCOM_CHANNELS",
                                                       range(2, 21, 2)),
                   env="INFOCOM_CHANNELS"),
        fixed=dict(M=20, km=2, B=20, T=100, gamma=0.9),
        n_value=1.0, c_value=2.0, penalty="empirical loss.mat (p1, p2)",
        weights_mode=WEIGHTS_DETERMINISTIC, policies=det_policies,
        subgradient=dict(method="harmonic (Episode1 M-step)", titer=det_titer,
                         titer_env="INFOCOM_TITER", seed=seed),
        profiles=None,
    ))
    entries.append(dict(
        stem="ErrorVsChannelsmodel", family="ErrorVsChannelsmodel",
        flavor="deterministic",
        description="Sweep channels N; synthetic 9-task penalty model.",
        sweep=dict(name="N", values=_default_int_list("INFOCOM_CHANNELS",
                                                       range(2, 21, 2)),
                   env="INFOCOM_CHANNELS"),
        fixed=dict(M=20, km=9, B=20, T=100, gamma=0.9),
        n_value=1.0, c_value=2.0,
        penalty="synthetic 9-task (j%3: linear / 10*log / exp(0.5 d))",
        weights_mode=WEIGHTS_DETERMINISTIC, policies=det_policies,
        subgradient=dict(method="harmonic (Episode1 M-step)", titer=det_titer,
                         titer_env="INFOCOM_TITER", seed=seed),
        profiles=None,
    ))
    entries.append(dict(
        stem="ErrorVsSources", family="ErrorVsSources", flavor="deterministic",
        description="Sweep sources M; synthetic 9-task penalty model.",
        sweep=dict(name="M", values=_default_int_list("INFOCOM_SOURCES",
                                                       range(2, 21, 2)),
                   env="INFOCOM_SOURCES"),
        fixed=dict(N=10, km=9, B=20, T=100, gamma=0.9),
        n_value=1.0, c_value=2.0,
        penalty="synthetic 9-task (j%3: linear / 10*log / exp(0.5 d))",
        weights_mode=WEIGHTS_DETERMINISTIC, policies=det_policies,
        subgradient=dict(method="harmonic (Episode1 M-step)", titer=det_titer,
                         titer_env="INFOCOM_TITER", seed=seed),
        profiles=None,
    ))
    entries.append(dict(
        stem="ErrorVsTasks", family="ErrorVsTasks", flavor="deterministic",
        description="Sweep tasks-per-source km; synthetic penalty model.",
        sweep=dict(name="km", values=_default_int_list("INFOCOM_TASKS",
                                                        range(3, 16, 3)),
                   env="INFOCOM_TASKS"),
        fixed=dict(M=20, N=10, B=20, T=100, gamma=0.9),
        n_value=1.0, c_value=2.0,
        penalty="synthetic model (j%3: linear / 10*log / exp(0.5 d))",
        weights_mode=WEIGHTS_DETERMINISTIC, policies=det_policies,
        subgradient=dict(method="harmonic (Episode1 M-step)", titer=det_titer,
                         titer_env="INFOCOM_TITER", seed=seed),
        profiles=None,
    ))

    # ---- Probabilistic sweeps (both weight modes) ------------------------
    prob_specs = [
        dict(family="ErrorVsChannel", base="ErrorVsChannel_probabilistic",
             description="Sweep channels N; empirical loss.mat penalties (km=2).",
             sweep=dict(name="N", values=_default_int_list("INFOCOM_CHANNELS",
                                                           range(2, 21, 2)),
                        env="INFOCOM_CHANNELS"),
             fixed=dict(M=20, km=2, B=20, T=100, gamma=0.9),
             penalty="empirical loss.mat (p1, p2)"),
        dict(family="ErrorVsChannelsmodel",
             base="ErrorVsChannelsmodel_probabilistic",
             description="Sweep channels N; synthetic 9-task penalty model.",
             sweep=dict(name="N", values=_default_int_list("INFOCOM_CHANNELS",
                                                           range(2, 21, 2)),
                        env="INFOCOM_CHANNELS"),
             fixed=dict(M=20, km=9, B=20, T=100, gamma=0.9),
             penalty="synthetic 9-task (j%3: linear / 10*log / exp(0.5 d))"),
        dict(family="ErrorVsSources", base="ErrorVsSources_probabilistic",
             description="Sweep sources M; synthetic 9-task penalty model.",
             sweep=dict(name="M", values=_default_int_list("INFOCOM_SOURCES",
                                                           range(2, 21, 2)),
                        env="INFOCOM_SOURCES"),
             fixed=dict(N=10, km=9, B=20, T=100, gamma=0.9),
             penalty="synthetic 9-task (j%3: linear / 10*log / exp(0.5 d))"),
        dict(family="ErrorVsTasks", base="ErrorVsTasks_probabilistic",
             description="Sweep tasks-per-source km; synthetic penalty model.",
             sweep=dict(name="km", values=_default_int_list("INFOCOM_TASKS",
                                                            range(3, 16, 3)),
                        env="INFOCOM_TASKS"),
             fixed=dict(M=20, N=10, B=20, T=100, gamma=0.9),
             penalty="synthetic model (j%3: linear / 10*log / exp(0.5 d))"),
    ]
    prob_method = os.environ.get("INFOCOM_SUBGRADIENT_METHOD", "episode1_mstep")
    for spec in prob_specs:
        for mode in WEIGHT_MODES:
            entries.append(dict(
                stem=spec["base"] + weights_suffix(mode),
                family=spec["family"], flavor="probabilistic",
                description=spec["description"],
                sweep=spec["sweep"], fixed=spec["fixed"],
                n_value=1.0, c_value=2.0, penalty=spec["penalty"],
                weights_mode=mode, policies=prob_policies,
                subgradient=dict(method=prob_method, titer=prob_titer,
                                 titer_env="INFOCOM_TITER",
                                 mc_trials=mc_trials, seed=seed),
                profiles=profiles,
            ))

    return entries
