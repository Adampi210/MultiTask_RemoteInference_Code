"""
ErrorVsChannel_steps.py - robot-loss ErrorVsChannel swept over the Episode1
M-step count (how many projected sub-update passes the subgradient method does
per timestep).

Same problem as ErrorVsChannel.py (penalties from the robot-collected loss.mat).
The outer subgradient iteration count (titer) is held fixed at the MATLAB
setting (10000); what we vary is Episode1's inner "M-step": the MATLAB M-pass
projection block is repeated `mult` times per timestep, for mult in
{1, 2, 4, 8, 16}. Since M=20 here, the total number of projected sub-updates per
timestep is mult * M = {20, 40, 80, 160, 320}. mult==1 (20 M-steps) is exactly
the MATLAB configuration.

Only MGF depends on the M-step count -- MAF, MIEF and Random never touch the
dual, so they are computed once and reused as the shared baseline in every plot
(this also keeps Random's RNG stream identical to ErrorVsChannel.py, preserving
the verified ~0.3% match to MATLAB).

For each mult we write
    data/deterministic/ErrorVsChannel_steps_{mult}_data.csv
    plots/deterministic/ErrorVsChannel_steps_{mult}.png
MATLAB only has the single 20-M-step (mult==1) configuration, so the side-by-side
MATLAB comparison (ErrorVsChannel_steps_1_compare.png + diff table) is produced
ONLY for mult==1 with titer==10000. A combined MGF-vs-channel overlay across all
M-step counts is also written for the convergence story.

Reads:  data/deterministic/loss.mat
Writes: data/deterministic/ErrorVsChannel_steps_{mult}_data.csv
        plots/deterministic/ErrorVsChannel_steps_{mult}.png
        plots/deterministic/ErrorVsChannel_steps_mgf_convergence.png
        plots/deterministic/ErrorVsChannel_steps_1_compare.png   (mult==1 only)
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
from multiprocessing import Pool

from _bootstrap import paths  # bootstraps sys.path to find src/main/

from MGF1 import MGF1
from MAF1 import MAF1
from MIEF1 import MIEF1
from randpolicy import randpolicy


TITER = int(os.environ.get("INFOCOM_TITER", "10000"))   # outer subgradient steps (fixed)
# mult = how many times Episode1's M-pass projection block is repeated per
# timestep. Total M-steps per timestep = mult * M.
MULTS = [int(x) for x in
         os.environ.get("INFOCOM_STEPS_MULTS", "1,2,4,8,16").split(",")]
SEED = int(os.environ.get("INFOCOM_SEED", "0"))
N_WORKERS = int(os.environ.get("INFOCOM_STEPS_WORKERS", "20"))


def _build_problem():
    """Identical setup to ErrorVsChannel.py (robot-loss penalties)."""
    M = 20
    km = 2
    B = 20
    T = 100
    K = T
    gamma = 0.9

    n = np.ones((M, km))
    c = np.ones((M, km)) * 2

    # MATLAB branches set w(m,j)=0.01 then override specific cells.
    w = np.full((M, km), 0.01)
    w[0, 1] = 1   # MATLAB w(1, 2) = 1
    w[4, 0] = 1   # MATLAB w(5, 1) = 1

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
        if j_py < p2.size:
            p[1, i] = p2[j_py]
        i += 1

    channel = np.arange(2, 21, 2)
    return dict(M=M, km=km, B=B, T=T, K=K, gamma=gamma,
                n=n, c=c, w=w, p=p, channel=channel)


def _mgf_worker(task):
    """Compute one MGF value for a (mult, channel) pair.

    save_multipliers=False so concurrent workers never touch multipliers.mat.
    m_step_repeats=mult sets the Episode1 M-step count for this run.
    """
    mult, ch_idx, N, titer, prob = task
    val = MGF1(prob['M'], N, prob['km'], prob['T'], prob['B'], prob['K'],
               prob['n'], prob['c'], prob['w'], prob['gamma'], prob['p'],
               titer=titer, verbose=False, save_multipliers=False,
               m_step_repeats=mult)
    return mult, ch_idx, val


def main():
    prob = _build_problem()
    M = prob['M']
    channel = prob['channel']
    nch = len(channel)

    # ---- 1) M-step-independent baselines, computed once ----------------
    # Seed once and call the policies in the same per-channel order as
    # ErrorVsChannel.py so randpolicy's RNG stream (and thus its MATLAB match)
    # is byte-identical. MAF/MIEF are RNG-free.
    np.random.seed(SEED)
    pmaf = np.zeros(nch)
    pmief = np.zeros(nch)
    prand = np.zeros(nch)
    for ch in range(nch):
        N = int(channel[ch])
        pmaf[ch] = MAF1(M, N, prob['km'], prob['T'], prob['B'], prob['K'],
                        prob['n'], prob['c'], prob['w'], prob['gamma'],
                        prob['p'], verbose=False)
        pmief[ch] = MIEF1(M, N, prob['km'], prob['T'], prob['B'], prob['K'],
                          prob['n'], prob['c'], prob['w'], prob['gamma'],
                          prob['p'], verbose=False)
        prand[ch] = randpolicy(M, N, prob['km'], prob['T'], prob['B'],
                               prob['K'], prob['n'], prob['c'], prob['w'],
                               prob['gamma'], prob['p'], verbose=False)
    print(f"baselines done (MAF/MIEF/Random); channels={list(channel)}")

    # ---- 2) MGF for every (mult, channel), in parallel -----------------
    tasks = []
    for mult in MULTS:
        for ch in range(nch):
            tasks.append((mult, ch, int(channel[ch]), TITER, prob))
    # Largest M-step count is slowest -> schedule those first.
    tasks.sort(key=lambda tk: tk[0], reverse=True)

    pmgf = {mult: np.zeros(nch) for mult in MULTS}
    nproc = min(N_WORKERS, len(tasks))
    print(f"running {len(tasks)} MGF jobs on {nproc} workers "
          f"(M={M}, m_step mults={MULTS} -> {[m*M for m in MULTS]} M-steps, "
          f"titer={TITER}) ...")
    t0 = time.perf_counter()
    done = 0
    with Pool(processes=nproc) as pool:
        for mult, ch_idx, val in pool.imap_unordered(_mgf_worker, tasks):
            pmgf[mult][ch_idx] = val
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} MGF jobs done "
                      f"({time.perf_counter() - t0:.0f}s)")
    print(f"all MGF jobs done in {time.perf_counter() - t0:.0f}s")

    # ---- 3) per-mult CSV + PNG -----------------------------------------
    for mult in MULTS:
        msteps = mult * M
        csv_out = paths.data_path(f'ErrorVsChannel_steps_{mult}_data.csv',
                                  probabilistic=False)
        np.savetxt(csv_out,
                   np.column_stack([channel, prand, pmaf, pmgf[mult], pmief]),
                   delimiter=',', header='channel,prand,pmaf,pmgf,pmief',
                   comments='')

        plt.figure()
        plt.plot(channel, prand, 'ro-', label='Random Policy',
                 linewidth=2, markersize=10)
        plt.plot(channel, pmaf, 'b*-', label='MAF Policy',
                 linewidth=2, markersize=10)
        plt.plot(channel, pmgf[mult], 'g-', label='MGF Policy',
                 linewidth=2, markersize=10)
        plt.plot(channel, pmief, 'ms--',
                 label='MIEF (Max Instantaneous Error First)',
                 linewidth=2, markersize=10)
        plt.xlabel('Number of Channels')
        plt.ylabel('Discounted Sum of Errors')
        plt.legend()
        plt.title(f'ErrorVsChannel  (Episode1 M-steps = {msteps} '
                  f'= {mult}x{M}, titer={TITER})')
        png_out = paths.plot_path(f'ErrorVsChannel_steps_{mult}.png',
                                  probabilistic=False)
        plt.savefig(png_out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  M-steps={msteps:>4} (mult={mult:>2}): saved "
              f"{os.path.basename(csv_out)} + {os.path.basename(png_out)}")

    # ---- 4) combined MGF-vs-channel overlay across M-step counts -------
    plt.figure()
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(MULTS)))
    for color, mult in zip(cmap, MULTS):
        plt.plot(channel, pmgf[mult], '-o', color=color, linewidth=2,
                 markersize=6, label=f'MGF, {mult*M} M-steps ({mult}x{M})')
    plt.xlabel('Number of Channels')
    plt.ylabel('Discounted Sum of Errors (MGF)')
    plt.title(f'MGF vs Episode1 M-step count (robot loss, titer={TITER})')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    conv_png = paths.plot_path('ErrorVsChannel_steps_mgf_convergence.png',
                               probabilistic=False)
    plt.savefig(conv_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {os.path.basename(conv_png)}")

    # ---- 5) MATLAB comparison ONLY for the MATLAB config (mult==1) -----
    # MATLAB uses exactly M (=20) M-steps and titer=10000.
    if 1 in MULTS and TITER == 10000:
        from compare_with_matlab import _compare_policy_plot
        _compare_policy_plot(
            'ErrorVsChannel_steps_1_data.csv', 'ErrorVsChannel1.fig',
            f'ErrorVsChannel (Episode1 M-steps={M}, MATLAB config)',
            'Number of Channels', 'Discounted Sum of Errors', log_y=False,
            out_name='ErrorVsChannel_steps_1_compare.png')
    else:
        print("  [matlab compare] skipped: MATLAB only matches mult==1 "
              "(M M-steps) with titer==10000.")


if __name__ == "__main__":
    main()
