"""
compare_with_matlab.py

For every Python-generated plot, build a *_compare.png that places the Python
plot and the MATLAB reference side-by-side (and overlays the numerical series
when MATLAB raw data is available in a .fig).

Outputs written next to this script:
    ErrorVsChannel_compare.png
    ErrorVsChannelsmodel_compare.png
    ErrorVsSources_compare.png
    ErrorVsSourcesM_compare.png
    ErrorVsTasks_compare.png
    lossread_seg_compare.png        (Segmentation errorbar vs MLexp.fig)
    lossread_det_compare.png        (Detection errorbar vs DetectionLoss.fig)
    lossread_yyaxis_compare.png     (yyaxis dual plot vs MLexp.fig + DetectionLoss.fig)

For policy plots (ErrorVsChannel*, ErrorVsSources*, ErrorVsTasks) the script
also prints a per-policy numerical diff table to stdout.

Notes on the MATLAB references:
- ErrorVsSourcesM has no MATLAB .fig with raw data, so its _compare just shows
  the Python plot next to the only intermediate MATLAB JPG (if any).
- SegmentationLoss.fig in the repo appears to come from a CSV that isn't in
  this directory (its y-values don't match any local file). We therefore use
  MLexp.fig as the canonical "segmentation" reference (its 'Segmentation'
  errorbar series matches the smoothed CSV used by lossread.py exactly).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_reader import extract_named_lines, extract_errorbars  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Policies that have a MATLAB reference (the original three) are listed first;
# MIEF was added in the Python port and has no MATLAB counterpart, so it shows
# up only on the Python panel of the side-by-side comparison.
POLICY_KEYS = ['prand', 'pmaf', 'pmgf']
POLICY_LABELS = ['Random Policy', 'MAF Policy', 'MGF Policy']
POLICY_COLORS = ['tab:red', 'tab:blue', 'tab:green']

PY_ONLY_KEY = 'pmief'
PY_ONLY_LABEL = 'MIEF (Python-only)'
PY_ONLY_COLOR = 'tab:purple'


def _load_csv(path):
    """Read a *_data.csv. Returns (x, dict-of-policies). Accepts both the
    legacy 4-column layout (prand, pmaf, pmgf) and the new 5-column layout
    that also has pmief at the end."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    out = {'prand': data[:, 1], 'pmaf': data[:, 2], 'pmgf': data[:, 3]}
    if data.shape[1] >= 5:
        out['pmief'] = data[:, 4]
    return data[:, 0], out


def _print_diff_table(title, xlabel, x_py, py, ml_series):
    print(f"\n=== {title} ===")
    for key, label in zip(POLICY_KEYS, POLICY_LABELS):
        if label not in ml_series:
            print(f"  {label}: not in MATLAB reference")
            continue
        x_ml, y_ml = ml_series[label]
        if np.array_equal(np.sort(x_ml), np.sort(x_py)):
            order_ml = np.argsort(x_ml)
            order_py = np.argsort(x_py)
            x_aligned = x_py[order_py]
            y_ml_aligned = y_ml[order_ml]
            y_py_aligned = py[key][order_py]
        else:
            x_aligned = x_py
            y_ml_aligned = np.interp(x_aligned, x_ml, y_ml)
            y_py_aligned = py[key]

        abs_err = np.abs(y_py_aligned - y_ml_aligned)
        rel_err = abs_err / np.maximum(np.abs(y_ml_aligned), 1e-12)
        print(f"  {label}:")
        print(f"    {xlabel:>5}  {'matlab':>14}  {'python':>14}  "
              f"{'abs_err':>10}  {'rel_err':>10}")
        for xv, yml, ypy, ae, re_ in zip(x_aligned, y_ml_aligned, y_py_aligned,
                                         abs_err, rel_err):
            print(f"    {xv:5g}  {yml:14.4f}  {ypy:14.4f}  "
                  f"{ae:10.4f}  {re_:10.2%}")


def _compare_policy_plot(py_csv, fig_file, title, xlabel, ylabel,
                         log_y, out_name):
    """Overlay + side-by-side Python vs MATLAB for a 3-line policy plot."""
    py_path = os.path.join(HERE, py_csv)
    fig_path = os.path.join(ROOT, fig_file)
    if not os.path.isfile(py_path):
        print(f"[skip] missing Python CSV: {py_path}")
        return
    if not os.path.isfile(fig_path):
        print(f"[skip] missing MATLAB .fig: {fig_path}")
        return

    x_py, py = _load_csv(py_path)
    ml_series = extract_named_lines(fig_path)
    _print_diff_table(title, xlabel, x_py, py, ml_series)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_py, ax_ml = axes

    for key, label, color in zip(POLICY_KEYS, POLICY_LABELS, POLICY_COLORS):
        ax_py.plot(x_py, py[key], color=color, marker='*', linestyle='-',
                   linewidth=2, label=label)
        if label in ml_series:
            x_ml, y_ml = ml_series[label]
            ax_ml.plot(x_ml, y_ml, color=color, marker='o', linestyle='-',
                       linewidth=2, label=label)

    # Extra Python-only policy (MIEF). No MATLAB counterpart, so it only
    # appears on the Python panel.
    if PY_ONLY_KEY in py:
        ax_py.plot(x_py, py[PY_ONLY_KEY], color=PY_ONLY_COLOR,
                   marker='s', linestyle='--', linewidth=2,
                   label=PY_ONLY_LABEL)

    for ax, who in [(ax_py, 'Python'), (ax_ml, 'MATLAB')]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale('log')
        ax.set_title(f'{title}  -  {who}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"  saved {out}")
    plt.close(fig)


def _emit_python_only_compare(py_png, title, note, out_name):
    """When MATLAB has no reference for this plot, emit a side-by-side image
    with the Python plot on the left and an explanatory text panel on the
    right so the file naming convention is preserved."""
    py_path = os.path.join(HERE, py_png)
    if not os.path.isfile(py_path):
        print(f"[skip] missing Python png: {py_path}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    img = mpimg.imread(py_path)
    axes[0].imshow(img)
    axes[0].set_axis_off()
    axes[0].set_title(f'{title}  -  Python')

    axes[1].set_axis_off()
    axes[1].set_title(f'{title}  -  MATLAB  (no reference)')
    axes[1].text(0.5, 0.5, note, ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=11,
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow',
                           edgecolor='gray'))
    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"  saved {out}")
    plt.close(fig)


def _compare_with_jpg(py_png, jpg_path, title, out_name):
    """Side-by-side: Python png on the left, MATLAB jpg on the right."""
    py_path = os.path.join(HERE, py_png)
    if not os.path.isfile(py_path):
        print(f"[skip] missing Python png: {py_path}")
        return
    if not os.path.isfile(jpg_path):
        print(f"[skip] missing MATLAB jpg: {jpg_path}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, img_path, who in [(axes[0], py_path, 'Python'),
                              (axes[1], jpg_path, 'MATLAB')]:
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f'{title}  -  {who}')
    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"  saved {out}")
    plt.close(fig)


def _compare_errorbar(py_csv_x, py_csv_y, py_csv_err, fig_file, eb_index,
                      title, xlabel, ylabel, out_name, x_is_y=False):
    """Compare an errorbar series in a MATLAB .fig against Python data.

    py_csv_* are arrays (in-memory). eb_index is which errorbarseries to use
    if the MATLAB fig has more than one.
    """
    fig_path = os.path.join(ROOT, fig_file)
    if not os.path.isfile(fig_path):
        print(f"[skip] missing MATLAB fig: {fig_path}")
        return
    ebs = extract_errorbars(fig_path)
    if eb_index >= len(ebs):
        print(f"[skip] {fig_file} only has {len(ebs)} errorbar series")
        return
    ml = ebs[eb_index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_py, ax_ml = axes

    if x_is_y:
        ax_py.errorbar(py_csv_y, py_csv_x, xerr=py_csv_err, fmt='o-', color='C0')
    else:
        ax_py.errorbar(py_csv_x, py_csv_y, yerr=py_csv_err, fmt='o-', color='C0')
    ax_py.set_xlabel(xlabel)
    ax_py.set_ylabel(ylabel)
    ax_py.set_title(f'{title}  -  Python')
    ax_py.grid(True, alpha=0.3)

    if x_is_y:
        ax_ml.errorbar(ml['y'], ml['x'], xerr=ml['L'], fmt='o-', color='C1')
    else:
        ax_ml.errorbar(ml['x'], ml['y'], yerr=ml['L'], fmt='o-', color='C1')
    ax_ml.set_xlabel(xlabel)
    ax_ml.set_ylabel(ylabel)
    ax_ml.set_title(f'{title}  -  MATLAB ({fig_file})')
    ax_ml.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"  saved {out}")
    plt.close(fig)


def main():
    # ---------------- policy plots ----------------
    _compare_policy_plot('ErrorVsChannel_data.csv', 'ErrorVsChannel1.fig',
                         'ErrorVsChannel', 'Number of Channels',
                         'Discounted Sum of Errors',
                         log_y=False, out_name='ErrorVsChannel_compare.png')

    _compare_policy_plot('ErrorVsChannelsmodel_data.csv',
                         'ErrorsVsChannelsmodel.fig',
                         'ErrorVsChannelsmodel', 'channel',
                         'Discounted Sum of Errors',
                         log_y=True,
                         out_name='ErrorVsChannelsmodel_compare.png')

    _compare_policy_plot('ErrorVsSources_data.csv', 'ErrorVsSourcesModel.fig',
                         'ErrorVsSources', 'Number of Sources',
                         'Discounted Sum of Errors',
                         log_y=True,
                         out_name='ErrorVsSources_compare.png')

    # ErrorVsSourcesM.m has NO saved MATLAB output (no .eps/.fig/.jpg with
    # name ErrorVsSourcesM.*). The closest .fig in the repo is for a
    # different script (ErrorVsSources.m) and has incompatible parameters
    # (sources 2..20 vs 3..21, linear y-range 144..160 vs MGF<1). Do not
    # cross-compare those — they would be misleading. We emit a
    # Python-only "_compare" panel with an explanatory note.
    _emit_python_only_compare(
        'ErrorVsSourcesM.png',
        'ErrorVsSourcesM',
        ("No MATLAB output for this script is saved in the repo.\n"
         "ErrorVsSourcesM.m exists but no .fig/.jpg/.eps was committed,\n"
         "so the Python output cannot be cross-checked numerically here.\n"
         "(Note: ErrorVsSources.jpg is from a different .m with different\n"
         "parameters and should NOT be used as a reference.)"),
        'ErrorVsSourcesM_compare.png',
    )

    _compare_policy_plot('ErrorVsTasks_data.csv', 'ErrorVsTasks.fig',
                         'ErrorVsTasks', 'Number of Tasks',
                         'Discounted Sum of Errors',
                         log_y=True,
                         out_name='ErrorVsTasks_compare.png')

    # ---------------- lossread plots ----------------
    seg_csv = os.path.join(ROOT,
                           'smooth_segmentation_averaged_multi_k_loss_pk_data.csv')
    det_csv = os.path.join(ROOT,
                           'smooth_averaged_detection_test_loss_pk_data.csv')
    if os.path.isfile(seg_csv) and os.path.isfile(det_csv):
        Mseg = np.loadtxt(seg_csv, delimiter=',', skiprows=1)
        Mdet = np.loadtxt(det_csv, delimiter=',', skiprows=1)
        AoI_seg = Mseg[1:, 0]
        p1 = 100.0 * Mseg[1:, 1]
        var1 = 100.0 * Mseg[1:, 2]
        AoI_det = Mdet[1:, 0]
        p2 = Mdet[1:, 1]
        var2 = Mdet[1:, 2]

        # Segmentation errorbar: compare to MLexp.fig (its 'Segmentation' eb
        # matches the smoothed CSV used by lossread.py exactly). Use vertical
        # errorbars on both sides for a clean visual comparison even though
        # lossread.m figure(1) uses horizontal style.
        _compare_errorbar(AoI_seg, p1, var1, 'MLexp.fig', 0,
                          'Segmentation Loss (lossread fig1)', 'AoI',
                          'Inference Error 100(1-IoU)',
                          'lossread_fig1_compare.png',
                          x_is_y=False)

        # Detection errorbar: compare to DetectionLoss.fig.
        _compare_errorbar(AoI_det, p2, var2, 'DetectionLoss.fig', 0,
                          'Detection Loss (lossread fig2)',
                          'AoI', 'Inference Error (MSE)',
                          'lossread_fig2_compare.png')

    # yyaxis-style combined plot: side-by-side Python png vs MLexp.jpg.
    mlexp_jpg = os.path.join(ROOT, 'MLexp.jpg')
    if os.path.isfile(mlexp_jpg):
        _compare_with_jpg('lossread_fig3.png', mlexp_jpg,
                          'Combined yyaxis (lossread fig3)',
                          'lossread_fig3_compare.png')


if __name__ == "__main__":
    main()
