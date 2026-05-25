"""
lossread.py - 1:1 Python port of lossread.m

Reads the smoothed segmentation / detection loss CSVs, plots them with error
bars, writes loss.mat-equivalent npz (and also writes loss.mat in this
directory so MGF/MAF scripts can find it via the project root).

MATLAB original:
  - figure 1: errorbar(AoI, p1, variance1, 'horizontal')
  - figure 2: errorbar(AoI, p2, variance2)
  - figure 3: yyaxis left / right errorbars
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def _read_csv(path):
    # Match MATLAB's readmatrix: skip the header row (first row of M is dropped
    # by `M(2:end,...)` in the .m file).
    return np.loadtxt(path, delimiter=',', skiprows=1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    seg_csv = os.path.join(root, 'smooth_segmentation_averaged_multi_k_loss_pk_data.csv')
    det_csv = os.path.join(root, 'smooth_averaged_detection_test_loss_pk_data.csv')

    Mseg = _read_csv(seg_csv)
    # MATLAB: p1 = 100*M(2:end, 2); variance1 = 100*M(2:end, 3); AoI = M(2:end, 1)
    # Our loadtxt already skipped the header row -> M(2:end, *) = our row 1: onward.
    p1 = 100.0 * Mseg[1:, 1]
    AoI1 = Mseg[1:, 0]
    variance1 = 100.0 * Mseg[1:, 2]

    plt.figure(1)
    plt.errorbar(AoI1, p1, xerr=variance1, fmt='o-')  # MATLAB 'horizontal' -> xerr
    plt.ylabel('Inference Error 100(1-IoU)')
    plt.xlabel('AoI')
    plt.title('Segmentation loss')
    plt.savefig(os.path.join(here, 'lossread_fig1.png'), dpi=150, bbox_inches='tight')

    Mdet = _read_csv(det_csv)
    p2 = Mdet[1:, 1]
    AoI2 = Mdet[1:, 0]
    variance2 = Mdet[1:, 2]

    plt.figure(2)
    plt.errorbar(AoI2, p2, yerr=variance2, fmt='o-')
    plt.ylabel('Inference Error (MSE)')
    plt.xlabel('AoI')
    plt.title('Detection loss')
    plt.savefig(os.path.join(here, 'lossread_fig2.png'), dpi=150, bbox_inches='tight')

    # MATLAB: save('loss.mat', 'p1', 'p2')
    out_mat = os.path.join(here, 'loss.mat')
    sio.savemat(out_mat,
                {'p1': p1.reshape(-1, 1), 'p2': p2.reshape(-1, 1)})

    fig3, ax_left = plt.subplots()
    ax_left.errorbar(AoI1, p1, yerr=variance1, fmt='o-', color='C0',
                     label='Segmentation')
    ax_left.set_xlabel('AoI')
    ax_left.set_ylabel('Inference Error 100(1-IoU)', color='C0')
    ax_left.tick_params(axis='y', labelcolor='C0')

    ax_right = ax_left.twinx()
    ax_right.errorbar(AoI2, p2, yerr=variance2, fmt='s-', color='C1',
                      label='Traffic Prediction')
    ax_right.set_ylabel('Inference Error (MSE)', color='C1')
    ax_right.tick_params(axis='y', labelcolor='C1')

    lines, labels = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines + lines2, labels + labels2, loc='best')
    fig3.tight_layout()
    fig3.savefig(os.path.join(here, 'lossread_fig3.png'), dpi=150, bbox_inches='tight')

    print(f"Wrote {out_mat}")
    plt.show()


if __name__ == "__main__":
    main()
