import matplotlib.pyplot as plt
import csv
import numpy as np

FONTSIZE_LARGE = 20
FONTSIZE_SMALL = 16

# Plot the pk loss for segmenter
def plot_pk_values(csv_file_path, output_image_path):
    # Read data from CSV
    k_values = []
    pk_values = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            k_values.append(int(row[0]))
            pk_values.append(float(row[1]))

    # Plotting
    plt.figure(figsize = (12, 8))
    plt.plot(k_values, pk_values, marker = 'o', linestyle = '-', linewidth = 2, markersize = 8)
    plt.title('Calculated Average IoU loss for different K', fontsize = FONTSIZE_LARGE, fontweight = 'bold')
    plt.xlabel('k', fontsize = FONTSIZE_SMALL)
    plt.ylabel('p(k)', fontsize = FONTSIZE_SMALL)
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.xlim(0, max(k_values))
    plt.ylim(0, max(pk_values) * 1.1)  # 10% margin on top
    plt.xticks(k_values, fontsize = int(FONTSIZE_SMALL * 0.75))
    plt.yticks(np.arange(0, max(pk_values), 0.1), fontsize = int(FONTSIZE_SMALL * 0.75))
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_image_path, dpi = 300, bbox_inches = 'tight')

    return

if __name__ == '__main__':
    csv_file_path = '../data/sb-camera5-0820am-0835am_multi_k_loss_result_seed_0_pk_values.csv'
    plotted_loss_file = 'plotted_pk_ex.png'
    plot_pk_values(csv_file_path, plotted_loss_file)
