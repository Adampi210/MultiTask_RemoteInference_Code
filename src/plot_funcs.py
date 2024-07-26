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

# Plot with variance
def plot_pk_values_with_variance(csv_file_path, output_image_path, title):
    # Read data from CSV
    k_values = []
    avg_values = []
    var_values = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            k_values.append(int(row[0]))
            avg_values.append(float(row[1]))
            var_values.append(float(row[2]))

    # Plotting
    plt.figure(figsize = (12, 8))
    
    # Plot average line
    plt.plot(k_values, avg_values, linestyle = '-', linewidth = 2, markersize = 8, color = 'blue')
    
    # Plot variance area
    std_dev = np.sqrt(var_values)
    plt.fill_between(k_values, np.array(avg_values) - std_dev, np.array(avg_values) + std_dev, 
                     alpha = 0.3, color = 'lightblue')

    plt.title(title, fontsize = FONTSIZE_LARGE)
    plt.xlabel('k', fontsize = FONTSIZE_SMALL)
    plt.ylabel('Loss', fontsize = FONTSIZE_SMALL)
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.xlim(0, max(k_values))
    plt.ylim(min(np.array(avg_values) - std_dev) * 0.99, max(np.array(avg_values)) * 1.01)  # 10% margin on top
    plt.xticks(fontsize = int(FONTSIZE_SMALL * 0.75))
    plt.yticks(fontsize = int(FONTSIZE_SMALL * 0.75))
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_image_path, dpi = 300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # csv_file_path = '../data/sb-camera5-0820am-0835am_multi_k_loss_result_seed_0_pk_values.csv'
    # plotted_loss_file = 'plotted_pk_ex.png'
    # plot_pk_values(csv_file_path, plotted_loss_file)
    csv_file_path = '../data/averaged_detection_test_loss_pk_data.csv'
    output_file = 'averaged_detection.png'
    plot_pk_values_with_variance(csv_file_path, output_file, 'Detection Loss For Different k')
    plot_pk_values_with_variance('../data/smooth_segmentation_averaged_multi_k_loss_pk_data.csv', 'smooth_seg_loss.png', 'Smooth Segmentation Loss for Different k')
    plot_pk_values_with_variance('../data/smooth_averaged_detection_test_loss_pk_data.csv', 'smooth_avg_detect_loss.png', 'Smooth Detection Loss for Different k')
    plot_pk_values_with_variance('../data/smooth_detection_simple_test_loss_pk_data.csv', 'smooth_detect_simple_loss.png', 'Smooth Simple Detection Loss for Different k')
    
