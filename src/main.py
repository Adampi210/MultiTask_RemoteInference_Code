import concurrent.futures
import os
import cv2
from data_funcs import *
from NN_funcs import *

dir_level = "../../.."
model_weight_path = "%s/models/" % dir_level
# results_dir = "/scratch/gilbreth/apiasecz/results/NGSIM_traffic_data_results/segmentation_masks"
results_dir = '../data/'

if __name__ == "__main__":
    N_FILES = 5
    seed = 0
    dir_data_path = '/scratch/gilbreth/apiasecz/data/NGSIM_traffic_data/'
    # video_files = sorted(get_video_files(dir_data_path))
    video_files = get_subset_video_files(dir_data_path, N_FILES, seed)
    i = 0
    for video_file in video_files:        
        i += 1
        if i >= 3:
            break
        print(video_file)
        calculate_multi_k_loss(video_file, seed = seed, start_frame = None, end_frame = None, results_dir = results_dir, max_k = 100)
        # calculate_vehicle_detection(video_file, seed = seed, start_frame = None, end_frame = None, results_dir = results_dir)
    exit()
