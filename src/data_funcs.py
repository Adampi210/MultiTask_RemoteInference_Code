import cv2
import os
import re 
import csv
import random
import numpy as np
import torch
import json
from collections import deque
from scipy.ndimage import gaussian_filter1d

# Set all seeds to specified value
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Return a generator object with video frames
def get_frames(video_path):
    # Get the video object
    vid_cv2 = cv2.VideoCapture(video_path)
    # read the frames (using yield to call consecutively for different frames)
    while vid_cv2.isOpened():
        success, frame = vid_cv2.read()
        if success:
            # cv2.imwrite('../data/example_frame.jpg', frame) # Example frame
            yield frame
            # return frame
        else:
            break
    # Close the object
    vid_cv2.release()
    return 

# Get a list of all video data
def get_video_files(dir_data):
    video_files = []
    for file in os.listdir(dir_data):
        if file.endswith('.avi'):
            video_files.append(os.path.join(dir_data, file))
    return video_files

# Save the image
def save_img(img, fname):
    cv2.imwrite(fname, img)

# For maintaining a sliding window of data and predictions
class DataHistory:
    def __init__(self, num_data, time_offset):
        self.num_data = num_data
        self.time_offset = time_offset
        self.data = deque(maxlen = num_data)
        self.segmentations = deque(maxlen = num_data)
        self.vehicle_counts = deque(maxlen = num_data)
        self.buffer = deque(maxlen = time_offset)
        self.curr_idx = 0
        
    # Add newest frame and predictions to the queue
    def update(self, frame, segmentation = None, vehicle_count = None):
        self.curr_idx += 1
        if self.time_offset == 0:
            self.data.append(frame)
            self.segmentations.append(segmentation)
            self.vehicle_counts.append(vehicle_count)
        else:
            if self.curr_idx <= self.time_offset:
                self.buffer.append((frame, segmentation, vehicle_count))
            else:
                if len(self.buffer) == self.time_offset:
                    old_frame, old_seg, old_count = self.buffer.popleft()
                    self.data.append(old_frame)
                    self.segmentations.append(old_seg)
                    self.vehicle_counts.append(old_count)
                self.buffer.append((frame, segmentation, vehicle_count))
            
    # Get current window
    def get_history(self):
        if self.curr_idx < self.num_data + self.time_offset:
            return None, None, None
        return list(self.data), list(self.segmentations), list(self.vehicle_counts)

def get_subset_video_files(dir_data, n = 5, seed = None):
    # For deterministic operation set all seeds to specified seed
    if seed is not None:
        set_seed(seed)
    
    # Draw at least 1 from all
    video_files = []
    prefix_files = {
        'sb': [],
        'nb': [],
        'peachtree': [],
        'lankershim': []
    }
    
    for file in os.listdir(dir_data):
        if file.endswith('.avi'):
            full_path = os.path.join(dir_data, file)
            video_files.append(full_path)
            
            if file.startswith('sb-'):
                prefix_files['sb'].append(full_path)
            elif file.startswith('nb-'):
                prefix_files['nb'].append(full_path)
            elif file.startswith('peachtree-'):
                prefix_files['peachtree'].append(full_path)
            elif file.startswith('lankershim-'):
                prefix_files['lankershim'].append(full_path)
    
    # Have at least 1 from all roads
    selected_files = []
    for prefix in prefix_files:
        if prefix_files[prefix]:
            selected_files.append(random.choice(prefix_files[prefix]))
    
    # The rest draw randomly from all
    remaining = sorted(list(set(video_files) - set(selected_files)))
    additional_needed = max(0, n - len(selected_files))
    selected_files.extend(random.sample(remaining, min(additional_needed, len(remaining))))
    
    return selected_files

def average_pk_values(data_dir, output_file):
    pk_files = [f for f in os.listdir(data_dir) if '_pk_values' in f]
    pk_sum = {k: 0 for k in range(101)}
    file_count = 0

    for pk_file in pk_files:
        file_path = os.path.join(data_dir, pk_file)
        with open(file_path, 'r') as file_pk:
            reader = csv.reader(file_pk)
            next(reader)
            for row in reader:
                k = int(row[0])
                pk = float(row[1])
                pk_sum[k] += pk
        file_count += 1

    pk_avg = {k: pk_sum[k] / file_count for k in pk_sum}
    print(file_count)
    output_path = os.path.join(data_dir, output_file)
    with open(output_path, 'w') as file_pk_avg:
        writer = csv.writer(file_pk_avg)
        writer.writerow(['k', 'p(k)'])
        for k in range(1, 101):
            writer.writerow([k, pk_avg[k]])

def detection_pk_test_loss(data_dir, output_file):
    detect_loss_files = [f for f in os.listdir(data_dir) if 'detection_test_loss_k_' in f]
    k_vals = []
    seed_vals = []
    k_val_file_dict = {}
    for f in detect_loss_files:
        match = re.search(r'_k_(.+?)_seed_(.+?)', f)
        k = int(match.group(1))
        seed = int(match.group(2))
        k_vals.append(k)
        if k in k_val_file_dict:
            k_val_file_dict[k].append(f)
        else:
            k_val_file_dict[k] = [f,]
        seed_vals.append(seed)
    k_vals = sorted(k_vals)
    loss_val = {k: {seed: 0 for seed in seed_vals} for k in k_vals}
    seed_vals = sorted(list(set(seed_vals)))
    
    for k in k_vals:
        for f in k_val_file_dict[k]:
            data_file = os.path.join(data_dir, f)
            match = re.search(r'seed_(.+?)', f)
            seed = int(match.group(1))
            with open(data_file, 'r') as df:
                lines = df.readlines()
                loss_val[k][seed] = float(lines[-1].split(',')[1])
                
    k_values, avg_losses, variances = [], [], []
    for k in sorted(loss_val.keys()):
        losses = [loss_val[k][seed] for seed in seed_vals]
        avg_loss = np.mean(losses)
        variance = np.var(losses)
        k_values.append(k)
        avg_losses.append(avg_loss)
        variances.append(variance)
    
    # Write results to CSV
    with open(output_file, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'average_loss', 'variance'])
        for k, avg, var in zip(k_values, avg_losses, variances):
            writer.writerow([k, avg, var])

def simple_detection_loss(data_dir, output_file, seed = 0):
    set_seed(seed)

    def get_vehicle_detection_csv_files(data_dir):
        csv_files = []
        for file_data in os.listdir(data_dir):
            if file_data.endswith('.csv') and 'vehicle_detection_result_seed' in file_data:
                csv_files.append(os.path.join(data_dir, file_data))
        return csv_files
    
    def load_vehicle_count_data(csv_file):
        with open(csv_file, 'r') as file_data:
            reader = csv.reader(file_data)
            next(reader)  # Skip header
            return [int(row[1]) for row in reader]

    csv_files = get_vehicle_detection_csv_files(data_dir)
    max_k = 100
    total_losses = np.zeros(max_k + 1)
    file_count = 0

    for csv_file in csv_files:
        vehicle_counts = load_vehicle_count_data(csv_file)
        file_losses = np.zeros(max_k + 1)

        for k in range(1, max_k + 1):
            predictions = vehicle_counts[:-k]
            actuals = vehicle_counts[k:]
            
            mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
            file_losses[k] = mse

        total_losses += file_losses
        file_count += 1

    average_losses = total_losses / file_count

    # Write results to output file
    with open(output_file, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'test loss'])
        for k in range(1, max_k + 1):
            writer.writerow([k, average_losses[k]])

    return average_losses

def calculate_moving_average_and_variance(data, window_size = 5):
    moving_avg = np.convolve(data, np.ones(window_size), 'valid') / window_size
    padding = np.full(window_size - 1, np.mean(data[-1 - window_size:-1]))
    moving_avg = np.concatenate((moving_avg, padding))
    variance = np.array([np.var(data[max(0, i - window_size) : i + 1]) for i in range(len(data))])
    
    return moving_avg, variance

def process_segmentation_data(input_file, output_file):
    k_values, pk_values = [], []
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            k_values.append(int(row[0]))
            pk_values.append(float(row[1]))
    
    moving_avg, variance = calculate_moving_average_and_variance(pk_values)
    
    with open(output_file, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'average_loss', 'variance'])
        for k, avg, var in zip(k_values, moving_avg, variance):
            writer.writerow([k, avg, var])

def process_simple_detection_data(input_file, output_file):
    k_values, loss_values = [], []
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            k_values.append(int(row[0]))
            loss_values.append(float(row[1]))
    
    moving_avg, variance = calculate_moving_average_and_variance(loss_values)
    
    with open(output_file, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'average_loss', 'variance'])
        for k, avg, var in zip(k_values, moving_avg, variance):
            writer.writerow([k, avg, var])

def smooth_lstm_detection_data(input_file, output_file, smoothing_factor = 2):
    k_values, avg_losses, variances = [], [], []
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            k_values.append(int(row[0]))
            avg_losses.append(float(row[1]))
            variances.append(float(row[2]))
    
    smoothed_avg = gaussian_filter1d(avg_losses, sigma = smoothing_factor)
    smoothed_var = gaussian_filter1d(variances, sigma = smoothing_factor)
    
    with open(output_file, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'average_loss', 'variance'])
        for k, avg, var in zip(k_values, smoothed_avg, smoothed_var):
            writer.writerow([k, avg, var])


# Test data processing code
if __name__ == "__main__":
    # Test Data history class
    average_pk_values('../data/', '../data/segmentation_averaged_multi_k_loss_pk_data.csv')
    process_segmentation_data('../data/segmentation_averaged_multi_k_loss_pk_data.csv', '../data/smooth_segmentation_averaged_multi_k_loss_pk_data.csv')
    detection_pk_test_loss('../data/', '../data/averaged_detection_test_loss_pk_data.csv')
    smooth_lstm_detection_data('../data/averaged_detection_test_loss_pk_data.csv', '../data/smooth_averaged_detection_test_loss_pk_data.csv')
    simple_detection_loss('../data/', '../data/detection_simple_test_loss_pk_data.csv')      
    process_simple_detection_data('../data/detection_simple_test_loss_pk_data.csv', '../data/smooth_detection_simple_test_loss_pk_data.csv')

# Try SAM
# Complete 1 task perfectly (Counting the vehicles)
# About the LSTM: 
#   - Count the number of vehicles, predict the number of vehicles for next frame
#   - Pretrain an LSTM model for the task
#   - The LSTM should be pretrained offline on the prediction data
#   - For the segmentation, potentially use past frame for future prediction
#   - Add inference error function of the LSTM for predicting the k (number of vehicles)
#   - Have multiple LSTMs for different k values, where k is the future predicted offset. 
# - Get the loss of the LSTM 
# 
