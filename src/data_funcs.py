import cv2
import os 
from collections import deque

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

# Test data processing code
if __name__ == "__main__":
    # Test Data history class
    frame_history = DataHistory(num_data = 5, time_offset = 1)
    
    # Simulate 10 frames
    for i in range(15):
        frame = f"Frame_{i}"  # Standin for the frame
        segmentation = f"Seg_{i}"  # Standin for the segmentation
        vehicle_count = i  # Standin for the number of vehicles
        
        frame_history.update(frame, segmentation, vehicle_count)
        
        # Check once filled
        if i >= 4: 
            frames, segs, counts = frame_history.get_history()
            print(f"Current frame: {i}")
            print(f"Stored frames: {frames}")
            print(f"Stored segmentations: {segs}")
            print(f"Stored vehicle counts: {counts}")
            print()

    # Test video file retrieval
    dir_level = '../../..'
    data_path = "%s/data/NGSIM_traffic_data/" % dir_level
    video_files = get_video_files(data_path)
    print(f"There are {len(video_files)} video files:")
    for fil in video_files:
        print(fil)

    # Test frame retreival
    if video_files:
        frames = get_frames(video_files[3])
        for i, frame in enumerate(frames):
            if i == 0:  
                save_img(frame, "../data/first_frame.jpg")
                print("Saved the image as first_frame.jpg")
                break
            
            
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
# email: preston smith and ask about the disk: https://www.rcac.purdue.edu/index.php/about/staff/psmith
# 