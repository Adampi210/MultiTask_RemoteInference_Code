import concurrent.futures
import os
import cv2
from data_funcs import get_frames
from NN_funcs import VehicleDetectorNN

def process_frame(frame_num, frame, detector):
    count = detector.count_vehicles(frame)
    annotated_frame = detector.draw_vehicles(frame)
    return frame_num, count, annotated_frame

def main(video_path):
    detector = VehicleDetectorNN()
    frame_generator = get_frames(video_path)

    data_dir = '../data/'
    os.makedirs(data_dir, exist_ok = True)

    with concurrent.futures.ThreadPoolExecutor(max_workers = 1) as executor:
        futures = []
        for frame_num, frame in enumerate(frame_generator):
            future = executor.submit(process_frame, frame_num, frame, detector)
            futures.append(future)
       
        for future in concurrent.futures.as_completed(futures):
            frame_num, count, annotated_frame = future.result()
            print(f"Frame {frame_num} has {count} vehicles")
            if frame_num % 10 == 0:
                cv2.putText(annotated_frame, f"Vehicles: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                filename = os.path.join(data_dir, f"video_frame_{frame_num}.jpg")
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved frame {frame_num} with {count} vehicles")

dir_level = '../../..'
data_dir = "%s/data/NGSIM_traffic_data/" % dir_level

if __name__ == "__main__":
    video_path = data_dir + "nb-camera2-0400pm-0415pm.avi"
    print(video_path)
    main(video_path)