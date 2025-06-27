import cv2
from ultralytics import YOLO
import numpy as np
import os
import glob
import json

# ======================================================================================
# Bounding Box Merging Function (Unchanged)
# ======================================================================================
def merge_close_bboxes(bboxes, distance_threshold):
    """
    Merges bounding boxes that are close to each other based on the Euclidean distance of their centroids.
    """
    if not bboxes:
        return np.array([])
    bboxes = np.array(bboxes)
    centroids = np.vstack([
        (bboxes[:, 0] + bboxes[:, 2]) / 2,
        (bboxes[:, 1] + bboxes[:, 3]) / 2
    ]).T
    groups = []
    visited = [False] * len(bboxes)
    for i in range(len(bboxes)):
        if visited[i]:
            continue
        current_group_indices = [i]
        visited[i] = True
        queue = [i]
        while queue:
            current_index = queue.pop(0)
            for j in range(len(bboxes)):
                if not visited[j]:
                    distance = np.linalg.norm(centroids[current_index] - centroids[j])
                    if distance < distance_threshold:
                        visited[j] = True
                        current_group_indices.append(j)
                        queue.append(j)
        groups.append(current_group_indices)
    merged_bboxes = []
    for group_indices in groups:
        group_bboxes = bboxes[group_indices]
        min_x = np.min(group_bboxes[:, 0])
        min_y = np.min(group_bboxes[:, 1])
        max_x = np.max(group_bboxes[:, 2])
        max_y = np.max(group_bboxes[:, 3])
        merged_bboxes.append([min_x, min_y, max_x, max_y])
    return np.array(merged_bboxes)

# ======================================================================================
# Updated Function to Process a Single Image
# ======================================================================================
def process_image(image_path, model, distance_threshold, y_cutoff):
    """
    Loads an image, performs object detection, filters by Y-coordinate, merges bboxes, 
    and returns structured data.

    Args:
        image_path (str): Path to the input image.
        model (YOLO): The loaded YOLO model object.
        distance_threshold (int): The distance threshold for merging boxes.
        y_cutoff (int): The Y-axis coordinate to ignore detections above.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected object.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"    - Warning: Could not read image at {image_path}")
        return []

    results = model.predict(source=img, conf=0.05, verbose=False)
    result = results[0]

    boxes_by_class = {}
    for box in result.boxes:
        # Extract coordinates and class ID
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        class_id = int(box.cls[0])

        # ==================================================
        # NEW: Filter out objects above the Y_CUTOFF line
        # If the top of the box is above the cutoff line, skip it.
        if y1 < y_cutoff:
            continue
        # ==================================================
            
        if class_id not in boxes_by_class:
            boxes_by_class[class_id] = []
        boxes_by_class[class_id].append([x1, y1, x2, y2])

    final_detections = []
    for class_id, bboxes in boxes_by_class.items():
        merged_bboxes = merge_close_bboxes(bboxes, distance_threshold)
        for mbox in merged_bboxes:
            final_detections.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "box_2d": [int(coord) for coord in mbox]
            })

    return final_detections

# ======================================================================================
# Main Script Logic
# ======================================================================================
def main():
    """
    Main function to run the batch processing.
    """
    # --- Configuration ---
    MODEL_PATH = 'yolov8m.pt'
    
    FOLDERS_TO_PROCESS = [
        '../../data/session1/',
        '../../data/session2/',
    ]
    
    DISTANCE_THRESHOLD = 150
    
    # ===============================================================================
    # NEW SETTING: Y-axis cutoff
    # Any object with a bounding box starting above this Y-pixel value will be ignored.
    # The Y-axis starts at 0 at the top of the image.
    # Set to 0 to disable this filter and process the whole image.
    Y_CUTOFF = 250
    # ===============================================================================

    print(f"Loading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for folder_path in FOLDERS_TO_PROCESS:
        if not os.path.isdir(folder_path):
            print(f"\nSkipping non-existent directory: {folder_path}")
            continue

        print(f"\nProcessing directory: {folder_path}")
        
        image_files = sorted(glob.glob(os.path.join(folder_path, 'frame_*.jpg')))
        
        if not image_files:
            print("  - No files matching 'frame_*.jpg' found in this directory.")
            continue
            
        all_results_for_folder = {}

        for image_path in image_files:
            filename = os.path.basename(image_path)
            print(f"  - Processing file: {filename}")
            
            # Pass the new Y_CUTOFF value to the processing function
            detected_objects = process_image(image_path, model, DISTANCE_THRESHOLD, Y_CUTOFF)
            
            all_results_for_folder[filename] = detected_objects

        json_output_path = os.path.join(folder_path, 'detection_results.json')
        print(f"\nSaving results for {folder_path} to {json_output_path}")
        with open(json_output_path, 'w') as f:
            json.dump(all_results_for_folder, f, indent=4)
        print("  - Save complete.")

    print("\nBatch processing finished.")


if __name__ == '__main__':
    main()