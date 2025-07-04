import json
import os
import numpy as np
import cv2
from ultralytics import YOLO

# List of folders to process (same as in your original code)
path_dirs = '../../..'
FOLDERS_TO_PROCESS = [
    f'{path_dirs}/robot_9/',
    # f'{path_dirs}/robot_1/',
    # f'{path_dirs}/robot_2/',
    # f'{path_dirs}/robot_3/',
    # f'{path_dirs}/robot_4/',
    # f'{path_dirs}/robot_5/',
    # f'{path_dirs}/robot_6/',
    # f'{path_dirs}/robot_7/',
]

y_cutoffs = [
    255, 
    # 300, 
    # 370, 
    # 370, 
    # 300, 
    # 300, 
    # 380, 
    # 380
]

# Number of previous frames to consider for average size comparison
N = 5

def identify_failed_detections(dir_to_process):
    """Identifies images where robot detection failed based on specified criteria."""
    didnt_work_list = []
    
    json_path = os.path.join(dir_to_process, 'detection_results.json')
    
    # Skip if JSON file doesn't exist
    if not os.path.exists(json_path):
        print(f"Warning: detection_results.json not found in {dir_to_process}")
        return
        
    # Load detection results
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Sort images to process them in sequence
    image_files = sorted(data.keys())
    
    # Track history of bbox sizes (only for frames with exactly 1 bbox)
    history_widths = []
    history_heights = []
    
    for filename in image_files:
        detections = data[filename]
        num_bboxes = len(detections)
        
        # Criterion 1: No bounding boxes detected
        # Criterion 2: More than one bounding box detected
        if num_bboxes == 0 or num_bboxes > 1:
            didnt_work_list.append(os.path.join(dir_to_process, filename))
            continue
            
        # Criterion 3: Size anomaly check (only if exactly 1 bbox)
        if num_bboxes == 1:
            box = detections[0]['box_2d']
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Check size against history if available
            if history_widths:
                avg_width = sum(history_widths[-N:]) / len(history_widths[-N:])
                avg_height = sum(history_heights[-N:]) / len(history_heights[-N:])
                if width >= 2 * avg_width or height >= 2 * avg_height:
                    didnt_work_list.append(os.path.join(dir_to_process, filename))
                    continue
            
            # Update history with current bbox size
            history_widths.append(width)
            history_heights.append(height)

    return didnt_work_list

def merge_close_bboxes(bboxes, distance_threshold):
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

def compute_hull_bbox(bboxes):
    if len(bboxes) == 0:
        return None
    points = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        points.append([x1, y1])
        points.append([x2, y2])
        points.append([x1, y2])
        points.append([x2, y1])
    points = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points)
    min_x = np.min(hull[:, 0, 0])
    min_y = np.min(hull[:, 0, 1])
    max_x = np.max(hull[:, 0, 0])
    max_y = np.max(hull[:, 0, 1])
    return [int(min_x), int(min_y), int(max_x), int(max_y)]


def draw_line_and_shade_above(image, m, b, alpha=0.4):
    height, width, _ = image.shape
    points = []
    u = 0
    x = u - width / 2
    y = m * x + b
    v = y + height / 2
    if 0 <= v <= height - 1:
        points.append((u, v))
    u = width - 1
    x = u - width / 2
    y = m * x + b
    v = y + height / 2
    if 0 <= v <= height - 1:
        points.append((u, v))
    v = 0
    y = v - height / 2
    if m != 0:
        x = (y - b) / m
        u = x + width / 2
        if 0 <= u <= width - 1:
            points.append((u, v))
    v = height - 1
    y = v - height / 2
    if m != 0:
        x = (y - b) / m
        u = x + width / 2
        if 0 <= u <= width - 1:
            points.append((u, v))
    if len(points) >= 2:
        pt1 = (int(points[0][0]), int(points[0][1]))
        pt2 = (int(points[1][0]), int(points[1][1]))
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    mask = v_grid < m * (u_grid - width / 2) + b + height / 2
    overlay = np.zeros_like(image)
    overlay[mask] = [255, 0, 0]
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def transform_and_detect(image_path, model, y_cutoff):
    """
    Applies enhanced preprocessing and detection to improve robot detection, saving the preprocessed image.

    Args:
        image_path (str): Path to the image file.
        model: Loaded YOLO model.
        y_cutoff (int): Y-coordinate threshold for filtering detections.

    Returns:
        list: List containing the merged bounding box detection (or empty if none).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"    - Warning: Could not read image at {image_path}")
        return []
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply CLAHE to L channel in LAB for adaptive contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # Apply unsharp masking to enhance edges
    blurred = cv2.GaussianBlur(enhanced_img, (0, 0), sigmaX=3, sigmaY=3)
    sharpened = cv2.addWeighted(enhanced_img, 1.5, blurred, -0.5, 0)
    
    # Create mask to focus on region below y_cutoff
    height, width, _ = sharpened.shape
    mask = np.zeros_like(sharpened)
    mask[y_cutoff:, :, :] = 255
    masked_img = cv2.bitwise_and(sharpened, mask)
    
    # Save the preprocessed image with 'test_' prefix
    output_dir = os.path.dirname(image_path)
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"test_{image_name}")
    cv2.imwrite(output_path, masked_img)
    print(f"    - Saved preprocessed image to {output_path}")
    
    # Perform detection with low confidence threshold
    results = model.predict(source=masked_img, conf=0.05, verbose=False)
    result = results[0]
    bboxes = []
    
    for box in result.boxes:
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        if y1 < y_cutoff:
            continue
        if 'robot_6' in image_path or 'robot_7' in image_path:
            m1, b1 = 0.65, -320
            m2, b2 = -0.55, -300
            line1_v = m1 * (x1 - width / 2) + b1 + height / 2
            line2_v = m2 * (x1 - width / 2) + b2 + height / 2
            if y1 < line1_v or y1 < line2_v:
                continue
        bboxes.append([x1, y1, x2, y2])
    
    hull_bbox = compute_hull_bbox(bboxes)
    if hull_bbox is None:
        return []
    return [{
        "class_id": -1,
        "class_name": "robot",
        "box_2d": hull_bbox
    }]


if __name__ == "__main__":
    MODEL_PATH = 'yolo11x.pt'
    # Load the YOLO model
    model = YOLO(MODEL_PATH)
    for directory, y_cutoff in zip(FOLDERS_TO_PROCESS, y_cutoffs):
        data = json.load(open(os.path.join(directory, 'detection_results.json')))
        print(f"Processing directory: {directory}")
        failed_images = identify_failed_detections(directory)
        if failed_images:
            print(f"Failed detections in {directory}:")
            for img_path in failed_images:
                print(f"- {img_path}")
        else:
            print(f"No failed detections in {directory}.")
        for failed_image in failed_images:
            image_name = os.path.basename(failed_image)
            new_bbox = transform_and_detect(failed_image, model, y_cutoff)
            if new_bbox:
                print(f" - Fixed detection for {image_name}: {new_bbox}")
                data[image_name] = new_bbox
            else:
                print(f" - No valid detection for {image_name}.")
        # Save the updated detection results
        with open(os.path.join(directory, 'detection_results.json'), 'w') as f:
            json.dump(data, f, indent=4)