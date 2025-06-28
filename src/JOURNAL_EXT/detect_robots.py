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
    Loads an image, performs object detection, filters, merges, and returns data and the image.

    Returns:
        tuple: A tuple containing (list_of_detections, image_object).
               The image_object is None if the image could not be read.
    """

    img = cv2.imread(image_path)
    height, width, _ = img. shape
    if img is None:
        print(f"    - Warning: Could not read image at {image_path}")
        return [], None

    results = model.predict(source=img, conf=0.01, verbose=False)
    result = results[0]

    boxes_by_class = {}
    for box in result.boxes:
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        class_id = int(box.cls[0])

        if y1 < y_cutoff:
            continue

        if 'robot_6' in image_path or 'robot_7' in image_path:
            # Define parameters for the two lines
            m1 = 0.65
            b1 = -320
            m2 = -0.55
            b2 = -300
            # Calculate y-values of the lines at x1
            line1_v = m1 * (x1 - width / 2) + b1 + height / 2
            line2_v = m2 * (x1 - width / 2) + b2 + height / 2
            # Skip if top-left corner is above either line
            if y1 < line1_v or y1 < line2_v:
                continue
            
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
    
    # Return both the structured data and the loaded image for drawing
    return final_detections, img

def draw_line_and_shade_above(image, m, b, alpha=0.4):
    """
    Draws a red line on the image with the equation y = mx + b, where (0,0) is the image center,
    and shades the region above the line with a blue transparent overlay.

    Parameters:
        image (numpy.ndarray): The input image in BGR format.
        m (float): Slope of the line.
        b (float): Y-intercept of the line at x=0, with origin at image center.
        alpha (float): Transparency factor for the blue overlay (default is 0.4).
    """
    # Get image dimensions
    height, width, _ = image.shape

    # Step 1: Find intersection points of the line with image boundaries
    points = []
    
    # Intersection with left edge (u=0)
    u = 0
    x = u - width / 2
    y = m * x + b
    v = y + height / 2
    if 0 <= v <= height - 1:
        points.append((u, v))

    # Intersection with right edge (u=width-1)
    u = width - 1
    x = u - width / 2
    y = m * x + b
    v = y + height / 2
    if 0 <= v <= height - 1:
        points.append((u, v))

    # Intersection with top edge (v=0)
    v = 0
    y = v - height / 2
    if m != 0:  # Avoid division by zero
        x = (y - b) / m
        u = x + width / 2
        if 0 <= u <= width - 1:
            points.append((u, v))

    # Intersection with bottom edge (v=height-1)
    v = height - 1
    y = v - height / 2
    if m != 0:
        x = (y - b) / m
        u = x + width / 2
        if 0 <= u <= width - 1:
            points.append((u, v))

    # Step 2: Draw the red line if there are at least two intersection points
    if len(points) >= 2:
        pt1 = (int(points[0][0]), int(points[0][1]))
        pt2 = (int(points[1][0]), int(points[1][1]))
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)  # Red line in BGR

    # Step 3: Shade the region above the line
    # Create coordinate grids for all pixels
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    # Line equation in image coordinates: v = m*(u - width/2) + b + height/2
    # Shade where v > m*(u - width/2) + b + height/2
    mask = v_grid < m * (u_grid - width / 2) + b + height / 2

    # Create overlay for shading
    overlay = np.zeros_like(image)
    overlay[mask] = [255, 0, 0]  # Blue in BGR

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

# ======================================================================================
# Main Script Logic
# ======================================================================================
def main():
    """
    Main function to run the batch processing.
    """
    # --- Configuration ---
    # NOTE: 'yolo11m.pt' appears to be a typo. Corrected to 'yolov8m.pt'.
    # Please update this to the correct path of your model.
    MODEL_PATH = 'yolov8m.pt' 
    
    FOLDERS_TO_PROCESS = [
        '../robot_0/',
        '../robot_1/',
        '../robot_2/',
        '../robot_3/',
        '../robot_4/',
        '../robot_5/',
        '../robot_6/',
        '../robot_7/',
    ]
    
    y_cutoffs = [300, 300, 370, 370, 300, 300, 380, 380]  # Example Y cutoff values for each folder
    
    DISTANCE_THRESHOLD = 150
    
    print(f"Loading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for folder_path, y_cutoff in zip(FOLDERS_TO_PROCESS, y_cutoffs):
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
            
            # Get detections and the loaded image object
            detected_objects, loaded_img = process_image(image_path, model, DISTANCE_THRESHOLD, y_cutoff)

            # If the image could not be loaded, skip to the next one
            if loaded_img is None:
                continue

            # Store the structured data for the JSON file
            all_results_for_folder[filename] = detected_objects

            # --- START VISUALIZATION ---
            
            # Draw the excluded region as a semi-transparent rectangle (only if cutoff is active)
            if y_cutoff > 0:
                height, width, _ = loaded_img.shape
                overlay = loaded_img.copy()
                cv2.rectangle(overlay, (0, 0), (width, y_cutoff), (0, 0, 0), -1)
                alpha = 0.4  # Transparency factor.
                cv2.addWeighted(overlay, alpha, loaded_img, 1 - alpha, 0, loaded_img)
            
            if 'robot_6' in image_path or 'robot_7' in image_path:
                draw_line_and_shade_above(loaded_img, m=0.65, b=-320)
                draw_line_and_shade_above(loaded_img, m=-0.55, b=-300)
                    
            # Draw the final, merged bounding boxes on the image
            for obj in detected_objects:
                box = obj['box_2d']
                class_name = obj['class_name']
                x1, y1, x2, y2 = box
                
                # Draw the bounding box
                cv2.rectangle(loaded_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare and draw the label
                label = f"{class_name}"
                cv2.putText(loaded_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save the new image with drawings
            output_filename = "processed_" + filename
            output_image_path = os.path.join(folder_path, output_filename)
            cv2.imwrite(output_image_path, loaded_img)
            # --- END VISUALIZATION ---

        # Save the JSON file for the entire folder
        json_output_path = os.path.join(folder_path, 'detection_results.json')
        print(f"\nSaving JSON results for {folder_path} to {json_output_path}")
        with open(json_output_path, 'w') as f:
            json.dump(all_results_for_folder, f, indent=4)
        print("  - Save complete.")

    print("\nBatch processing finished.")


if __name__ == '__main__':
    main()