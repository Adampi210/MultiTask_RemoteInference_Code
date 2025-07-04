import json
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    # Calculate intersection area
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate areas of both boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    
    # Calculate union area
    union_area = box1_area + box2_area - inter_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def calculate_average_error(bboxes, aoi):
    """Calculate average 1 - IoU for a given AoI across all relevant frames."""
    N = len(bboxes)
    errors = []
    
    # For each time t from AoI to the last frame
    for t in range(aoi, N):
        iou = calculate_iou(bboxes[t], bboxes[t - aoi])
        errors.append(1 - iou)
    
    # Return average error or 0 if no errors calculated
    return sum(errors) / len(errors) if errors else 0

def plot_error_vs_aoi(bboxes, filename, max_aoi):
    """Plot average error (1 - IoU) versus AoI."""
    aoi_values = list(range(max_aoi + 1))
    errors = [calculate_average_error(bboxes, k) for k in aoi_values]
    
    with open(filename + '_error_function.json', 'w') as f:
        json.dump({'aoi_values': aoi_values, 'errors': errors}, f, indent=4)
    
    plt.figure(figsize=(10, 6))
    plt.plot(aoi_values, errors, marker='o', linestyle='-', color='b')
    plt.xlabel('Age of Information (AoI)')
    plt.ylabel('Average Error (1 - IoU)')
    plt.title('Error Function vs AoI')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    filename = '../../data/detection_results_robot_8'
    # Load the JSON data
    with open(filename + '.json', 'r') as f:
        data = json.load(f)

    # Sort frame names by numerical order
    frame_names = sorted(data.keys(), key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Extract bounding boxes in sorted order
    bboxes = [data[frame][0]['box_2d'] for frame in frame_names]

    # Plot error vs AoI for AoI from 0 to 100
    plot_error_vs_aoi(bboxes, filename, 40)
    
    # Plot the current bbox and predicted bbox for a given AoI for different frames
    
    