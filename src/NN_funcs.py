import numpy as np
import torch
import cv2
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from data_funcs import *

dir_level = "../../.."
model_weight_path = "%s/models/" % dir_level
# results_dir = "/scratch/gilbreth/apiasecz/results/NGSIM_traffic_data_results/segmentation_masks"
results_dir = '../data/'

############## DETECTOR  ###############
# Detector class
class VehicleDetectorNN():
    def __init__(self, yolo_model_path = '../../../models/yolov8x.pt', sam_model_path = '../../../models/sam_vit_l.pth', device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.yolo_model = YOLO(yolo_model_path)
        self.vehicle_classes = [2, 5, 7]  # Indices for car, truck, bus in COCO dataset
        self.input_size = (1280, 720)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.50

        # Initialize SAM
        sam = sam_model_registry["vit_l"](checkpoint = sam_model_path)
        sam.to(device = device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        
    def filter_vehicle_masks(self, masks, image_shape):
        filtered_masks = []
        height, width = image_shape[:2]
        for mask in masks:
            area = mask['area']
            if area < 0.0001 * height * width or area > 0.1 * height * width:
                continue
            
            bbox = mask['bbox']
            if bbox[3] - bbox[1] != 0:
                aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) 
                if aspect_ratio < 0.5 or aspect_ratio > 2.5:
                    continue
            
            filtered_masks.append(mask)
        return filtered_masks

    def generate_attention_mask(self, image):
        masks = self.mask_generator.generate(image)
        filtered_masks = self.filter_vehicle_masks(masks, image.shape)
        
        # Combine all masks into a single attention map
        attention_mask = np.zeros(image.shape[:2], dtype = np.float32)
        for mask in filtered_masks:
            attention_mask += mask['segmentation'].astype(np.float32)
        
        # Normalize the attention mask
        attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())
        return attention_mask

    def preprocess_image_normal(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        #clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(3,3))
        #equalized = clahe.apply(gray)
        
        # Edge enhancement
        edge_enhanced = cv2.Laplacian(gray, cv2.CV_8U)
        
        # Combine edge-enhanced image with original grayscale
        sharpened = cv2.addWeighted(gray, 0.8, edge_enhanced, 0.2, 0)
        
        # Resize the image
        frame_resized = cv2.resize(sharpened, self.input_size, interpolation = cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        frame_blurred = cv2.GaussianBlur(frame_resized, (9, 9), 0)
        
        # Convert back to 3-channel image (YOLO expects 3 channels)
        frame_3ch = cv2.cvtColor(frame_blurred, cv2.COLOR_GRAY2BGR)
        
        return frame_3ch

    def preprocess_image_sam(self, frame):
        # Generate attention mask using SAM
        attention_mask = self.generate_attention_mask(frame)
        
        # Apply attention mask to the original image
        enhanced_frame = frame * attention_mask[:,:,np.newaxis] * 4
        enhanced_frame = enhanced_frame.astype(np.uint8)
        
        # Resize the image
        frame_resized = cv2.resize(enhanced_frame, self.input_size, interpolation = cv2.INTER_CUBIC)
        
        return frame_resized

    def non_max_suppression(self, boxes, overlapThresh = 0.5):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        pick = []

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(boxes[:,5])

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        return boxes[pick].tolist()

    def detect_vehicles(self, frame):
        original_h, original_w = frame.shape[:2]
        
        # Detect on SAM-enhanced image
        preprocessed_frame_sam = self.preprocess_image_sam(frame)
        results_sam = self.yolo_model(preprocessed_frame_sam, conf = self.conf_threshold, iou = self.iou_threshold, classes = self.vehicle_classes)
        
        # Detect on normally preprocessed image
        preprocessed_frame_normal = self.preprocess_image_normal(frame)
        results_normal = self.yolo_model(preprocessed_frame_normal, conf = self.conf_threshold, iou=self.iou_threshold, classes = self.vehicle_classes)
        
        # Combine detections
        vehicles = []
        for results in [results_sam, results_normal]:
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Scale bounding box coordinates back to original image size
                    x1, x2 = [int(x * original_w / self.input_size[0]) for x in [x1, x2]]
                    y1, y2 = [int(y * original_h / self.input_size[1]) for y in [y1, y2]]
                    
                    vehicles.append([x1, y1, x2, y2, cls, conf])

        # Remove overlap
        vehicles = self.non_max_suppression(vehicles, overlapThresh = 0.5)
        
        return vehicles
    
    def count_vehicles(self, frame):
        return len(self.detect_vehicles(frame))

    def draw_vehicles(self, frame):
        vehicles = self.detect_vehicles(frame)
        for vehicle in vehicles:
            try:
                x1, y1, x2, y2, cls, conf = vehicle

                x1 = max(0, min(int(x1), frame.shape[1] - 1))
                y1 = max(0, min(int(y1), frame.shape[0] - 1))
                x2 = max(0, min(int(x2), frame.shape[1] - 1))
                y2 = max(0, min(int(y2), frame.shape[0] - 1))
                
                color = (0, 255, 0) if cls == 2 else (0, 0, 255) if cls == 5 else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f''
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                print(f"Error drawing vehicle: {e}")
                print(f"Vehicle data: {vehicle}")
        return frame

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Unable to read image at {image_path}")
        
        frame_with_vehicles = self.draw_vehicles(frame)
        vehicle_count = self.count_vehicles(frame)
        
        return frame_with_vehicles, vehicle_count
        
# Get the testing result
def calculate_vehicle_detection(video_path, seed = 0, start_frame = None, end_frame = None, results_dir = './results'):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = VehicleDetectorNN(device = device)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(results_dir, f"{video_name}_vehicle_detection_result_seed_{seed}.csv")
    os.makedirs(results_dir, exist_ok = True)

    frame_generator = get_frames(video_path)
    
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'Vehicle_Count', 'Vehicle_Locations'])

        for frame_num, frame in enumerate(frame_generator):
            if frame_num == 0:
                frame_height, frame_width = frame.shape[:2]
                print(f"Frame size: {frame_width}x{frame_height}")
                
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num >= end_frame:
                break

            # Detect vehicles
            vehicles = detector.detect_vehicles(frame)
            
            # Count vehicles
            vehicle_count = len(vehicles)
            
            # Format vehicle locations
            vehicle_locations = []
            for vehicle in vehicles:
                x1, y1, x2, y2, cls, conf = vehicle
                vehicle_locations.append(f"({x1},{y1},{x2},{y2})")
            
            # Join vehicle locations into a string
            vehicle_locations_str = "|".join(vehicle_locations)
            
            # Write to CSV
            csv_writer.writerow([frame_num, vehicle_count, vehicle_locations_str])

            if frame_num % 10 == 0:
                print(f"Processed frame {frame_num}")

    print(f"Vehicle detection results saved to {output_file}")

############## SEGMENTER ###############
# Segmenter class              
class SAMSegmenter:
    def __init__(self, model_type = "vit_l", device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.model_type = model_type
        self.model = self.load_model()
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        
    def load_model(self):
        checkpoint_path = f"{model_weight_path}/sam_{self.model_type}.pth"
        sam = sam_model_registry[self.model_type](checkpoint = checkpoint_path)
        sam.to(device = self.device)
        return sam
    
    def preprocess_image(self, image):
        # Convert to RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def segment(self, image):
        processed = self.preprocess_image(image)
        masks = self.mask_generator.generate(processed)
        return masks
    
    def colorize_segmentation(self, masks):
        # Get a colormap for the segmentation output
        colormap = plt.get_cmap('tab20b')
        # Colored segmentation
        colored_seg = np.zeros((*masks[0]['segmentation'].shape, 3), dtype = np.uint8)
        
        # Assign colors to the generated masks
        for i, mask in enumerate(masks):
            color = np.array(colormap(i % 20)[:3]) * 255
            colored_seg[mask['segmentation']] = color
        
        return colored_seg
    
# Test the segmenter
def test_segmentation(image_path, model_type = "vit_l"):
    segmenter = SAMSegmenter(model_type = model_type)
    image = cv2.imread(image_path)
    
    masks = segmenter.segment(image)
    colored_segmentation = segmenter.colorize_segmentation(masks)
    
    # Overlay segmentation on original image
    overlay = cv2.addWeighted(image, 0.3, colored_segmentation, 0.7, 0)
    
    cv2.imwrite(f'segmentation_result_{model_type}.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"\nNumber of segments found: {len(masks)}")

# Get masks for ground truth and the test
def process_frame_both_models(frame, vit_l_segmenter, vit_b_segmenter):
    vit_l_masks = vit_l_segmenter.segment(frame)
    vit_b_masks = vit_b_segmenter.segment(frame)
    return vit_l_masks, vit_b_masks

# Calculate loss
def calculate_inverted_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return 1 - iou  

# Get the loss
def segmentation_loss_calculation(video_path, seed = 0, start_frame = None, end_frame = None, results_dir = './results'):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init the 2 model
    vit_l_segmenter = SAMSegmenter(model_type = "vit_l", device = device)
    vit_b_segmenter = SAMSegmenter(model_type = "vit_l", device = device)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(results_dir, f"{video_name}_segmentation_IoU_loss_result_seed_{seed}.csv")
    os.makedirs(results_dir, exist_ok = True)

    frame_generator = get_frames(video_path)
    
    with open(output_file, 'w', newline = '') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'IoU_Loss'])

        for frame_num, frame in enumerate(frame_generator):
            if frame_num == 0:
                frame_height, frame_width = frame.shape[:2]
                print(f"Frame size: {frame_width}x{frame_height}")
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num >= end_frame:
                break
    
            # Get masks
            vit_l_masks, vit_b_masks = process_frame_both_models(frame, vit_l_segmenter, vit_b_segmenter)
            
            # Optional get the frames with segmentation for checking
            if frame_num % 500 == 0:
                colored_vit_l_masks = vit_l_segmenter.colorize_segmentation(vit_l_masks)
                colored_vit_b_masks = vit_b_segmenter.colorize_segmentation(vit_b_masks)
                overlay_l = cv2.addWeighted(frame, 0.3, colored_vit_l_masks, 0.7, 0)
                overlay_b = cv2.addWeighted(frame, 0.3, colored_vit_b_masks, 0.7, 0)
                cv2.imwrite(f'../data/{video_name}_frame_{frame_num}_segmentation_result_vit_l.jpg', cv2.cvtColor(overlay_l, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'../data/{video_name}_frame_{frame_num}_segmentation_result_vit_b.jpg', cv2.cvtColor(overlay_b, cv2.COLOR_RGB2BGR))

            # Calculate loss
            vit_l_combined = np.zeros((frame_height, frame_width), dtype = bool)
            vit_b_combined = np.zeros((frame_height, frame_width), dtype = bool)

            for mask in vit_l_masks:
                vit_l_combined |= mask['segmentation']
            for mask in vit_b_masks:
                vit_b_combined |= mask['segmentation']

            iou_loss = calculate_inverted_iou(vit_l_combined, vit_b_combined)

            csv_writer.writerow([frame_num, iou_loss])

            if frame_num % 10 == 0:
                print(f"Processed frame {frame_num}")

            # Clear masks from memory
            del vit_l_masks, vit_b_masks, vit_l_combined, vit_b_combined

    print(f"IoU loss results saved to {output_file}")

def calculate_multi_k_loss(video_path, seed = 0, start_frame = None, end_frame = None, results_dir = './results', max_k = 20):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_l_segmenter = SAMSegmenter(model_type = "vit_l", device = device)
    vit_b_segmenter = SAMSegmenter(model_type = "vit_l", device = device)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(results_dir, f"{video_name}_multi_k_loss_result_seed_{seed}.csv")
    os.makedirs(results_dir, exist_ok = True)

    frame_generator = get_frames(video_path)
    
    # Initialize a deque to store the last max_k+1 frames
    frame_history = deque(maxlen = max_k + 1)
    
    with open(output_file, 'w', newline = '') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['Frame'] + [f'k={i}' for i in range(max_k + 1)]
        csv_writer.writerow(header)

        for frame_num, frame in enumerate(frame_generator):
            if frame_num == 0:
                frame_height, frame_width = frame.shape[:2]
                print(f"Frame size: {frame_width}x{frame_height}")
                
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num >= end_frame:
                break

            # Get masks for current frame
            vit_l_masks, vit_b_masks = process_frame_both_models(frame, vit_l_segmenter, vit_b_segmenter)
            
            # Combine masks
            vit_l_combined = np.zeros((frame_height, frame_width), dtype = bool)
            vit_b_combined = np.zeros((frame_height, frame_width), dtype = bool)
            for mask in vit_l_masks:
                vit_l_combined |= mask['segmentation']
            for mask in vit_b_masks:
                vit_b_combined |= mask['segmentation']
            
            # Add current frame masks to history
            frame_history.append((vit_l_combined, vit_b_combined))
            
            # Calculate all k values IoU losses
            losses = [frame_num]  
            for k in range(max_k + 1): 
                if len(frame_history) > k:
                    Y_t, Y_hat_t_minus_k = frame_history[-1][0], frame_history[-k-1][1]
                    loss = calculate_inverted_iou(Y_t, Y_hat_t_minus_k)
                    losses.append(loss)
                else:
                    losses.append(1)
            
            csv_writer.writerow(losses)

            if frame_num % 10 == 0:
                print(f"Processed frame {frame_num}")

    print(f"Multi-k loss results saved to {output_file}")

    calculate_final_pk_values(output_file, max_k)

def calculate_final_pk_values(input_file, max_k):
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        
        sum_losses = [0] * (max_k + 1)
        frame_count = 0
        
        for row in reader:
            frame_count += 1
            for k in range(max_k + 1):
                sum_losses[k] += float(row[k + 1])
    
    pk_values = [sum_loss / frame_count for sum_loss in sum_losses]
    
    output_file = os.path.splitext(input_file)[0] + "_pk_values.csv"
    with open(output_file, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['k', 'p(k)'])
        for k, pk in enumerate(pk_values):
            writer.writerow([k, pk])
    
    print(f"Final p(k) values saved to {output_file}")
    
if __name__ == "__main__":
    N_FILES = 5
    seed = 0
    i = 0
    dir_data_path = '/scratch/gilbreth/apiasecz/data/NGSIM_traffic_data/'
    # video_files = sorted(get_video_files(dir_data_path))
    video_files = get_subset_video_files(dir_data_path, N_FILES, seed)
    for video_file in video_files:
        print(video_file)
        calculate_multi_k_loss(video_file, seed = seed, start_frame = None, end_frame = None, results_dir = results_dir, max_k = 100)
        break
    exit()
