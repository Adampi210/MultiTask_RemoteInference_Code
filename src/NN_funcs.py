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
results_dir = '.'

# Detector class
class VehicleDetectorNN():
    def __init__(self, model_path='../../../models/yolov8x.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 5, 7]  # Indices for car, truck, bus in COCO dataset
        self.input_size = (1280, 1280)  # Increased input size for better small object detection
        self.conf_threshold = 0.25
        self.iou_threshold = 0.40
        self.min_area = 15
        
    def preprocess_image(self, frame):
        # Convert to RGB (YOLO expects RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the image
        frame_resized = cv2.resize(frame_rgb, self.input_size, interpolation=cv2.INTER_CUBIC)

        # Apply Gaussian blur to reduce noise
        frame_blurred = cv2.GaussianBlur(frame_resized, (5, 5), 0)

        return frame_rgb

    def detect_vehicles(self, frame):
        original_h, original_w = frame.shape[:2]
        preprocessed_frame = self.preprocess_image(frame)
        
        results = self.model(preprocessed_frame, conf=self.conf_threshold, iou=self.iou_threshold, classes=self.vehicle_classes)
        
        vehicles = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Scale bounding box coordinates back to original image size
                x1, x2 = [int(x * original_w / self.input_size[0]) for x in [x1, x2]]
                y1, y2 = [int(y * original_h / self.input_size[1]) for y in [y1, y2]]
                
                # Filter out small detections
                if (x2 - x1) * (y2 - y1) >= self.min_area:
                    vehicles.append([x1, y1, x2, y2, cls, conf])
        
        return vehicles

    def count_vehicles(self, frame):
        return len(self.detect_vehicles(frame))

    def draw_vehicles(self, frame):
        vehicles = self.detect_vehicles(frame)
        for vehicle in vehicles:
            x1, y1, x2, y2, cls, conf = vehicle
            color = (0, 255, 0) if cls == 2 else (0, 0, 255) if cls == 5 else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f''
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Unable to read image at {image_path}")
        
        frame_with_vehicles = self.draw_vehicles(frame)
        vehicle_count = self.count_vehicles(frame)
        
        return frame_with_vehicles, vehicle_count

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_with_vehicles = self.draw_vehicles(frame)
            out.write(frame_with_vehicles)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        print(f"Video processing completed. Total frames: {frame_count}")
  
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
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

# Get the loss
def segmentation_loss_calculation(video_path, seed = 0, start_frame = None, end_frame = None, results_dir = './results'):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init the 2 model
    vit_l_segmenter = SAMSegmenter(model_type = "vit_l", device = device)
    vit_b_segmenter = SAMSegmenter(model_type = "vit_b", device = device)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(results_dir, f"{video_name}_segmentation_IoU_loss_result_seed_{seed}.csv")

    os.makedirs(results_dir, exist_ok = True)

    frame_generator = get_frames(video_path)
    
    with open(output_file, 'w', newline = '') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'IoU_Loss'])

        for frame_num, frame in enumerate(frame_generator):
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num >= end_frame:
                break

            if frame_num == 0:
                frame_height, frame_width = frame.shape[:2]
                print(f"Frame size: {frame_width}x{frame_height}")

            # Get masks
            vit_l_masks, vit_b_masks = process_frame_both_models(frame, vit_l_segmenter, vit_b_segmenter)
            
            # Optional get the frames with segmentation for checking
            if frame_num % 10 == 0:
                colored_vit_l_masks = vit_l_segmenter.colorize_segmentation(vit_l_masks)
                colored_vit_b_masks = vit_b_segmenter.colorize_segmentation(vit_b_masks)
                overlay_l = cv2.addWeighted(frame, 0.3, colored_vit_l_masks, 0.7, 0)
                overlay_b = cv2.addWeighted(frame, 0.3, colored_vit_b_masks, 0.7, 0)
                cv2.imwrite(f'../data/frame_{frame_num}_segmentation_result_vit_l.jpg', cv2.cvtColor(overlay_l, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'../data/frame_{frame_num}_segmentation_result_vit_b.jpg', cv2.cvtColor(overlay_b, cv2.COLOR_RGB2BGR))

            # Calculate loss
            vit_l_combined = np.zeros((frame_height, frame_width), dtype=bool)
            vit_b_combined = np.zeros((frame_height, frame_width), dtype=bool)

            for mask in vit_l_masks:
                vit_l_combined |= mask['segmentation']
            for mask in vit_b_masks:
                vit_b_combined |= mask['segmentation']

            iou_loss = calculate_iou(vit_l_combined, vit_b_combined)

            csv_writer.writerow([frame_num, iou_loss])

            if frame_num % 10 == 0:
                print(f"Processed frame {frame_num}")

            # Clear masks from memory
            del vit_l_masks, vit_b_masks, vit_l_combined, vit_b_combined

    print(f"IoU loss results saved to {output_file}")


if __name__ == "__main__":
    # test_segmentation('../data/first_frame.jpg', 'vit_b')
    video_path = '/scratch/gilbreth/apiasecz/data/NGSIM_traffic_data/nb-camera1-0400pm-0415pm.avi'
    segmentation_loss_calculation(video_path, seed = 0, start_frame = None, end_frame = 11, results_dir = results_dir)
