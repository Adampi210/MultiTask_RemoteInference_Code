import numpy as np
import torch
import cv2
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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
  

img_path = '../data/first_frame.jpg'

if __name__ == '__main__':
    detector = VehicleDetectorNN()    
    image = cv2.imread(img_path)
    vehicles_frame = detector.draw_vehicles(image)
    cv2.imwrite('detected_vehicles.jpg', vehicles_frame)

    attention_mask = detector.generate_attention_mask(image)
    plt.imshow(attention_mask, cmap = 'hot')
    plt.savefig('attention_mask.jpg')