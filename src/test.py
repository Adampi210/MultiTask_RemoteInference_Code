import numpy as np
import torch
import cv2
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from ultralytics import YOLO
from data_funcs import *

# Detector class
class VehicleDetectorNN():
    def __init__(self, model_path='../../../models/yolov8x.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 5, 7]  # Indices for car, truck, bus in COCO dataset
        self.input_size = (1280, 720)  # Increased input size for better small object detection
        self.conf_threshold = 0.15
        self.iou_threshold = 0.50
        
    '''
    def preprocess_image(self, frame):
        # Resize the image
        frame_resized = cv2.resize(frame, self.input_size, interpolation = cv2.INTER_CUBIC)
        # Apply Gaussian blur to reduce noise
        frame_blurred = cv2.GaussianBlur(frame_resized, (9, 9), 0)
        
        return frame_blurred
    '''
    def preprocess_image(self, frame):
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
        frame_resized = cv2.resize(sharpened, self.input_size, interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        frame_blurred = cv2.GaussianBlur(frame_resized, (9, 9), 0)
        
        # Convert back to 3-channel image (YOLO expects 3 channels)
        frame_3ch = cv2.cvtColor(frame_blurred, cv2.COLOR_GRAY2BGR)
        
        return frame_3ch
    
    def detect_vehicles(self, frame):
        original_h, original_w = frame.shape[:2]
        preprocessed_frame = self.preprocess_image(frame)
        cv2.imwrite('after_preprocessing.jpg', preprocessed_frame)
      
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
  

img_path = '../data/first_frame.jpg'

if __name__ == '__main__':
    detector = VehicleDetectorNN()    
    image = cv2.imread(img_path)
    vehicles_frame = detector.draw_vehicles(image)
    cv2.imwrite('detected_vehicles.jpg', vehicles_frame)