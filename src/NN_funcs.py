import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from data_funcs import *

dir_level = '../../..'
model_weight_path = "%s/models/" % dir_level

class VehicleDetectorNN():
    def __init__(self, model_path='../../../models/yolov8x.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.vehicle_classes = ['car', 'truck', 'bus']
        self.input_size = (640, 640)  # Standard input size for YOLOv8

    def preprocess_image(self, frame):
        # Convert to RGB (YOLO expects RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the image
        frame_resized = cv2.resize(frame_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Apply Gaussian blur to reduce noise
        frame_blurred = cv2.GaussianBlur(frame_resized, (5, 5), 0)



        return frame_blurred

    def detect_vehicles(self, frame):
        preprocessed_frame = self.preprocess_image(frame)
        results = self.model(preprocessed_frame)
        vehicles = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if self.model.names[cls] in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    if conf > 0.25:  # Confidence threshold
                        # Scale bounding box coordinates back to original image size
                        h, w = frame.shape[:2]
                        x1 = int(x1 * w / self.input_size[0])
                        y1 = int(y1 * h / self.input_size[1])
                        x2 = int(x2 * w / self.input_size[0])
                        y2 = int(y2 * h / self.input_size[1])
                        vehicles.append([x1, y1, x2, y2])
        return vehicles

    def count_vehicles(self, frame):
        return len(self.detect_vehicles(frame))

    def draw_vehicles(self, frame):
        vehicles = self.detect_vehicles(frame)
        for vehicle in vehicles:
            cv2.rectangle(frame, (vehicle[0], vehicle[1]), (vehicle[2], vehicle[3]), (0, 255, 0), 2)
        return frame

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_with_vehicles = self.draw_vehicles(frame)
            out.write(frame_with_vehicles)
        
        cap.release()
        out.release()
                
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
    
# For testing the segmenter
def test_segmentation(image_path, model_type = "vit_l"):
    segmenter = SAMSegmenter(model_type = model_type)
    image = cv2.imread(image_path)
    
    masks = segmenter.segment(image)
    colored_segmentation = segmenter.colorize_segmentation(masks)
    
    # Overlay segmentation on original image
    overlay = cv2.addWeighted(image, 0.3, colored_segmentation, 0.7, 0)
    
    cv2.imwrite(f'segmentation_result_{model_type}.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"\nNumber of segments found: {len(masks)}")

if __name__ == "__main__":
    test_segmentation("../data/first_frame.jpg", model_type = "vit_l")