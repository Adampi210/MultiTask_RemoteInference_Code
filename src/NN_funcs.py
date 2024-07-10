import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from data_funcs import *

dir_level = '../../..'
model_weight_path = "%s/models/" % dir_level

class VehicleDetectorNN():
    def __init__(self, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        # Use a pretrained model, NOTE: if want to change the model, change here
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.model.eval()
        self.vehicle_classes = [3, 6, 8]  # Indices for car, bus, truck in COCO dataset
    
    # To find all vehicles in a frame
    def detect_vehicles(self, frame):
        # Get the image tensor from the frame
        img_tns = F.to_tensor(frame).unsqueeze(0).to(self.device)
        # Get prediction on the frame
        with torch.no_grad():
            prediction_tns = self.model(img_tns)[0]  
            # Get boxes for each vehicle  
            vehicles = []
            for i, label in enumerate(prediction_tns['labels']):
                if label in self.vehicle_classes and prediction_tns['scores'][i] > 0.5:
                    box = prediction_tns['boxes'][i].cpu().numpy().astype(int)
                    vehicles.append(box)
        return vehicles
    
    # To count vehicles
    def count_vehicles(self, frame):
        vehicles = self.detect_vehicles(frame)
        return len(vehicles)
    
    # To draw boxes over a vehicles in a frame
    def draw_vehicles(self, frame):
        vehicles = self.detect_vehicles(frame)
        for vehicle in vehicles:
            cv2.rectangle(frame, (vehicle[0], vehicle[1]), (vehicle[2], vehicle[3]), (0, 0, 255), 2)
        return frame

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
    test_segmentation("../data/first_frame.jpg", model_type = "vit_b")