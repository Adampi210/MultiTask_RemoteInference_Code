import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
import albumentations as A
import torchvision.transforms as T
import torchvision.transforms.functional as F
import segmentation_models_pytorch as smp

from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from data_funcs import *

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

class DDRNetSegmenter:
    def __init__(self, model_path, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Classes from the Cityscapes dataset
        self.classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        # Need to resize to Cityscapes resolution, also normalize
        self.preprocess = T.Compose([
            T.Resize((1024, 2048)),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
    
    # Load the pretrained model
    def load_model(self, model_path):
        model = torch.load(model_path, map_location = self.device)
        return model
    
    # Preprocess the image for the input to the neural net
    def preprocess_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        preprocessed = self.preprocess(pil_image)
        return preprocessed.unsqueeze(0)
    
    # Create the segmentation
    def segment(self, image):
        processed = self.preprocess_image(image).to(self.device)
        with torch.no_grad():
            output = self.model(processed)
        pred = output.argmax(1).squeeze(0).cpu().numpy()
        return pred
    
    # Create color map
    def colorize_segmentation(self, pred):
        colormap = np.array([
            [128, 64, 128],  # road
            [244, 35, 232],  # sidewalk
            [70, 70, 70],    # building
            [102, 102, 156], # wall
            [190, 153, 153], # fence
            [153, 153, 153], # pole
            [250, 170, 30],  # traffic light
            [220, 220, 0],   # traffic sign
            [107, 142, 35],  # vegetation
            [152, 251, 152], # terrain
            [70, 130, 180],  # sky
            [220, 20, 60],   # person
            [255, 0, 0],     # rider
            [0, 0, 142],     # car
            [0, 60, 100],    # truck
            [0, 80, 100],    # bus
            [0, 0, 230],     # train
            [119, 11, 32],   # motorcycle
            [0, 0, 142],     # bicycle
        ], dtype = np.uint8)
        
        colored_pred = colormap[pred]
        return colored_pred

def test_segmentation(image_path, model_path):
    segmenter = DDRNetSegmenter(model_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    segmentation = segmenter.segment(image)
    colored_segmentation = segmenter.colorize_segmentation(segmentation)
    
    # Resize colored_segmentation to match original image size
    colored_segmentation = cv2.resize(colored_segmentation, (image.shape[1], image.shape[0]))
    
    # Overlay segmentation on original image
    overlay = cv2.addWeighted(image, 0.1, colored_segmentation, 0.9, 0)
    
    cv2.imwrite('segmentation_result.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Segmentation result saved as segmentation_result.jpg")

    # Print unique classes in the segmentation
    unique_classes = np.unique(segmentation)
    print("\nClasses found in the image:")
    for class_id in unique_classes:
        if class_id < len(segmenter.classes):
            print(f"- {segmenter.classes[class_id]}")
        else:
            print(f"- Unknown class {class_id}")

if __name__ == "__main__":
    test_segmentation("../first_frame.jpg", "../../../models/DDRNet_23_Cityscapes.pth")