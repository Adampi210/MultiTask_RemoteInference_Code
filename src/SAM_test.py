import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# VIT-L - 308M parameters (Baseline)
# VIT-B - 91M parameters (On-drone)

# model_type = "vit_l"
model_type = "vit_b"
# sam_l = sam_model_registry[model_type](checkpoint = "../../../models/sam_vit_l_0b3195.pth")
sam_b = sam_model_registry[model_type](checkpoint = "../../../models/sam_vit_b_01ec64.pth")
sam.to(device = device)
predictor = SamPredictor(sam)

image = cv2.imread('../data/first_frame.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)