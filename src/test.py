from PIL import Image
import cv2
import numpy as np
import requests

from NN_funcs import *

img_path = '../data/first_frame.jpg'

if __name__ == '__main__':
    detector = VehicleDetectorNN()    
    image = cv2.imread(img_path)
    vehicles_frame = detector.draw_vehicles(image)
    cv2.imwrite('detected_vehicles.jpg', vehicles_frame)