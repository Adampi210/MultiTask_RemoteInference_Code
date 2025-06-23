import cv2
from ultralytics import YOLO

# load the YOLOv8 model
# Choices: ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
print("Loading YOLOv8 model...")
model = YOLO('yolov8m.pt')

# Load the image
dataset_path = '../../data/'
image_name = 'frame_00090.jpg'
image_path = dataset_path + image_name
print(f"Loading image from: {image_path}")
img = cv2.imread(image_path)

# Check if the image was loaded correctly
if img is None:
    print(f"Error: Could not read the image at {image_path}")
else:
    # Detect objects
    print("Performing object detection...")
    results = model.predict(source=img, conf=0.30) # Confidence threshold set to 0.30
    result = results[0]

    # Draw bounding boxes and labels on the image
    for box in result.boxes:
        # Get the coordinates of the bounding box (x1, y1, x2, y2) (xyxy format)
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]

        # Get the confidence score
        confidence = float(box.conf[0])
        # Get the class
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Draw the bounding box and label
        label = f"{class_name} {confidence:.2f}"
        print(f"Detected '{class_name}' with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
        area = (x2 - x1) * (y2 - y1)
        print(f"Area of bounding box: {area} pixels")
        # Draw the bounding box for the robots
        if area > 15000 and area < 150000:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # Save results
    output_path = 'detection_result.jpg'
    cv2.imwrite(output_path, img)