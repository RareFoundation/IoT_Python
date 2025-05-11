from ultralytics import YOLO
import cv2
import numpy as np
import requests

# Load the model
model = YOLO("yolov8s.pt")

# Get one frame from ESP32 CAM
while 1:
    response = requests.get("http://192.168.100.64/capture")
    img_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Run inference on the frame
    results = model.predict(source=image)

# Optionally save the result manually (if show doesn't work on macOS)
# results[0].save(filename="result.jpg")
