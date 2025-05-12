from ultralytics import YOLO
import cv2
import numpy as np
import requests
import paho.mqtt.client as mqtt
import json

# --- MQTT CONFIGURATION ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/camera/yolo"

# Load the model and class-name map
model = YOLO("yolov8s.pt")
names = model.names  # e.g. {0: 'person', 1: 'bicycle', …}

# Create and connect the MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Get one frame from ESP32 CAM
while 1:
    response = requests.get("http://192.168.100.67/capture")
    img_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Run inference on the frame
    # results = model.predict(source=image)

    results = model.predict(source=image, conf=0.5)
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        cls_id     = int(box.cls[0])
        class_name = names[cls_id]                # ← get human-readable name
        # conf       = float(box.conf[0])
        # xyxy       = box.xyxy[0].tolist()
        detections.append({
            # "class_id":    cls_id,
            "class_name":  class_name,
            # "confidence":  conf,
            # "bbox":        xyxy
        })

    # Serialize and publish
    payload = json.dumps({"detections": detections})
    mqtt_client.publish(MQTT_TOPIC, payload)
    print(f"[MQTT] Published:\n{payload}")
