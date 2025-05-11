import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
from collections import Counter
import time
import threading

# ----- MQTT CONFIG -----
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/camera/yolo"

# ----- Global state -----
detection_log = []
counts = Counter()


# ----- MQTT Callbacks -----
def on_connect(client, userdata, flags, rc):
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    labels = [d["class_name"] for d in payload.get("detections", [])]
    detection_log.extend(labels)
    counts.update(labels)


# ----- Start MQTT Client in Background -----
def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()


mqtt_thread = threading.Thread(target=start_mqtt)
mqtt_thread.daemon = True
mqtt_thread.start()

# ----- Streamlit UI -----
st.title("YOLOv8 Object Detection via MQTT")
st.markdown("📡 Listening to MQTT topic: `" + MQTT_TOPIC + "`")

placeholder = st.empty()

# Auto-refresh every 2 seconds
while True:
    with placeholder.container():
        st.subheader("Live Object Count")
        if counts:
            df = pd.DataFrame(counts.items(), columns=["Object", "Count"])
            st.bar_chart(df.set_index("Object"))
        else:
            st.info("Waiting for detections...")

        st.subheader("Last 20 Detections")
        st.write(detection_log[-20:])

    time.sleep(2)