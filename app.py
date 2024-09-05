import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
from PIL import Image
import re

# Define the paths
weight_path = "yolov3.weights"
cfg_path = "yolov3.cfg"
coco_path = "coco.names"

# Download YOLO files if they don't exist
if not os.path.exists(weight_path):
    st.text("Downloading YOLOv3 weights...")
    urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", weight_path)

if not os.path.exists(cfg_path):
    st.text("Downloading YOLOv3 config...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", cfg_path)

if not os.path.exists(coco_path):
    st.text("Downloading COCO names...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", coco_path)

# Load YOLO model
net = cv2.dnn.readNet(weight_path, cfg_path)

# Load class names
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Streamlit app
st.title("Real-Time Object Detection")

# Initialize webcam
cap = cv2.VideoCapture(0)

def show_webcam_error():
    st.error("Could not open webcam. Please ensure that the webcam is connected and enabled.")
    st.text("To use this application, you need to allow access to your webcam. Please check your browser or device settings to ensure the webcam is enabled.")
    st.button("Retry Webcam", on_click=lambda: st.experimental_rerun())

# Check if the webcam is opened correctly
if not cap.isOpened():
    show_webcam_error()
else:
    freeze = st.checkbox("Freeze")

    if not freeze:
        ret, frame = cap.read()

        # Check if a frame was successfully captured
        if not ret:
            st.error("Failed to capture frame from webcam. Please try again.")
            show_webcam_error()
        else:
            height, width, channels = frame.shape

            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (256, 256), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())

            # Process the detection results
            class_ids = []
            confidences = []
            boxes = []
            total_price = 0
            object_counts = {}

            for out in outs:
                for detection in out:
                    for obj in detection:
                        scores = obj[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(obj[0] * width)
                            center_y = int(obj[1] * height)
                            w = int(obj[2] * width)
                            h = int(obj[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                price = extract_price(label)
                total_price += price

                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1

            # Display results in Streamlit
            st.image(frame, channels="BGR")

            y_offset = 30
            for obj_label, count in object_counts.items():
                st.text(f"{obj_label}: {count}")
                y_offset += 30

            st.text(f"Total: RM{total_price}")

    cap.release()
