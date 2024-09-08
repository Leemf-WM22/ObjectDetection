import streamlit as st
import cv2
import numpy as np
import re
from PIL import Image
import io
import os
import urllib.request

# Function to download YOLOv3 files
def download_yolo_files():
    files = {
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    for file_name, url in files.items():
        if not os.path.exists(file_name):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(url, file_name)
            print(f"{file_name} downloaded successfully.")
        else:
            print(f"{file_name} already exists.")

# Check and download YOLOv3 files
download_yolo_files()

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Helper function to extract price from the label
def extract_price(label):
    """Extract price from the label."""
    match = re.search(r"RM(\d+)", label)
    return int(match.group(1)) if match else 0

# Initialize Streamlit UI
st.title("Real-Time Object Detection")

# Control buttons
run = st.checkbox('Run Object Detection')

FRAME_WINDOW = st.image([])
export = st.button('Checkout')

cap = cv2.VideoCapture(0)  # Initialize webcam

total_price = 0
object_counts = {}

# Placeholder for storing item details
item_details = []

if run:
    while True:
        ret, frame = cap.read()
        height, width, channels = frame.shape

        if not run:
            break
        
        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (256, 256), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process the detection results
        class_ids = []
        confidences = []
        boxes = []
        total_price = 0  # Initialize total price for this frame
        object_counts = {}  # Reset object counts for this frame

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] != 'person':  # Filter weak detections and exclude 'person'
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (255, 0, 0)  # Red color for the box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Extract and display price
                price = extract_price(label)
                total_price += price  # Add price to the total

                # Update object count
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1

                # Add to item details
                item_details.append({"Item": label, "Price": price})

        # Display the quantity and total price
        y_offset = 30  # Start position for the first line of text
        for obj_label, count in object_counts.items():
            # Display the object name and its count
            cv2.putText(frame, f"{obj_label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30  # Move down for the next line

        # Display the total price
        cv2.putText(frame, f"Total: RM{total_price}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert OpenCV frame to PIL image for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Display the frame in Streamlit UI
        FRAME_WINDOW.image(img)

        # Handle export button
        if export:
            # Export item details to a .txt file
            item_details_str = "\n".join([f"Item: {item['Item']}, Price: RM{item['Price']}" for item in item_details])
            
            # Create a BytesIO buffer
            buffer = io.BytesIO()
            buffer.write(item_details_str.encode())
            buffer.seek(0)
            
            st.download_button(
                label="Download TXT file",
                data=buffer,
                file_name="detected_items.txt",
                mime="text/plain"
            )

            st.write("TXT file has been created and is ready for download.")
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
