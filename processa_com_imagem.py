import cv2
import requests
import numpy as np
import queue
import threading
import time
import datetime
import logging
import os
import random
from ultralytics import YOLO  # Import the YOLO model

# Configure the logging
logging.basicConfig(level=logging.INFO)

# Environment variables
node_red_endpoint = "http://100.100.210.26:1880/teste1"
heartbeat_endpoint = "http://100.100.210.26:1880/teste2"
camera_ip = "rtsp://teste:Ambev123@192.168.0.109:554/cam/realmonitor?channel=1&subtype=1"
heartbeat_interval = 60  # Heartbeat interval in seconds

# Load the YOLO model
model = YOLO(r'best.pt')

class VideoCapture:
    def __init__(self, url):
        self.url = url
        self.connect()
        self.q = queue.Queue(maxsize=10)
        t = threading.Thread(target=self._reader)
        t.daemon = True  # Set the thread as a daemon
        t.start()

    def connect(self):
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            logging.error("Failed to connect to the camera.")

    def reconnect(self):
        self.cap.release()
        self.connect()
        if self.cap.isOpened():
            logging.info("Successfully reconnected to the camera.")
        else:
            logging.error("Failed to reconnect to the camera.")

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.critical("Camera disconnected. Attempting to reconnect.")
                self.reconnect()
                time.sleep(5)
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

def object_detection(frame):
    results = model(frame)
    detection_scores = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            detection_scores.append(score)

            # Draw bounding box and score
            
            # Prepare the score text
            score_text = f"colaborador: {score * 100:.0f}%"
            # Calculate the width of the text
            (text_width, text_height), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Create a rectangle with red color for the text background
            cv2.rectangle(frame, (int(x1) -3, int(y2)), (int(x1) + text_width + 10, int(y2) + text_height + 10), (0, 0, 255), -1) 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)

            # Put text on the rectangle, change the text color to white (255,255,255)
            cv2.putText(frame, score_text, (int(x1) + 5, int(y2) + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  
    return detection_scores, frame

def send_to_node_red(scores):
    try:
        response = requests.post(node_red_endpoint, json={'scores': scores}, timeout=3)
        response.close()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send data to Node-RED: {e}")

def send_heartbeat_to_node_red(heartbeat):
    try:
        response = requests.post(heartbeat_endpoint, json={'heartbeat': heartbeat}, timeout=3)
        response.close()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send heartbeat to Node-RED: {e}")

def process_frames():
    global cap
    last_heartbeat_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.critical("Failed to read a frame. Skipping this iteration.")
            time.sleep(5)
            continue

        # Object detection
        detection_scores, processed_frame = object_detection(frame)
        if detection_scores:
            logging.info("Detection Scores: %s", detection_scores)
            send_to_node_red(detection_scores)

        # Show the frame
        cv2.imshow('Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

        # Heartbeat
        current_time = time.time()
        if current_time - last_heartbeat_time >= heartbeat_interval:
            send_heartbeat_to_node_red(1)
            last_heartbeat_time = current_time

        time.sleep(1)  # Delay before processing the next frame

    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows
    logging.info("Camera released and program terminated.")

if __name__ == "__main__":
    cap = VideoCapture(camera_ip)
    try:
        process_frames()
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting...")
