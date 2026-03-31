import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import time

# Configuration
SAVE_PATH = "dataset/"
POSTURE_LABEL = "sitting" # Change this manual label while recording different sessions
target_count = 500 # Number of samples per posture

# Ensure directories exist
os.makedirs(SAVE_PATH, exist_ok=True)

class DatasetGenerator:
    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        self.data_list = []

    def capture_samples(self, label):
        cap = cv2.VideoCapture(0)
        count = 0
        print(f"Starting capture for: {label}. Please perform the posture.")
        
        while count < target_count:
            ret, frame = cap.read()
            if not ret: break
            
            results = self.model(frame, verbose=False)
            if results and len(results[0].keypoints.data) > 0:
                # Get the first person's keypoints
                kp = results[0].keypoints.data[0].cpu().numpy()
                
                # Normalize keypoints by bounding box
                box = results[0].boxes.xyxy[0].cpu().numpy()
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                normalized_kp = []
                for point in kp:
                    # x_norm = (x - box_x) / width, y_norm = (y - box_y) / height
                    normalized_kp.extend([(point[0] - box[0])/width, (point[1] - box[1])/height, point[2]])
                
                self.data_list.append(normalized_kp + [label])
                count += 1
                
                # Visual feedback
                cv2.putText(frame, f"Capturing {label}: {count}/{target_count}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow("Dataset Generator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Save to CSV
        df = pd.DataFrame(self.data_list)
        csv_file = f"{SAVE_PATH}/posture_data.csv"
        # Append if exists, else write
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
        print(f"Saved {count} samples for {label}")

if __name__ == "__main__":
    gen = DatasetGenerator()
    # To use: gen.capture_samples("standing"), then gen.capture_samples("sitting"), etc.
    gen.capture_samples(POSTURE_LABEL)
