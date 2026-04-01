import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import math
import joblib
import os

class PostureEngine:
    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        self.track_history = {} # person_id -> {posture: last_posture, start_time: timestamp, last_alert: timestamp}
        self.alert_threshold = 30 # 30 seconds threshold
        self.completed_sessions = [] # List of {track_id, posture, start, end, duration, confidence}
        
        # Disabled custom classifier as requested to enforce HIGH LEVEL skeletal math mapping
        self.clf = None

    def calculate_angle(self, a, b, c):
        """Calculates the angle between three points (a, b, c) at point b."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def classify_posture(self, keypoints):
        """Classifies posture purely using high-accuracy skeletal angular mapping."""
        try:
            # Extract relevant keypoints
            l_shoulder = keypoints[5][:2]
            r_shoulder = keypoints[6][:2]
            l_hip = keypoints[11][:2]
            r_hip = keypoints[12][:2]
            l_knee = keypoints[13][:2]
            r_knee = keypoints[14][:2]
            l_ankle = keypoints[15][:2]
            r_ankle = keypoints[16][:2]

            # Calculate midpoints for simpler analysis
            hip_mid = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
            shou_mid = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
            
            # Hip angle (Shoulder-Hip-Knee)
            hip_angle_l = self.calculate_angle(l_shoulder, l_hip, l_knee)
            hip_angle_r = self.calculate_angle(r_shoulder, r_hip, r_knee)
            avg_hip_angle = (hip_angle_l + hip_angle_r) / 2

            # Knee angle (Hip-Knee-Ankle)
            knee_angle_l = self.calculate_angle(l_hip, l_knee, l_ankle)
            knee_angle_r = self.calculate_angle(r_hip, r_knee, r_ankle)
            avg_knee_angle = (knee_angle_l + knee_angle_r) / 2

            # Metric: Body height vs Width
            head_y = keypoints[0][1] # Nose
            avg_hip_y = (l_hip[1] + r_hip[1]) / 2
            avg_knee_y = (l_knee[1] + r_knee[1]) / 2
            avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2

            # Falling detection: low height, horizontal layout
            box = [min(keypoints[:,0]), min(keypoints[:,1]), max(keypoints[:,0]), max(keypoints[:,1])]
            width = box[2] - box[0]
            height = box[3] - box[1]

            if width > height * 1.5 and head_y > avg_hip_y - (height * 0.2):
                return "falling"

            # Logic:
            # Standing: hip and knee angles near 180
            if avg_hip_angle > 150 and avg_knee_angle > 150:
                return "standing"
            # Sitting: hip and knee angles near 90
            elif 60 < avg_hip_angle < 120 and 60 < avg_knee_angle < 130:
                return "sitting"
            # Lying Down: check if spine is horizontal
            elif abs(shou_mid[1] - hip_mid[1]) < abs(shou_mid[0] - hip_mid[0]) * 0.6:
                return "lying down"
            # Bending
            elif avg_hip_angle < 100:
                return "bending"
            else:
                return "idle"
        except Exception:
            return "unknown"

    def process_frame(self, frame):
        """Processes a frame, detects people, tracks them steadily, and returns posture data."""
        # Removed verbose output to stop terminal spam log, increased confidence for stable high-level ID mapping
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.45)
        
        current_detections = []
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, kp, conf in zip(boxes, track_ids, keypoints, confidences):
                posture = self.classify_posture(kp)
                
                # Persistence logic
                now = time.time()
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        "posture": posture,
                        "start_time": now,
                        "last_alert": 0,
                        "total_time": 0
                    }
                else:
                    hist = self.track_history[track_id]
                    if hist["posture"] != posture:
                        # Record completed session
                        end_time = now
                        start_time = hist["start_time"]
                        self.completed_sessions.append({
                            "track_id": int(track_id),
                            "posture": hist["posture"],
                            "start": datetime.fromtimestamp(start_time),
                            "end": datetime.fromtimestamp(end_time),
                            "duration": end_time - start_time,
                            "confidence": float(conf)
                        })
                        # Reset for new posture
                        hist["posture"] = posture
                        hist["start_time"] = now
                
                duration = now - self.track_history[track_id]["start_time"]
                
                current_detections.append({
                    "id": int(track_id),
                    "box": box.tolist(),
                    "keypoints": kp.tolist(),
                    "posture": posture,
                    "duration": round(duration, 2),
                    "confidence": float(conf)
                })
        
        return current_detections

# Singleton instance
engine = PostureEngine()
