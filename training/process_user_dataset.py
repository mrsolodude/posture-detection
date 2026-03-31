import cv2
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

# Configuration
DATASET_PATH = "datasets/human_data/train_data"
OUTPUT_CSV = "dataset/user_posture_data.csv"
MODEL_PATH = "yolo11n-pose.pt"

def extract_features():
    model = YOLO(MODEL_PATH)
    data = []
    
    # Get all categories
    categories = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    print(f"Categories found: {categories}")
    
    for category in categories:
        cat_path = os.path.join(DATASET_PATH, category)
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {category}: {len(images)} images")
        
        # Limit processing if needed for speed testing, but we'll try to process a batch
        for img_name in tqdm(images[:500], desc=category): # Processing 500 per class for speed
            img_path = os.path.join(cat_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # Inference
            results = model(frame, verbose=False)
            if results and len(results[0].keypoints.data) > 0:
                # Get the first person's keypoints
                kp = results[0].keypoints.data[0].cpu().numpy()
                box = results[0].boxes.xyxy[0].cpu().numpy()
                
                # Normalize keypoints
                width = max(1, box[2] - box[0])
                height = max(1, box[3] - box[1])
                
                norm_kp = []
                for p in kp:
                    # x_norm, y_norm, confidence
                    norm_kp.extend([(p[0] - box[0])/width, (p[1] - box[1])/height, p[2]])
                
                data.append(norm_kp + [category])
                
    # Save to CSV
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Feature extraction complete. Saved {len(data)} samples to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_features()
