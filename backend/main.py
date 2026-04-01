import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from sqlalchemy.orm import Session
from datetime import timedelta
import cv2
import base64
import json
import asyncio
import re
from typing import Optional
from database.models import Base, User, PostureActivity, AlertLog
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from alerts.alert_system import alert_system
from backend.services.pose_engine import engine
import bcrypt
import jwt
from pydantic import BaseModel
import time
import random
import numpy as np

# Security configuration
SECRET_KEY = "SUPER_SECRET_KEY_REPLACE_IN_PROD"
ALGORITHM = "HS256"

# Database setup
DATABASE_URL = "sqlite:////Users/apple/Mini Projects/posture new/posture_monitoring.db"
engine_db = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# Enable WAL mode for SQLite to prevent locking
from sqlalchemy import event
@event.listens_for(engine_db, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_db)
Base.metadata.create_all(bind=engine_db)

app = FastAPI(title="PostureGuard AI - Backend")

# DASHBOARD EXPLICIT SERVE (Priority)
@app.get("/")
async def read_index(request: Request):
    print(f"Index requested from {request.client.host}")
    index_path = os.path.join(os.getcwd(), "frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": f"Frontend index.html NOT FOUND at {index_path}"}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class LoginRequest(BaseModel):
    phone_number: str
    password: str

# Shared state
monitoring_active = False

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Background task for alert monitoring
async def alert_monitor_task():
    global monitoring_active
    while True:
        if monitoring_active:
            db = SessionLocal()
            try:
                now = time.time()
                for track_id, hist in engine.track_history.items():
                    duration = now - hist["start_time"]
                    curr_posture = hist["posture"]
                    
                    is_incident = False
                    msg = ""
                    
                    # 1. Immediate incident: Falling
                    if curr_posture == "falling":
                        if now - hist.get("last_fall_log", 0) > 30: # Log at most every 30s
                            is_incident = True
                            msg = f"INCIDENT: Person {track_id} has FALLEN!"
                            hist["last_fall_log"] = now
                    
                    # 2. Prolonged posture incident
                    elif duration >= engine.alert_threshold:
                        if now - hist["last_alert"] > 30: # Log every 30s if still in same posture over threshold
                            is_incident = True
                            msg = f"INCIDENT: Person {track_id} in '{curr_posture}' for {int(duration)}s"
                            hist["last_alert"] = now
                    
                    if is_incident:
                        log = AlertLog(user_id=None, person_track_id=track_id, posture_type=curr_posture, message=msg)
                        db.add(log)
                        db.commit()
                        print(f"Logged Incident: {msg}")
            except Exception as e:
                print(f"Error in incident monitor: {e}")
            finally:
                db.close()
        await asyncio.sleep(5) # check every 5 seconds for incidents

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(alert_monitor_task())

# Registration and login endpoints removed

# Mock OTP storage
otp_store = {}

@app.post("/request_otp")
def request_otp(phone_number: str):
    code = str(random.randint(100000, 999999))
    otp_store[phone_number] = code
    # Simulate sending SMS
    msg = f"Your PostureGuard OTP code is: {code}"
    alert_system.send_sms(phone_number, msg)
    return {"message": "OTP sent successfully (Simulated)"}

@app.post("/verify_otp")
def verify_otp(phone_number: str, code: str):
    if otp_store.get(phone_number) == code:
        del otp_store[phone_number]
        return {"message": "Verification successful"}
    raise HTTPException(status_code=400, detail="Invalid OTP")

@app.get("/start_monitoring")
def start_monitoring():
    global monitoring_active
    monitoring_active = True
    return {"status": "Monitoring started"}

@app.get("/stop_monitoring")
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    return {"status": "Monitoring stopped"}

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(PostureActivity).order_by(PostureActivity.start_time.desc()).limit(100).all()

@app.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    return db.query(AlertLog).order_by(AlertLog.created_at.desc()).limit(100).all()

import csv
from fastapi.responses import StreamingResponse
import io

@app.get("/download_csv")
def download_csv(db: Session = Depends(get_db)):
    # Combine history and alerts
    history = db.query(PostureActivity).order_by(PostureActivity.start_time.desc()).all()
    alerts = db.query(AlertLog).order_by(AlertLog.created_at.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["Type", "Person ID", "Posture/Message", "Start Time/Created At", "Duration (s)", "Confidence"])
    
    for a in alerts:
        writer.writerow(["ALERT", a.person_track_id, a.message, a.created_at, "-", "-"])
        
    for h in history:
        writer.writerow(["ACTIVITY", h.person_track_id, h.posture_type, h.start_time, h.duration, h.confidence])
    
    output.seek(0)
    
    headers = {
        'Content-Disposition': 'attachment; filename="posture_logs.csv"'
    }
    return StreamingResponse(output, media_type="text/csv", headers=headers)

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    
    cap = None
    # 1. Force strict physical laptop camera initialization
    # Test up to index 3 to bypass iPhone Continuity Camera (which returns pitch-black frames)
    for index in [0, 1, 2]:
        print(f"Attempting to open Laptop Camera at index {index}...")
        temp_cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) # Force AVFoundation for Mac
        if not temp_cap.isOpened():
             temp_cap.release()
             temp_cap = cv2.VideoCapture(index) # Fallback to default
             
        if temp_cap.isOpened():
            # Warm up and check for totally black frames (Continuity Camera issue)
            is_valid_cam = False
            for _ in range(5):
                ret, frame = temp_cap.read()
                if ret and frame is not None and np.mean(frame) > 2.0:
                    is_valid_cam = True
                    break
            
            if is_valid_cam:
                cap = temp_cap
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"SUCCESS: Connected to LIVE Laptop Camera at index {index}")
                break
            else:
                print(f"Camera at index {index} is blank or offline. Skipping...")
        temp_cap.release()
    
    if cap is None or not cap.isOpened():
        error_msg = "CRITICAL: MacOS Blocked Camera Access or Camera is OFF. Please run the server from your own visible Terminal to grant Camera Permissions!"
        print(error_msg)
        await websocket.send_text(json.dumps({"error": error_msg}))
        await websocket.close()
        return

    print(f"Active Camera Backend: {cap.getBackendName()}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab LIVE frame from laptop camera.")
                break
                
            frame = cv2.resize(frame, (640, 480))
            
            # AI Inference
            detections = engine.process_frame(frame)
            
            # Draw on frame for visualization
            for d in detections:
                box = d["box"]
                keypoints = d.get("keypoints", [])
                is_alert = d["duration"] >= 30 or d["posture"] == "falling"
                color = (0, 0, 255) if is_alert else (0, 255, 0) # Red if alert, Green if not
                
                # HIGH-LEVEL SKELETON REPRESENTATION (Mediapipe-Style)
                if keypoints:
                    skeleton_links = [
                        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
                        (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
                    ]
                    # Draw connecting lines
                    sk_color = (0, 0, 255) if is_alert else (255, 144, 30) # Red for alert, Deep Blue/Cyan for base
                    for p1_i, p2_i in skeleton_links:
                        if p1_i < len(keypoints) and p2_i < len(keypoints):
                            k1, k2 = keypoints[p1_i], keypoints[p2_i]
                            if k1[2] > 0.4 and k2[2] > 0.4: # Confidence threshold
                                pt1 = (int(k1[0]), int(k1[1]))
                                pt2 = (int(k2[0]), int(k2[1]))
                                cv2.line(frame, pt1, pt2, sk_color, 2)
                    
                    # Draw joint nodes
                    node_color = (0, 0, 255) if is_alert else (0, 255, 255)
                    for k in keypoints:
                        if k[2] > 0.4:
                            cv2.circle(frame, (int(k[0]), int(k[1])), 4, node_color, -1)

                # Draw Bounding Box & Label
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3 if is_alert else 2)
                label = f"ID:{d['id']} {d['posture']} ({d['duration']}s)"
                
                # Background for text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (int(box[0]), int(box[1]) - 20), (int(box[0]) + w, int(box[1])), color, -1)
                cv2.putText(frame, label, (int(box[0]), int(box[1])-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Convert to base64
            _, buffer = cv2.imencode(".jpg", frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            
            # Sync completed sessions to DB
            if engine.completed_sessions:
                db = SessionLocal()
                try:
                    for ses in engine.completed_sessions[:]: # Copy to safely remove
                        new_act = PostureActivity(
                            person_track_id=ses["track_id"],
                            posture_type=ses["posture"],
                            start_time=ses["start"],
                            end_time=ses["end"],
                            duration=ses["duration"],
                            confidence=ses["confidence"]
                        )
                        db.add(new_act)
                    db.commit()
                    engine.completed_sessions.clear()
                except Exception as ex:
                    print(f"Sync DB Error: {ex}")
                finally:
                    db.close()

            # Send results
            data = {
                "frame": jpg_as_text,
                "detections": detections
            }
            await websocket.send_text(json.dumps(data))
            
            # Use appropriate sleep cycle to maintain ~30 fps without overwhelming UI
            await asyncio.sleep(0.001)
    except Exception as e:
        print(f"WS Exception: {e}")
    finally:
        if cap is not None:
            cap.release()
        await websocket.close()


# Static files should still be available for other assets if any
try:
    if os.path.exists("frontend"):
        app.mount("/static", StaticFiles(directory="frontend"), name="static")
except:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
