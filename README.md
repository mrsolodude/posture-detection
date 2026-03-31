# PostureGuard AI

Intelligent Human Posture Monitoring system using YOLOv11-Pose and FastAPI.

## Features
- Real-time posture tracking and classification (Sitting, Standing, Bending, etc.)
- Multi-person tracking using ByteTrack
- Incident logging and alerts
- Interactive Dashboard (FastAPI + React-style UI)
- SQLite Database integration

## Tech Stack
- **Backend**: Python (FastAPI, SQLAlchemy, Ultralytics YOLOv11)
- **Frontend**: Vanilla CSS, Tailwind, JavaScript
- **Database**: SQLite

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the backend:
   ```bash
   PYTHONPATH=. python backend/main.py
   ```
3. Open `http://localhost:8080` in your browser.

## Database Models
- `User`: User management
- `PostureActivity`: Logged sessions for each posture
- `AlertLog`: Incident logs (falls, etc.)
