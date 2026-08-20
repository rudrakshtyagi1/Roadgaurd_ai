import os

base_dir = "/Users/rudrakshtyagi/Desktop/roadgaurdai/backend"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "core"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "ws"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "routers"), exist_ok=True)

files = {}

files["requirements.txt"] = """fastapi==0.111.0
uvicorn[standard]==0.30.1
websockets==12.0
opencv-python==4.10.0.84
mediapipe==0.10.14
ultralytics==8.2.48
numpy==1.26.4
scikit-learn==1.5.0
python-multipart==0.0.9
pillow==10.4.0
aiofiles==23.2.1
pydantic==2.7.4
httpx==0.27.0
"""

files["core/__init__.py"] = ""

files["core/temporal_buffer.py"] = '''import collections
import numpy as np

class TemporalBuffer:
    """A rolling buffer for the last 5 seconds of inference data."""
    
    def __init__(self, maxlen: int = 150):
        self.buffer = collections.deque(maxlen=maxlen)
        
    def push(self, frame_data: dict) -> None:
        """Push a new frame data dictionary into the buffer."""
        self.buffer.append(frame_data)
        
    def get_stats(self) -> dict:
        """Calculate and return statistics over the buffer."""
        if not self.buffer:
            return {
                "ear_mean": 0.0,
                "ear_trend": 0.0,
                "mar_mean": 0.0,
                "perclos": 0.0,
                "yaw_mean": 0.0,
                "pitch_mean": 0.0,
                "pothole_conf_mean": 0.0,
                "vehicle_count_mean": 0.0,
                "blink_rate": 0.0
            }
            
        ears = [d.get("ear", 0.0) for d in self.buffer if d.get("ear") is not None]
        mars = [d.get("mar", 0.0) for d in self.buffer if d.get("mar") is not None]
        yaws = [d.get("head_pose", {}).get("yaw", 0.0) for d in self.buffer if d.get("head_pose")]
        pitches = [d.get("head_pose", {}).get("pitch", 0.0) for d in self.buffer if d.get("head_pose")]
        potholes = [d.get("pothole_confidence", 0.0) for d in self.buffer if d.get("pothole_confidence") is not None]
        vehicles = [d.get("vehicle_count", 0) for d in self.buffer if d.get("vehicle_count") is not None]
        
        ear_mean = float(np.mean(ears)) if ears else 0.0
        mar_mean = float(np.mean(mars)) if mars else 0.0
        yaw_mean = float(np.mean(yaws)) if yaws else 0.0
        pitch_mean = float(np.mean(pitches)) if pitches else 0.0
        pothole_conf_mean = float(np.mean(potholes)) if potholes else 0.0
        vehicle_count_mean = float(np.mean(vehicles)) if vehicles else 0.0
        
        # PERCLOS: % of frames where EAR < 0.25
        perclos = sum(1 for e in ears if e < 0.25) / len(ears) if ears else 0.0
        
        # EAR Trend on last 30 samples
        last_30_ears = ears[-30:]
        if len(last_30_ears) >= 2:
            ear_trend = float(np.polyfit(range(len(last_30_ears)), last_30_ears, 1)[0])
        else:
            ear_trend = 0.0
            
        # Blink rate (total blinks in this window)
        blink_counts = [d.get("blink_count", 0) for d in self.buffer if d.get("blink_count") is not None]
        blink_rate = float(blink_counts[-1] - blink_counts[0]) if blink_counts else 0.0
        
        return {
            "ear_mean": ear_mean,
            "ear_trend": ear_trend,
            "mar_mean": mar_mean,
            "perclos": perclos,
            "yaw_mean": yaw_mean,
            "pitch_mean": pitch_mean,
            "pothole_conf_mean": pothole_conf_mean,
            "vehicle_count_mean": vehicle_count_mean,
            "blink_rate": blink_rate
        }
'''

files["core/driver_monitor.py"] = '''import cv2
import numpy as np
import mediapipe as mp
import time

class DriverMonitor:
    """MediaPipe FaceMesh based driver monitoring."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1
        )
        self.blink_count = 0
        self.is_blinking = False
        self.blink_start_time = 0.0
        self.avg_blink_duration_ms = 0.0
        self.blink_durations = []
        
        # 3D model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye
            (225.0, 170.0, -135.0),      # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth
            (150.0, -150.0, -125.0)      # Right mouth
        ])
        self.pose_landmarks = [1, 152, 226, 446, 57, 287]

    def _calc_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _calc_ear(self, landmarks, eye_indices):
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]
        
        ear = (self._calc_distance(p2, p6) + self._calc_distance(p3, p5)) / (2.0 * self._calc_distance(p1, p4))
        return ear

    def process(self, frame: np.ndarray) -> dict:
        """Process frame and return driver state."""
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        default_return = {
            "ear": 0.0,
            "mar": 0.0,
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "eye_state": "OPEN",
            "blink_detected": False,
            "blink_count": self.blink_count,
            "avg_blink_duration_ms": self.avg_blink_duration_ms,
            "face_detected": False
        }
        
        if not results.multi_face_landmarks:
            return default_return
            
        landmarks = [(lm.x * w, lm.y * h) for lm in results.multi_face_landmarks[0].landmark]
        
        # EAR
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        left_ear = self._calc_ear(landmarks, left_eye_indices)
        right_ear = self._calc_ear(landmarks, right_eye_indices)
        ear = (left_ear + right_ear) / 2.0
        
        # MAR
        mouth_indices = [61, 291, 39, 181, 0, 17, 269, 405]
        # Approximation for MAR: height / width
        mouth_w = self._calc_distance(landmarks[61], landmarks[291])
        mouth_h = self._calc_distance(landmarks[0], landmarks[17])
        mar = mouth_h / mouth_w if mouth_w > 0 else 0.0
        
        # Head Pose
        image_points = np.array([landmarks[idx] for idx in self.pose_landmarks], dtype="double")
        camera_matrix = np.array([
            [w, 0.0, w/2],
            [0.0, w, h/2],
            [0.0, 0.0, 1.0]
        ])
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs
        )
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rotation_matrix, translation_vector))
            euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
            pitch = euler_angles[0][0]
            yaw = euler_angles[1][0]
            roll = euler_angles[2][0]
        
        # Eye State & Blinks
        blink_detected = False
        if ear <= 0.15:
            eye_state = "CLOSED"
        elif 0.15 < ear <= 0.25:
            eye_state = "CLOSING"
        else:
            eye_state = "OPEN"
            
        if ear < 0.25:
            if not self.is_blinking:
                self.is_blinking = True
                self.blink_start_time = time.time()
        else:
            if self.is_blinking:
                self.is_blinking = False
                blink_detected = True
                self.blink_count += 1
                duration = (time.time() - self.blink_start_time) * 1000.0
                self.blink_durations.append(duration)
                if len(self.blink_durations) > 20:
                    self.blink_durations.pop(0)
                self.avg_blink_duration_ms = sum(self.blink_durations) / len(self.blink_durations)
                
        return {
            "ear": ear,
            "mar": mar,
            "head_pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "eye_state": eye_state,
            "blink_detected": blink_detected,
            "blink_count": self.blink_count,
            "avg_blink_duration_ms": self.avg_blink_duration_ms,
            "face_detected": True
        }
'''

files["core/road_monitor.py"] = '''from ultralytics import YOLO
import numpy as np

class RoadMonitor:
    """YOLO-based road monitoring."""
    
    def __init__(self, model_path='yolov8n.pt', pothole_model_path=None):
        self.model = YOLO(model_path)
        self.pothole_model = YOLO(pothole_model_path) if pothole_model_path else None
        self.VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 0: 'person'}
        
    def process(self, frame: np.ndarray) -> dict:
        """Process frame and return road state."""
        results = self.model(frame, verbose=False)
        h, w = frame.shape[:2]
        
        detections = []
        vehicles_count = 0
        nearest_distance = float('inf')
        
        for det in results[0].boxes:
            conf = float(det.conf[0])
            if conf < 0.4:
                continue
            cls_id = int(det.cls[0])
            if cls_id in self.VEHICLE_CLASSES:
                vehicles_count += 1
                bbox = det.xyxy[0].cpu().numpy().tolist()
                bbox_h = bbox[3] - bbox[1]
                
                # Distance heuristic
                dist_m = (h / max(bbox_h, 1)) * 5.0
                dist_m = max(2.0, min(100.0, dist_m))
                nearest_distance = min(nearest_distance, dist_m)
                
                detections.append({
                    "class_name": self.VEHICLE_CLASSES[cls_id],
                    "confidence": conf,
                    "bbox": bbox,
                    "distance_m": dist_m
                })
                
        pothole_detected = False
        pothole_confidence = 0.0
        
        if self.pothole_model:
            p_results = self.pothole_model(frame, verbose=False)
            for det in p_results[0].boxes:
                conf = float(det.conf[0])
                if conf > 0.4:
                    pothole_detected = True
                    pothole_confidence = max(pothole_confidence, conf)
                    bbox = det.xyxy[0].cpu().numpy().tolist()
                    bbox_h = bbox[3] - bbox[1]
                    dist_m = max(2.0, min(100.0, (h / max(bbox_h, 1)) * 5.0))
                    nearest_distance = min(nearest_distance, dist_m)
                    
        # Hazard score
        hazard_score = 0.0
        for d in detections:
            if d['distance_m'] <= 10.0:
                hazard_score += 20
        if pothole_detected:
            hazard_score += 30 + (pothole_confidence * 40)
        hazard_score = max(0.0, min(100.0, hazard_score))
        
        if nearest_distance == float('inf'):
            nearest_distance = 100.0
            
        return {
            "vehicles": vehicles_count,
            "detections": detections,
            "pothole_detected": pothole_detected,
            "pothole_confidence": pothole_confidence,
            "hazard_score": hazard_score,
            "nearest_distance_m": nearest_distance
        }
'''

files["core/risk_engine.py"] = '''class RiskEngine:
    """Rule-based risk scoring."""
    
    def compute(self, driver_state: dict, road_state: dict, temporal_stats: dict) -> dict:
        ear_mean = temporal_stats.get('ear_mean', 0.0)
        perclos = temporal_stats.get('perclos', 0.0)
        mar_mean = temporal_stats.get('mar_mean', 0.0)
        pitch_mean = temporal_stats.get('pitch_mean', 0.0)
        yaw_mean = temporal_stats.get('yaw_mean', 0.0)
        ear_trend = temporal_stats.get('ear_trend', 0.0)
        
        # Fatigue score
        f_part1 = 40 * perclos
        f_part2 = 30 * (1 - ear_mean) if ear_mean < 0.4 else 0
        f_part3 = 20 if mar_mean > 0.5 else 0
        f_part4 = 10 if abs(pitch_mean) > 15 else 0
        fatigue_score = f_part1 + f_part2 + f_part3 + f_part4
        fatigue_score = min(100, max(0, fatigue_score))
        
        # Attention score
        attention_score = 0
        if abs(yaw_mean) > 15:
            attention_score += 30
        if abs(pitch_mean) > 10:
            attention_score += 20
        if ear_trend < 0:
            attention_score += 20
        attention_score = min(100, max(0, attention_score))
        
        # Hazard score
        hazard_score = road_state.get('hazard_score', 0.0)
        
        # Proximity score
        nearest = road_state.get('nearest_distance_m', 100.0)
        if nearest < 3:
            proximity_score = 100
        elif nearest < 5:
            proximity_score = 80
        elif nearest < 8:
            proximity_score = 60
        elif nearest < 15:
            proximity_score = 40
        elif nearest < 30:
            proximity_score = 20
        else:
            proximity_score = 0
            
        final_risk = int(0.35 * fatigue_score + 0.20 * attention_score + 0.25 * hazard_score + 0.20 * proximity_score)
        final_risk = min(100, max(0, final_risk))
        
        if final_risk <= 30:
            status = "SAFE"
            alert = "Normal driving"
        elif final_risk <= 55:
            status = "CAUTION"
            alert = "Stay alert"
        elif final_risk <= 75:
            status = "HIGH"
            alert = "Warning: High Risk!"
        else:
            status = "CRITICAL"
            alert = "CRITICAL WARNING: Take Action!"
            
        return {
            "risk": final_risk,
            "status": status,
            "alert": alert,
            "components": {
                "fatigue_score": fatigue_score,
                "attention_score": attention_score,
                "hazard_score": hazard_score,
                "proximity_score": proximity_score
            }
        }
'''

files["core/alert_engine.py"] = '''import collections
import time
import uuid
from datetime import datetime, timezone

class AlertEngine:
    """Alert and event logging."""
    
    def __init__(self):
        self._events = collections.deque(maxlen=100)
        self._last_alerts = {}
        self.DEBOUNCE_SECONDS = 3.0
        
    def generate_alert(self, risk_data: dict) -> dict:
        status = risk_data.get('status', 'SAFE')
        if status in ['SAFE', 'CAUTION']:
            return None
            
        alert_type = 'CRITICAL_RISK' if status == 'CRITICAL' else 'HIGH_RISK'
        now = time.time()
        
        if alert_type in self._last_alerts:
            if now - self._last_alerts[alert_type] < self.DEBOUNCE_SECONDS:
                return None
                
        self._last_alerts[alert_type] = now
        return self.log_event(alert_type, risk_data.get('alert', 'Warning'), status)
        
    def log_event(self, event_type: str, message: str, risk_level: str) -> dict:
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "message": message,
            "risk_level": risk_level
        }
        self._events.append(event)
        return event
        
    def get_recent_events(self, n: int = 20) -> list:
        events_list = list(self._events)
        return events_list[-n:]
        
    def clear_events(self):
        self._events.clear()
'''

files["ws/__init__.py"] = ""

files["ws/stream.py"] = '''import json
from fastapi import WebSocket

class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def broadcast(self, message: dict):
        text = json.dumps(message)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(text)
            except Exception:
                disconnected.append(connection)
        for c in disconnected:
            self.disconnect(c)
            
    async def send_personal(self, websocket: WebSocket, message: dict):
        await websocket.send_text(json.dumps(message))
'''

files["ws/inference_loop.py"] = '''import asyncio
import time
import cv2
import math
import traceback
from ..core.driver_monitor import DriverMonitor
from ..core.road_monitor import RoadMonitor
from ..core.temporal_buffer import TemporalBuffer
from ..core.risk_engine import RiskEngine
from ..core.alert_engine import AlertEngine
from .stream import ConnectionManager

class InferenceLoop:
    """Background inference loop."""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.driver_monitor = DriverMonitor()
        self.road_monitor = RoadMonitor()
        self.temporal_buffer = TemporalBuffer()
        self.risk_engine = RiskEngine()
        self.alert_engine = AlertEngine()
        self.sim_start = time.time()
        
    def generate_synthetic_data(self):
        """Generates realistic varying synthetic data."""
        t = time.time() - self.sim_start
        
        # Driver state oscillating
        ear = 0.25 + 0.1 * math.sin(t * 0.5)
        mar = 0.1 + 0.2 * abs(math.cos(t * 0.3))
        yaw = 10 * math.sin(t)
        pitch = 5 * math.cos(t * 1.5)
        
        driver_state = {
            "ear": ear,
            "mar": mar,
            "head_pose": {"yaw": yaw, "pitch": pitch, "roll": 0.0},
            "eye_state": "OPEN" if ear > 0.25 else ("CLOSING" if ear > 0.15 else "CLOSED"),
            "blink_detected": False,
            "blink_count": int(t / 5),
            "avg_blink_duration_ms": 150.0,
            "face_detected": True
        }
        
        # Road state
        pothole = math.sin(t) > 0.8
        hazard = 50.0 if pothole else 10.0
        
        road_state = {
            "vehicles": max(0, int(3 + 2 * math.sin(t * 0.2))),
            "detections": [],
            "pothole_detected": pothole,
            "pothole_confidence": 0.85 if pothole else 0.0,
            "hazard_score": hazard,
            "nearest_distance_m": max(5.0, 20.0 + 15 * math.sin(t * 0.8))
        }
        
        return driver_state, road_state

    async def run(self):
        driver_cap = cv2.VideoCapture(0)
        road_cap = cv2.VideoCapture(1)
        if not road_cap.isOpened():
            road_cap = cv2.VideoCapture(0)
            
        fps_buffer = []
        
        while True:
            start_t = time.perf_counter()
            try:
                driver_success, d_frame = driver_cap.read() if driver_cap.isOpened() else (False, None)
                road_success, r_frame = road_cap.read() if road_cap.isOpened() else (False, None)
                
                if driver_success and road_success:
                    driver_state = self.driver_monitor.process(d_frame)
                    road_state = self.road_monitor.process(r_frame)
                else:
                    driver_state, road_state = self.generate_synthetic_data()
                    
                self.temporal_buffer.push({
                    "ear": driver_state.get("ear"),
                    "mar": driver_state.get("mar"),
                    "head_pose": driver_state.get("head_pose"),
                    "blink_count": driver_state.get("blink_count"),
                    "pothole_confidence": road_state.get("pothole_confidence"),
                    "vehicle_count": road_state.get("vehicles")
                })
                
                stats = self.temporal_buffer.get_stats()
                risk = self.risk_engine.compute(driver_state, road_state, stats)
                alert = self.alert_engine.generate_alert(risk)
                
                end_t = time.perf_counter()
                latency = (end_t - start_t) * 1000
                fps = 1000.0 / latency if latency > 0 else 30.0
                fps_buffer.append(fps)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer)
                
                msg = {
                    "driver_state": driver_state,
                    "road_state": road_state,
                    "stats": stats,
                    "risk": risk,
                    "latency_ms": latency,
                    "fps": avg_fps,
                    "alert": alert
                }
                
                await self.manager.broadcast(msg)
                
            except Exception as e:
                print("Error in inference loop:", e)
                traceback.print_exc()
                
            await asyncio.sleep(1/30)
'''

files["routers/__init__.py"] = ""

files["routers/video.py"] = '''import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/video', tags=['video'])

@router.post('/driver')
async def process_driver(request: Request, frame: UploadFile):
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
        
    state = request.app.state.driver_monitor.process(img)
    return JSONResponse(content=state)

@router.post('/road')
async def process_road(request: Request, frame: UploadFile):
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
        
    state = request.app.state.road_monitor.process(img)
    return JSONResponse(content=state)
'''

files["routers/inference.py"] = '''import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/inference', tags=['inference'])

@router.post('/')
async def full_inference(request: Request, driver_frame: UploadFile, road_frame: UploadFile):
    d_contents = await driver_frame.read()
    d_nparr = np.frombuffer(d_contents, np.uint8)
    d_img = cv2.imdecode(d_nparr, cv2.IMREAD_COLOR)
    
    r_contents = await road_frame.read()
    r_nparr = np.frombuffer(r_contents, np.uint8)
    r_img = cv2.imdecode(r_nparr, cv2.IMREAD_COLOR)
    
    d_state = request.app.state.driver_monitor.process(d_img)
    r_state = request.app.state.road_monitor.process(r_img)
    
    request.app.state.temporal_buffer.push({
        "ear": d_state.get("ear"),
        "mar": d_state.get("mar"),
        "head_pose": d_state.get("head_pose"),
        "blink_count": d_state.get("blink_count"),
        "pothole_confidence": r_state.get("pothole_confidence"),
        "vehicle_count": r_state.get("vehicles")
    })
    
    stats = request.app.state.temporal_buffer.get_stats()
    risk = request.app.state.risk_engine.compute(d_state, r_state, stats)
    
    state = {
        "driver_state": d_state,
        "road_state": r_state,
        "stats": stats,
        "risk": risk
    }
    request.app.state.current_state = state
    return JSONResponse(content=state)

@router.get('/status')
async def get_status(request: Request):
    if hasattr(request.app.state, 'current_state'):
        return JSONResponse(content=request.app.state.current_state)
    return JSONResponse(content={"status": "no data yet"})
'''

files["routers/risk.py"] = '''from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/risk', tags=['risk'])

@router.get('/current')
async def get_current_risk(request: Request):
    if hasattr(request.app.state, 'current_state') and 'risk' in request.app.state.current_state:
        return JSONResponse(content=request.app.state.current_state['risk'])
    return JSONResponse(content={"error": "no data"})

@router.get('/history')
async def get_risk_history(request: Request):
    if hasattr(request.app.state, 'risk_history'):
        return JSONResponse(content=list(request.app.state.risk_history))
    return JSONResponse(content=[])
'''

files["routers/events.py"] = '''from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/events', tags=['events'])

@router.get('/')
async def get_events(request: Request):
    events = request.app.state.alert_engine.get_recent_events()
    return JSONResponse(content=events)

@router.post('/clear')
async def clear_events(request: Request):
    request.app.state.alert_engine.clear_events()
    return JSONResponse(content={"status": "cleared"})
'''

files["main.py"] = '''import asyncio
import uvicorn
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.driver_monitor import DriverMonitor
from core.road_monitor import RoadMonitor
from core.temporal_buffer import TemporalBuffer
from core.risk_engine import RiskEngine
from core.alert_engine import AlertEngine
from ws.stream import ConnectionManager
from ws.inference_loop import InferenceLoop

from routers import video, inference, risk, events

app = FastAPI(title='RoadGuard AI', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video.router)
app.include_router(inference.router)
app.include_router(risk.router)
app.include_router(events.router)

manager = ConnectionManager()
loop_task = None

@app.on_event('startup')
async def startup_event():
    app.state.driver_monitor = DriverMonitor()
    app.state.road_monitor = RoadMonitor()
    app.state.temporal_buffer = TemporalBuffer()
    app.state.risk_engine = RiskEngine()
    app.state.alert_engine = AlertEngine()
    app.state.risk_history = deque(maxlen=60)
    
    inference_loop = InferenceLoop(manager)
    inference_loop.driver_monitor = app.state.driver_monitor
    inference_loop.road_monitor = app.state.road_monitor
    inference_loop.temporal_buffer = app.state.temporal_buffer
    inference_loop.risk_engine = app.state.risk_engine
    inference_loop.alert_engine = app.state.alert_engine
    
    global loop_task
    loop_task = asyncio.create_task(inference_loop.run())

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get('/health')
async def health_check():
    return {"status": "ok", "version": "0.1.0"}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
'''

for path, content in files.items():
    with open(os.path.join(base_dir, path), "w") as f:
        f.write(content)
