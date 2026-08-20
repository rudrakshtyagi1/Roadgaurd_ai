import asyncio
import time
import cv2
import math
import traceback
from core.driver_monitor import DriverMonitor
from core.road_monitor import RoadMonitor
from core.temporal_buffer import TemporalBuffer
from core.risk_engine import RiskEngine
from core.alert_engine import AlertEngine
from ws.stream import ConnectionManager

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
                
                # Derive attention level from head pose + EAR
                yaw = driver_state.get("head_pose", {}).get("yaw", 0.0)
                ear_val = driver_state.get("ear", 0.3)
                if abs(yaw) > 25 or ear_val < 0.15:
                    attention = "DISTRACTED"
                elif abs(yaw) > 15 or ear_val < 0.22:
                    attention = "LOW"
                elif abs(yaw) > 8:
                    attention = "MEDIUM"
                else:
                    attention = "HIGH"

                # Derive fatigue 0.0-1.0 from components
                comps = risk.get("components", {})
                fatigue_norm = min(1.0, comps.get("fatigue_score", 0) / 100.0)

                # Build potholes list
                potholes = []
                if road_state.get("pothole_detected"):
                    conf = road_state.get("pothole_confidence", 0.5)
                    sev = "HIGH" if conf > 0.75 else ("MEDIUM" if conf > 0.5 else "LOW")
                    potholes = [{"confidence": conf, "severity": sev, "distance_m": road_state.get("nearest_distance_m", 25.0)}]

                msg = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "driver": {
                        "ear": driver_state.get("ear", 0.0),
                        "mar": driver_state.get("mar", 0.0),
                        "head_pose": driver_state.get("head_pose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}),
                        "fatigue": fatigue_norm,
                        "attention": attention,
                        "eye_state": driver_state.get("eye_state", "OPEN"),
                        "blink_rate": stats.get("blink_rate", 0),
                        "face_detected": driver_state.get("face_detected", True),
                    },
                    "road": {
                        "potholes": potholes,
                        "vehicles": road_state.get("vehicles", 0),
                        "hazard_score": road_state.get("hazard_score", 0.0),
                        "detections": road_state.get("detections", []),
                        "nearest_distance_m": road_state.get("nearest_distance_m", 100.0),
                        "pothole_detected": road_state.get("pothole_detected", False),
                        "pothole_confidence": road_state.get("pothole_confidence", 0.0),
                    },
                    "risk_data": risk,
                    "metrics": {
                        "fps": round(avg_fps, 1),
                        "latency_ms": round(latency, 1),
                        "driver_connected": driver_cap.isOpened(),
                        "road_connected": road_cap.isOpened(),
                    },
                    "events": self.alert_engine.get_recent_events(10),
                    "temporal": {
                        "ear_mean": stats.get("ear_mean", 0.0),
                        "ear_trend": stats.get("ear_trend", 0.0),
                        "perclos": stats.get("perclos", 0.0),
                        "mar_mean": stats.get("mar_mean", 0.0),
                        "blink_rate": stats.get("blink_rate", 0),
                        "vehicle_count_mean": stats.get("vehicle_count_mean", 0.0),
                    },
                }
                
                await self.manager.broadcast(msg)
                
            except Exception as e:
                print("Error in inference loop:", e)
                traceback.print_exc()
                
            await asyncio.sleep(1/30)
