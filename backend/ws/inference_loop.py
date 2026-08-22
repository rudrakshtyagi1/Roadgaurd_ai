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
    """Background inference loop integrating live pothole model processing."""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.driver_monitor = DriverMonitor()
        self.road_monitor = RoadMonitor()
        self.temporal_buffer = TemporalBuffer()
        self.risk_engine = RiskEngine()
        self.alert_engine = AlertEngine()
        self.sim_start = time.time()
        
    def generate_synthetic_data(self):
        """Generates realistic varying synthetic data for the Driver Monitor branch."""
        t = time.time() - self.sim_start
        
        # Driver state oscillating (mock/prototype indicator)
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
        
        return driver_state

    async def run(self):
        video_path = '/Users/rudrakshtyagi/Desktop/roadgaurdai/potholes/sample_video.mp4'
        print(f"[InferenceLoop] Loading road feed from local video: {video_path}")
        road_cap = cv2.VideoCapture(video_path)
        
        fps_buffer = []
        
        while True:
            start_t = time.perf_counter()
            try:
                # 1. Driver monitoring: use live camera frames if posted recently, otherwise fallback to synthetic data
                latest_driver = getattr(self.driver_monitor, "latest_state", None)
                latest_driver_t = getattr(self.driver_monitor, "latest_timestamp", 0.0)

                if latest_driver is not None and (time.time() - latest_driver_t) < 3.0:
                    driver_state = latest_driver
                    is_driver_live = True
                else:
                    driver_state = self.generate_synthetic_data()
                    is_driver_live = False
                
                # 2. Road monitoring runs real-time inference on the test video
                road_success, r_frame = road_cap.read() if road_cap.isOpened() else (False, None)
                
                # Automatic loop reset when video ends
                if not road_success or r_frame is None:
                    if road_cap.isOpened():
                        road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        road_success, r_frame = road_cap.read()
                        
                if road_success and r_frame is not None:
                    # Run actual pothole detector model
                    road_state = self.road_monitor.process(r_frame)
                else:
                    # Fail-safe empty road state if video cannot load
                    road_state = {
                        "vehicles": 0,
                        "detections": [],
                        "pothole_detected": False,
                        "pothole_confidence": 0.0,
                        "hazard_score": 0.0,
                        "nearest_distance_m": 100.0,
                        "relative_proximity": "FAR",
                        "potholes": [],
                        "frame_data": ""
                    }
                    
                # Push data to temporal buffer for stats
                self.temporal_buffer.push({
                    "ear": driver_state.get("ear"),
                    "mar": driver_state.get("mar"),
                    "head_pose": driver_state.get("head_pose"),
                    "blink_count": driver_state.get("blink_count"),
                    "pothole_confidence": road_state.get("pothole_confidence"),
                    "vehicle_count": road_state.get("vehicles")
                })
                
                stats = self.temporal_buffer.get_stats()
                
                # Compute risk engine components
                risk = self.risk_engine.compute(driver_state, road_state, stats)
                
                # Handle alert logs (potholes + overall risk alerts)
                self.alert_engine.generate_alert(risk)
                self.alert_engine.generate_road_alert(road_state)
                
                end_t = time.perf_counter()
                latency = (end_t - start_t) * 1000
                fps = 1000.0 / latency if latency > 0 else 30.0
                fps_buffer.append(fps)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer)
                
                # Derive attention level for mock driver
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

                fatigue_norm = min(1.0, risk.get("components", {}).get("fatigue_score", 0) / 100.0)

                # Broadcast WS message
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
                        "drowsy_probability": driver_state.get("drowsy_probability", 0.0),
                        "smoothed_fatigue": driver_state.get("smoothed_fatigue", 0.0),
                        "state": driver_state.get("state", "ALERT"),
                        "latency_ms": driver_state.get("latency_ms", 3.1),
                        "is_mock": not is_driver_live
                    },
                    "road": {
                        "potholes": road_state.get("potholes", []),
                        "vehicles": road_state.get("vehicles", 0),
                        "hazard_score": road_state.get("hazard_score", 0.0),
                        "detections": road_state.get("detections", []),
                        "nearest_distance_m": road_state.get("nearest_distance_m", 100.0),
                        "pothole_detected": road_state.get("pothole_detected", False),
                        "pothole_confidence": road_state.get("pothole_confidence", 0.0),
                        "relative_proximity": road_state.get("relative_proximity", "FAR"),
                        "frame_data": road_state.get("frame_data", ""),
                        "is_mock": False
                    },
                    "risk_data": risk,
                    "metrics": {
                        "fps": round(avg_fps, 1),
                        "latency_ms": round(latency, 1),
                        "driver_connected": True,
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
