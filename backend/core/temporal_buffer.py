import collections
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
