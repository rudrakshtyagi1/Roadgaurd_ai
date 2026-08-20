from ultralytics import YOLO
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
