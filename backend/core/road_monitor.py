import cv2
import numpy as np
import torch
import base64
from ultralytics import YOLO

class RoadMonitor:
    """YOLO-based road monitoring with trained pothole detector integration."""
    
    def __init__(self, model_path='/Users/rudrakshtyagi/Desktop/roadgaurdai/potholes/runs/pothole_yolov8n_v1/weights/best.pt'):
        # Check device support
        if torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        print(f"[RoadMonitor] Loading model from {model_path} on device {self.device}...")
        self.model = YOLO(model_path)
        
        # Configuration parameters
        self.confidence_threshold = 0.5
        self.minimum_persistence_frames = 3
        self.alert_cooldown = 3.0  # seconds
        
        # Temporal stability tracking
        self.frames_pothole_detected = 0
        self.last_alert_time = 0.0
        
    def process(self, frame: np.ndarray) -> dict:
        """Process frame, detect potholes, draw bounding boxes, and compute metrics."""
        if frame is None:
            return {
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
            
        h, w = frame.shape[:2]
        
        # Run inference using the trained model
        results = self.model(frame, conf=self.confidence_threshold, device=self.device, verbose=False)
        
        potholes = []
        raw_pothole_detected = False
        max_confidence = 0.0
        nearest_distance = 100.0
        relative_proximity = "FAR"
        
        # Process detections
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Since this is single-class model, class_id must be 0 (pothole)
            if cls_id == 0:
                raw_pothole_detected = True
                max_confidence = max(max_confidence, conf)
                
                # Get bounding box coordinates in xyxy
                bbox = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = bbox
                
                # Bounding box relative height & area for relative proximity estimation
                box_w = x2 - x1
                box_h = y2 - y1
                area_pct = (box_w * box_h) / (w * h)
                bottom_pct = y2 / h
                
                # Defensive relative proximity logic
                if area_pct >= 0.08 or bottom_pct >= 0.75:
                    proximity = "NEAR"
                    dist_val = 5.0
                    box_color = (30, 45, 225)      # Crisp High Hazard Red
                elif area_pct >= 0.025 or bottom_pct >= 0.5:
                    proximity = "MEDIUM"
                    dist_val = 15.0
                    box_color = (25, 140, 245)     # Amber/Orange
                else:
                    proximity = "FAR"
                    dist_val = 35.0
                    box_color = (35, 180, 235)     # Warm Yellow
                    
                nearest_distance = min(nearest_distance, dist_val)
                if proximity == "NEAR":
                    relative_proximity = "NEAR"
                elif proximity == "MEDIUM" and relative_proximity != "NEAR":
                    relative_proximity = "MEDIUM"
                    
                severity = "HIGH" if proximity == "NEAR" else ("MEDIUM" if proximity == "MEDIUM" else "LOW")
                
                potholes.append({
                    "confidence": conf,
                    "severity": severity,
                    "distance_m": dist_val
                })

                # ── CLEAN AUTOMOTIVE BOUNDING BOX OVERLAY ──
                # 1. Bounding Box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2, cv2.LINE_AA)
                
                # 2. Sleek Compact Label Badge (prevents overlapping clutter)
                label = f"POTHOLE {int(conf * 100)}%"
                font_scale = 0.38
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Place badge cleanly at top-left of box
                tag_y1 = max(0, int(y1) - th - 5)
                tag_y2 = max(th + 5, int(y1))
                tag_x2 = min(w, int(x1) + tw + 6)
                
                cv2.rectangle(frame, (int(x1), tag_y1), (tag_x2, tag_y2), box_color, -1)
                cv2.putText(
                    frame, 
                    label, 
                    (int(x1) + 3, tag_y2 - 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (255, 255, 255), 
                    thickness, 
                    cv2.LINE_AA
                )
                
        # Temporal stability/persistence tracking
        if raw_pothole_detected:
            self.frames_pothole_detected = min(self.minimum_persistence_frames + 5, self.frames_pothole_detected + 1)
        else:
            self.frames_pothole_detected = max(0, self.frames_pothole_detected - 1)
            
        confirmed_pothole = self.frames_pothole_detected >= self.minimum_persistence_frames
        
        # ── REALISTIC, NON-SATURATING ROAD HAZARD SCORE ──
        hazard_score = 0.0
        if confirmed_pothole and potholes:
            # Base score scaled by closest proximity
            if relative_proximity == "NEAR":
                base_score = 55.0
            elif relative_proximity == "MEDIUM":
                base_score = 35.0
            else:
                base_score = 18.0
                
            # Soft logarithmic multi-pothole penalty (max +15)
            mult_penalty = min(15.0, (len(potholes) - 1) * 2.5)
            # Confidence bonus (max +12)
            conf_bonus = max_confidence * 12.0
            
            hazard_score = base_score + mult_penalty + conf_bonus
            # Clamped realistically around 85 max so 9 potholes doesn't trivialize to 100
            hazard_score = min(85.0, max(0.0, hazard_score))
            
        # Encode frame to base64 jpeg image
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        frame_data = f"data:image/jpeg;base64,{frame_base64}"
        
        return {
            "vehicles": 0,  # Single-class pothole model only
            "detections": [],
            "pothole_detected": confirmed_pothole,
            "pothole_confidence": max_confidence,
            "hazard_score": hazard_score,
            "nearest_distance_m": nearest_distance,
            "relative_proximity": relative_proximity,
            "potholes": potholes,
            "frame_data": frame_data
        }
