"""
RoadGuard AI — Road Object Detector
YOLOv8-based detector for vehicles, cyclists, and pedestrians.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from core.collision_config import DetectorConfig
from core.collision.types import ObjectDetection


class RoadObjectDetector:
    """
    Detects road vehicles and vulnerable road users using YOLOv8.
    Supported classes: car, motorcycle, bus, truck, bicycle, person.
    """

    CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self.model = None
        self.device = "cpu"
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            from ultralytics import YOLO

            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            self.model = YOLO(self.config.model_path)
            print(f"[RoadObjectDetector] ✅ Loaded {self.config.model_path} on {self.device}")
        except Exception as e:
            print(f"[RoadObjectDetector] ⚠️ Model initialization fallback/mock: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        """
        Run inference on image frame and extract standardized ObjectDetection records.
        """
        if frame is None or frame.size == 0 or self.model is None:
            return []

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        try:
            results = self.model(frame, conf=self.config.confidence_threshold, device=self.device, verbose=False)
            detections: list[ObjectDetection] = []

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id not in self.CLASS_NAMES:
                        continue

                    conf = float(box.conf[0].item())
                    if conf < self.config.confidence_threshold:
                        continue

                    # Normalized coordinates [0.0 - 1.0]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    nx1, ny1, nx2, ny2 = max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h)
                    
                    bw = nx2 - nx1
                    bh = ny2 - ny1
                    cx = nx1 + bw / 2.0
                    cy = ny1 + bh / 2.0
                    area = bw * bh

                    detections.append(ObjectDetection(
                        class_name=self.CLASS_NAMES[cls_id],
                        class_id=cls_id,
                        confidence=round(conf, 4),
                        bbox_xyxy=[round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
                        bbox_center=[round(cx, 4), round(cy, 4)],
                        bbox_width=round(bw, 4),
                        bbox_height=round(bh, 4),
                        bbox_area=round(area, 4),
                    ))

            return detections
        except Exception as e:
            print(f"[RoadObjectDetector] Detection error: {e}")
            return []
