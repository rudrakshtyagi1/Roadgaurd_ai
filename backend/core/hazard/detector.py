"""
RoadGuard AI — Pothole & Road Hazard Detector
Wraps YOLOv8 pothole model for real-time inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.hazard_config import HazardDetectorConfig
from core.hazard.types import HazardDetection

logger = logging.getLogger("RoadGuard.HazardDetector")


class PotholeDetector:
    """Detects potholes and road surface anomalies."""

    def __init__(self, config: Optional[HazardDetectorConfig] = None) -> None:
        self.config = config or HazardDetectorConfig()
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        p = Path(self.config.model_path)
        if not p.exists():
            # Fallback to relative path from backend
            alt = Path(__file__).parent.parent.parent / "potholes" / "runs" / "pothole_yolov8n_v1" / "weights" / "best.pt"
            if alt.exists():
                p = alt

        try:
            from ultralytics import YOLO
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = YOLO(str(p))
            logger.info(f"[PotholeDetector] ✅ Loaded pothole model from {p} on {device}")
        except Exception as e:
            logger.warning(f"[PotholeDetector] ⚠️ Could not load pothole model: {e}")
            self._model = None

    def detect(self, frame: np.ndarray) -> list[HazardDetection]:
        if self._model is None or frame is None or frame.size == 0:
            return []

        try:
            results = self._model(frame, verbose=False, conf=self.config.confidence_threshold)
            if not results or not results[0].boxes:
                return []

            h, w = frame.shape[:2]
            detections: list[HazardDetection] = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id not in self.config.target_classes or conf < self.config.confidence_threshold:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                # Normalize coordinates
                nx1, ny1, nx2, ny2 = x1 / w, y1 / h, x2 / w, y2 / h
                bw = nx2 - nx1
                bh = ny2 - ny1
                cx = (nx1 + nx2) / 2.0
                cy = (ny1 + ny2) / 2.0
                area = bw * bh

                detections.append(HazardDetection(
                    class_name="pothole",
                    class_id=cls_id,
                    confidence=round(conf, 4),
                    bbox_xyxy=[round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
                    bbox_center=[round(cx, 4), round(cy, 4)],
                    bbox_width=round(bw, 4),
                    bbox_height=round(bh, 4),
                    bbox_area=round(area, 6),
                ))
            return detections
        except Exception as e:
            logger.error(f"[PotholeDetector] Inference error: {e}")
            return []
