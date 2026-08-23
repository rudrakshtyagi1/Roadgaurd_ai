"""
DriverMonitor — MediaPipe FaceLandmarker & PyTorch Driver Drowsiness Perception Pipeline.
Integrates geometric facial landmarks (EAR, MAR, 3D Head Pose), PyTorch CustomDriverCNN,
and DriverStateEngineV2 for calibrated temporal state evaluation.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    from core.driver_classifier import DriverClassifier
    from core.driver_state_engine import DriverStateEngineV2
except ImportError:
    from driver_classifier import DriverClassifier
    from driver_state_engine import DriverStateEngineV2


class DriverMonitor:
    """
    Monitors driver state using:
      1. PyTorch CustomDriverCNN for raw deep facial drowsiness probability.
      2. DriverStateEngineV2 for monotonic temporal PERCLOS, blink/yawn hysteresis,
         and reason-coded state machine (NORMAL -> FATIGUE_RISK -> DROWSY -> CRITICAL).
      3. MediaPipe FaceLandmarker for EAR, MAR, and 3D head pose estimation.
    """

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),           # Nose tip (lm 1)
        (0.0, -330.0, -65.0),      # Chin (lm 152)
        (-225.0, 170.0, -135.0),   # Left eye corner (lm 226)
        (225.0, 170.0, -135.0),    # Right eye corner (lm 446)
        (-150.0, -150.0, -125.0),  # Left mouth (lm 57)
        (150.0, -150.0, -125.0),   # Right mouth (lm 287)
    ], dtype=np.float64)
    POSE_LM_INDICES = [1, 152, 226, 446, 57, 287]

    def __init__(self, face_model_path: Optional[str] = None) -> None:
        self._landmarker = None
        self._mesh_failed = False

        # Initialize PyTorch Drowsiness Classifier
        try:
            self.classifier = DriverClassifier()
            print("[DriverMonitor] ✅ PyTorch DriverClassifier initialized")
        except Exception as e:
            print(f"[DriverMonitor] ⚠️ DriverClassifier init error: {e}")
            self.classifier = None

        # Initialize Driver State Engine V2
        self.state_engine = DriverStateEngineV2()
        print("[DriverMonitor] ✅ DriverStateEngineV2 initialized")

        # Initialize MediaPipe FaceLandmarker Task
        model_file = face_model_path or "/Users/rudrakshtyagi/Desktop/roadgaurdai/backend/models/face_landmarker.task"
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if Path(model_file).exists():
                base_options = python.BaseOptions(model_asset_path=model_file)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=1
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(options)
                print(f"[DriverMonitor] ✅ MediaPipe FaceLandmarker task initialized from {model_file}")
            else:
                print(f"[DriverMonitor] ⚠️ MediaPipe task model not found at {model_file}")
                self._mesh_failed = True
        except Exception as e:
            print(f"[DriverMonitor] ⚠️ MediaPipe unavailable: {e}")
            self._mesh_failed = True

        self.latest_state: Optional[dict[str, Any]] = None
        self.latest_timestamp: float = 0.0
        self.last_frame_time: float = 0.0
        self._sim_t = time.time()
        self._frame_count = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dist(p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _ear(self, lm: list, indices: list) -> float:
        pts = [lm[i] for i in indices]
        num = self._dist(pts[1], pts[5]) + self._dist(pts[2], pts[4])
        den = 2.0 * self._dist(pts[0], pts[3])
        return num / den if den > 0 else 0.3

    def _mar(self, lm: list) -> float:
        h = self._dist(lm[0], lm[17])
        w = self._dist(lm[61], lm[291])
        return h / w if w > 0 else 0.1

    def _head_pose(self, lm: list, frame_w: int, frame_h: int) -> tuple[float, float, float]:
        img_pts = np.array(
            [[lm[i][0] * frame_w, lm[i][1] * frame_h] for i in self.POSE_LM_INDICES],
            dtype=np.float64,
        )
        cam = np.array([[frame_w, 0, frame_w / 2],
                        [0, frame_w, frame_h / 2],
                        [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(self.MODEL_POINTS, img_pts, cam, np.zeros((4, 1)))
        if not ok:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec)))
        pitch, yaw, roll = angles.flatten()[:3]
        return float(yaw), float(pitch), float(roll)

    def _extract_face_crop(self, frame: np.ndarray, lm: list, w: int, h: int) -> tuple[np.ndarray, list[int]]:
        """Extract square face crop centered around landmark coordinates matching training distribution."""
        xs = [int(p[0] * w) for p in lm]
        ys = [int(p[1] * h) for p in lm]
        min_x, max_x = max(0, min(xs)), min(w, max(xs))
        min_y, max_y = max(0, min(ys)), min(h, max(ys))

        box_w = max_x - min_x
        box_h = max_y - min_y
        side = max(box_w, box_h)
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        pad_side = int(side * 1.35)

        x1 = max(0, cx - pad_side // 2)
        y1 = max(0, cy - pad_side // 2)
        x2 = min(w, cx + pad_side // 2)
        y2 = min(h, cy + pad_side // 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return frame, [0, 0, w, h]
        return crop, [x1, y1, x2, y2]

    # ── synthetic fallback ────────────────────────────────────────────────────

    def _synthetic(self) -> dict[str, Any]:
        """Oscillating synthetic driver state for demo when no camera active."""
        t = time.time() - self._sim_t
        ear = 0.28 + 0.07 * math.sin(t * 0.7)
        mar = max(0.0, 0.08 + 0.15 * abs(math.sin(t * 0.25)))
        yaw = 8.0 * math.sin(t * 0.4)
        pitch = 4.0 * math.cos(t * 0.6)
        roll = 2.0 * math.sin(t * 0.3)
        prob = 0.05 + 0.35 * max(0.0, math.sin(t * 0.3))

        self._frame_count += 1
        mono_ms = time.monotonic() * 1000.0
        v2_res = self.state_engine.update(
            ear=ear,
            mar=mar,
            head_pose={"yaw": yaw, "pitch": pitch, "roll": roll},
            cnn_drowsy_prob=prob,
            face_detected=True,
            monotonic_ms=mono_ms,
            frame_index=self._frame_count,
        )

        eye_state = "OPEN" if ear > 0.25 else ("CLOSING" if ear > 0.15 else "CLOSED")

        return {
            "ear": round(ear, 4),
            "mar": round(mar, 4),
            "head_pose": {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)},
            "eye_state": eye_state,
            "blink_detected": False,
            "blink_count": v2_res["metrics"]["total_blink_count"],
            "avg_blink_duration_ms": v2_res["metrics"]["avg_blink_duration_ms"],
            "face_detected": True,
            "cnn_drowsy_probability": round(prob, 4),
            "cnn_predicted_class": "NON_DROWSY" if prob < 0.50 else "DROWSY",
            "cnn_predicted_class_probability": round(1.0 - prob if prob < 0.50 else prob, 4),
            "drowsy_probability": round(prob, 4),
            "smoothed_fatigue": v2_res["fatigue_risk_score"],
            "state": v2_res["state"],
            "reason_codes": v2_res["reason_codes"],
            "fatigue_risk_score": v2_res["fatigue_risk_score"],
            "signal_strength": v2_res["signal_strength"],
            "tracking_state": v2_res["tracking_state"],
            "driver_state_v2": v2_res,
            "latency_ms": 3.1,
            "is_mock": True,
        }

    # ── main API ──────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Process a BGR video frame from live camera.
        Returns unified dictionary with MediaPipe landmarks, deep CNN prediction on face crop,
        and DriverStateEngineV2 temporal evaluation.
        """
        if frame is None or frame.size == 0:
            return self._synthetic()

        t_start = time.perf_counter()
        mono_ms = time.monotonic() * 1000.0
        self._frame_count += 1
        h, w = frame.shape[:2]

        try:
            face_detected = False
            ear = 0.30
            mar = 0.10
            yaw, pitch, roll = 0.0, 0.0, 0.0
            face_crop = None
            crop_bbox = [0, 0, w, h]

            # 1. MediaPipe Geometric Landmarks
            if self._landmarker is not None:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self._landmarker.detect(mp_img)

                if results.face_landmarks and len(results.face_landmarks) > 0:
                    face_detected = True
                    raw = results.face_landmarks[0]
                    lm = [(l.x, l.y) for l in raw]

                    ear = (self._ear(lm, self.LEFT_EYE) + self._ear(lm, self.RIGHT_EYE)) / 2.0
                    mar = self._mar(lm)
                    yaw, pitch, roll = self._head_pose(lm, w, h)
                    face_crop, crop_bbox = self._extract_face_crop(frame, lm, w, h)

            # 2. PyTorch CustomDriverCNN Drowsiness Inference (ONLY on detected face crops!)
            drowsy_prob = 0.0
            cnn_pred_class = "NON_DROWSY"
            cnn_pred_prob = 1.0
            cnn_debug_info = {}

            if face_detected and face_crop is not None and self.classifier is not None:
                cnn_res = self.classifier.predict(face_crop, return_debug=True)
                drowsy_prob = cnn_res["cnn_drowsy_probability"]
                cnn_pred_class = cnn_res["cnn_predicted_class"]
                cnn_pred_prob = cnn_res["cnn_predicted_class_probability"]
                cnn_debug_info = cnn_res.get("cnn_debug", {})
            elif not face_detected:
                # When face is not detected, CNN should not infer on background
                drowsy_prob = 0.0
                cnn_pred_class = "NO_FACE"
                cnn_pred_prob = 0.0

            # 3. DriverStateEngineV2 Temporal Evaluation
            v2_res = self.state_engine.update(
                ear=ear,
                mar=mar,
                head_pose={"yaw": yaw, "pitch": pitch, "roll": roll},
                cnn_drowsy_prob=drowsy_prob,
                face_detected=face_detected,
                monotonic_ms=mono_ms,
                frame_index=self._frame_count,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            eye_state = "CLOSED" if ear <= 0.15 else ("CLOSING" if ear <= 0.25 else "OPEN")

            res = {
                "ear": round(ear, 4),
                "mar": round(mar, 4),
                "head_pose": {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)},
                "eye_state": eye_state,
                "blink_detected": (v2_res["metrics"]["eye_closure_duration_ms"] > 0),
                "blink_count": v2_res["metrics"]["total_blink_count"],
                "avg_blink_duration_ms": v2_res["metrics"]["avg_blink_duration_ms"],
                "face_detected": face_detected,
                "cnn_drowsy_probability": round(drowsy_prob, 4),
                "cnn_predicted_class": cnn_pred_class,
                "cnn_predicted_class_probability": round(cnn_pred_prob, 4),
                "drowsy_probability": round(drowsy_prob, 4),
                "smoothed_fatigue": v2_res["fatigue_risk_score"],
                "state": v2_res["state"],
                "reason_codes": v2_res["reason_codes"],
                "fatigue_risk_score": v2_res["fatigue_risk_score"],
                "signal_strength": v2_res["signal_strength"],
                "tracking_state": v2_res["tracking_state"],
                "wakefulness_support": v2_res.get("wakefulness_support", 1.0),
                "signal_disagreement": v2_res.get("signal_disagreement", False),
                "active_signals": v2_res.get("active_signals", {}),
                "recent_signals": v2_res.get("recent_signals", {}),
                "driver_state_v2": v2_res,
                "latency_ms": round(latency_ms, 2),
                "is_mock": False,
            }
            self.latest_state = res
            self.latest_timestamp = time.time()
            self.last_frame_time = self.latest_timestamp
            return res

        except Exception as e:
            print(f"[DriverMonitor] process() error: {e}")
            return self._synthetic()
