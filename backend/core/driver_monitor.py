"""
DriverMonitor — MediaPipe FaceMesh & PyTorch Driver Drowsiness Perception Pipeline.
Combines geometric facial landmarks (EAR, MAR, 3D Head Pose) with the trained
PyTorch CustomDriverCNN model and TemporalDriverBuffer.
"""
import time
import math
import cv2
import numpy as np

try:
    from core.driver_classifier import DriverClassifier, TemporalDriverBuffer
except ImportError:
    from driver_classifier import DriverClassifier, TemporalDriverBuffer


class DriverMonitor:
    """
    Monitors driver state using:
      1. PyTorch CustomDriverCNN for raw deep facial drowsiness probability
      2. TemporalDriverBuffer for rolling hysteresis & debounced alert state
      3. MediaPipe FaceMesh for EAR, MAR, and 3D head pose estimation
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

    def __init__(self):
        self._face_mesh = None
        self._mesh_failed = False

        # Initialize PyTorch Drowsiness Classifier & Temporal Buffer
        try:
            self.classifier = DriverClassifier()
            self.temporal_buffer = TemporalDriverBuffer(window_size=15, drowsy_threshold=0.55, consecutive_frames_required=5)
            print("[DriverMonitor] ✅ PyTorch DriverClassifier & Temporal Buffer initialized")
        except Exception as e:
            print(f"[DriverMonitor] ⚠️ DriverClassifier init error: {e}")
            self.classifier = None
            self.temporal_buffer = None

        # Initialize MediaPipe FaceMesh
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=1,
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[DriverMonitor] ✅ MediaPipe FaceMesh initialized (mp.solutions)")
        except Exception as e:
            print(f"[DriverMonitor] ⚠️ MediaPipe unavailable: {e}")
            self._mesh_failed = True

        self.blink_count = 0
        self.is_blinking = False
        self.blink_start_time = 0.0
        self.blink_durations: list[float] = []
        self.avg_blink_duration_ms = 0.0
        self._sim_t = time.time()

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

    def _head_pose(self, lm: list, frame_w: int, frame_h: int):
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

    def _extract_face_crop(self, frame: np.ndarray, lm: list, w: int, h: int) -> np.ndarray:
        """Extract padded square face crop from landmarks for the CNN classifier."""
        xs = [int(p[0] * w) for p in lm]
        ys = [int(p[1] * h) for p in lm]
        min_x, max_x = max(0, min(xs)), min(w, max(xs))
        min_y, max_y = max(0, min(ys)), min(h, max(ys))

        box_w = max_x - min_x
        box_h = max_y - min_y
        pad = int(max(box_w, box_h) * 0.15)

        x1 = max(0, min_x - pad)
        y1 = max(0, min_y - pad)
        x2 = min(w, max_x + pad)
        y2 = min(h, max_y + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return frame
        return crop

    # ── synthetic fallback ────────────────────────────────────────────────────

    def _synthetic(self) -> dict:
        """Oscillating synthetic driver state for demo when no camera/MediaPipe."""
        t = time.time() - self._sim_t
        ear = 0.28 + 0.07 * math.sin(t * 0.7)
        mar = max(0.0, 0.08 + 0.15 * abs(math.sin(t * 0.25)))
        yaw = 8.0 * math.sin(t * 0.4)
        pitch = 4.0 * math.cos(t * 0.6)
        roll = 2.0 * math.sin(t * 0.3)
        eye_state = "OPEN" if ear > 0.25 else ("CLOSING" if ear > 0.15 else "CLOSED")
        blink = math.sin(t * 3.5) > 0.97
        if blink:
            self.blink_count += 1

        prob = 0.15 + 0.7 * max(0.0, math.sin(t * 0.3))

        return {
            "ear": round(ear, 4),
            "mar": round(mar, 4),
            "head_pose": {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)},
            "eye_state": eye_state,
            "blink_detected": blink,
            "blink_count": self.blink_count,
            "avg_blink_duration_ms": 145.0,
            "face_detected": True,
            "drowsy_probability": round(prob, 4),
            "smoothed_fatigue": round(prob, 4),
            "state": "DROWSY" if prob > 0.65 else ("LOW_VIGILANCE" if prob > 0.4 else "ALERT"),
            "latency_ms": 3.1,
            "is_mock": True
        }

    # ── main API ──────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> dict:
        """
        Process a BGR video frame from live camera.
        Returns unified dictionary with deep CNN prediction, temporal buffer state, and geometric landmarks.
        """
        if frame is None or frame.size == 0:
            return self._synthetic()

        t_start = time.perf_counter()
        h, w = frame.shape[:2]

        default_result = {
            "ear": 0.3, "mar": 0.1,
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "eye_state": "OPEN", "blink_detected": False,
            "blink_count": self.blink_count,
            "avg_blink_duration_ms": self.avg_blink_duration_ms,
            "face_detected": False,
            "drowsy_probability": 0.0,
            "smoothed_fatigue": 0.0,
            "state": "ALERT",
            "latency_ms": 0.0,
            "is_mock": False
        }

        try:
            face_detected = False
            ear = 0.3
            mar = 0.1
            yaw, pitch, roll = 0.0, 0.0, 0.0
            eye_state = "OPEN"
            blink_detected = False
            face_crop = frame

            # 1. MediaPipe Geometric Landmarks
            if self._face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_detected = True
                    raw = results.multi_face_landmarks[0].landmark
                    lm = [(l.x, l.y) for l in raw]

                    ear = (self._ear(lm, self.LEFT_EYE) + self._ear(lm, self.RIGHT_EYE)) / 2.0
                    mar = self._mar(lm)
                    yaw, pitch, roll = self._head_pose(lm, w, h)
                    eye_state = "CLOSED" if ear <= 0.15 else ("CLOSING" if ear <= 0.25 else "OPEN")

                    # Blink logic
                    if ear < 0.25:
                        if not self.is_blinking:
                            self.is_blinking = True
                            self.blink_start_time = time.time()
                    else:
                        if self.is_blinking:
                            self.is_blinking = False
                            blink_detected = True
                            self.blink_count += 1
                            dur = (time.time() - self.blink_start_time) * 1000.0
                            self.blink_durations.append(dur)
                            if len(self.blink_durations) > 20:
                                self.blink_durations.pop(0)
                            self.avg_blink_duration_ms = sum(self.blink_durations) / len(self.blink_durations)

                    # Extract face crop for CNN
                    face_crop = self._extract_face_crop(frame, lm, w, h)

            # 2. PyTorch CustomDriverCNN Drowsiness Inference
            drowsy_prob = 0.0
            smoothed_fatigue = 0.0
            driver_state_label = "ALERT"

            if self.classifier is not None:
                # If face not detected by mediapipe, still run on center crop/frame
                drowsy_prob = self.classifier.predict_proba(face_crop)
                if self.temporal_buffer is not None:
                    buf_stat = self.temporal_buffer.push(drowsy_prob)
                    smoothed_fatigue = buf_stat["smoothed_fatigue"]
                    driver_state_label = buf_stat["state"]
                else:
                    smoothed_fatigue = drowsy_prob
                    driver_state_label = "DROWSY" if drowsy_prob >= 0.65 else ("LOW_VIGILANCE" if drowsy_prob >= 0.40 else "ALERT")

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            res = {
                "ear": round(ear, 4),
                "mar": round(mar, 4),
                "head_pose": {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)},
                "eye_state": eye_state,
                "blink_detected": blink_detected,
                "blink_count": self.blink_count,
                "avg_blink_duration_ms": round(self.avg_blink_duration_ms, 1),
                "face_detected": face_detected,
                "drowsy_probability": round(drowsy_prob, 4),
                "smoothed_fatigue": round(smoothed_fatigue, 4),
                "state": driver_state_label,
                "latency_ms": round(latency_ms, 2),
                "is_mock": False
            }
            self.latest_state = res
            self.latest_timestamp = time.time()
            return res

        except Exception as e:
            print(f"[DriverMonitor] process() error: {e}")
            return default_result
