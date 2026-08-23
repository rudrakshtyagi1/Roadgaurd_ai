"""
RoadGuard AI — Unified Real-Time Inference Loop (Phases 0 to 8)
30 Hz async perception loop with multi-module failure isolation,
unified temporal risk fusion (RoadGuard Brain), intelligent decision prioritzation,
edge performance profiling, and backward-compatible WebSocket broadcasting.
"""

from __future__ import annotations

import asyncio
import math
import time
import traceback

import cv2

from core.alert_engine import AlertEngine
from core.brain import RoadGuardBrain, UnifiedRiskAssessment
from core.collision import CollisionIntelligenceEngine
from core.decision import AlertPrioritizerEngine, AlertIntervention
from core.driver_monitor import DriverMonitor
from core.edge import AdaptivePipelineScheduler, EdgeProfiler
from core.flywheel import HardCaseMiner
from core.hazard import RoadHazardEngine
from core.inference_logger import InferenceLogger
from core.perception_event import PerceptionEvent
from core.risk_engine import RiskEngine
from core.road_monitor import RoadMonitor
from core.temporal_buffer import TemporalBuffer
from pipeline.runner import PipelineRunner
from ws.stream import ConnectionManager

_ROAD_VIDEO_PATH = "/Users/rudrakshtyagi/Desktop/roadgaurdai/potholes/sample_video.mp4"
_DRIVER_FRESHNESS_S = 3.0


class InferenceLoop:
    """Master background inference loop orchestrating all perception engines."""

    def __init__(self, manager: ConnectionManager) -> None:
        self.manager = manager

        # Core Engines
        self.driver_monitor = DriverMonitor()
        self.road_monitor = RoadMonitor()
        self.collision_engine = CollisionIntelligenceEngine()
        self.hazard_engine = RoadHazardEngine()
        self.temporal_buffer = TemporalBuffer()
        self.risk_engine = RiskEngine()
        self.alert_engine = AlertEngine()

        # Brain & Decision Engines
        self.brain = RoadGuardBrain()
        self.decision_engine = AlertPrioritizerEngine(debounce_cooldown_s=2.5)

        # Edge & Flywheel
        self.scheduler = AdaptivePipelineScheduler(base_fps=30.0)
        self.profiler = EdgeProfiler()
        self.miner = HardCaseMiner(max_queue_size=100)

        # Telemetry logger
        self._logger = InferenceLogger()

        # Pipeline runner registering isolated modules
        self._pipeline = PipelineRunner(logger=self._logger)
        self._pipeline.register(
            source="road_monitor",
            module=self.road_monitor,
            event_type="road_hazard",
            confidence_key="pothole_confidence",
        )
        self._pipeline.register(
            source="collision_engine",
            module=self.collision_engine,
            event_type="collision_state",
            confidence_key="global_collision_risk_score",
        )
        self._pipeline.register(
            source="hazard_engine",
            module=self.hazard_engine,
            event_type="hazard_state",
            confidence_key="global_hazard_risk_score",
        )

        self._sim_start = time.time()
        self._frame_index = 0

    def _synthetic_driver(self) -> dict:
        t = time.time() - self._sim_start
        ear = 0.25 + 0.1 * math.sin(t * 0.5)
        mar = 0.1 + 0.2 * abs(math.cos(t * 0.3))
        yaw = 10 * math.sin(t)
        pitch = 5 * math.cos(t * 1.5)
        return {
            "ear": ear,
            "mar": mar,
            "head_pose": {"yaw": yaw, "pitch": pitch, "roll": 0.0},
            "eye_state": "OPEN" if ear > 0.25 else ("CLOSING" if ear > 0.15 else "CLOSED"),
            "blink_count": int(t * 0.3),
            "face_detected": True,
            "drowsy_probability": 0.05,
            "smoothed_fatigue": 0.05,
            "state": "ALERT",
            "reason_codes": ["BASELINE_NORMAL"],
            "fatigue_risk_score": 0.05,
            "signal_strength": 1.0,
            "tracking_state": "TRACKED",
            "driver_state_v2": {},
            "latency_ms": 1.5,
        }

    async def run(self) -> None:
        print(f"[InferenceLoop] Loading road feed: {_ROAD_VIDEO_PATH}")
        road_cap = cv2.VideoCapture(_ROAD_VIDEO_PATH)

        while True:
            loop_t0 = time.perf_counter()
            self._frame_index += 1
            now_mono_ms = time.monotonic() * 1000.0

            try:
                # ── 1. Driver state ───────────────────────────────────────
                live_driver = self.driver_monitor.latest_state
                last_time = getattr(self.driver_monitor, "latest_timestamp", getattr(self.driver_monitor, "last_frame_time", 0.0))
                is_driver_live = (
                    live_driver is not None
                    and (time.time() - last_time) < _DRIVER_FRESHNESS_S
                )
                driver_state = live_driver if is_driver_live else self._synthetic_driver()

                # ── 2. Road video frame ──────────────────────────────────
                road_success, r_frame = (
                    road_cap.read() if road_cap.isOpened() else (False, None)
                )
                if not road_success or r_frame is None:
                    if road_cap.isOpened():
                        road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        road_success, r_frame = road_cap.read()

                # ── 3. Pipeline tick (Road Hazard + Collision Engines) ──
                if road_success and r_frame is not None:
                    events, _ = self._pipeline.tick(r_frame, frame_index=self._frame_index)
                    road_event = PipelineRunner.get(events, "road_monitor")
                    road_state = road_event.data if road_event and not road_event.is_error else _empty_road_state()

                    collision_event = PipelineRunner.get(events, "collision_engine")
                    collision_state = collision_event.data if collision_event and not collision_event.is_error else {}

                    hazard_event = PipelineRunner.get(events, "hazard_engine")
                    hazard_state = hazard_event.data if hazard_event and not hazard_event.is_error else {}
                else:
                    road_state = _empty_road_state()
                    collision_state = {}
                    hazard_state = {}

                # ── 4. Temporal buffer + legacy risk + alerts ─────────────
                self.temporal_buffer.push({
                    "ear": driver_state.get("ear"),
                    "mar": driver_state.get("mar"),
                    "head_pose": driver_state.get("head_pose"),
                    "blink_count": driver_state.get("blink_count"),
                    "pothole_confidence": road_state.get("pothole_confidence"),
                    "vehicle_count": road_state.get("vehicles"),
                })
                stats = self.temporal_buffer.get_stats()
                risk = self.risk_engine.compute(driver_state, road_state, stats)
                self.alert_engine.generate_alert(risk)
                self.alert_engine.generate_road_alert(road_state)

                # ── 5. Phase 4: RoadGuard Brain (Unified Fusion) ─────────
                brain_assessment = self.brain.compute(
                    driver_data=driver_state,
                    collision_data=collision_state,
                    hazard_data=hazard_state,
                    monotonic_ms=now_mono_ms,
                )

                # ── 6. Phase 5: Intelligent Decision & Intervention ──────
                intervention = self.decision_engine.decide(
                    brain_assessment=brain_assessment,
                    monotonic_ms=now_mono_ms,
                )

                # ── 7. Phase 8: Hard Case Auto-Mining ────────────────────
                if brain_assessment.unified_state == "CRITICAL" or 0.25 <= brain_assessment.unified_risk_score <= 0.40:
                    self.miner.inspect_and_mine(
                        event_source="brain_fusion",
                        confidence=brain_assessment.unified_risk_score,
                        state=brain_assessment.unified_state,
                        metadata={"primary": brain_assessment.primary_risk_source, "reasons": brain_assessment.reason_codes},
                    )

                # ── 8. Phase 7: Edge Profiler ────────────────────────────
                latency_ms = (time.perf_counter() - loop_t0) * 1000.0
                telemetry = self.profiler.record_frame(latency_ms)

                # Attention heuristic
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

                # ── 9. WebSocket broadcast (All phases backward-compatible) ──
                msg = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "driver": {
                        "ear": driver_state.get("ear", 0.0),
                        "mar": driver_state.get("mar", 0.0),
                        "head_pose": driver_state.get(
                            "head_pose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
                        ),
                        "fatigue": fatigue_norm,
                        "attention": attention,
                        "eye_state": driver_state.get("eye_state", "OPEN"),
                        "blink_rate": stats.get("blink_rate", 0),
                        "face_detected": driver_state.get("face_detected", True),
                        "cnn_drowsy_probability": driver_state.get("cnn_drowsy_probability", driver_state.get("drowsy_probability", 0.0)),
                        "cnn_predicted_class": driver_state.get("cnn_predicted_class", "NON_DROWSY"),
                        "cnn_predicted_class_probability": driver_state.get("cnn_predicted_class_probability", 1.0),
                        "drowsy_probability": driver_state.get("cnn_drowsy_probability", driver_state.get("drowsy_probability", 0.0)),
                        "smoothed_fatigue": driver_state.get("smoothed_fatigue", 0.0),
                        "state": driver_state.get("state", "ALERT"),
                        "reason_codes": driver_state.get("reason_codes", ["BASELINE_NORMAL"]),
                        "fatigue_risk_score": driver_state.get("fatigue_risk_score", 0.0),
                        "signal_strength": driver_state.get("signal_strength", 1.0),
                        "tracking_state": driver_state.get("tracking_state", "TRACKED"),
                        "wakefulness_support": driver_state.get("wakefulness_support", 1.0),
                        "signal_disagreement": driver_state.get("signal_disagreement", False),
                        "active_signals": driver_state.get("active_signals", {}),
                        "recent_signals": driver_state.get("recent_signals", {}),
                        "driver_state_v2": driver_state.get("driver_state_v2", {}),
                        "latency_ms": driver_state.get("latency_ms", 3.1),
                        "is_mock": not is_driver_live,
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
                        "is_mock": False,
                    },
                    "collision": collision_state,
                    "hazard": hazard_state,
                    "brain": brain_assessment.__dict__,
                    "intervention": intervention.__dict__,
                    "risk_data": risk,
                    "metrics": {
                        "fps": telemetry.rolling_fps,
                        "latency_ms": telemetry.mean_latency_ms,
                        "p95_latency_ms": telemetry.p95_latency_ms,
                        "memory_mb": telemetry.memory_rss_mb,
                        "cpu_percent": telemetry.cpu_percent,
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

            except Exception as exc:
                print(f"[InferenceLoop] Error on frame {self._frame_index}: {exc}", flush=True)
                traceback.print_exc()

            await asyncio.sleep(1 / 30)


def _empty_road_state() -> dict:
    return {
        "potholes": [],
        "vehicles": 0,
        "hazard_score": 0.0,
        "detections": [],
        "nearest_distance_m": 100.0,
        "pothole_detected": False,
        "pothole_confidence": 0.0,
        "relative_proximity": "FAR",
        "frame_data": "",
    }
