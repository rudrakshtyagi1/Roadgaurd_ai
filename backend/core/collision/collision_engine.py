"""
RoadGuard AI — Collision Intelligence Engine
Orchestrates object detection, multi-object tracking, perspective ego-path relevance,
relative motion, image-space TTC, risk fusion, and hysteretic state management.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.collision_config import CollisionEngineConfig
from core.collision.detector import RoadObjectDetector
from core.collision.ego_path import PerspectiveEgoCorridor
from core.collision.motion import RelativeMotionEstimator
from core.collision.risk_engine import CollisionRiskEngine
from core.collision.state_machine import CollisionStateMachine
from core.collision.tracker import MultiObjectTracker
from core.collision.ttc import TTCEstimator
from core.collision.types import (
    GlobalCollisionResult,
    ObjectDetection,
    TrackCollisionAssessment,
    TrackedObject,
)
from core.perception_event import PerceptionEvent


class CollisionIntelligenceEngine:
    """
    Central real-time Collision Intelligence Engine for RoadGuard AI Phase 2.
    """

    def __init__(self, config: Optional[CollisionEngineConfig] = None) -> None:
        self.config = config or CollisionEngineConfig.from_yaml()
        self.detector = RoadObjectDetector(config=self.config.detector)
        self.tracker = MultiObjectTracker(config=self.config.tracker)
        self.ego_corridor = PerspectiveEgoCorridor(config=self.config.ego_corridor)
        self.motion_estimator = RelativeMotionEstimator(config=self.config.motion)
        self.ttc_estimator = TTCEstimator(config=self.config.ttc)
        self.risk_engine = CollisionRiskEngine(config=self.config)
        self.state_machine = CollisionStateMachine(config=self.config.state_machine)

        self._frame_count = 0

    def reset(self) -> None:
        """Reset all tracking and state buffers."""
        self.tracker.reset()
        self.state_machine.reset()
        self._frame_count = 0

    def process(
        self,
        frame: Optional[np.ndarray] = None,
        detections: Optional[list[ObjectDetection]] = None,
        monotonic_ms: Optional[float] = None,
        frame_index: int = 0,
    ) -> dict[str, Any]:
        """
        Process a single video frame or explicit list of detections.
        Returns serialized dictionary representation of GlobalCollisionResult.
        """
        t0_total = time.perf_counter()
        t_now = monotonic_ms if monotonic_ms is not None else time.monotonic() * 1000.0
        self._frame_count += 1
        latencies: dict[str, float] = {}

        # 1. Detection
        t0 = time.perf_counter()
        if detections is None:
            if frame is not None and frame.size > 0:
                raw_detections = self.detector.detect(frame)
            else:
                raw_detections = []
        else:
            raw_detections = detections
        latencies["detection_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 2. Tracking
        t0 = time.perf_counter()
        active_tracks: list[TrackedObject] = self.tracker.update(raw_detections, t_now)
        latencies["tracking_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 3. Per-Track Collision Intelligence
        t0 = time.perf_counter()
        assessments: list[TrackCollisionAssessment] = []
        reliable_ttc_count = 0

        for track in active_tracks:
            # A. Ego-path relevance
            relevance, relation = self.ego_corridor.evaluate_bbox(track.current_bbox)

            # B. Relative motion
            motion, m_strength, growth_rate, m_rel = self.motion_estimator.estimate_motion(
                track.history, t_now
            )

            # C. TTC estimation
            ttc_s, ttc_rel, ttc_sig = self.ttc_estimator.estimate_ttc(
                track.history, motion, growth_rate, track.age_ms
            )
            if ttc_rel:
                reliable_ttc_count += 1

            # D. Risk scoring & assessment
            assessment = self.risk_engine.assess_track(
                track=track,
                ego_path_relevance=relevance,
                path_relation=relation,
                relative_motion=motion,
                motion_strength=m_strength,
                growth_rate=growth_rate,
                ttc_seconds=ttc_s,
                ttc_reliable=ttc_rel,
                ttc_signal_strength=ttc_sig,
            )
            assessments.append(assessment)

        latencies["per_track_analysis_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 4. Primary Threat Selection & State Machine
        t0 = time.perf_counter()
        primary_threat = self.risk_engine.select_primary_threat(assessments)
        global_state, global_risk_score, global_reasons = self.state_machine.update(
            primary_threat, t_now
        )
        latencies["risk_and_state_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies["total_pipeline_ms"] = round((time.perf_counter() - t0_total) * 1000.0, 2)

        # 5. Format Output
        serialized_tracks = [
            {
                "track_id": a.track_id,
                "class_name": a.class_name,
                "detector_confidence": a.detector_confidence,
                "tracking_state": a.tracking_state,
                "track_age_ms": round(a.track_age_ms, 1),
                "ego_path_relevance": a.ego_path_relevance,
                "path_relation": a.path_relation,
                "relative_motion": a.relative_motion,
                "motion_strength": a.motion_strength,
                "bbox_growth_rate": a.bbox_growth_rate,
                "ttc_seconds": a.ttc_seconds,
                "ttc_reliable": a.ttc_reliable,
                "ttc_signal_strength": a.ttc_signal_strength,
                "collision_risk_score": a.collision_risk_score,
                "collision_state": a.collision_state,
                "reason_codes": a.reason_codes,
            }
            for a in assessments
        ]

        primary_threat_dict = None
        if primary_threat is not None:
            primary_threat_dict = next(
                (t for t in serialized_tracks if t["track_id"] == primary_threat.track_id),
                None,
            )

        return {
            "global_collision_state": global_state,
            "global_collision_risk_score": global_risk_score,
            "primary_threat_track_id": primary_threat.track_id if primary_threat else None,
            "primary_threat": primary_threat_dict,
            "active_tracks_count": len(active_tracks),
            "reliable_ttc_tracks_count": reliable_ttc_count,
            "all_tracks": serialized_tracks,
            "reason_codes": global_reasons,
            "monotonic_ms": round(t_now, 3),
            "frame_index": frame_index or self._frame_count,
            "latencies": latencies,
        }

    def to_perception_event(self, result: dict[str, Any], frame_index: int = 0) -> PerceptionEvent:
        """Convert collision result dict into a standardized PerceptionEvent."""
        return PerceptionEvent.ok(
            source="collision_intelligence",
            event_type="collision_state",
            frame_index=frame_index or result.get("frame_index", 0),
            latency_ms=result.get("latencies", {}).get("total_pipeline_ms", 0.0),
            confidence=result.get("global_collision_risk_score", 0.0),
            data=result,
            monotonic_ms=result.get("monotonic_ms"),
        )
