"""
RoadGuard AI — Central Road Hazard Intelligence Engine (Phase 3)
Orchestrates detector, tracker, ego-corridor, severity estimator, and risk engine.
Conforms to the RoadHazardDetector protocol.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.hazard_config import HazardEngineConfig
from core.hazard.detector import PotholeDetector
from core.hazard.ego_path import HazardEgoPath
from core.hazard.risk_engine import HazardRiskEngine
from core.hazard.severity import HazardSeverityEstimator
from core.hazard.tracker import HazardTracker
from core.hazard.types import GlobalHazardResult, HazardAssessment, HazardDetection
from core.perception_event import PerceptionEvent


class RoadHazardEngine:
    """Road hazard perception, tracking, and severity engine."""

    def __init__(self, config: Optional[HazardEngineConfig] = None) -> None:
        self.config = config or HazardEngineConfig.from_yaml()
        self.detector = PotholeDetector(self.config.detector)
        self.tracker = HazardTracker(self.config.tracker)
        self.ego_path = HazardEgoPath(self.config.ego_corridor)
        self.severity_estimator = HazardSeverityEstimator(self.config.severity)
        self.risk_engine = HazardRiskEngine(self.config)
        self._frame_index = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.risk_engine.reset()
        self._frame_index = 0

    def process(
        self,
        frame: Optional[np.ndarray] = None,
        detections: Optional[list[HazardDetection]] = None,
        monotonic_ms: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Process camera frame or direct synthetic detections.
        """
        t0 = time.perf_counter()
        mono_ms = monotonic_ms if monotonic_ms is not None else (time.monotonic() * 1000.0)
        self._frame_index += 1

        # 1. Detection
        det_t0 = time.perf_counter()
        if detections is None:
            raw_detections = self.detector.detect(frame) if frame is not None else []
        else:
            raw_detections = detections
        det_lat_ms = (time.perf_counter() - det_t0) * 1000.0

        # 2. Tracking
        trk_t0 = time.perf_counter()
        active_tracks = self.tracker.update(raw_detections, monotonic_ms=mono_ms)
        trk_lat_ms = (time.perf_counter() - trk_t0) * 1000.0

        # 3. Assessment
        assess_t0 = time.perf_counter()
        assessments: list[HazardAssessment] = []
        for trk in active_tracks:
            cx, cy = trk.history[-1].center_x, trk.history[-1].center_y
            rel, rel_str = self.ego_path.evaluate(cx, cy)
            sev_lvl, sev_score, g_rate = self.severity_estimator.estimate(trk)
            assessment = self.risk_engine.assess_hazard(
                track=trk,
                ego_path_relevance=rel,
                path_relation=rel_str,
                severity_level=sev_lvl,
                severity_score=sev_score,
                growth_rate=g_rate,
            )
            assessments.append(assessment)

        g_state, g_score, primary_haz, r_codes = self.risk_engine.update_global_state(
            assessments=assessments,
            monotonic_ms=mono_ms,
        )
        assess_lat_ms = (time.perf_counter() - assess_t0) * 1000.0
        total_lat_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "global_hazard_state": g_state,
            "global_hazard_risk_score": g_score,
            "primary_hazard_id": primary_haz.hazard_id if primary_haz else None,
            "primary_hazard": primary_haz.__dict__ if primary_haz else None,
            "active_hazards_count": len(assessments),
            "all_hazards": [h.__dict__ for h in assessments],
            "reason_codes": r_codes,
            "monotonic_ms": mono_ms,
            "frame_index": self._frame_index,
            "latency_ms": round(total_lat_ms, 2),
            "latencies": {
                "detection_ms": round(det_lat_ms, 2),
                "tracking_ms": round(trk_lat_ms, 2),
                "assessment_ms": round(assess_lat_ms, 2),
                "total_ms": round(total_lat_ms, 2),
            },
        }

    def to_perception_event(self, result: dict[str, Any], frame_index: Optional[int] = None) -> PerceptionEvent:
        f_idx = frame_index if frame_index is not None else result.get("frame_index", self._frame_index)
        return PerceptionEvent.ok(
            source="road_hazard_engine",
            event_type="hazard_state",
            monotonic_ms=result.get("monotonic_ms", time.monotonic() * 1000.0),
            frame_index=f_idx,
            latency_ms=result.get("latency_ms", 1.0),
            confidence=result.get("global_hazard_risk_score", 0.0),
            data=result,
        )
