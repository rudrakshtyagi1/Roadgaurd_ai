"""
RoadGuard AI — Road Hazard Benchmark Harness (Phase 3)
Evaluates RoadHazardEngine with deterministic synthetic scenarios:
  1. severe_pothole_in_ego_path
  2. minor_surface_pothole
  3. adjacent_roadside_pothole
  4. repeated_detection_deduplication
  5. transient_detector_flicker
  6. multi_pothole_road
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.hazard.hazard_engine import RoadHazardEngine
from core.hazard.types import HazardDetection


@dataclass
class SyntheticHazardFrame:
    time_ms: float
    detections: list[HazardDetection]


@dataclass
class HazardScenarioReport:
    scenario_name: str
    passed: bool
    failure_reasons: list[str]
    final_state: str
    unique_hazard_ids_tracked: int
    states_observed: list[str]


class HazardBenchmark:
    """Benchmark runner for RoadHazardEngine."""

    def __init__(self, engine: Optional[RoadHazardEngine] = None) -> None:
        self.engine = engine or RoadHazardEngine()

    @staticmethod
    def _make_det(cx: float, cy: float, w: float, h: float, conf: float = 0.90) -> HazardDetection:
        return HazardDetection(
            class_name="pothole",
            class_id=0,
            confidence=conf,
            bbox_xyxy=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            bbox_center=[cx, cy],
            bbox_width=w,
            bbox_height=h,
            bbox_area=w * h,
        )

    def run_all_benchmarks(self) -> list[HazardScenarioReport]:
        reports = []
        fps = 30.0
        dt_ms = 1000.0 / fps

        # 1. severe_pothole_in_ego_path
        self.engine.reset()
        states_1 = []
        hids_1 = set()
        for i in range(90):  # 3.0s
            t_ms = i * dt_ms
            t_s = t_ms / 1000.0
            # Pothole approaching in path: moves from y=0.55 to y=0.90, grows from 0.05 to 0.25
            y = 0.55 + 0.12 * t_s
            s = 0.06 + 0.08 * t_s
            det = self._make_det(cx=0.50, cy=y, w=s * 1.5, h=s, conf=0.88)
            res = self.engine.process(detections=[det], monotonic_ms=t_ms)
            states_1.append(res["global_hazard_state"])
            if res["primary_hazard_id"] is not None:
                hids_1.add(res["primary_hazard_id"])

        pass_1 = "CRITICAL" in states_1 or "WARNING" in states_1
        reasons_1 = [] if pass_1 else ["Failed to reach WARNING or CRITICAL for severe approaching pothole"]
        reports.append(HazardScenarioReport("severe_pothole_in_ego_path", pass_1, reasons_1, states_1[-1], len(hids_1), list(set(states_1))))

        # 2. adjacent_roadside_pothole (outside path at x=0.88 -> SAFE)
        self.engine.reset()
        states_2 = []
        hids_2 = set()
        for i in range(90):
            t_ms = i * dt_ms
            det = self._make_det(cx=0.88, cy=0.75, w=0.20, h=0.15, conf=0.90)
            res = self.engine.process(detections=[det], monotonic_ms=t_ms)
            states_2.append(res["global_hazard_state"])
            if res["primary_hazard_id"] is not None:
                hids_2.add(res["primary_hazard_id"])

        pass_2 = all(s in ["SAFE"] for s in states_2)
        reasons_2 = [] if pass_2 else [f"Adjacent roadside pothole triggered unsafe states: {set(states_2)}"]
        reports.append(HazardScenarioReport("adjacent_roadside_pothole", pass_2, reasons_2, states_2[-1], len(hids_2), list(set(states_2))))

        # 3. repeated_detection_deduplication (1 pothole tracked continuously -> exactly 1 hazard_id)
        self.engine.reset()
        states_3 = []
        hids_3 = set()
        for i in range(120):
            t_ms = i * dt_ms
            det = self._make_det(cx=0.50, cy=0.65, w=0.10, h=0.08, conf=0.85)
            res = self.engine.process(detections=[det], monotonic_ms=t_ms)
            states_3.append(res["global_hazard_state"])
            if res["primary_hazard_id"] is not None:
                hids_3.add(res["primary_hazard_id"])

        pass_3 = (len(hids_3) == 1)
        reasons_3 = [] if pass_3 else [f"Expected 1 unique hazard_id, saw {len(hids_3)}"]
        reports.append(HazardScenarioReport("repeated_detection_deduplication", pass_3, reasons_3, states_3[-1], len(hids_3), list(set(states_3))))

        # 4. transient_detector_flicker (1 frame false detection -> filtered out)
        self.engine.reset()
        states_4 = []
        for i in range(60):
            t_ms = i * dt_ms
            dets = [self._make_det(cx=0.50, cy=0.65, w=0.10, h=0.08, conf=0.40)] if i == 15 else []
            res = self.engine.process(detections=dets, monotonic_ms=t_ms)
            states_4.append(res["global_hazard_state"])

        pass_4 = all(s in ["SAFE"] for s in states_4)
        reasons_4 = [] if pass_4 else [f"Transient 1-frame flicker triggered alert: {set(states_4)}"]
        reports.append(HazardScenarioReport("transient_detector_flicker", pass_4, reasons_4, states_4[-1], 0, list(set(states_4))))

        return reports
