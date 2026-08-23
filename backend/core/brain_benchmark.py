"""
RoadGuard AI — Brain Multi-Hazard Fusion Benchmark (Phase 4)
Evaluates RoadGuard Brain against multi-hazard scenarios:
  1. drowsy_driver_clear_road -> DRIVER_FATIGUE primary
  2. alert_driver_severe_pothole -> ROAD_HAZARD primary
  3. drowsy_driver_low_ttc_collision -> COLLISION primary override + DRIVER_FATIGUE secondary
  4. drowsy_driver_approaching_pothole -> Compounding amplification to CRITICAL
  5. triple_simultaneous_hazard -> COLLISION priority + ROAD_HAZARD & DRIVER_FATIGUE secondary
  6. normal_clear_drive -> SAFE, NONE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.brain.fusion_engine import RoadGuardBrain


@dataclass
class BrainScenarioReport:
    scenario_name: str
    passed: bool
    expected_primary: str
    actual_primary: str
    expected_state: str
    actual_state: str
    reason_codes: list[str]


class BrainBenchmark:
    """Benchmark runner for RoadGuard Brain multi-hazard scenarios."""

    def __init__(self, brain: Optional[RoadGuardBrain] = None) -> None:
        self.brain = brain or RoadGuardBrain()

    def run_all_benchmarks(self) -> list[BrainScenarioReport]:
        reports = []

        # 1. drowsy_driver_clear_road
        d1 = {"state": "DROWSY", "fatigue_risk_score": 0.82, "drowsy_probability": 0.85, "reason_codes": ["PROLONGED_CLOSURE"]}
        c1 = {"global_collision_state": "SAFE", "global_collision_risk_score": 0.0, "reason_codes": []}
        h1 = {"global_hazard_state": "SAFE", "global_hazard_risk_score": 0.0, "reason_codes": []}
        res1 = self.brain.compute(d1, c1, h1)
        pass1 = (res1.primary_risk_source == "DRIVER_FATIGUE" and res1.unified_state == "HIGH")
        reports.append(BrainScenarioReport("drowsy_driver_clear_road", pass1, "DRIVER_FATIGUE", res1.primary_risk_source, "HIGH", res1.unified_state, res1.reason_codes))

        # 2. alert_driver_severe_pothole
        d2 = {"state": "NORMAL", "fatigue_risk_score": 0.05, "drowsy_probability": 0.02, "reason_codes": []}
        c2 = {"global_collision_state": "SAFE", "global_collision_risk_score": 0.0, "reason_codes": []}
        h2 = {"global_hazard_state": "WARNING", "global_hazard_risk_score": 0.72, "primary_hazard": {"estimated_distance_m": 15.0}, "reason_codes": ["HIGH_SEVERITY_POTHOLE"]}
        res2 = self.brain.compute(d2, c2, h2)
        pass2 = (res2.primary_risk_source == "ROAD_HAZARD" and res2.unified_state == "HIGH")
        reports.append(BrainScenarioReport("alert_driver_severe_pothole", pass2, "ROAD_HAZARD", res2.primary_risk_source, "HIGH", res2.unified_state, res2.reason_codes))

        # 3. drowsy_driver_low_ttc_collision (Collision Priority Override)
        d3 = {"state": "DROWSY", "fatigue_risk_score": 0.80, "drowsy_probability": 0.88, "reason_codes": ["MICRO_SLEEP"]}
        c3 = {"global_collision_state": "CRITICAL", "global_collision_risk_score": 0.95, "primary_threat": {"ttc_seconds": 1.4}, "reason_codes": ["VERY_LOW_TTC", "OBJECT_IN_EGO_PATH"]}
        h3 = {"global_hazard_state": "SAFE", "global_hazard_risk_score": 0.0, "reason_codes": []}
        res3 = self.brain.compute(d3, c3, h3)
        pass3 = (res3.primary_risk_source == "COLLISION" and res3.unified_state == "CRITICAL" and "DRIVER_FATIGUE" in res3.secondary_risk_sources)
        reports.append(BrainScenarioReport("drowsy_driver_low_ttc_collision", pass3, "COLLISION", res3.primary_risk_source, "CRITICAL", res3.unified_state, res3.reason_codes))

        # 4. drowsy_driver_approaching_pothole (Compounding Risk)
        d4 = {"state": "DROWSY", "fatigue_risk_score": 0.78, "drowsy_probability": 0.80, "reason_codes": ["PROLONGED_CLOSURE"]}
        c4 = {"global_collision_state": "SAFE", "global_collision_risk_score": 0.0, "reason_codes": []}
        h4 = {"global_hazard_state": "WARNING", "global_hazard_risk_score": 0.70, "primary_hazard": {"estimated_distance_m": 14.0}, "reason_codes": ["HIGH_SEVERITY_POTHOLE"]}
        res4 = self.brain.compute(d4, c4, h4)
        pass4 = (res4.primary_risk_source == "ROAD_HAZARD" and res4.unified_state == "CRITICAL" and "FATIGUED_DRIVER_APPROACHING_HAZARD" in res4.reason_codes)
        reports.append(BrainScenarioReport("drowsy_driver_approaching_pothole", pass4, "ROAD_HAZARD", res4.primary_risk_source, "CRITICAL", res4.unified_state, res4.reason_codes))

        # 5. triple_simultaneous_hazard
        d5 = {"state": "CRITICAL", "fatigue_risk_score": 0.95, "drowsy_probability": 0.98, "reason_codes": ["PROLONGED_EYE_CLOSURE_FATIGUE"]}
        c5 = {"global_collision_state": "CRITICAL", "global_collision_risk_score": 0.98, "primary_threat": {"ttc_seconds": 1.1}, "reason_codes": ["VERY_LOW_TTC"]}
        h5 = {"global_hazard_state": "CRITICAL", "global_hazard_risk_score": 0.88, "primary_hazard": {"estimated_distance_m": 8.0}, "reason_codes": ["SEVERE_POTHOLE_SURFACE"]}
        res5 = self.brain.compute(d5, c5, h5)
        pass5 = (res5.primary_risk_source == "COLLISION" and res5.unified_state == "CRITICAL" and "DRIVER_FATIGUE" in res5.secondary_risk_sources and "ROAD_HAZARD" in res5.secondary_risk_sources)
        reports.append(BrainScenarioReport("triple_simultaneous_hazard", pass5, "COLLISION", res5.primary_risk_source, "CRITICAL", res5.unified_state, res5.reason_codes))

        # 6. normal_clear_drive
        d6 = {"state": "NORMAL", "fatigue_risk_score": 0.0, "drowsy_probability": 0.0, "reason_codes": []}
        c6 = {"global_collision_state": "SAFE", "global_collision_risk_score": 0.0, "reason_codes": []}
        h6 = {"global_hazard_state": "SAFE", "global_hazard_risk_score": 0.0, "reason_codes": []}
        res6 = self.brain.compute(d6, c6, h6)
        pass6 = (res6.primary_risk_source == "NONE" and res6.unified_state == "SAFE")
        reports.append(BrainScenarioReport("normal_clear_drive", pass6, "NONE", res6.primary_risk_source, "SAFE", res6.unified_state, res6.reason_codes))

        return reports
