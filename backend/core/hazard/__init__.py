"""
RoadGuard AI — Road Hazard Intelligence Package (Phase 3)
"""

from core.hazard.types import (
    HazardDetection,
    HazardSample,
    TrackedHazard,
    HazardAssessment,
    GlobalHazardResult,
)
from core.hazard.detector import PotholeDetector
from core.hazard.tracker import HazardTracker
from core.hazard.ego_path import HazardEgoPath
from core.hazard.severity import HazardSeverityEstimator
from core.hazard.risk_engine import HazardRiskEngine
from core.hazard.hazard_engine import RoadHazardEngine

__all__ = [
    "HazardDetection",
    "HazardSample",
    "TrackedHazard",
    "HazardAssessment",
    "GlobalHazardResult",
    "PotholeDetector",
    "HazardTracker",
    "HazardEgoPath",
    "HazardSeverityEstimator",
    "HazardRiskEngine",
    "RoadHazardEngine",
]
