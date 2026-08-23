"""
RoadGuard AI — Hazard Intelligence Data Types (Phase 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class HazardDetection:
    """Raw hazard detection from YOLO pothole detector."""
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: list[float] # [x1, y1, x2, y2]
    bbox_center: list[float] # [cx, cy]
    bbox_width: float
    bbox_height: float
    bbox_area: float


@dataclass
class HazardSample:
    """Temporal historical sample for tracked road hazard."""
    monotonic_ms: float
    center_x: float
    center_y: float
    bottom_center_x: float
    bottom_center_y: float
    width: float
    height: float
    area: float
    confidence: float


@dataclass
class TrackedHazard:
    """Active road hazard track across frames."""
    hazard_id: int
    class_name: str
    tracking_state: str # "NEW", "TRACKED", "DEGRADED", "LOST"
    current_bbox: list[float]
    first_seen_mono_ms: float
    last_seen_mono_ms: float
    frames_seen: int
    frames_missing: int
    history: list[HazardSample] = field(default_factory=list)

    @property
    def age_ms(self) -> float:
        return max(0.0, self.last_seen_mono_ms - self.first_seen_mono_ms)


@dataclass
class HazardAssessment:
    """Single hazard threat evaluation."""
    hazard_id: int
    class_name: str
    confidence: float
    tracking_state: str
    track_age_ms: float

    ego_path_relevance: float
    path_relation: str # "IN_PATH", "NEAR_PATH", "OUT_OF_PATH"

    severity_level: str # "LOW", "MEDIUM", "HIGH", "SEVERE"
    severity_score: float # [0.0 - 1.0]

    growth_rate: float
    estimated_distance_m: float

    hazard_risk_score: float # [0.0 - 1.0]
    hazard_state: str # "SAFE", "CAUTION", "WARNING", "CRITICAL"
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class GlobalHazardResult:
    """Frame-level aggregated road hazard intelligence result."""
    global_hazard_state: str # "SAFE", "CAUTION", "WARNING", "CRITICAL"
    global_hazard_risk_score: float
    primary_hazard_id: Optional[int]
    primary_hazard: Optional[HazardAssessment]
    active_hazards_count: int
    all_hazards: list[HazardAssessment]
    reason_codes: list[str]
    monotonic_ms: float
    frame_index: int
    latencies: dict[str, float] = field(default_factory=dict)
