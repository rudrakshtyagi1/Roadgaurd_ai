"""
RoadGuard AI — Collision Intelligence Data Types
Shared dataclasses and schemas for Phase 2 Collision Intelligence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ObjectDetection:
    """Single frame object detection from visual detector."""
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: list[float]  # [x1, y1, x2, y2] normalized or pixels
    bbox_center: list[float]  # [cx, cy]
    bbox_width: float
    bbox_height: float
    bbox_area: float


@dataclass
class TrackSample:
    """Single historical temporal observation for a tracked object."""
    monotonic_ms: float
    center_x: float
    center_y: float
    bottom_center_x: float
    bottom_center_y: float
    width: float
    height: float
    area: float
    scale: float  # fused or selected scale in normalized units
    confidence: float


@dataclass
class TrackedObject:
    """Active multi-object track with persistent identity and temporal buffer."""
    track_id: int
    class_name: str
    class_id: int
    tracking_state: str  # "NEW", "TRACKED", "DEGRADED", "LOST"
    current_bbox: list[float]  # [x1, y1, x2, y2]
    first_seen_mono_ms: float
    last_seen_mono_ms: float
    frames_seen: int
    frames_missing: int
    history: list[TrackSample] = field(default_factory=list)

    @property
    def age_ms(self) -> float:
        return max(0.0, self.last_seen_mono_ms - self.first_seen_mono_ms)


@dataclass
class TrackCollisionAssessment:
    """Per-track collision threat and trajectory evaluation."""
    track_id: int
    class_name: str
    detector_confidence: float
    tracking_state: str  # "NEW", "TRACKED", "DEGRADED", "LOST"
    track_age_ms: float

    ego_path_relevance: float  # [0.0 - 1.0]
    path_relation: str         # "IN_PATH", "NEAR_PATH", "OUT_OF_PATH", "UNKNOWN"

    relative_motion: str       # "APPROACHING", "STABLE", "RECEDING", "UNKNOWN"
    motion_strength: float     # [0.0 - 1.0]
    bbox_growth_rate: float    # scale change per second

    ttc_seconds: Optional[float]
    ttc_reliable: bool
    ttc_signal_strength: float

    collision_risk_score: float # [0.0 - 1.0]
    collision_state: str        # "UNKNOWN", "SAFE", "CAUTION", "WARNING", "CRITICAL"
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class GlobalCollisionResult:
    """Global frame-level collision intelligence state."""
    global_collision_state: str  # "UNKNOWN", "SAFE", "CAUTION", "WARNING", "CRITICAL"
    global_collision_risk_score: float
    primary_threat_track_id: Optional[int]
    primary_threat: Optional[TrackCollisionAssessment]
    active_tracks_count: int
    reliable_ttc_tracks_count: int
    all_tracks: list[TrackCollisionAssessment]
    reason_codes: list[str]
    monotonic_ms: float
    frame_index: int
    latencies: dict[str, float] = field(default_factory=dict)
