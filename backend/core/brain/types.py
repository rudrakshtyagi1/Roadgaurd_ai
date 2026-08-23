"""
RoadGuard AI — Unified RoadGuard Brain Data Types (Phase 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UnifiedRiskAssessment:
    """Unified multi-hazard risk assessment produced by RoadGuard Brain."""
    unified_state: str # "SAFE", "ELEVATED", "HIGH", "CRITICAL"
    unified_risk_score: float # [0.0 - 1.0]

    primary_risk_source: str # "COLLISION", "ROAD_HAZARD", "DRIVER_FATIGUE", "NONE"
    secondary_risk_sources: list[str] = field(default_factory=list)

    reason_codes: list[str] = field(default_factory=list)
    driver_state: str = "NORMAL"
    collision_state: str = "SAFE"
    hazard_state: str = "SAFE"

    ttc_seconds: Optional[float] = None
    hazard_distance_m: Optional[float] = None
    drowsy_probability: float = 0.0

    monotonic_ms: float = 0.0
    frame_index: int = 0
    contributing_signals: dict[str, float] = field(default_factory=dict)
