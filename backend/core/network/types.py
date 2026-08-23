"""
RoadGuard AI — Road Intelligence Network Data Types (Phase 6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeoLocation:
    latitude: float
    longitude: float
    accuracy_m: float = 5.0


@dataclass
class NetworkHazardReport:
    """Individual hazard telemetry event reported by an ego vehicle."""
    report_id: str
    hazard_type: str # "pothole", "debris", "speed_breaker"
    severity: str # "LOW", "MEDIUM", "HIGH", "SEVERE"
    confidence: float
    location: GeoLocation
    timestamp_utc: str
    vehicle_id: str


@dataclass
class ClusteredRoadHazard:
    """Aggregated & verified road hazard stored in RoadGuard network."""
    cluster_id: str
    hazard_type: str
    severity: str
    aggregated_confidence: float
    latitude: float
    longitude: float
    report_count: int
    first_reported_utc: str
    last_reported_utc: str
    is_active: bool = True


@dataclass
class RouteOption:
    """Route candidate evaluated with road safety metrics."""
    route_id: str
    name: str
    distance_km: float
    duration_minutes: float
    road_safety_score: float # [0 - 100] (higher = safer)
    hazard_count: int
    severe_hazard_count: int
    recommendation_reason: str
