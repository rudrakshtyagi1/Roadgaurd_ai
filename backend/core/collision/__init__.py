"""
RoadGuard AI — Collision Intelligence Engine Package
"""

from core.collision.collision_engine import CollisionIntelligenceEngine
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
    TrackSample,
    TrackedObject,
)

__all__ = [
    "CollisionIntelligenceEngine",
    "RoadObjectDetector",
    "PerspectiveEgoCorridor",
    "RelativeMotionEstimator",
    "TTCEstimator",
    "CollisionRiskEngine",
    "CollisionStateMachine",
    "MultiObjectTracker",
    "GlobalCollisionResult",
    "ObjectDetection",
    "TrackCollisionAssessment",
    "TrackSample",
    "TrackedObject",
]
