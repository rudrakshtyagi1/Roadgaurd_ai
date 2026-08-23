"""
RoadGuard AI — Road Hazard Intelligence Configuration
Dataclasses and parser for Phase 3 Hazard Intelligence Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class HazardDetectorConfig:
    confidence_threshold: float = 0.25
    model_path: str = "/Users/rudrakshtyagi/Desktop/roadgaurdai/potholes/runs/pothole_yolov8n_v1/weights/best.pt"
    target_classes: list[int] = field(default_factory=lambda: [0])


@dataclass
class HazardTrackerConfig:
    iou_threshold: float = 0.20
    spatial_distance_threshold: float = 0.15
    max_missing_frames: int = 8
    min_hits_to_confirm: int = 2
    max_track_history_ms: float = 3000.0


@dataclass
class HazardEgoCorridorConfig:
    horizon_y: float = 0.45
    bottom_y: float = 1.00
    top_half_width: float = 0.08
    bottom_half_width: float = 0.28
    center_x: float = 0.50
    in_path_threshold: float = 0.70
    near_path_threshold: float = 0.30


@dataclass
class HazardSeverityConfig:
    area_severe_threshold: float = 0.040
    area_high_threshold: float = 0.020
    area_medium_threshold: float = 0.008
    expansion_rate_severe: float = 0.05
    min_confidence_high_severity: float = 0.45


@dataclass
class HazardWeightsConfig:
    ego_path: float = 0.40
    severity: float = 0.35
    approach_motion: float = 0.15
    proximity_scale: float = 0.10


@dataclass
class HazardStateMachineConfig:
    caution_score_threshold: float = 0.35
    warning_score_threshold: float = 0.60
    critical_score_threshold: float = 0.80
    cooldown_critical_to_warning_ms: float = 1200.0
    cooldown_warning_to_caution_ms: float = 1800.0
    cooldown_caution_to_safe_ms: float = 2200.0


@dataclass
class HazardEngineConfig:
    detector: HazardDetectorConfig = field(default_factory=HazardDetectorConfig)
    tracker: HazardTrackerConfig = field(default_factory=HazardTrackerConfig)
    ego_corridor: HazardEgoCorridorConfig = field(default_factory=HazardEgoCorridorConfig)
    severity: HazardSeverityConfig = field(default_factory=HazardSeverityConfig)
    weights: HazardWeightsConfig = field(default_factory=HazardWeightsConfig)
    state_machine: HazardStateMachineConfig = field(default_factory=HazardStateMachineConfig)

    @classmethod
    def from_yaml(cls, path: Optional[Path | str] = None) -> "HazardEngineConfig":
        if path is None:
            default_path = Path(__file__).parent.parent / "configs" / "hazard_engine.yaml"
            if default_path.exists():
                path = default_path
            else:
                return cls()

        path = Path(path)
        if not path.exists():
            return cls()

        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        except Exception:
            return cls()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HazardEngineConfig":
        cfg = cls()
        if "detector" in d:
            cfg.detector = HazardDetectorConfig(**{k: v for k, v in d["detector"].items() if hasattr(cfg.detector, k)})
        if "tracker" in d:
            cfg.tracker = HazardTrackerConfig(**{k: (int(v) if k in ["max_missing_frames", "min_hits_to_confirm"] else float(v)) for k, v in d["tracker"].items() if hasattr(cfg.tracker, k)})
        if "ego_corridor" in d:
            cfg.ego_corridor = HazardEgoCorridorConfig(**{k: float(v) for k, v in d["ego_corridor"].items() if hasattr(cfg.ego_corridor, k)})
        if "severity" in d:
            cfg.severity = HazardSeverityConfig(**{k: float(v) for k, v in d["severity"].items() if hasattr(cfg.severity, k)})
        if "weights" in d:
            cfg.weights = HazardWeightsConfig(**{k: float(v) for k, v in d["weights"].items() if hasattr(cfg.weights, k)})
        if "state_machine" in d:
            cfg.state_machine = HazardStateMachineConfig(**{k: float(v) for k, v in d["state_machine"].items() if hasattr(cfg.state_machine, k)})
        return cfg
