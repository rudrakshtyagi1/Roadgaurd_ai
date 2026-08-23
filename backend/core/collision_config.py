"""
RoadGuard AI — Collision Engine Configuration
Strongly-typed configuration schema and parser for Collision Intelligence Engine.
Loads from YAML file with fallback to calibrated ADAS safety defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class DetectorConfig:
    confidence_threshold: float = 0.35
    target_classes: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    model_path: str = "yolov8n.pt"


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.30
    max_missing_frames: int = 10
    min_hits_to_track: int = 3
    max_track_history_ms: float = 3000.0
    history_prune_interval_ms: float = 1000.0


@dataclass
class EgoCorridorConfig:
    horizon_y: float = 0.45
    bottom_y: float = 1.00
    top_half_width: float = 0.06
    bottom_half_width: float = 0.26
    center_x: float = 0.50
    in_path_threshold: float = 0.70
    near_path_threshold: float = 0.30


@dataclass
class MotionConfig:
    window_ms: float = 800.0
    min_samples: int = 5
    approaching_slope_threshold: float = 0.04
    receding_slope_threshold: float = -0.04
    smoothing_alpha: float = 0.30


@dataclass
class TTCConfig:
    scale_signal_mode: str = "fused"  # "fused", "sqrt_area", "bbox_height", "bbox_area"
    min_track_age_ms: float = 500.0
    min_samples: int = 8
    rolling_window_ms: float = 600.0
    min_r_squared: float = 0.50
    min_valid_ttc_s: float = 0.2
    max_valid_ttc_s: 10.0 = 10.0
    caution_ttc_s: float = 4.5
    warning_ttc_s: float = 2.8
    critical_ttc_s: float = 1.6


@dataclass
class CollisionWeightsConfig:
    ego_path: float = 0.35
    ttc: float = 0.35
    closing_motion: float = 0.15
    proximity_scale: float = 0.10
    object_vulnerability: float = 0.05


@dataclass
class CollisionStateMachineConfig:
    caution_score_threshold: float = 0.45
    warning_score_threshold: float = 0.65
    critical_score_threshold: float = 0.85
    cooldown_critical_to_warning_ms: float = 1500.0
    cooldown_warning_to_caution_ms: float = 2000.0
    cooldown_caution_to_safe_ms: float = 2500.0


@dataclass
class CollisionEngineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ego_corridor: EgoCorridorConfig = field(default_factory=EgoCorridorConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    ttc: TTCConfig = field(default_factory=TTCConfig)
    weights: CollisionWeightsConfig = field(default_factory=CollisionWeightsConfig)
    state_machine: CollisionStateMachineConfig = field(default_factory=CollisionStateMachineConfig)

    @classmethod
    def from_yaml(cls, path: Optional[Path | str] = None) -> "CollisionEngineConfig":
        """Load from YAML file or return default instance."""
        if path is None:
            default_path = Path(__file__).parent.parent / "configs" / "collision_engine.yaml"
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
            return cls._parse_simple_yaml(path)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CollisionEngineConfig":
        """Construct from raw nested dictionary."""
        cfg = cls()
        if "detector" in d:
            cfg.detector = DetectorConfig(**{k: v for k, v in d["detector"].items() if hasattr(cfg.detector, k)})
        if "tracker" in d:
            cfg.tracker = TrackerConfig(**{k: (int(v) if k in ["max_missing_frames", "min_hits_to_track"] else float(v)) for k, v in d["tracker"].items() if hasattr(cfg.tracker, k)})
        if "ego_corridor" in d:
            cfg.ego_corridor = EgoCorridorConfig(**{k: float(v) for k, v in d["ego_corridor"].items() if hasattr(cfg.ego_corridor, k)})
        if "motion" in d:
            cfg.motion = MotionConfig(**{k: (int(v) if k == "min_samples" else float(v)) for k, v in d["motion"].items() if hasattr(cfg.motion, k)})
        if "ttc" in d:
            cfg.ttc = TTCConfig(**{k: (int(v) if k == "min_samples" else (str(v) if k == "scale_signal_mode" else float(v))) for k, v in d["ttc"].items() if hasattr(cfg.ttc, k)})
        if "weights" in d:
            cfg.weights = CollisionWeightsConfig(**{k: float(v) for k, v in d["weights"].items() if hasattr(cfg.weights, k)})
        if "state_machine" in d:
            cfg.state_machine = CollisionStateMachineConfig(**{k: float(v) for k, v in d["state_machine"].items() if hasattr(cfg.state_machine, k)})
        return cfg

    @classmethod
    def _parse_simple_yaml(cls, path: Path) -> "CollisionEngineConfig":
        """Fallback line parser."""
        text = path.read_text(encoding="utf-8")
        current_section: Optional[str] = None
        data: dict[str, dict[str, Any]] = {}

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = [p.strip() for p in line.split(":", 1)]
                val = val.split("#")[0].strip()
                if not val:
                    current_section = key
                    data[current_section] = {}
                elif current_section:
                    try:
                        if val.startswith("[") and val.endswith("]"):
                            data[current_section][key] = [int(x.strip()) for x in val[1:-1].split(",") if x.strip()]
                        elif "." in val:
                            data[current_section][key] = float(val)
                        elif val.isdigit():
                            data[current_section][key] = int(val)
                        else:
                            data[current_section][key] = val.strip('"\'')
                    except ValueError:
                        data[current_section][key] = val.strip('"\'')

        return cls.from_dict(data)
