"""
RoadGuard AI — Driver State Configuration
Strongly-typed configuration schema and parser for Driver State Engine V2.0.
Loads from YAML file with fallback to calibrated ADAS safety defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class WindowConfig:
    perclos_window_ms: float = 30000.0       # 30s rolling window for PERCLOS
    blink_history_window_ms: float = 60000.0 # 60s window for blink rate
    quality_window_ms: float = 10000.0       # 10s window for tracking quality
    min_valid_history_ms: float = 5000.0     # Min 5s face data before PERCLOS is reliable
    max_sample_dt_ms: float = 500.0          # Max dt clamp between consecutive samples


@dataclass
class EvidenceConfig:
    long_blink_history_window_ms: float = 30000.0
    long_blink_strong_window_ms: float = 6000.0
    yawn_history_window_ms: float = 45000.0
    yawn_strong_window_ms: float = 8000.0
    head_nod_history_window_ms: float = 10000.0
    moderate_cnn_threshold: float = 0.50
    strong_cnn_threshold: float = 0.80
    cnn_confirmation_window_ms: float = 1500.0
    wakefulness_ear_min: float = 0.28
    wakefulness_perclos_max: float = 0.08


@dataclass
class EyeConfig:
    ear_close_threshold: float = 0.20        # EAR below this starts eye closure
    ear_reopen_threshold: float = 0.25       # EAR above this confirms eye reopened (hysteresis)
    normal_blink_min_ms: float = 80.0        # Minimum duration for valid physiological blink
    normal_blink_max_ms: float = 400.0       # Maximum duration for normal blink
    long_blink_min_ms: float = 500.0         # Threshold to classify as long blink / microsleep
    prolonged_closure_critical_ms: float = 1500.0 # Eye closed >= this fires CRITICAL override
    prolonged_closure_warning_ms: float = 800.0   # Eye closed >= this contributes to high fatigue


@dataclass
class YawnConfig:
    yawn_start_mar: float = 0.55             # MAR above this starts yawn state
    yawn_end_mar: float = 0.38               # MAR below this ends yawn (hysteresis)
    min_yawn_duration_ms: float = 1000.0     # Minimum duration to count as completed yawn
    sustained_yawn_ms: float = 2000.0        # Ongoing yawn duration >= this triggers FATIGUE_RISK


@dataclass
class HeadPoseConfig:
    inattention_yaw_deg: float = 25.0
    inattention_pitch_up_deg: float = 20.0
    nodding_pitch_down_deg: float = -18.0
    head_turn_duration_warning_ms: float = 2000.0
    nodding_duration_warning_ms: float = 1500.0


@dataclass
class QualityConfig:
    tracked_ratio_threshold: float = 0.70    # >= 70% valid face in window -> TRACKED
    degraded_ratio_threshold: float = 0.30   # 30-70% valid face -> DEGRADED, < 30% -> LOST
    max_consecutive_dropouts_ms: float = 2000.0 # Time with no face before marking tracking LOST


@dataclass
class FusionWeightsConfig:
    perclos: float = 0.35
    cnn_drowsy: float = 0.25
    long_blinks: float = 0.15
    yawn_persistence: float = 0.10
    eye_closure: float = 0.10
    head_nodding: float = 0.05


@dataclass
class StateMachineConfig:
    fatigue_risk_threshold: float = 0.35
    drowsy_threshold: float = 0.60
    critical_score_threshold: float = 0.85
    cooldown_critical_to_drowsy_ms: float = 2000.0
    cooldown_drowsy_to_fatigue_ms: float = 3000.0
    cooldown_fatigue_to_normal_ms: float = 3000.0


@dataclass
class RecoveryConfig:
    reopen_confirmation_ms: float = 400.0
    drowsy_recovery_ms: float = 2000.0
    normal_recovery_ms: float = 4000.0
    strong_open_ear_margin: float = 0.05


@dataclass
class DriverStateConfig:
    windows: WindowConfig = field(default_factory=WindowConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)
    yawn: YawnConfig = field(default_factory=YawnConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    weights: FusionWeightsConfig = field(default_factory=FusionWeightsConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    @classmethod
    def from_yaml(cls, path: Optional[Path | str] = None) -> "DriverStateConfig":
        """Load from YAML file or return default instance."""
        if path is None:
            default_path = Path(__file__).parent.parent / "configs" / "driver_state.yaml"
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
    def from_dict(cls, d: dict[str, Any]) -> "DriverStateConfig":
        """Construct from raw nested dictionary."""
        cfg = cls()
        if "windows" in d:
            cfg.windows = WindowConfig(**{k: float(v) for k, v in d["windows"].items() if hasattr(cfg.windows, k)})
        if "evidence" in d:
            cfg.evidence = EvidenceConfig(**{k: float(v) for k, v in d["evidence"].items() if hasattr(cfg.evidence, k)})
        if "eye" in d:
            cfg.eye = EyeConfig(**{k: float(v) for k, v in d["eye"].items() if hasattr(cfg.eye, k)})
        if "yawn" in d:
            cfg.yawn = YawnConfig(**{k: float(v) for k, v in d["yawn"].items() if hasattr(cfg.yawn, k)})
        if "head_pose" in d:
            cfg.head_pose = HeadPoseConfig(**{k: float(v) for k, v in d["head_pose"].items() if hasattr(cfg.head_pose, k)})
        if "quality" in d:
            cfg.quality = QualityConfig(**{k: float(v) for k, v in d["quality"].items() if hasattr(cfg.quality, k)})
        if "weights" in d:
            cfg.weights = FusionWeightsConfig(**{k: float(v) for k, v in d["weights"].items() if hasattr(cfg.weights, k)})
        if "state_machine" in d:
            cfg.state_machine = StateMachineConfig(**{k: float(v) for k, v in d["state_machine"].items() if hasattr(cfg.state_machine, k)})
        if "recovery" in d:
            cfg.recovery = RecoveryConfig(**{k: float(v) for k, v in d["recovery"].items() if hasattr(cfg.recovery, k)})
        return cfg

    @classmethod
    def _parse_simple_yaml(cls, path: Path) -> "DriverStateConfig":
        """Simple line-by-line fallback parser for the config structure."""
        text = path.read_text(encoding="utf-8")
        current_section: Optional[str] = None
        data: dict[str, dict[str, float]] = {}

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
                        data[current_section][key] = float(val)
                    except ValueError:
                        pass

        return cls.from_dict(data)
