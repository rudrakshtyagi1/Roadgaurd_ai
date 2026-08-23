"""
RoadGuard AI — Driver State Engine V2.0 Benchmark Suite
Evaluates DriverStateEngineV2 against explicit temporal ground-truth intervals.

Metrics computed:
  - false_alerts_per_minute (strictly over non-event intervals where forbidden states occur)
  - false_critical_time_ms
  - false_drowsy_time_ms
  - false_warning_time_ms
  - under_warning_time_ms (expected severe, predicted mild)
  - over_warning_time_ms (expected mild, predicted severe)
  - detection_delay_ms (guaranteed >= 0)
  - recovery_delay_ms (guaranteed >= 0)
  - state_transition_count
  - valid_observation_ratio
  - time-in-state confusion matrix
"""

from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from core.driver_state_config import DriverStateConfig
from core.driver_state_engine import DriverStateEngineV2


@dataclass
class ExpectedInterval:
    """Explicit ground-truth temporal interval."""
    start_ms: float
    end_ms: float
    allowed_states: set[str]       # States permitted without penalty
    forbidden_states: set[str] = field(default_factory=set)
    is_event: bool = False
    event_type: str = ""           # e.g. "MICROSLEEP", "YAWN", "HEAD_NOD"
    required_states: set[str] = field(default_factory=set) # Must see at least one of these


@dataclass
class ScenarioExpectations:
    """Quantitative evaluation thresholds for scenario pass/fail."""
    max_false_alerts_per_minute: float = 0.0
    max_false_critical_time_ms: float = 0.0
    max_false_drowsy_time_ms: float = 0.0
    max_detection_delay_ms: Optional[float] = None
    max_recovery_delay_ms: Optional[float] = None
    max_state_transitions: int = 4
    min_valid_observation_ratio: float = 0.0
    required_states: set[str] = field(default_factory=set)
    forbidden_states: set[str] = field(default_factory=set)


@dataclass
class ScenarioFrame:
    """Single synthetic frame observation for benchmarking."""
    time_ms: float
    ear: float
    mar: float
    yaw: float
    pitch: float
    roll: float
    cnn_prob: float
    face_detected: bool


@dataclass
class BenchmarkScenario:
    """Full scenario specification containing observations, intervals, and expectations."""
    name: str
    duration_s: float
    fps: float
    frames: list[ScenarioFrame]
    intervals: list[ExpectedInterval]
    expectations: ScenarioExpectations


@dataclass
class BenchmarkReport:
    """Comprehensive quantitative report for a benchmark scenario."""
    scenario_name: str
    total_duration_s: float
    total_frames: int
    passed: bool
    failure_reasons: list[str]
    false_alerts_per_minute: float
    false_critical_time_ms: float
    false_drowsy_time_ms: float
    false_warning_time_ms: float
    under_warning_time_ms: float
    over_warning_time_ms: float
    detection_delay_ms: Optional[float]
    recovery_delay_ms: Optional[float]
    state_transition_count: int
    valid_observation_ratio: float
    final_engine_state: str
    states_observed: list[str]
    confusion_matrix: dict[str, dict[str, float]] # expected -> predicted -> duration_ms


class DriverBenchmark:
    """Benchmark harness evaluating DriverStateEngineV2 against temporal ground-truth intervals."""

    def __init__(self, engine: Optional[DriverStateEngineV2] = None) -> None:
        self.engine = engine or DriverStateEngineV2()

    @staticmethod
    def build_scenario(name: str, duration_s: float = 20.0, fps: float = 30.0) -> BenchmarkScenario:
        """Construct deterministic synthetic benchmark scenarios with labeled temporal intervals."""
        dt_ms = 1000.0 / fps
        total_frames = int(duration_s * fps)
        rng = random.Random(42)

        frames: list[ScenarioFrame] = []
        intervals: list[ExpectedInterval] = []
        expectations = ScenarioExpectations()

        # ─────────────────────────────────────────────────────────────
        # 1. normal_driving: 30s of clean attentive baseline driving
        # ─────────────────────────────────────────────────────────────
        if name == "normal_driving":
            duration_s = 20.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                ear = 0.30 + 0.02 * math.sin(t_s * 0.5)
                # physiological 120ms blink every 4s
                if (t_ms % 4000.0) < 120.0:
                    ear = 0.08
                frames.append(ScenarioFrame(t_ms, ear, 0.12, 0.0, 0.0, 0.0, 0.08, True))
            intervals = [ExpectedInterval(0.0, duration_s * 1000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"})]
            expectations = ScenarioExpectations(max_false_alerts_per_minute=0.0, max_false_critical_time_ms=0.0, max_state_transitions=0, min_valid_observation_ratio=1.0)

        # ─────────────────────────────────────────────────────────────
        # 2. startup_with_closed_eyes: 400ms blink within first 1.5s
        # ─────────────────────────────────────────────────────────────
        elif name == "startup_with_closed_eyes":
            duration_s = 10.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                ear = 0.08 if (400.0 <= t_ms <= 800.0) else 0.30
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, 0.0, 0.0, 0.05, True))
            intervals = [ExpectedInterval(0.0, 10000.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, forbidden_states={"CRITICAL"})]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, forbidden_states={"CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 3. rapid_normal_blinks: 20 rapid physiological blinks (140ms each)
        # ─────────────────────────────────────────────────────────────
        elif name == "rapid_normal_blinks":
            duration_s = 20.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                # 140ms blink every 1500ms
                ear = 0.08 if (t_ms % 1500.0) < 140.0 else 0.32
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, 0.0, 0.0, 0.05, True))
            intervals = [ExpectedInterval(0.0, 20000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"})]
            expectations = ScenarioExpectations(max_false_alerts_per_minute=0.0, forbidden_states={"DROWSY", "CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 4. slow_benign_blinks: lazy/tired but benign blinks (320ms each)
        # ─────────────────────────────────────────────────────────────
        elif name == "slow_benign_blinks":
            duration_s = 20.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                # 320ms blink every 3500ms (below long_blink threshold of 500ms)
                ear = 0.08 if (t_ms % 3500.0) < 320.0 else 0.30
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, 0.0, 0.0, 0.10, True))
            intervals = [ExpectedInterval(0.0, 20000.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, forbidden_states={"CRITICAL"})]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, forbidden_states={"CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 5. repeated_long_blinks: 4 long blinks (650ms) during t=4-16s
        # ─────────────────────────────────────────────────────────────
        elif name == "repeated_long_blinks":
            duration_s = 25.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                if 4000.0 <= t_ms <= 16000.0:
                    cycle = (t_ms - 4000.0) % 3000.0
                    ear = 0.08 if cycle < 650.0 else 0.30
                else:
                    ear = 0.30
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, 0.0, 0.0, 0.20, True))
            intervals = [
                ExpectedInterval(0.0, 4000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"}),
                ExpectedInterval(4000.0, 16000.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY"}, required_states={"FATIGUE_RISK", "DROWSY"}, is_event=True, event_type="LONG_BLINKS"),
                ExpectedInterval(16000.0, 25000.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY"}),
            ]
            expectations = ScenarioExpectations(required_states={"FATIGUE_RISK", "DROWSY"}, max_false_critical_time_ms=0.0)

        # ─────────────────────────────────────────────────────────────
        # 6. microsleep: 2.0s eye closure at t=5s -> immediate CRITICAL
        # ─────────────────────────────────────────────────────────────
        elif name == "microsleep":
            duration_s = 15.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                ear = 0.08 if (5000.0 <= t_ms <= 7000.0) else 0.30
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, 0.0, 0.0, 0.10, True))
            intervals = [
                ExpectedInterval(0.0, 5000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"}),
                ExpectedInterval(5000.0, 7000.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY", "CRITICAL"}, required_states={"CRITICAL"}, is_event=True, event_type="MICROSLEEP"),
                ExpectedInterval(7000.0, 15000.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY", "CRITICAL"}),
            ]
            expectations = ScenarioExpectations(required_states={"CRITICAL"}, max_detection_delay_ms=1600.0)

        # ─────────────────────────────────────────────────────────────
        # 7. recovery_after_microsleep: 1.8s microsleep then 10s recovery
        # ─────────────────────────────────────────────────────────────
        elif name == "recovery_after_microsleep":
            duration_s = 20.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                ear = 0.08 if (5000.0 <= t_ms <= 6800.0) else 0.32
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, 0.0, 0.0, 0.10, True))
            intervals = [
                ExpectedInterval(0.0, 5000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"}),
                ExpectedInterval(5000.0, 6800.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY", "CRITICAL"}, required_states={"CRITICAL"}, is_event=True, event_type="MICROSLEEP"),
                ExpectedInterval(6800.0, 20000.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY", "CRITICAL"}),
            ]
            expectations = ScenarioExpectations(required_states={"CRITICAL"}, max_detection_delay_ms=1600.0, max_recovery_delay_ms=10000.0)

        # ─────────────────────────────────────────────────────────────
        # 8. sustained_yawn: 2.8s yawn (MAR=0.65) during t=4-6.8s
        # ─────────────────────────────────────────────────────────────
        elif name == "sustained_yawn":
            duration_s = 15.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                mar = 0.65 if (4000.0 <= t_ms <= 6800.0) else 0.10
                frames.append(ScenarioFrame(t_ms, 0.30, mar, 0.0, 0.0, 0.0, 0.10, True))
            intervals = [
                ExpectedInterval(0.0, 4000.0, allowed_states={"NORMAL"}, forbidden_states={"FATIGUE_RISK", "DROWSY", "CRITICAL"}),
                ExpectedInterval(4000.0, 6800.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, required_states={"FATIGUE_RISK"}, is_event=True, event_type="YAWN"),
                ExpectedInterval(6800.0, 15000.0, allowed_states={"NORMAL", "FATIGUE_RISK"}),
            ]
            expectations = ScenarioExpectations(required_states={"FATIGUE_RISK"}, max_detection_delay_ms=2100.0, max_false_critical_time_ms=0.0)

        # ─────────────────────────────────────────────────────────────
        # 9. yawn_without_eye_closure: yawn with clear open eyes (EAR=0.32)
        # ─────────────────────────────────────────────────────────────
        elif name == "yawn_without_eye_closure":
            duration_s = 15.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                mar = 0.60 if (3000.0 <= t_ms <= 5500.0) else 0.10
                frames.append(ScenarioFrame(t_ms, 0.32, mar, 0.0, 0.0, 0.0, 0.05, True))
            intervals = [
                ExpectedInterval(0.0, 3000.0, allowed_states={"NORMAL"}, forbidden_states={"FATIGUE_RISK", "DROWSY", "CRITICAL"}),
                ExpectedInterval(3000.0, 5500.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, required_states={"FATIGUE_RISK"}, is_event=True, event_type="YAWN"),
                ExpectedInterval(5500.0, 15000.0, allowed_states={"NORMAL", "FATIGUE_RISK"}),
            ]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, forbidden_states={"CRITICAL", "DROWSY"})

        # ─────────────────────────────────────────────────────────────
        # 10. brief_mirror_check: 0.5s head turn (yaw=35 deg)
        # ─────────────────────────────────────────────────────────────
        elif name == "brief_mirror_check":
            duration_s = 10.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                yaw = 35.0 if (3000.0 <= t_ms <= 3500.0) else 0.0
                frames.append(ScenarioFrame(t_ms, 0.30, 0.10, yaw, 0.0, 0.0, 0.05, True))
            intervals = [ExpectedInterval(0.0, 10000.0, allowed_states={"NORMAL"}, forbidden_states={"FATIGUE_RISK", "DROWSY", "CRITICAL"})]
            expectations = ScenarioExpectations(max_false_alerts_per_minute=0.0, forbidden_states={"FATIGUE_RISK", "DROWSY", "CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 11. looking_left_1s: 1.0s head turn left (yaw=30 deg)
        # ─────────────────────────────────────────────────────────────
        elif name == "looking_left_1s":
            duration_s = 10.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                yaw = 30.0 if (3000.0 <= t_ms <= 4000.0) else 0.0
                frames.append(ScenarioFrame(t_ms, 0.30, 0.10, yaw, 0.0, 0.0, 0.05, True))
            intervals = [ExpectedInterval(0.0, 10000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"})]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, forbidden_states={"DROWSY", "CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 12. looking_right_1s: 1.0s head turn right (yaw=-30 deg)
        # ─────────────────────────────────────────────────────────────
        elif name == "looking_right_1s":
            duration_s = 10.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                yaw = -30.0 if (3000.0 <= t_ms <= 4000.0) else 0.0
                frames.append(ScenarioFrame(t_ms, 0.30, 0.10, yaw, 0.0, 0.0, 0.05, True))
            intervals = [ExpectedInterval(0.0, 10000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"})]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, forbidden_states={"DROWSY", "CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 13. looking_down_briefly: 0.8s pitch=-20 deg (dashboard glance)
        # ─────────────────────────────────────────────────────────────
        elif name == "looking_down_briefly":
            duration_s = 10.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                pitch = -20.0 if (3000.0 <= t_ms <= 3800.0) else 0.0
                frames.append(ScenarioFrame(t_ms, 0.30, 0.10, 0.0, pitch, 0.0, 0.05, True))
            intervals = [ExpectedInterval(0.0, 10000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"})]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, forbidden_states={"DROWSY", "CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 14. head_nod_without_eye_closure: pitch=-25 deg for 2.2s with eyes open
        # ─────────────────────────────────────────────────────────────
        elif name == "head_nod_without_eye_closure":
            duration_s = 12.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                pitch = -25.0 if (3000.0 <= t_ms <= 5200.0) else 0.0
                frames.append(ScenarioFrame(t_ms, 0.30, 0.10, 0.0, pitch, 0.0, 0.05, True))
            intervals = [
                ExpectedInterval(0.0, 3000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"}),
                ExpectedInterval(3000.0, 5200.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, required_states={"FATIGUE_RISK"}, is_event=True, event_type="HEAD_NOD"),
                ExpectedInterval(5200.0, 12000.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, forbidden_states={"DROWSY", "CRITICAL"}),
            ]
            # Head nod ALONE must NEVER escalate to DROWSY or CRITICAL!
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, forbidden_states={"DROWSY", "CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 15. sustained_head_drop_with_eye_closure: pitch=-25 + eyes closed
        # ─────────────────────────────────────────────────────────────
        elif name == "sustained_head_drop_with_eye_closure":
            duration_s = 12.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                is_drop = (3000.0 <= t_ms <= 5500.0)
                pitch = -25.0 if is_drop else 0.0
                ear = 0.08 if is_drop else 0.30
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, pitch, 0.0, 0.20, True))
            intervals = [
                ExpectedInterval(0.0, 3000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"}),
                ExpectedInterval(3000.0, 5500.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY", "CRITICAL"}, required_states={"CRITICAL"}, is_event=True, event_type="CORROBORATED_HEAD_DROP"),
                ExpectedInterval(5500.0, 12000.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY", "CRITICAL"}),
            ]
            expectations = ScenarioExpectations(required_states={"CRITICAL"}, max_detection_delay_ms=1600.0)

        # ─────────────────────────────────────────────────────────────
        # 16. cnn_probability_noise: fluctuating CNN prob with open eyes
        # ─────────────────────────────────────────────────────────────
        elif name == "cnn_probability_noise":
            duration_s = 15.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                cnn_prob = rng.uniform(0.10, 0.85)
                frames.append(ScenarioFrame(t_ms, 0.30, 0.10, 0.0, 0.0, 0.0, cnn_prob, True))
            intervals = [ExpectedInterval(0.0, 15000.0, allowed_states={"NORMAL", "FATIGUE_RISK", "DROWSY"}, forbidden_states={"CRITICAL"})]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, max_state_transitions=2, forbidden_states={"CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 17. fluttering_ear_noise: high-frequency noise around open eyes
        # ─────────────────────────────────────────────────────────────
        elif name == "fluttering_ear_noise":
            duration_s = 10.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                ear = 0.27 + 0.03 * math.sin(t_s * 10.0) + (0.01 if i % 3 != 0 else -0.08)
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, 0.0, 0.0, 0.05, True))
            intervals = [ExpectedInterval(0.0, 10000.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, forbidden_states={"CRITICAL"})]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, max_state_transitions=2, forbidden_states={"CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 18. alternating_face_detection: 50% intermittent face tracking
        # ─────────────────────────────────────────────────────────────
        elif name == "alternating_face_detection":
            duration_s = 10.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                face_detected = (int(t_ms / 500.0) % 2 == 0)
                frames.append(ScenarioFrame(t_ms, 0.30, 0.10, 0.0, 0.0, 0.0, 0.05, face_detected))
            intervals = [ExpectedInterval(0.0, 10000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"})]
            expectations = ScenarioExpectations(max_false_alerts_per_minute=0.0, forbidden_states={"DROWSY", "CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 19. eyes_closed_then_face_dropout: 1s closure then face lost 3s
        # ─────────────────────────────────────────────────────────────
        elif name == "eyes_closed_then_face_dropout":
            duration_s = 12.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                if 3000.0 <= t_ms < 4000.0:
                    ear = 0.08
                    face_det = True
                elif 4000.0 <= t_ms < 7000.0:
                    ear = 0.0
                    face_det = False
                else:
                    ear = 0.30
                    face_det = True
                frames.append(ScenarioFrame(t_ms, ear, 0.10, 0.0, 0.0, 0.0, 0.10, face_det))
            intervals = [
                ExpectedInterval(0.0, 3000.0, allowed_states={"NORMAL"}, forbidden_states={"CRITICAL"}),
                ExpectedInterval(3000.0, 4000.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, is_event=True, event_type="SHORT_CLOSURE"),
                ExpectedInterval(4000.0, 7000.0, allowed_states={"NORMAL", "FATIGUE_RISK"}, forbidden_states={"CRITICAL"}),
                ExpectedInterval(7000.0, 12000.0, allowed_states={"NORMAL"}),
            ]
            expectations = ScenarioExpectations(max_false_critical_time_ms=0.0, forbidden_states={"CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # 20. long_face_dropout: 10s missing face -> tracking LOST
        # ─────────────────────────────────────────────────────────────
        elif name == "long_face_dropout":
            duration_s = 15.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                face_det = not (3000.0 <= t_ms <= 13000.0)
                frames.append(ScenarioFrame(t_ms, 0.30 if face_det else 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, face_det))
            intervals = [ExpectedInterval(0.0, 15000.0, allowed_states={"NORMAL"}, forbidden_states={"DROWSY", "CRITICAL"})]
            expectations = ScenarioExpectations(max_false_alerts_per_minute=0.0, forbidden_states={"DROWSY", "CRITICAL"})

        return BenchmarkScenario(name, duration_s, fps, frames, intervals, expectations)

    def run_benchmark(self, scenario_name: str, duration_s: float = 20.0, fps: float = 30.0) -> BenchmarkReport:
        """Evaluate a scenario against labeled temporal intervals and compute exact timing invariants."""
        scenario = self.build_scenario(scenario_name, duration_s=duration_s, fps=fps)
        self.engine.reset()

        dt_ms = 1000.0 / scenario.fps
        states_observed: list[str] = []
        state_transitions = 0
        prev_state: Optional[str] = None

        false_critical_ms = 0.0
        false_drowsy_ms = 0.0
        false_warning_ms = 0.0
        under_warning_ms = 0.0
        over_warning_ms = 0.0
        false_alert_frames = 0
        non_event_frames = 0

        valid_face_frames = sum(1 for f in scenario.frames if f.face_detected)
        valid_observation_ratio = valid_face_frames / len(scenario.frames) if scenario.frames else 1.0

        state_rank = {"NORMAL": 0, "FATIGUE_RISK": 1, "DROWSY": 2, "CRITICAL": 3}
        confusion_matrix: dict[str, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))

        event_interval = next((iv for iv in scenario.intervals if iv.is_event), None)
        first_alert_t_ms: Optional[float] = None
        recovery_t_ms: Optional[float] = None

        for f in scenario.frames:
            res = self.engine.update(
                ear=f.ear,
                mar=f.mar,
                head_pose={"yaw": f.yaw, "pitch": f.pitch, "roll": f.roll},
                cnn_drowsy_prob=f.cnn_prob,
                face_detected=f.face_detected,
                monotonic_ms=f.time_ms,
            )
            current_state = res["state"]
            if current_state not in states_observed:
                states_observed.append(current_state)

            if prev_state is not None and current_state != prev_state:
                state_transitions += 1
            prev_state = current_state

            # Find matching ground-truth interval
            current_iv = next((iv for iv in scenario.intervals if iv.start_ms <= f.time_ms <= iv.end_ms), None)

            # Confusion matrix accounting
            expected_primary = list(current_iv.allowed_states)[0] if current_iv and current_iv.allowed_states else "NORMAL"
            confusion_matrix[expected_primary][current_state] += dt_ms

            if current_iv:
                # Check forbidden states in interval
                if current_state in current_iv.forbidden_states:
                    if current_state == "CRITICAL":
                        false_critical_ms += dt_ms
                    elif current_state == "DROWSY":
                        false_drowsy_ms += dt_ms
                    elif current_state == "FATIGUE_RISK":
                        false_warning_ms += dt_ms

                    if not current_iv.is_event:
                        false_alert_frames += 1

                if not current_iv.is_event:
                    non_event_frames += 1

                # Severity-aware comparison
                exp_rank = max(state_rank.get(s, 0) for s in current_iv.allowed_states)
                pred_rank = state_rank.get(current_state, 0)
                if pred_rank < min(state_rank.get(s, 0) for s in current_iv.allowed_states):
                    under_warning_ms += dt_ms
                elif pred_rank > exp_rank:
                    over_warning_ms += dt_ms

            # ── Event-Level Exact Timing Invariants ──────────────────
            if event_interval:
                # 1. Detection delay: first frame >= event.start_ms where state is in event's target/required states
                target_event_states = event_interval.required_states or {"FATIGUE_RISK", "DROWSY", "CRITICAL"}
                if f.time_ms >= event_interval.start_ms and first_alert_t_ms is None:
                    if current_state in target_event_states:
                        first_alert_t_ms = f.time_ms

                # 2. Recovery delay: first frame >= event.end_ms where state has returned to NORMAL
                if f.time_ms >= event_interval.end_ms and recovery_t_ms is None:
                    if current_state == "NORMAL":
                        recovery_t_ms = f.time_ms

        # Calculate exact non-negative delays
        detection_delay_ms: Optional[float] = None
        recovery_delay_ms: Optional[float] = None

        if event_interval and first_alert_t_ms is not None:
            detection_delay_ms = max(0.0, first_alert_t_ms - event_interval.start_ms)

        if event_interval and recovery_t_ms is not None:
            recovery_delay_ms = max(0.0, recovery_t_ms - event_interval.end_ms)

        # Rate of false alerts per minute during non-event baseline intervals
        non_event_minutes = (non_event_frames * dt_ms) / 60000.0
        false_alerts_per_min = (false_alert_frames / scenario.fps) / non_event_minutes if non_event_minutes > 0 else 0.0

        # ── Quantitative Pass/Fail Assertion Verification ───────────
        exp = scenario.expectations
        reasons: list[str] = []

        if false_alerts_per_min > exp.max_false_alerts_per_minute + 0.01:
            reasons.append(f"false_alerts_per_min {false_alerts_per_min:.2f} > allowed {exp.max_false_alerts_per_minute:.2f}")

        if false_critical_ms > exp.max_false_critical_time_ms:
            reasons.append(f"false_critical_ms {false_critical_ms:.0f} > allowed {exp.max_false_critical_time_ms:.0f}")

        if false_drowsy_ms > exp.max_false_drowsy_time_ms:
            reasons.append(f"false_drowsy_ms {false_drowsy_ms:.0f} > allowed {exp.max_false_drowsy_time_ms:.0f}")

        if exp.max_detection_delay_ms is not None:
            if detection_delay_ms is None:
                reasons.append(f"event was not detected (target states {exp.required_states})")
            elif detection_delay_ms > exp.max_detection_delay_ms:
                reasons.append(f"detection_delay_ms {detection_delay_ms:.0f} > allowed {exp.max_detection_delay_ms:.0f}")

        if exp.max_recovery_delay_ms is not None:
            if recovery_delay_ms is None:
                reasons.append("state failed to recover to NORMAL")
            elif recovery_delay_ms > exp.max_recovery_delay_ms:
                reasons.append(f"recovery_delay_ms {recovery_delay_ms:.0f} > allowed {exp.max_recovery_delay_ms:.0f}")

        if exp.required_states and not any(s in states_observed for s in exp.required_states):
            reasons.append(f"required state {exp.required_states} was not observed in {states_observed}")

        if exp.forbidden_states and any(s in states_observed for s in exp.forbidden_states):
            forbidden_seen = set(states_observed).intersection(exp.forbidden_states)
            reasons.append(f"forbidden state {forbidden_seen} was observed")

        if state_transitions > exp.max_state_transitions:
            reasons.append(f"state_transitions {state_transitions} > allowed {exp.max_state_transitions}")

        passed = (len(reasons) == 0)

        return BenchmarkReport(
            scenario_name=scenario_name,
            total_duration_s=scenario.duration_s,
            total_frames=len(scenario.frames),
            passed=passed,
            failure_reasons=reasons,
            false_alerts_per_minute=round(false_alerts_per_min, 2),
            false_critical_time_ms=round(false_critical_ms, 1),
            false_drowsy_time_ms=round(false_drowsy_ms, 1),
            false_warning_time_ms=round(false_warning_ms, 1),
            under_warning_time_ms=round(under_warning_ms, 1),
            over_warning_time_ms=round(over_warning_ms, 1),
            detection_delay_ms=round(detection_delay_ms, 1) if detection_delay_ms is not None else None,
            recovery_delay_ms=round(recovery_delay_ms, 1) if recovery_delay_ms is not None else None,
            state_transition_count=state_transitions,
            valid_observation_ratio=round(valid_observation_ratio, 3),
            final_engine_state=current_state,
            states_observed=states_observed,
            confusion_matrix={k: dict(v) for k, v in confusion_matrix.items()},
        )

    def run_all_benchmarks(self) -> list[BenchmarkReport]:
        """Run all 20 scenario benchmarks and return report list."""
        scenarios = [
            "normal_driving",
            "startup_with_closed_eyes",
            "rapid_normal_blinks",
            "slow_benign_blinks",
            "repeated_long_blinks",
            "microsleep",
            "recovery_after_microsleep",
            "sustained_yawn",
            "yawn_without_eye_closure",
            "brief_mirror_check",
            "looking_left_1s",
            "looking_right_1s",
            "looking_down_briefly",
            "head_nod_without_eye_closure",
            "sustained_head_drop_with_eye_closure",
            "cnn_probability_noise",
            "fluttering_ear_noise",
            "alternating_face_detection",
            "eyes_closed_then_face_dropout",
            "long_face_dropout",
        ]
        return [self.run_benchmark(s) for s in scenarios]
