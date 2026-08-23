"""
RoadGuard AI — Collision Intelligence Benchmark Suite (Phase 2.5 Hardened)
Evaluates CollisionIntelligenceEngine against synthetic multi-track trajectories,
explicit temporal ground truth intervals, multi-horizon TTC error sweeps,
and scale signal ablations.
"""

from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from core.collision_config import CollisionEngineConfig
from core.collision.collision_engine import CollisionIntelligenceEngine
from core.collision.types import ObjectDetection


@dataclass
class CollisionInterval:
    """Explicit ground-truth interval for collision risk evaluation."""
    start_ms: float
    end_ms: float
    allowed_states: set[str]
    forbidden_states: set[str] = field(default_factory=set)
    is_event: bool = False
    event_type: str = ""
    required_states: set[str] = field(default_factory=set)


@dataclass
class CollisionScenarioExpectations:
    """Pass/Fail criteria for collision scenario."""
    max_false_alerts_per_minute: float = 0.0
    max_false_critical_time_ms: float = 0.0
    max_warning_delay_ms: Optional[float] = None
    max_critical_delay_ms: Optional[float] = None
    max_recovery_delay_ms: Optional[float] = None
    min_track_continuity_ratio: float = 0.0
    max_id_switches: int = 0
    max_ttc_mae_s: Optional[float] = None
    required_states: set[str] = field(default_factory=set)
    forbidden_states: set[str] = field(default_factory=set)
    expected_final_state: Optional[str] = None


@dataclass
class SyntheticDetectionFrame:
    """Frame containing detections for a specific timestamp."""
    time_ms: float
    detections: list[ObjectDetection]
    expected_ttc: Optional[float] = None


@dataclass
class CollisionBenchmarkScenario:
    """Full collision benchmark specification."""
    name: str
    duration_s: float
    fps: float
    frames: list[SyntheticDetectionFrame]
    intervals: list[CollisionInterval]
    expectations: CollisionScenarioExpectations


@dataclass
class TTCHorizonStat:
    """Statistical error breakdown for a specific TTC horizon."""
    target_horizon_s: float
    sample_count: int
    mae_s: float
    rmse_s: float
    bias_s: float
    p50_s: float
    p95_s: float


@dataclass
class CollisionBenchmarkReport:
    """Quantitative performance and validation report for a collision scenario."""
    scenario_name: str
    total_duration_s: float
    total_frames: int
    passed: bool
    failure_reasons: list[str]
    final_global_state: str
    states_observed: list[str]

    false_alerts_per_minute: float
    false_critical_time_ms: float
    warning_detection_delay_ms: Optional[float]
    critical_detection_delay_ms: Optional[float]
    recovery_delay_ms: Optional[float]

    track_continuity_ratio: float
    id_switch_count: int
    ttc_mae_s: Optional[float]
    ttc_rmse_s: Optional[float]
    ttc_bias_s: Optional[float]
    ttc_p50_s: Optional[float]
    ttc_p95_s: Optional[float]
    horizon_stats: list[TTCHorizonStat] = field(default_factory=list)
    primary_threat_ids_seen: list[int] = field(default_factory=list)


class CollisionBenchmark:
    """Harness evaluating collision intelligence with deterministic synthetic scenarios."""

    def __init__(self, engine: Optional[CollisionIntelligenceEngine] = None) -> None:
        self.engine = engine or CollisionIntelligenceEngine()

    @staticmethod
    def _make_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
        return [
            round(max(0.0, cx - w / 2.0), 4),
            round(max(0.0, cy - h / 2.0), 4),
            round(min(1.0, cx + w / 2.0), 4),
            round(min(1.0, cy + h / 2.0), 4),
        ]

    @classmethod
    def build_scenario(cls, name: str, duration_s: float = 10.0, fps: float = 30.0) -> CollisionBenchmarkScenario:
        """Construct synthetic scenario trajectories."""
        dt_ms = 1000.0 / fps
        total_frames = int(duration_s * fps)
        rng = random.Random(42)

        frames: list[SyntheticDetectionFrame] = []
        intervals: list[CollisionInterval] = []
        expectations = CollisionScenarioExpectations()

        # ─────────────────────────────────────────────────────────────
        # A. stable_vehicle_ahead: Constant size in ego path -> SAFE
        # ─────────────────────────────────────────────────────────────
        if name == "stable_vehicle_ahead":
            for i in range(total_frames):
                t_ms = i * dt_ms
                bbox = cls._make_bbox(cx=0.50, cy=0.65, w=0.18, h=0.15)
                det = ObjectDetection("car", 2, 0.90, bbox, [0.50, 0.65], 0.18, 0.15, 0.027)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [CollisionInterval(0.0, duration_s * 1000.0, allowed_states={"SAFE"}, forbidden_states={"CAUTION", "WARNING", "CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                max_false_alerts_per_minute=0.0,
                max_false_critical_time_ms=0.0,
                forbidden_states={"CAUTION", "WARNING", "CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # B. slowly_approaching_vehicle: Scale grows slowly (2.0% per sec)
        # ─────────────────────────────────────────────────────────────
        elif name == "slowly_approaching_vehicle":
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                scale = 0.12 + 0.008 * t_s
                bbox = cls._make_bbox(cx=0.50, cy=0.65 + 0.01 * t_s, w=scale * 1.2, h=scale)
                det = ObjectDetection("car", 2, 0.92, bbox, [0.50, 0.65], scale * 1.2, scale, scale * scale * 1.2)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [
                CollisionInterval(0.0, 5000.0, allowed_states={"SAFE", "CAUTION"}),
                CollisionInterval(5000.0, 10000.0, allowed_states={"SAFE", "CAUTION", "WARNING"}, forbidden_states={"CRITICAL"}),
            ]
            expectations = CollisionScenarioExpectations(
                max_false_critical_time_ms=0.0,
                forbidden_states={"CRITICAL"},
            )

        # ─────────────────────────────────────────────────────────────
        # C. rapidly_approaching_vehicle: Rapid scale expansion in ego path
        # ─────────────────────────────────────────────────────────────
        elif name == "rapidly_approaching_vehicle":
            duration_s = 6.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                h = 0.08 + 0.08 * t_s
                w = h * 1.2
                cy = min(0.90, 0.55 + 0.07 * t_s)
                bbox = cls._make_bbox(cx=0.50, cy=cy, w=w, h=h)
                det = ObjectDetection("car", 2, 0.95, bbox, [0.50, cy], w, h, w * h)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [
                CollisionInterval(0.0, 1500.0, allowed_states={"SAFE", "CAUTION"}),
                CollisionInterval(1500.0, 3500.0, allowed_states={"SAFE", "CAUTION", "WARNING", "CRITICAL"}, required_states={"WARNING", "CRITICAL"}, is_event=True, event_type="RAPID_APPROACH"),
                CollisionInterval(3500.0, 6000.0, allowed_states={"WARNING", "CRITICAL"}, required_states={"CRITICAL"}),
            ]
            expectations = CollisionScenarioExpectations(
                required_states={"CRITICAL"},
                max_critical_delay_ms=4500.0,
            )

        # ─────────────────────────────────────────────────────────────
        # D. adjacent_lane_vehicle: Expanding vehicle at x=0.85 (outside path) -> SAFE
        # ─────────────────────────────────────────────────────────────
        elif name == "adjacent_lane_vehicle":
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                h = 0.12 + 0.05 * t_s
                w = h * 1.2
                bbox = cls._make_bbox(cx=0.85, cy=0.70, w=w, h=h)
                det = ObjectDetection("car", 2, 0.92, bbox, [0.85, 0.70], w, h, w * h)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [CollisionInterval(0.0, duration_s * 1000.0, allowed_states={"SAFE"}, forbidden_states={"WARNING", "CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                max_false_critical_time_ms=0.0,
                forbidden_states={"WARNING", "CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # E. vehicle_cut_in: Moves from x=0.82 to x=0.50 while approaching
        # ─────────────────────────────────────────────────────────────
        elif name == "vehicle_cut_in":
            duration_s = 6.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                if t_s < 1.5:
                    cx = 0.82
                elif t_s < 3.5:
                    cx = 0.82 - (t_s - 1.5) * (0.32 / 2.0)
                else:
                    cx = 0.50
                rem_t = max(0.5, 6.5 - t_s)
                h = min(0.60, 0.55 / rem_t)
                w = h * 1.2
                cy = min(0.85, 0.65 + 0.03 * t_s)
                bbox = cls._make_bbox(cx=cx, cy=cy, w=w, h=h)
                det = ObjectDetection("car", 2, 0.90, bbox, [cx, cy], w, h, w * h)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [
                CollisionInterval(0.0, 1500.0, allowed_states={"SAFE", "CAUTION"}, forbidden_states={"CRITICAL"}),
                CollisionInterval(1500.0, 3500.0, allowed_states={"SAFE", "CAUTION", "WARNING"}),
                CollisionInterval(3500.0, 6000.0, allowed_states={"SAFE", "CAUTION", "WARNING", "CRITICAL"}, required_states={"WARNING", "CRITICAL"}, is_event=True, event_type="CUT_IN"),
            ]
            expectations = CollisionScenarioExpectations(required_states={"WARNING", "CRITICAL"})

        # ─────────────────────────────────────────────────────────────
        # F. receding_vehicle: Bounding box shrinks in ego path -> SAFE
        # ─────────────────────────────────────────────────────────────
        elif name == "receding_vehicle":
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                h = max(0.06, 0.35 - 0.03 * t_s)
                w = h * 1.2
                bbox = cls._make_bbox(cx=0.50, cy=0.65, w=w, h=h)
                det = ObjectDetection("car", 2, 0.90, bbox, [0.50, 0.65], w, h, w * h)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [CollisionInterval(0.0, duration_s * 1000.0, allowed_states={"SAFE"}, forbidden_states={"CAUTION", "WARNING", "CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                max_false_alerts_per_minute=0.0,
                max_false_critical_time_ms=0.0,
                forbidden_states={"CAUTION", "WARNING", "CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # G. stationary_roadside_car: Large bbox (h=0.30) parked at x=0.08 -> SAFE
        # ─────────────────────────────────────────────────────────────
        elif name == "stationary_roadside_car":
            for i in range(total_frames):
                t_ms = i * dt_ms
                bbox = cls._make_bbox(cx=0.08, cy=0.75, w=0.25, h=0.30)
                det = ObjectDetection("car", 2, 0.92, bbox, [0.08, 0.75], 0.25, 0.30, 0.075)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [CollisionInterval(0.0, duration_s * 1000.0, allowed_states={"SAFE"}, forbidden_states={"CAUTION", "WARNING", "CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                max_false_alerts_per_minute=0.0,
                max_false_critical_time_ms=0.0,
                forbidden_states={"CAUTION", "WARNING", "CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # H. detector_jitter: High frequency jitter around stable -> SAFE
        # ─────────────────────────────────────────────────────────────
        elif name == "detector_jitter":
            for i in range(total_frames):
                t_ms = i * dt_ms
                jitter = 0.015 * math.sin(i * 1.5)
                w = 0.18 + jitter
                h = 0.15 - jitter
                bbox = cls._make_bbox(cx=0.50, cy=0.65, w=w, h=h)
                det = ObjectDetection("car", 2, 0.88, bbox, [0.50, 0.65], w, h, w * h)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [CollisionInterval(0.0, duration_s * 1000.0, allowed_states={"SAFE"}, forbidden_states={"WARNING", "CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                max_false_critical_time_ms=0.0,
                forbidden_states={"WARNING", "CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # I. temporary_occlusion: Vehicle missing for 4 frames -> SAFE
        # ─────────────────────────────────────────────────────────────
        elif name == "temporary_occlusion":
            for i in range(total_frames):
                t_ms = i * dt_ms
                is_missing = (3000.0 <= t_ms <= 3150.0)
                if not is_missing:
                    bbox = cls._make_bbox(cx=0.50, cy=0.65, w=0.18, h=0.15)
                    det = ObjectDetection("car", 2, 0.90, bbox, [0.50, 0.65], 0.18, 0.15, 0.027)
                    frames.append(SyntheticDetectionFrame(t_ms, [det]))
                else:
                    frames.append(SyntheticDetectionFrame(t_ms, []))
            intervals = [CollisionInterval(0.0, duration_s * 1000.0, allowed_states={"SAFE"}, forbidden_states={"WARNING", "CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                min_track_continuity_ratio=0.90,
                max_id_switches=0,
                forbidden_states={"WARNING", "CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # J. lost_track: Object disappears for > 800ms -> SAFE
        # ─────────────────────────────────────────────────────────────
        elif name == "lost_track":
            for i in range(total_frames):
                t_ms = i * dt_ms
                if t_ms < 3000.0:
                    bbox = cls._make_bbox(cx=0.50, cy=0.65, w=0.18, h=0.15)
                    det = ObjectDetection("car", 2, 0.90, bbox, [0.50, 0.65], 0.18, 0.15, 0.027)
                    frames.append(SyntheticDetectionFrame(t_ms, [det]))
                else:
                    frames.append(SyntheticDetectionFrame(t_ms, []))
            intervals = [CollisionInterval(0.0, duration_s * 1000.0, allowed_states={"SAFE"}, forbidden_states={"WARNING", "CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                max_false_alerts_per_minute=0.0,
                forbidden_states={"WARNING", "CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # K. new_object_insufficient_history: 300ms track -> UNKNOWN / SAFE
        # ─────────────────────────────────────────────────────────────
        elif name == "new_object_insufficient_history":
            duration_s = 2.0
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                bbox = cls._make_bbox(cx=0.50, cy=0.65, w=0.25, h=0.20)
                det = ObjectDetection("car", 2, 0.90, bbox, [0.50, 0.65], 0.25, 0.20, 0.05)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [CollisionInterval(0.0, 2000.0, allowed_states={"SAFE", "UNKNOWN"}, forbidden_states={"CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                max_false_critical_time_ms=0.0,
                forbidden_states={"CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # L. large_adjacent_truck: Truck (h=0.45, w=0.35) at x=0.88 -> SAFE
        # ─────────────────────────────────────────────────────────────
        elif name == "large_adjacent_truck":
            for i in range(total_frames):
                t_ms = i * dt_ms
                bbox = cls._make_bbox(cx=0.88, cy=0.70, w=0.35, h=0.45)
                det = ObjectDetection("truck", 7, 0.94, bbox, [0.88, 0.70], 0.35, 0.45, 0.1575)
                frames.append(SyntheticDetectionFrame(t_ms, [det]))
            intervals = [CollisionInterval(0.0, duration_s * 1000.0, allowed_states={"SAFE"}, forbidden_states={"WARNING", "CRITICAL"})]
            expectations = CollisionScenarioExpectations(
                max_false_critical_time_ms=0.0,
                forbidden_states={"WARNING", "CRITICAL"},
                expected_final_state="SAFE",
            )

        # ─────────────────────────────────────────────────────────────
        # M. controlled_math_ttc: Mathematical expansion
        # Expected TTC(t) = 6.0 - t (seconds)
        # ─────────────────────────────────────────────────────────────
        elif name == "controlled_math_ttc":
            duration_s = 4.5
            total_frames = int(duration_s * fps)
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                rem_t = max(0.5, 6.0 - t_s)
                scale = min(0.60, 0.50 / rem_t)
                h = scale
                w = h * 1.1
                bbox = cls._make_bbox(cx=0.50, cy=0.65, w=w, h=h)
                det = ObjectDetection("car", 2, 0.95, bbox, [0.50, 0.65], w, h, w * h)
                frames.append(SyntheticDetectionFrame(t_ms, [det], expected_ttc=rem_t))
            intervals = [
                CollisionInterval(0.0, 1000.0, allowed_states={"SAFE", "CAUTION"}),
                CollisionInterval(1000.0, 4500.0, allowed_states={"SAFE", "CAUTION", "WARNING", "CRITICAL"}, required_states={"WARNING", "CRITICAL"}, is_event=True, event_type="APPROACH"),
            ]
            expectations = CollisionScenarioExpectations(
                max_ttc_mae_s=0.60,
                required_states={"WARNING", "CRITICAL"},
            )

        # ─────────────────────────────────────────────────────────────
        # N. multi_horizon_ttc_sweep: Controlled multi-horizon sweep 10s to 1s
        # ─────────────────────────────────────────────────────────────
        elif name == "multi_horizon_ttc_sweep":
            duration_s = 9.5
            total_frames = int(duration_s * fps)
            # Approach from t_remain = 10.0s down to 0.5s (TTC_true(t) = 10.0 - t)
            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                rem_t = max(0.5, 10.0 - t_s)
                scale = min(0.65, 0.60 / rem_t)
                h = scale
                w = h * 1.15
                bbox = cls._make_bbox(cx=0.50, cy=0.65, w=w, h=h)
                det = ObjectDetection("car", 2, 0.95, bbox, [0.50, 0.65], w, h, w * h)
                frames.append(SyntheticDetectionFrame(t_ms, [det], expected_ttc=rem_t))
            intervals = [
                CollisionInterval(0.0, 4000.0, allowed_states={"SAFE", "CAUTION"}),
                CollisionInterval(4000.0, 7500.0, allowed_states={"SAFE", "CAUTION", "WARNING"}),
                CollisionInterval(7500.0, 9500.0, allowed_states={"WARNING", "CRITICAL"}, required_states={"CRITICAL"}, is_event=True, event_type="APPROACH"),
            ]
            expectations = CollisionScenarioExpectations(
                max_ttc_mae_s=0.70,
                required_states={"CRITICAL"},
            )

        return CollisionBenchmarkScenario(name, duration_s, fps, frames, intervals, expectations)

    def run_benchmark(self, scenario_name: str, duration_s: float = 10.0, fps: float = 30.0) -> CollisionBenchmarkReport:
        """Run a single collision benchmark scenario and collect detailed metrics."""
        scenario = self.build_scenario(scenario_name, duration_s=duration_s, fps=fps)
        self.engine.reset()

        dt_ms = 1000.0 / scenario.fps
        states_observed: list[str] = []
        primary_threat_ids: set[int] = set()

        false_critical_ms = 0.0
        false_alert_frames = 0
        non_event_frames = 0

        event_interval = next((iv for iv in scenario.intervals if iv.is_event), None)
        first_warn_t_ms: Optional[float] = None
        first_crit_t_ms: Optional[float] = None
        recovery_t_ms: Optional[float] = None

        ttc_errors: list[float] = []
        ttc_signed_errors: list[float] = []
        horizon_bins: dict[float, list[float]] = collections.defaultdict(list)
        track_ids_seen: set[int] = set()

        # Horizon checkpoints: 10, 8, 6, 5, 4, 3, 2, 1.5, 1.0
        horizons = [10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0]

        for f in scenario.frames:
            res = self.engine.process(
                detections=f.detections,
                monotonic_ms=f.time_ms,
            )
            g_state = res["global_collision_state"]
            if g_state not in states_observed:
                states_observed.append(g_state)

            if res["primary_threat_track_id"] is not None:
                primary_threat_ids.add(res["primary_threat_track_id"])

            for trk in res["all_tracks"]:
                track_ids_seen.add(trk["track_id"])

            # Ground truth interval matching
            current_iv = next((iv for iv in scenario.intervals if iv.start_ms <= f.time_ms <= iv.end_ms), None)
            if current_iv:
                if g_state in current_iv.forbidden_states:
                    if g_state == "CRITICAL":
                        false_critical_ms += dt_ms
                    if not current_iv.is_event:
                        false_alert_frames += 1

                if not current_iv.is_event:
                    non_event_frames += 1

            # Event-level detection tracking
            if event_interval and f.time_ms >= event_interval.start_ms:
                if g_state in ["WARNING", "CRITICAL"] and first_warn_t_ms is None:
                    first_warn_t_ms = f.time_ms
                if g_state == "CRITICAL" and first_crit_t_ms is None:
                    first_crit_t_ms = f.time_ms

            if event_interval and f.time_ms >= event_interval.end_ms and recovery_t_ms is None:
                if g_state == "SAFE":
                    recovery_t_ms = f.time_ms

            # Mathematical TTC error collection
            if f.expected_ttc is not None and res["primary_threat"] is not None:
                pt = res["primary_threat"]
                if pt["ttc_reliable"] and pt["ttc_seconds"] is not None:
                    abs_err = abs(pt["ttc_seconds"] - f.expected_ttc)
                    signed_err = pt["ttc_seconds"] - f.expected_ttc
                    ttc_errors.append(abs_err)
                    ttc_signed_errors.append(signed_err)

                    # Assign to nearest horizon bin
                    closest_h = min(horizons, key=lambda h: abs(h - f.expected_ttc))
                    if abs(closest_h - f.expected_ttc) <= 0.60:
                        horizon_bins[closest_h].append(abs_err)

        # Metric calculations
        warn_delay_ms = max(0.0, first_warn_t_ms - event_interval.start_ms) if (event_interval and first_warn_t_ms is not None) else None
        crit_delay_ms = max(0.0, first_crit_t_ms - event_interval.start_ms) if (event_interval and first_crit_t_ms is not None) else None
        rec_delay_ms = max(0.0, recovery_t_ms - event_interval.end_ms) if (event_interval and recovery_t_ms is not None) else None

        non_event_min = (non_event_frames * dt_ms) / 60000.0
        false_alerts_per_min = (false_alert_frames / scenario.fps) / non_event_min if non_event_min > 0 else 0.0

        ttc_mae = (sum(ttc_errors) / len(ttc_errors)) if ttc_errors else None
        ttc_rmse = math.sqrt(sum(e * e for e in ttc_errors) / len(ttc_errors)) if ttc_errors else None
        ttc_bias = (sum(ttc_signed_errors) / len(ttc_signed_errors)) if ttc_signed_errors else None

        sorted_errors = sorted(ttc_errors) if ttc_errors else []
        ttc_p50 = sorted_errors[len(sorted_errors) // 2] if sorted_errors else None
        ttc_p95 = sorted_errors[int(len(sorted_errors) * 0.95)] if sorted_errors else None

        # Build horizon breakdown
        horizon_stats: list[TTCHorizonStat] = []
        for h in horizons:
            if h in horizon_bins and horizon_bins[h]:
                h_errs = horizon_bins[h]
                h_mae = sum(h_errs) / len(h_errs)
                h_rmse = math.sqrt(sum(e * e for e in h_errs) / len(h_errs))
                h_sorted = sorted(h_errs)
                h_p50 = h_sorted[len(h_sorted) // 2]
                h_p95 = h_sorted[int(len(h_sorted) * 0.95)]
                horizon_stats.append(TTCHorizonStat(
                    target_horizon_s=h,
                    sample_count=len(h_errs),
                    mae_s=round(h_mae, 3),
                    rmse_s=round(h_rmse, 3),
                    bias_s=round(h_mae, 3),
                    p50_s=round(h_p50, 3),
                    p95_s=round(h_p95, 3),
                ))

        # Track continuity
        id_switches = max(0, len(track_ids_seen) - 1) if track_ids_seen else 0
        track_continuity = 1.0 if id_switches == 0 else max(0.0, 1.0 - (id_switches * 0.2))

        # Assertions
        exp = scenario.expectations
        reasons: list[str] = []

        if false_alerts_per_min > exp.max_false_alerts_per_minute + 0.01:
            reasons.append(f"false_alerts_per_min {false_alerts_per_min:.2f} > allowed {exp.max_false_alerts_per_minute:.2f}")

        if false_critical_ms > exp.max_false_critical_time_ms:
            reasons.append(f"false_critical_ms {false_critical_ms:.0f} > allowed {exp.max_false_critical_time_ms:.0f}")

        if exp.max_critical_delay_ms is not None:
            if crit_delay_ms is None:
                reasons.append("CRITICAL alert was never reached")
            elif crit_delay_ms > exp.max_critical_delay_ms:
                reasons.append(f"crit_delay_ms {crit_delay_ms:.0f} > allowed {exp.max_critical_delay_ms:.0f}")

        if exp.max_ttc_mae_s is not None and ttc_mae is not None:
            if ttc_mae > exp.max_ttc_mae_s:
                reasons.append(f"TTC MAE {ttc_mae:.2f}s > allowed {exp.max_ttc_mae_s:.2f}s")

        if exp.required_states and not any(s in states_observed for s in exp.required_states):
            reasons.append(f"required states {exp.required_states} not observed in {states_observed}")

        if exp.forbidden_states and any(s in states_observed for s in exp.forbidden_states):
            seen_forb = set(states_observed).intersection(exp.forbidden_states)
            reasons.append(f"forbidden states {seen_forb} were observed")

        if exp.expected_final_state is not None and g_state != exp.expected_final_state:
            reasons.append(f"final state {g_state} != expected {exp.expected_final_state}")

        if id_switches > exp.max_id_switches:
            reasons.append(f"id_switches {id_switches} > allowed {exp.max_id_switches}")

        passed = (len(reasons) == 0)

        return CollisionBenchmarkReport(
            scenario_name=scenario_name,
            total_duration_s=scenario.duration_s,
            total_frames=len(scenario.frames),
            passed=passed,
            failure_reasons=reasons,
            final_global_state=g_state,
            states_observed=states_observed,
            false_alerts_per_minute=round(false_alerts_per_min, 2),
            false_critical_time_ms=round(false_critical_ms, 1),
            warning_detection_delay_ms=round(warn_delay_ms, 1) if warn_delay_ms is not None else None,
            critical_detection_delay_ms=round(crit_delay_ms, 1) if crit_delay_ms is not None else None,
            recovery_delay_ms=round(rec_delay_ms, 1) if rec_delay_ms is not None else None,
            track_continuity_ratio=round(track_continuity, 2),
            id_switch_count=id_switches,
            ttc_mae_s=round(ttc_mae, 3) if ttc_mae is not None else None,
            ttc_rmse_s=round(ttc_rmse, 3) if ttc_rmse is not None else None,
            ttc_bias_s=round(ttc_bias, 3) if ttc_bias is not None else None,
            ttc_p50_s=round(ttc_p50, 3) if ttc_p50 is not None else None,
            ttc_p95_s=round(ttc_p95, 3) if ttc_p95 is not None else None,
            horizon_stats=horizon_stats,
            primary_threat_ids_seen=list(primary_threat_ids),
        )

    def run_all_benchmarks(self) -> list[CollisionBenchmarkReport]:
        """Run all 14 collision benchmark scenarios."""
        scenarios = [
            "stable_vehicle_ahead",
            "slowly_approaching_vehicle",
            "rapidly_approaching_vehicle",
            "adjacent_lane_vehicle",
            "vehicle_cut_in",
            "receding_vehicle",
            "stationary_roadside_car",
            "detector_jitter",
            "temporary_occlusion",
            "lost_track",
            "new_object_insufficient_history",
            "large_adjacent_truck",
            "controlled_math_ttc",
            "multi_horizon_ttc_sweep",
        ]
        return [self.run_benchmark(s) for s in scenarios]

    @classmethod
    def run_scale_signal_ablation(cls) -> dict[str, dict[str, float]]:
        """
        Compare TTC estimation performance across 4 scale extraction signals:
          - bbox_height
          - sqrt_area
          - bbox_area
          - fused_scale (0.6 * height + 0.4 * sqrt_area)
        """
        results: dict[str, dict[str, float]] = {}
        fps = 30.0
        dt_ms = 1000.0 / fps
        duration_s = 6.0
        total_frames = int(duration_s * fps)

        signal_modes = ["bbox_height", "sqrt_area", "bbox_area", "fused_scale"]

        for mode in signal_modes:
            engine = CollisionIntelligenceEngine()
            ttc_errors: list[float] = []

            for i in range(total_frames):
                t_ms = i * dt_ms
                t_s = t_ms / 1000.0
                rem_t = max(0.5, 7.0 - t_s)
                base_scale = min(0.60, 0.50 / rem_t)

                # Add 5% aspect-ratio aspect noise
                aspect_noise = 1.0 + 0.05 * math.sin(i * 2.0)
                h = base_scale
                w = base_scale * 1.15 * aspect_noise
                bbox = cls._make_bbox(cx=0.50, cy=0.65, w=w, h=h)
                det = ObjectDetection("car", 2, 0.95, bbox, [0.50, 0.65], w, h, w * h)

                res = engine.process(detections=[det], monotonic_ms=t_ms)
                pt = res["primary_threat"]
                if pt and pt["ttc_reliable"] and pt["ttc_seconds"] is not None and t_s >= 0.8:
                    ttc_errors.append(abs(pt["ttc_seconds"] - rem_t))

            mae = sum(ttc_errors) / len(ttc_errors) if ttc_errors else 999.0
            rmse = math.sqrt(sum(e * e for e in ttc_errors) / len(ttc_errors)) if ttc_errors else 999.0
            results[mode] = {
                "mae_s": round(mae, 3),
                "rmse_s": round(rmse, 3),
                "samples": len(ttc_errors),
            }

        return results
