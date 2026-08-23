"""
RoadGuard AI — Adaptive Pipeline Scheduler (Phase 7 Edge AI)
Manages multi-rate pipeline execution and dynamic load shedding to maintain 30 FPS on edge hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class SchedulerDecision:
    frame_index: int
    run_driver: bool
    run_vehicle_detector: bool
    run_hazard_detector: bool
    run_brain: bool
    target_fps: float


class AdaptivePipelineScheduler:
    """
    Schedules vision and inference workloads dynamically:
      - Driver landmarks: Every frame (30 Hz)
      - Vehicle detection: Every 2 frames (15 Hz) (tracking runs every frame)
      - Hazard detection: Every 3 frames (10 Hz) (tracking runs every frame)
      - Brain & Decision: Every frame (30 Hz)
      - Dynamic load shedding if p95 latency > 33ms.
    """

    def __init__(self, base_fps: float = 30.0) -> None:
        self.base_fps = base_fps
        self._frame_count = 0
        self._recent_latencies: list[float] = []
        self._load_shed_mode = False

    def schedule_tick(self, frame_latency_ms: Optional[float] = None) -> SchedulerDecision:
        self._frame_count += 1
        idx = self._frame_count

        if frame_latency_ms is not None:
            self._recent_latencies.append(frame_latency_ms)
            if len(self._recent_latencies) > 30:
                self._recent_latencies.pop(0)

            # Check for overload (> 33ms per frame)
            avg_lat = sum(self._recent_latencies) / len(self._recent_latencies)
            self._load_shed_mode = (avg_lat > 33.0)

        # Workload stride allocation
        run_driver = True  # Safety critical: always runs
        run_brain = True   # Lightweight: always runs

        if self._load_shed_mode:
            # Down-throttle detectors under heavy compute load
            run_veh = (idx % 3 == 0)
            run_haz = (idx % 4 == 0)
        else:
            run_veh = (idx % 2 == 0)
            run_haz = (idx % 3 == 0)

        return SchedulerDecision(
            frame_index=idx,
            run_driver=run_driver,
            run_vehicle_detector=run_veh,
            run_hazard_detector=run_haz,
            run_brain=run_brain,
            target_fps=self.base_fps,
        )
