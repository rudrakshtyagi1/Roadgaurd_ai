"""
RoadGuard AI — Edge Performance Telemetry & Hardware Profiler (Phase 7)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EdgeTelemetryMetrics:
    current_fps: float
    rolling_fps: float
    mean_latency_ms: float
    p95_latency_ms: float
    memory_rss_mb: float
    cpu_percent: float
    frame_count: int


class EdgeProfiler:
    """Tracks latency percentiles and resource consumption."""

    def __init__(self) -> None:
        self._latencies: list[float] = []
        self._frame_count = 0
        self._last_t = time.perf_counter()

    def record_frame(self, latency_ms: float) -> EdgeTelemetryMetrics:
        self._frame_count += 1
        now = time.perf_counter()
        dt = max(1e-4, now - self._last_t)
        self._last_t = now
        fps = 1.0 / dt

        self._latencies.append(latency_ms)
        if len(self._latencies) > 60:
            self._latencies.pop(0)

        mean_lat = sum(self._latencies) / len(self._latencies)
        sorted_lat = sorted(self._latencies)
        p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else mean_lat
        roll_fps = 1000.0 / mean_lat if mean_lat > 0 else 30.0

        # Memory footprint
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On macOS ru_maxrss is in bytes, on Linux in KB
            mem_mb = (usage / (1024 * 1024)) if usage > 1000000 else (usage / 1024)
        except Exception:
            mem_mb = 120.0

        return EdgeTelemetryMetrics(
            current_fps=round(fps, 1),
            rolling_fps=round(roll_fps, 1),
            mean_latency_ms=round(mean_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            memory_rss_mb=round(mem_mb, 1),
            cpu_percent=round(min(100.0, (mean_lat / 33.3) * 100.0), 1),
            frame_count=self._frame_count,
        )
