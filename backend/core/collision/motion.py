"""
RoadGuard AI — Relative Motion Estimator
Estimates temporal closing and receding trajectories based on scale expansion rates.
"""

from __future__ import annotations

import math
from typing import Optional

from core.collision_config import MotionConfig
from core.collision.types import TrackSample


class RelativeMotionEstimator:
    """
    Computes scale growth rates (ds/dt) and classifies relative motion:
      - APPROACHING: object scale expanding temporally
      - STABLE: object scale stationary
      - RECEDING: object scale shrinking temporally
      - UNKNOWN: insufficient history or noisy observation
    """

    def __init__(self, config: Optional[MotionConfig] = None) -> None:
        self.config = config or MotionConfig()

    def estimate_motion(self, history: list[TrackSample], current_mono_ms: float) -> tuple[str, float, float, float]:
        """
        Estimate motion from history buffer.
        Returns: (relative_motion: str, motion_strength: float, bbox_growth_rate: float, motion_reliability: float).
        """
        if not history or len(history) < self.config.min_samples:
            return "UNKNOWN", 0.0, 0.0, 0.0

        cutoff_ms = current_mono_ms - self.config.window_ms
        samples = [s for s in history if s.monotonic_ms >= cutoff_ms]
        if len(samples) < self.config.min_samples:
            samples = history[-self.config.min_samples:]

        n = len(samples)
        if n < 3:
            return "UNKNOWN", 0.0, 0.0, 0.0

        # Time in seconds relative to first sample
        t0 = samples[0].monotonic_ms / 1000.0
        ts = [(s.monotonic_ms / 1000.0) - t0 for s in samples]
        scales = [s.scale for s in samples]

        # Total time span
        dt_span = ts[-1] - ts[0]
        if dt_span < 0.10:  # < 100ms
            return "UNKNOWN", 0.0, 0.0, 0.0

        # Linear regression: scale = slope * t + intercept
        mean_t = sum(ts) / n
        mean_s = sum(scales) / n

        num = sum((ts[i] - mean_t) * (scales[i] - mean_s) for i in range(n))
        den = sum((ts[i] - mean_t) ** 2 for i in range(n))

        if den <= 1e-6:
            return "STABLE", 0.0, 0.0, 0.5

        slope = num / den  # scale change per second

        # Correlation coefficient R^2
        ss_tot = sum((scales[i] - mean_s) ** 2 for i in range(n))
        if ss_tot > 1e-6:
            ss_res = sum((scales[i] - (mean_s + slope * (ts[i] - mean_t))) ** 2 for i in range(n))
            r_squared = max(0.0, 1.0 - (ss_res / ss_tot))
        else:
            r_squared = 0.50

        # Normalized growth rate relative to current scale
        curr_scale = max(1e-4, scales[-1])
        rel_growth_rate = slope / curr_scale  # (1/sec)

        cfg = self.config
        if rel_growth_rate >= cfg.approaching_slope_threshold and r_squared >= 0.35:
            motion = "APPROACHING"
            strength = min(1.0, max(0.2, rel_growth_rate / 0.50))
        elif rel_growth_rate <= cfg.receding_slope_threshold and r_squared >= 0.35:
            motion = "RECEDING"
            strength = min(1.0, max(0.2, abs(rel_growth_rate) / 0.50))
        else:
            motion = "STABLE"
            strength = max(0.0, 1.0 - abs(rel_growth_rate) / cfg.approaching_slope_threshold)

        reliability = min(1.0, max(0.0, r_squared * min(1.0, n / 10.0)))

        return motion, round(strength, 4), round(slope, 4), round(reliability, 4)
