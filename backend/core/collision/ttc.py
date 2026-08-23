"""
RoadGuard AI — Time-To-Collision (TTC) Estimator
Image-space TTC estimation based on scale expansion rates (s / ds_dt) with rigorous reliability gating.
"""

from __future__ import annotations

import math
from typing import Optional

from core.collision_config import TTCConfig
from core.collision.types import TrackSample


class TTCEstimator:
    """
    Computes image-space Time-to-Collision (TTC) for approaching tracks:
      TTC = scale(t) / (d_scale / dt)
    
    Reliability Gating:
      - Requires minimum track age (e.g. >= 500ms)
      - Requires minimum samples (e.g. >= 8 samples)
      - Requires approaching trajectory (slope > 0)
      - Returns None for receding, stationary, or noisy tracks.
    """

    def __init__(self, config: Optional[TTCConfig] = None) -> None:
        self.config = config or TTCConfig()

    def estimate_ttc(
        self,
        history: list[TrackSample],
        relative_motion: str,
        growth_rate: float,
        track_age_ms: float,
    ) -> tuple[Optional[float], bool, float]:
        """
        Estimate TTC in seconds from historical samples.
        Returns: (ttc_seconds: Optional[float], ttc_reliable: bool, ttc_signal_strength: float).
        """
        cfg = self.config

        # 1. Gate: Trajectory must be APPROACHING
        if relative_motion != "APPROACHING" or growth_rate <= 1e-4:
            return None, False, 0.0

        # 2. Gate: Minimum track age and sample count
        if track_age_ms < cfg.min_track_age_ms or len(history) < cfg.min_samples:
            return None, False, 0.0

        cutoff_ms = history[-1].monotonic_ms - 600.0
        samples = [s for s in history if s.monotonic_ms >= cutoff_ms]
        if len(samples) < cfg.min_samples:
            samples = history[-cfg.min_samples:]

        n = len(samples)
        if n < 4:
            return None, False, 0.0

        # Compute regression over samples
        t0 = samples[0].monotonic_ms / 1000.0
        ts = [(s.monotonic_ms / 1000.0) - t0 for s in samples]
        scales = [s.scale for s in samples]

        mean_t = sum(ts) / n
        mean_s = sum(scales) / n

        num = sum((ts[i] - mean_t) * (scales[i] - mean_s) for i in range(n))
        den = sum((ts[i] - mean_t) ** 2 for i in range(n))

        if den <= 1e-6:
            return None, False, 0.0

        slope = num / den
        if slope <= 1e-4:
            return None, False, 0.0

        # Correlation R^2
        ss_tot = sum((scales[i] - mean_s) ** 2 for i in range(n))
        if ss_tot > 1e-6:
            ss_res = sum((scales[i] - (mean_s + slope * (ts[i] - mean_t))) ** 2 for i in range(n))
            r_squared = max(0.0, 1.0 - (ss_res / ss_tot))
        else:
            r_squared = 0.50

        if r_squared < cfg.min_r_squared:
            # Fit too noisy / inconsistent expansion
            return None, False, 0.0

        # TTC = current_scale / slope
        current_scale = max(1e-4, scales[-1])
        ttc_raw = current_scale / slope

        if ttc_raw < cfg.min_valid_ttc_s or ttc_raw > cfg.max_valid_ttc_s:
            return None, False, 0.0

        ttc_seconds = round(ttc_raw, 2)
        ttc_reliable = True
        ttc_signal_strength = min(1.0, max(0.2, r_squared * (n / 12.0)))

        return ttc_seconds, ttc_reliable, round(ttc_signal_strength, 4)
