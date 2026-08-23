"""
RoadGuard AI — Hazard Severity Estimator (Phase 3)
Estimates pothole severity (LOW, MEDIUM, HIGH, SEVERE) based on size, geometry, and approach.
"""

from __future__ import annotations

from typing import Optional

from core.hazard_config import HazardSeverityConfig
from core.hazard.types import TrackedHazard


class HazardSeverityEstimator:
    """Classifies road hazard severity."""

    def __init__(self, config: Optional[HazardSeverityConfig] = None) -> None:
        self.config = config or HazardSeverityConfig()

    def estimate(self, track: TrackedHazard) -> tuple[str, float, float]:
        """
        Returns: (severity_level: str, severity_score: float, growth_rate: float).
        """
        if not track.history:
            return "LOW", 0.20, 0.0

        latest = track.history[-1]
        area = latest.area
        conf = latest.confidence
        cfg = self.config

        # 1. Compute growth / approach rate if history exists
        growth_rate = 0.0
        if len(track.history) >= 3:
            dt = max(1e-3, (track.history[-1].monotonic_ms - track.history[0].monotonic_ms) / 1000.0)
            d_area = track.history[-1].area - track.history[0].area
            growth_rate = d_area / dt

        # 2. Score calculation
        if area >= cfg.area_severe_threshold and conf >= cfg.min_confidence_high_severity:
            level = "SEVERE"
            score = 1.0
        elif area >= cfg.area_high_threshold:
            level = "HIGH"
            score = 0.75 + min(0.25, (area - cfg.area_high_threshold) / (cfg.area_severe_threshold - cfg.area_high_threshold) * 0.25)
        elif area >= cfg.area_medium_threshold:
            level = "MEDIUM"
            score = 0.45 + min(0.30, (area - cfg.area_medium_threshold) / (cfg.area_high_threshold - cfg.area_medium_threshold) * 0.30)
        else:
            level = "LOW"
            score = min(0.40, (area / max(1e-4, cfg.area_medium_threshold)) * 0.40)

        # Bonus for rapidly expanding hazard right in front
        if growth_rate > cfg.expansion_rate_severe:
            score = min(1.0, score + 0.15)
            if level == "HIGH":
                level = "SEVERE"
            elif level == "MEDIUM":
                level = "HIGH"

        return level, round(score, 4), round(growth_rate, 4)
