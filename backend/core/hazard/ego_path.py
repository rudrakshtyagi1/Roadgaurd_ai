"""
RoadGuard AI — Hazard Ego Path Evaluator
Evaluates if a detected surface hazard is within the vehicle's driving corridor.
"""

from __future__ import annotations

from typing import Optional

from core.hazard_config import HazardEgoCorridorConfig


class HazardEgoPath:
    """Evaluates perspective corridor relevance for road hazards."""

    def __init__(self, config: Optional[HazardEgoCorridorConfig] = None) -> None:
        self.config = config or HazardEgoCorridorConfig()

    def evaluate(self, cx: float, cy: float) -> tuple[float, str]:
        cfg = self.config
        if cy < cfg.horizon_y:
            return 0.0, "OUT_OF_PATH"

        span = max(1e-4, cfg.bottom_y - cfg.horizon_y)
        u = min(1.0, max(0.0, (cy - cfg.horizon_y) / span))

        corridor_half_w = cfg.top_half_width + u * (cfg.bottom_half_width - cfg.top_half_width)
        lat_dist = abs(cx - cfg.center_x)

        if lat_dist <= corridor_half_w:
            # Fully in-path
            rel = 1.0 - 0.30 * (lat_dist / max(1e-4, corridor_half_w))
            return round(rel, 4), "IN_PATH"
        elif lat_dist <= corridor_half_w * 1.5:
            # Near-path buffer
            excess = lat_dist - corridor_half_w
            band_w = max(1e-4, corridor_half_w * 0.5)
            rel = 0.70 - 0.40 * (excess / band_w)
            return round(max(0.30, rel), 4), "NEAR_PATH"
        else:
            # Out of path
            return 0.0, "OUT_OF_PATH"
