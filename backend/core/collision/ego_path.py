"""
RoadGuard AI — Perspective Ego Corridor
Evaluates object relevance against a trapezoidal perspective driving corridor.
"""

from __future__ import annotations

from typing import Optional

from core.collision_config import EgoCorridorConfig


class PerspectiveEgoCorridor:
    """
    Computes perspective-aware path relevance and classification:
      - IN_PATH: directly in ego driving corridor
      - NEAR_PATH: adjacent lane / encroaching boundary
      - OUT_OF_PATH: roadside / peripheral object
      - UNKNOWN: invalid or above-horizon coordinates
    """

    def __init__(self, config: Optional[EgoCorridorConfig] = None) -> None:
        self.config = config or EgoCorridorConfig()

    def evaluate_bbox(self, bbox_xyxy: list[float]) -> tuple[float, str]:
        """
        Evaluate normalized bounding box [x1, y1, x2, y2].
        Returns (relevance: float [0.0 - 1.0], path_relation: str).
        """
        if not bbox_xyxy or len(bbox_xyxy) < 4:
            return 0.0, "UNKNOWN"

        x1, y1, x2, y2 = bbox_xyxy
        bottom_cx = (x1 + x2) / 2.0
        bottom_cy = y2

        return self.evaluate_point(bottom_cx, bottom_cy)

    def evaluate_point(self, bottom_x: float, bottom_y: float) -> tuple[float, str]:
        """
        Evaluate bottom ground-contact point in normalized coordinates [0.0 - 1.0].
        """
        cfg = self.config
        if bottom_y < cfg.horizon_y:
            # Above horizon -> distant / sky / indeterminate
            return 0.0, "UNKNOWN"

        # Normalized vertical progress along the road corridor u in [0.0, 1.0]
        denom = max(1e-4, cfg.bottom_y - cfg.horizon_y)
        u = min(1.0, max(0.0, (bottom_y - cfg.horizon_y) / denom))

        # Corridor half-width at this vertical line: narrows towards horizon
        corridor_half_w = cfg.top_half_width + u * (cfg.bottom_half_width - cfg.top_half_width)

        # Lateral deviation from center line
        center_x = cfg.center_x
        lat_dist = abs(bottom_x - center_x)

        # Relevance score: 1.0 if perfectly centered, decreases laterally
        if lat_dist <= corridor_half_w:
            # Inside core trapezoid
            rel_ratio = lat_dist / max(1e-4, corridor_half_w)
            relevance = 1.0 - 0.30 * rel_ratio
            relevance = max(cfg.in_path_threshold, min(1.0, relevance))
        elif lat_dist <= corridor_half_w * 1.5:
            # Near boundary / adjacent lane
            excess = lat_dist - corridor_half_w
            span = corridor_half_w * 0.5
            relevance = cfg.in_path_threshold - (excess / max(1e-4, span)) * (cfg.in_path_threshold - cfg.near_path_threshold)
            relevance = max(cfg.near_path_threshold, min(cfg.in_path_threshold - 0.001, relevance))
        else:
            # Far outside ego corridor
            excess = lat_dist - (corridor_half_w * 1.5)
            relevance = max(0.0, cfg.near_path_threshold - (excess / max(1e-4, corridor_half_w)) * cfg.near_path_threshold)

        relevance = round(max(0.0, min(1.0, relevance)), 4)

        if relevance >= cfg.in_path_threshold:
            relation = "IN_PATH"
        elif relevance >= cfg.near_path_threshold:
            relation = "NEAR_PATH"
        else:
            relation = "OUT_OF_PATH"

        return relevance, relation
