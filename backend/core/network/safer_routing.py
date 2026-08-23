"""
RoadGuard AI — Safer Route Engine (Phase 6)
Compares routing options and calculates Road Safety Scores (0-100) based on hazard density.
"""

from __future__ import annotations

from typing import Optional

from core.network.hazard_database import HazardDatabase
from core.network.types import ClusteredRoadHazard, RouteOption


class SaferRouteEngine:
    """Evaluates routes and computes Road Safety Score."""

    def __init__(self, hazard_db: Optional[HazardDatabase] = None) -> None:
        self.hazard_db = hazard_db or HazardDatabase()

    def evaluate_routes(
        self,
        route_a_coords: list[tuple[float, float]],
        route_b_coords: list[tuple[float, float]],
        route_a_dist_km: float = 10.2,
        route_a_min: float = 18.0,
        route_b_dist_km: float = 11.5,
        route_b_min: float = 21.0,
    ) -> list[RouteOption]:
        """
        Evaluate two alternative routes against the Road Hazard Database.
        """
        # Count hazards along each route
        all_hazards = self.hazard_db.get_all_hazards()

        def score_route(coords: list[tuple[float, float]], dist_km: float) -> tuple[int, int, float]:
            hazards_along_route = 0
            severe_along_route = 0
            for h in all_hazards:
                for lat, lon in coords:
                    if HazardDatabase.haversine_distance_m(lat, lon, h.latitude, h.longitude) <= 30.0:
                        hazards_along_route += 1
                        if h.severity in ["HIGH", "SEVERE"]:
                            severe_along_route += 1
                        break

            # Safety Score: 100 - penalties
            penalty = (hazards_along_route * 8.0) + (severe_along_route * 15.0)
            score = max(10.0, min(100.0, 100.0 - penalty))
            return hazards_along_route, severe_along_route, round(score, 1)

        haz_a, sev_a, score_a = score_route(route_a_coords, route_a_dist_km)
        haz_b, sev_b, score_b = score_route(route_b_coords, route_b_dist_km)

        res_a = RouteOption(
            route_id="route_fastest",
            name="Fastest Route (Main Arterial)",
            distance_km=route_a_dist_km,
            duration_minutes=route_a_min,
            road_safety_score=score_a,
            hazard_count=haz_a,
            severe_hazard_count=sev_a,
            recommendation_reason="Shortest travel time, but contains higher surface hazard density.",
        )

        res_b = RouteOption(
            route_id="route_safer",
            name="RoadGuard Safer Route",
            distance_km=route_b_dist_km,
            duration_minutes=route_b_min,
            road_safety_score=score_b,
            hazard_count=haz_b,
            severe_hazard_count=sev_b,
            recommendation_reason=f"Adds {round(route_b_min - route_a_min, 0):.0f} mins but avoids {max(0, haz_a - haz_b)} hazards ({score_b}/100 safety score).",
        )

        return [res_a, res_b]
