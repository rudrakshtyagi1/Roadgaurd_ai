"""
RoadGuard AI — Road Hazard Network Database & Clustering (Phase 6)
Aggregates crowd-sourced hazard telemetry, computes multi-vehicle confidence,
and decays stale road hazards over time.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from core.network.types import ClusteredRoadHazard, GeoLocation, NetworkHazardReport


class HazardDatabase:
    """Stores, clusters, and aggregates road hazards."""

    def __init__(self, cluster_radius_m: float = 20.0, decay_half_life_days: float = 14.0) -> None:
        self.cluster_radius_m = cluster_radius_m
        self.decay_half_life_days = decay_half_life_days
        self._clusters: dict[str, ClusteredRoadHazard] = {}
        self._next_id: int = 1

    def reset(self) -> None:
        self._clusters.clear()
        self._next_id = 1

    @staticmethod
    def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine formula for distance in meters between two GPS coordinates."""
        R = 6371000.0 # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def ingest_report(self, report: NetworkHazardReport) -> ClusteredRoadHazard:
        """Ingest a new hazard report and merge into existing spatial cluster or create new."""
        lat = report.location.latitude
        lon = report.location.longitude

        # Find closest existing cluster
        closest_cluster: Optional[ClusteredRoadHazard] = None
        min_dist = float("inf")
        for cluster in self._clusters.values():
            if cluster.hazard_type == report.hazard_type and cluster.is_active:
                d = self.haversine_distance_m(lat, lon, cluster.latitude, cluster.longitude)
                if d <= self.cluster_radius_m and d < min_dist:
                    min_dist = d
                    closest_cluster = cluster

        if closest_cluster is not None:
            # Merge into cluster: Bayesian multi-observation reinforcement
            # P_new = 1 - (1 - P_old) * (1 - P_new_obs)
            p_old = closest_cluster.aggregated_confidence
            p_new_obs = report.confidence
            p_combined = 1.0 - (1.0 - p_old) * (1.0 - p_new_obs)

            # Update coordinates (weighted average)
            n = closest_cluster.report_count
            new_lat = (closest_cluster.latitude * n + lat) / (n + 1)
            new_lon = (closest_cluster.longitude * n + lon) / (n + 1)

            # Severity upgrade if new observation is higher
            sev_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "SEVERE": 4}
            new_sev = report.severity if sev_rank.get(report.severity, 1) > sev_rank.get(closest_cluster.severity, 1) else closest_cluster.severity

            closest_cluster.aggregated_confidence = round(min(0.999, p_combined), 4)
            closest_cluster.latitude = round(new_lat, 6)
            closest_cluster.longitude = round(new_lon, 6)
            closest_cluster.severity = new_sev
            closest_cluster.report_count += 1
            closest_cluster.last_reported_utc = report.timestamp_utc
            return closest_cluster
        else:
            cid = f"haz_{self._next_id}"
            self._next_id += 1
            cluster = ClusteredRoadHazard(
                cluster_id=cid,
                hazard_type=report.hazard_type,
                severity=report.severity,
                aggregated_confidence=round(report.confidence, 4),
                latitude=round(lat, 6),
                longitude=round(lon, 6),
                report_count=1,
                first_reported_utc=report.timestamp_utc,
                last_reported_utc=report.timestamp_utc,
                is_active=True,
            )
            self._clusters[cid] = cluster
            return cluster

    def get_hazards_near(self, lat: float, lon: float, radius_m: float = 1000.0) -> list[ClusteredRoadHazard]:
        results = []
        for cluster in self._clusters.values():
            if cluster.is_active:
                d = self.haversine_distance_m(lat, lon, cluster.latitude, cluster.longitude)
                if d <= radius_m:
                    results.append(cluster)
        return results

    def get_all_hazards(self) -> list[ClusteredRoadHazard]:
        return list(self._clusters.values())
