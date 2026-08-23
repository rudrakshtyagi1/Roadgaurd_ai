"""
Unit tests for Road Intelligence Network & Safer Routing (Phase 6).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.network.hazard_database import HazardDatabase
from core.network.safer_routing import SaferRouteEngine
from core.network.types import GeoLocation, NetworkHazardReport


class TestRoadIntelligenceNetwork(unittest.TestCase):

    def setUp(self):
        self.db = HazardDatabase(cluster_radius_m=25.0)
        self.routing_engine = SaferRouteEngine(self.db)

    def test_multi_vehicle_hazard_clustering(self):
        # Vehicle 1 reports pothole at (28.7041, 77.1025)
        r1 = NetworkHazardReport("r1", "pothole", "MEDIUM", 0.80, GeoLocation(28.70410, 77.10250), "2026-08-23T10:00:00Z", "veh_A")
        c1 = self.db.ingest_report(r1)
        self.assertEqual(c1.report_count, 1)
        self.assertEqual(c1.aggregated_confidence, 0.80)

        # Vehicle 2 reports same pothole 10m away at (28.70415, 77.10255)
        r2 = NetworkHazardReport("r2", "pothole", "HIGH", 0.85, GeoLocation(28.70415, 77.10255), "2026-08-23T10:05:00Z", "veh_B")
        c2 = self.db.ingest_report(r2)

        # Same cluster ID, count increased, Bayesian confidence combined > 0.95
        self.assertEqual(c2.cluster_id, c1.cluster_id)
        self.assertEqual(c2.report_count, 2)
        self.assertGreater(c2.aggregated_confidence, 0.95)
        self.assertEqual(c2.severity, "HIGH")

    def test_safer_routing_safety_score(self):
        # Ingest 3 hazards along Route A coords
        for i in range(3):
            r = NetworkHazardReport(f"h{i}", "pothole", "SEVERE", 0.90, GeoLocation(28.7041 + i * 0.001, 77.1025), "2026-08-23T10:00:00Z", "v1")
            self.db.ingest_report(r)

        coords_a = [(28.7041 + i * 0.001, 77.1025) for i in range(5)]
        coords_b = [(28.7500 + i * 0.001, 77.2000) for i in range(5)] # Clear route

        routes = self.routing_engine.evaluate_routes(coords_a, coords_b)
        self.assertEqual(len(routes), 2)
        route_a, route_b = routes[0], routes[1]

        # Route B should have higher safety score and lower hazard count
        self.assertGreater(route_a.hazard_count, route_b.hazard_count)
        self.assertGreater(route_b.road_safety_score, route_a.road_safety_score)


if __name__ == "__main__":
    unittest.main()
