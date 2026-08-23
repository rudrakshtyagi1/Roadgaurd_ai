"""
Unit tests for PerspectiveEgoCorridor.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.collision.ego_path import PerspectiveEgoCorridor


class TestEgoPath(unittest.TestCase):

    def setUp(self):
        self.corridor = PerspectiveEgoCorridor()

    def test_centered_in_path(self):
        # Bottom center is (0.50, 0.80) -> perfectly in path
        relevance, relation = self.corridor.evaluate_bbox([0.40, 0.60, 0.60, 0.80])
        self.assertGreaterEqual(relevance, 0.70)
        self.assertEqual(relation, "IN_PATH")

    def test_adjacent_lane_vehicle(self):
        # Bottom center is (0.85, 0.80) -> adjacent/out of path
        relevance, relation = self.corridor.evaluate_bbox([0.75, 0.60, 0.95, 0.80])
        self.assertLess(relevance, 0.70)
        self.assertIn(relation, ["NEAR_PATH", "OUT_OF_PATH"])

    def test_far_roadside_car(self):
        # Bottom center is (0.05, 0.85) -> clearly OUT_OF_PATH
        relevance, relation = self.corridor.evaluate_bbox([0.00, 0.70, 0.10, 0.85])
        self.assertLess(relevance, 0.30)
        self.assertEqual(relation, "OUT_OF_PATH")

    def test_above_horizon(self):
        # Sky/distant object y2 < 0.45
        relevance, relation = self.corridor.evaluate_bbox([0.45, 0.20, 0.55, 0.35])
        self.assertEqual(relevance, 0.0)
        self.assertEqual(relation, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
