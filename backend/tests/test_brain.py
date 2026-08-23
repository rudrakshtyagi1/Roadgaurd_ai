"""
Unit tests for RoadGuard Brain (Phase 4).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.brain.fusion_engine import RoadGuardBrain
from core.brain_benchmark import BrainBenchmark


class TestRoadGuardBrain(unittest.TestCase):

    def setUp(self):
        self.brain = RoadGuardBrain()

    def test_all_brain_benchmarks_pass(self):
        bm = BrainBenchmark(self.brain)
        reports = bm.run_all_benchmarks()
        for r in reports:
            self.assertTrue(
                r.passed,
                f"Brain benchmark '{r.scenario_name}' FAILED!\n"
                f"Expected primary={r.expected_primary}, actual={r.actual_primary}\n"
                f"Expected state={r.expected_state}, actual={r.actual_state}\n"
                f"Reason codes={r.reason_codes}"
            )

    def test_perception_event_generation(self):
        res = self.brain.compute(
            driver_data={"state": "NORMAL"},
            collision_data={"global_collision_state": "SAFE"},
            hazard_data={"global_hazard_state": "SAFE"},
            monotonic_ms=1000.0,
        )
        event = self.brain.to_perception_event(res)
        self.assertEqual(event.source, "roadguard_brain")
        self.assertEqual(event.event_type, "unified_risk")
        self.assertFalse(event.is_error)


if __name__ == "__main__":
    unittest.main()
