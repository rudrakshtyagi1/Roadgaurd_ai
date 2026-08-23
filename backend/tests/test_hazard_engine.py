"""
Unit tests for RoadHazardEngine and HazardBenchmark (Phase 3).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hazard.hazard_engine import RoadHazardEngine
from core.hazard_benchmark import HazardBenchmark


class TestRoadHazardEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RoadHazardEngine()

    def test_all_hazard_benchmarks_pass(self):
        bm = HazardBenchmark(self.engine)
        reports = bm.run_all_benchmarks()
        for r in reports:
            self.assertTrue(
                r.passed,
                f"Hazard benchmark '{r.scenario_name}' FAILED: {r.failure_reasons}"
            )

    def test_protocol_perception_event(self):
        res = self.engine.process(detections=[], monotonic_ms=1000.0)
        event = self.engine.to_perception_event(res, frame_index=1)
        self.assertEqual(event.source, "road_hazard_engine")
        self.assertEqual(event.event_type, "hazard_state")
        self.assertFalse(event.is_error)


if __name__ == "__main__":
    unittest.main()
