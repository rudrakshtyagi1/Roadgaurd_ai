"""
Unit tests for Edge AI (Phase 7) and Data Flywheel (Phase 8).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.edge.scheduler import AdaptivePipelineScheduler
from core.edge.profiler import EdgeProfiler
from core.flywheel.hard_case_miner import HardCaseMiner
from core.flywheel.ablation_suite import FullAblationSuite


class TestEdgeAndFlywheel(unittest.TestCase):

    def test_adaptive_scheduler_multi_rate(self):
        sched = AdaptivePipelineScheduler(base_fps=30.0)
        d1 = sched.schedule_tick()
        d2 = sched.schedule_tick()
        d3 = sched.schedule_tick()

        # Driver always runs
        self.assertTrue(d1.run_driver)
        self.assertTrue(d2.run_driver)
        self.assertTrue(d3.run_driver)

        # Vehicle runs on frame 2 (15Hz), Hazard on frame 3 (10Hz)
        self.assertTrue(d2.run_vehicle_detector)
        self.assertTrue(d3.run_hazard_detector)

    def test_edge_profiler_telemetry(self):
        prof = EdgeProfiler()
        for _ in range(10):
            telemetry = prof.record_frame(latency_ms=18.5)
        self.assertGreater(telemetry.rolling_fps, 20.0)
        self.assertEqual(telemetry.frame_count, 10)
        self.assertGreater(telemetry.memory_rss_mb, 0.0)

    def test_hard_case_miner(self):
        miner = HardCaseMiner()
        # Marginal confidence detection should be mined
        case = miner.inspect_and_mine(
            event_source="pothole_detector",
            confidence=0.32,
            state="SAFE",
            metadata={"box": [0.5, 0.6, 0.6, 0.7]},
        )
        self.assertIsNotNone(case)
        self.assertEqual(case.category, "LOW_CONFIDENCE")
        self.assertEqual(len(miner.get_mined_cases()), 1)

    def test_full_ablation_suite(self):
        results = FullAblationSuite.run_collision_ablation()
        self.assertEqual(len(results), 5)
        # Full engine should have 0.0 false alerts
        self.assertEqual(results[-1].false_alert_rate_per_min, 0.0)
        self.assertEqual(results[-1].continuity_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
