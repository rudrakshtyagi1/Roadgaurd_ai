"""
Validation suite running all CollisionBenchmark scenarios (Phase 2.5 Hardened).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.collision_benchmark import CollisionBenchmark


class TestCollisionBenchmarkSuite(unittest.TestCase):

    def test_all_14_collision_benchmarks_pass(self):
        bm = CollisionBenchmark()
        reports = bm.run_all_benchmarks()

        self.assertEqual(len(reports), 14)
        for r in reports:
            self.assertTrue(
                r.passed,
                f"Collision benchmark '{r.scenario_name}' FAILED!\n"
                f"Reasons: {r.failure_reasons}\n"
                f"Observed states: {r.states_observed}\n"
                f"False alerts/min: {r.false_alerts_per_minute}\n"
                f"False critical time: {r.false_critical_time_ms}ms"
            )

    def test_scale_signal_ablation(self):
        """Verify scale signal ablation executes and provides comparative metrics."""
        results = CollisionBenchmark.run_scale_signal_ablation()
        self.assertIn("fused_scale", results)
        self.assertIn("sqrt_area", results)
        self.assertIn("bbox_height", results)
        self.assertLess(results["fused_scale"]["mae_s"], 0.75)


if __name__ == "__main__":
    unittest.main()
