"""
Unit tests for RelativeMotionEstimator.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.collision.motion import RelativeMotionEstimator
from core.collision.types import TrackSample


class TestRelativeMotion(unittest.TestCase):

    def setUp(self):
        self.estimator = RelativeMotionEstimator()

    def test_approaching_trajectory(self):
        history = []
        for i in range(15):
            t_ms = i * 50.0
            scale = 0.10 + 0.01 * (i / 5.0)  # expanding scale
            history.append(TrackSample(t_ms, 0.5, 0.6, 0.5, 0.7, scale, scale, scale*scale, scale, 0.9))

        motion, strength, growth, rel = self.estimator.estimate_motion(history, current_mono_ms=750.0)
        self.assertEqual(motion, "APPROACHING")
        self.assertGreater(growth, 0.0)
        self.assertGreater(rel, 0.5)

    def test_receding_trajectory(self):
        history = []
        for i in range(15):
            t_ms = i * 50.0
            scale = 0.30 - 0.01 * (i / 5.0)  # shrinking scale
            history.append(TrackSample(t_ms, 0.5, 0.6, 0.5, 0.7, scale, scale, scale*scale, scale, 0.9))

        motion, strength, growth, rel = self.estimator.estimate_motion(history, current_mono_ms=750.0)
        self.assertEqual(motion, "RECEDING")
        self.assertLess(growth, 0.0)

    def test_stable_trajectory(self):
        history = []
        for i in range(15):
            t_ms = i * 50.0
            scale = 0.20
            history.append(TrackSample(t_ms, 0.5, 0.6, 0.5, 0.7, scale, scale, scale*scale, scale, 0.9))

        motion, strength, growth, rel = self.estimator.estimate_motion(history, current_mono_ms=750.0)
        self.assertEqual(motion, "STABLE")


if __name__ == "__main__":
    unittest.main()
