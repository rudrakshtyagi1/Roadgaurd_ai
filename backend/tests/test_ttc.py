"""
Unit tests for TTCEstimator.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.collision.ttc import TTCEstimator
from core.collision.types import TrackSample


class TestTTCEstimator(unittest.TestCase):

    def setUp(self):
        self.ttc_estimator = TTCEstimator()

    def test_mathematical_ttc_accuracy(self):
        # Scale s(t) = 0.20 + 0.10 * t (t in seconds).
        # At t = 1.0s, scale = 0.30, ds/dt = 0.10 -> TTC = 0.30 / 0.10 = 3.0 seconds.
        history = []
        for i in range(20):
            t_s = i * 0.05  # 0 to 1.0s
            t_ms = t_s * 1000.0
            scale = 0.20 + 0.10 * t_s
            history.append(TrackSample(t_ms, 0.5, 0.6, 0.5, 0.7, scale, scale, scale*scale, scale, 0.95))

        ttc_s, reliable, sig = self.ttc_estimator.estimate_ttc(
            history=history,
            relative_motion="APPROACHING",
            growth_rate=0.10,
            track_age_ms=1000.0,
        )

        self.assertTrue(reliable)
        self.assertIsNotNone(ttc_s)
        self.assertAlmostEqual(ttc_s, 3.0, delta=0.2)

    def test_receding_returns_none(self):
        history = [TrackSample(i*50.0, 0.5, 0.6, 0.5, 0.7, 0.3-0.01*i, 0.3, 0.09, 0.3, 0.9) for i in range(15)]
        ttc_s, reliable, _ = self.ttc_estimator.estimate_ttc(history, "RECEDING", -0.05, 750.0)
        self.assertIsNone(ttc_s)
        self.assertFalse(reliable)

    def test_insufficient_history_returns_none(self):
        history = [TrackSample(i*50.0, 0.5, 0.6, 0.5, 0.7, 0.2+0.01*i, 0.2, 0.04, 0.2, 0.9) for i in range(3)]
        ttc_s, reliable, _ = self.ttc_estimator.estimate_ttc(history, "APPROACHING", 0.05, 150.0)
        self.assertIsNone(ttc_s)
        self.assertFalse(reliable)


if __name__ == "__main__":
    unittest.main()
