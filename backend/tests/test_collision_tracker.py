"""
Unit tests for MultiObjectTracker.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.collision.tracker import MultiObjectTracker, compute_iou
from core.collision.types import ObjectDetection


class TestCollisionTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = MultiObjectTracker()

    def test_compute_iou(self):
        boxA = [0.1, 0.1, 0.5, 0.5]
        boxB = [0.1, 0.1, 0.5, 0.5]
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)

        boxC = [0.6, 0.6, 0.9, 0.9]
        self.assertEqual(compute_iou(boxA, boxC), 0.0)

    def test_track_creation_and_lifecycle(self):
        # Frame 1: New detection -> NEW track
        det1 = ObjectDetection("car", 2, 0.9, [0.4, 0.5, 0.6, 0.7], [0.5, 0.6], 0.2, 0.2, 0.04)
        tracks1 = self.tracker.update([det1], monotonic_ms=100.0)
        self.assertEqual(len(tracks1), 1)
        self.assertEqual(tracks1[0].tracking_state, "NEW")
        tid = tracks1[0].track_id

        # Frame 2: Same detection
        tracks2 = self.tracker.update([det1], monotonic_ms=133.3)
        self.assertEqual(tracks2[0].track_id, tid)
        self.assertEqual(tracks2[0].frames_seen, 2)

        # Frame 3: Hits reached -> TRACKED
        tracks3 = self.tracker.update([det1], monotonic_ms=166.6)
        self.assertEqual(tracks3[0].tracking_state, "TRACKED")

    def test_temporary_occlusion_and_recovery(self):
        det = ObjectDetection("car", 2, 0.9, [0.4, 0.5, 0.6, 0.7], [0.5, 0.6], 0.2, 0.2, 0.04)
        # Establish track
        for i in range(4):
            self.tracker.update([det], monotonic_ms=i * 33.3)

        # Miss 2 frames -> DEGRADED
        self.tracker.update([], monotonic_ms=150.0)
        tracks = self.tracker.update([], monotonic_ms=183.3)
        self.assertEqual(tracks[0].tracking_state, "DEGRADED")
        self.assertEqual(tracks[0].frames_missing, 2)

        # Reacquired -> TRACKED with same track ID
        recovered = self.tracker.update([det], monotonic_ms=216.6)
        self.assertEqual(recovered[0].tracking_state, "TRACKED")
        self.assertEqual(recovered[0].frames_missing, 0)
        self.assertEqual(recovered[0].track_id, 1)


if __name__ == "__main__":
    unittest.main()
