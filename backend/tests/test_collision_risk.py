"""
Unit tests for CollisionRiskEngine and CollisionStateMachine (Phase 2.5 Hardened).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.collision.risk_engine import CollisionRiskEngine
from core.collision.state_machine import CollisionStateMachine
from core.collision.types import TrackSample, TrackedObject


class TestCollisionRisk(unittest.TestCase):

    def setUp(self):
        self.risk_engine = CollisionRiskEngine()
        self.state_machine = CollisionStateMachine()

    def test_insufficient_history_produces_unknown(self):
        # New track with only 100ms history
        sample = TrackSample(100.0, 0.50, 0.70, 0.50, 0.80, 0.2, 0.2, 0.04, 0.2, 0.9)
        track = TrackedObject(1, "car", 2, "NEW", [0.40, 0.6, 0.60, 0.8], 0.0, 100.0, 3, 0, [sample])

        assessment = self.risk_engine.assess_track(
            track=track,
            ego_path_relevance=0.95,
            path_relation="IN_PATH",
            relative_motion="UNKNOWN",
            motion_strength=0.0,
            growth_rate=0.0,
            ttc_seconds=None,
            ttc_reliable=False,
            ttc_signal_strength=0.0,
        )

        self.assertEqual(assessment.collision_state, "UNKNOWN")
        self.assertEqual(assessment.collision_risk_score, 0.0)
        self.assertIn("INSUFFICIENT_TRACK_HISTORY", assessment.reason_codes)

    def test_low_ttc_outside_path_suppressed(self):
        # Low TTC (1.2s) but OUT_OF_PATH (relevance = 0.10)
        sample = TrackSample(1000.0, 0.85, 0.70, 0.85, 0.80, 0.2, 0.2, 0.04, 0.2, 0.9)
        track = TrackedObject(1, "car", 2, "TRACKED", [0.75, 0.6, 0.95, 0.8], 0.0, 1000.0, 30, 0, [sample]*10)

        assessment = self.risk_engine.assess_track(
            track=track,
            ego_path_relevance=0.10,
            path_relation="OUT_OF_PATH",
            relative_motion="APPROACHING",
            motion_strength=0.90,
            growth_rate=0.15,
            ttc_seconds=1.2,
            ttc_reliable=True,
            ttc_signal_strength=0.90,
        )

        self.assertEqual(assessment.collision_state, "SAFE")
        self.assertEqual(assessment.collision_risk_score, 0.0)
        self.assertIn("ADJACENT_OBJECT_SUPPRESSED", assessment.reason_codes)

    def test_low_ttc_in_path_triggers_critical(self):
        sample = TrackSample(1000.0, 0.50, 0.70, 0.50, 0.80, 0.2, 0.2, 0.04, 0.2, 0.9)
        track = TrackedObject(1, "car", 2, "TRACKED", [0.40, 0.6, 0.60, 0.8], 0.0, 1000.0, 30, 0, [sample]*10)

        assessment = self.risk_engine.assess_track(
            track=track,
            ego_path_relevance=0.95,
            path_relation="IN_PATH",
            relative_motion="APPROACHING",
            motion_strength=0.90,
            growth_rate=0.15,
            ttc_seconds=1.2,
            ttc_reliable=True,
            ttc_signal_strength=0.90,
        )

        self.assertEqual(assessment.collision_state, "CRITICAL")
        self.assertIn("OBJECT_IN_EGO_PATH", assessment.reason_codes)
        self.assertIn("VERY_LOW_TTC", assessment.reason_codes)

    def test_state_machine_hysteresis_cooldown(self):
        sample = TrackSample(1000.0, 0.50, 0.70, 0.50, 0.80, 0.2, 0.2, 0.04, 0.2, 0.9)
        track = TrackedObject(1, "car", 2, "TRACKED", [0.40, 0.6, 0.60, 0.8], 0.0, 1000.0, 30, 0, [sample]*10)

        crit_assessment = self.risk_engine.assess_track(
            track=track, ego_path_relevance=0.95, path_relation="IN_PATH",
            relative_motion="APPROACHING", motion_strength=0.9, growth_rate=0.15,
            ttc_seconds=1.2, ttc_reliable=True, ttc_signal_strength=0.9
        )

        # Immediate escalation to CRITICAL
        state, _, _ = self.state_machine.update(crit_assessment, monotonic_ms=1000.0)
        self.assertEqual(state, "CRITICAL")

        # Threat disappears at t=1100ms -> Cooldown holds state
        state_hold, _, _ = self.state_machine.update(None, monotonic_ms=1100.0)
        self.assertEqual(state_hold, "CRITICAL")

        # After cooldown elapsed (t=3000ms) -> transitions down
        state_down, _, _ = self.state_machine.update(None, monotonic_ms=3000.0)
        self.assertIn(state_down, ["WARNING", "CAUTION", "SAFE"])


if __name__ == "__main__":
    unittest.main()
