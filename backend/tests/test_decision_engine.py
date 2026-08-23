"""
Unit tests for Intelligent Decision & Alert Prioritizer (Phase 5).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.brain.types import UnifiedRiskAssessment
from core.decision.alert_prioritizer import AlertPrioritizerEngine


class TestDecisionEngine(unittest.TestCase):

    def setUp(self):
        self.decision_engine = AlertPrioritizerEngine(debounce_cooldown_s=2.0)

    def test_collision_critical_priority(self):
        assessment = UnifiedRiskAssessment(
            unified_state="CRITICAL",
            unified_risk_score=0.95,
            primary_risk_source="COLLISION",
            secondary_risk_sources=["DRIVER_FATIGUE"],
            reason_codes=["VERY_LOW_TTC"],
            monotonic_ms=1000.0,
        )
        intervention = self.decision_engine.decide(assessment, monotonic_ms=1000.0)
        self.assertEqual(intervention.priority_level, "CRITICAL")
        self.assertEqual(intervention.alert_type, "COLLISION_IMMINENT")
        self.assertEqual(intervention.action_directive, "BRAKE IMMEDIATELY")
        self.assertEqual(intervention.audio_cue, "CRITICAL_COLLISION_ALARM")
        self.assertTrue(intervention.suppress_other_alerts)

    def test_pothole_priority_and_anti_spam(self):
        assessment = UnifiedRiskAssessment(
            unified_state="HIGH",
            unified_risk_score=0.75,
            primary_risk_source="ROAD_HAZARD",
            reason_codes=["HIGH_SEVERITY_POTHOLE"],
            monotonic_ms=1000.0,
        )
        # First trigger -> Plays audio
        int1 = self.decision_engine.decide(assessment, monotonic_ms=1000.0)
        self.assertEqual(int1.alert_type, "POTHOLE_AHEAD")
        self.assertEqual(int1.audio_cue, "POTHOLE_ALERT_CHIME")

        # Second trigger 500ms later -> Visual retained, but audio debounced (NONE)
        int2 = self.decision_engine.decide(assessment, monotonic_ms=1500.0)
        self.assertEqual(int2.alert_type, "POTHOLE_AHEAD")
        self.assertEqual(int2.audio_cue, "NONE")
        self.assertEqual(int2.visual_cue, "FLASH_AMBER")

    def test_clear_road_standby(self):
        assessment = UnifiedRiskAssessment(
            unified_state="SAFE",
            unified_risk_score=0.0,
            primary_risk_source="NONE",
            reason_codes=["ALL_SYSTEMS_NORMAL"],
            monotonic_ms=1000.0,
        )
        intervention = self.decision_engine.decide(assessment, monotonic_ms=1000.0)
        self.assertEqual(intervention.priority_level, "NONE")
        self.assertEqual(intervention.alert_type, "STANDBY")
        self.assertEqual(intervention.visual_cue, "STEADY_GREEN")
        self.assertEqual(intervention.audio_cue, "NONE")


if __name__ == "__main__":
    unittest.main()
