"""
RoadGuard AI — Driver State Engine V2.0 Lifecycle, Decay & Fusion Regression Tests
Verifies that stale temporal evidence decays, physiological contradiction prevents false drowsiness,
and multi-hazard RoadGuard Brain fusion is protected from false driver risk.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.driver_state_config import DriverStateConfig
from core.driver_state_engine import DriverStateEngineV2
from core.brain.fusion_engine import RoadGuardBrain


class TestDriverLifecycleAndFusionRegression(unittest.TestCase):

    def setUp(self):
        self.engine = DriverStateEngineV2()
        self.brain = RoadGuardBrain()

    def test_exact_observed_bug_awake_with_moderate_cnn(self):
        """
        Exact bug scenario from screenshot:
        EAR = 0.42, eye_state = OPEN, PERCLOS = 0.02 (reliable after 6s baseline),
        CNN drowsy = 0.64, no active closure, no active yawn.
        Engine MUST output NORMAL and must NOT include LONG_BLINK_DETECTED or SUSTAINED_YAWN.
        """
        # Feed 6 seconds of awake baseline to establish reliable low PERCLOS
        t = 1000.0
        for _ in range(180): # 6s @ 30fps
            self.engine.update(ear=0.42, mar=0.10, cnn_drowsy_prob=0.10, monotonic_ms=t)
            t += 33.33

        # Now simulate frame matching the screenshot: CNN = 0.64, EAR = 0.42
        res = self.engine.update(ear=0.42, mar=0.10, cnn_drowsy_prob=0.64, monotonic_ms=t)

        self.assertEqual(res["state"], "NORMAL")
        self.assertLess(res["fatigue_risk_score"], 0.25)
        self.assertNotIn("LONG_BLINK_DETECTED", res["reason_codes"])
        self.assertNotIn("SUSTAINED_YAWN", res["reason_codes"])
        self.assertNotIn("HIGH_PERCLOS", res["reason_codes"])
        self.assertTrue(res["wakefulness_support"] >= 0.70)
        self.assertTrue(res["signal_disagreement"])

    def test_stale_long_blink_decays_and_expires(self):
        """
        One long blink (600ms) occurs at t=2s.
        After several seconds of open eyes, the evidence weight must decay and the state recover to NORMAL.
        """
        # Baseline 2s
        t = 1000.0
        for _ in range(60):
            self.engine.update(ear=0.32, mar=0.10, monotonic_ms=t)
            t += 33.33

        # 600ms long blink
        for _ in range(18):
            self.engine.update(ear=0.08, mar=0.10, monotonic_ms=t)
            t += 33.33

        # Reopen eyes
        res_reopen = self.engine.update(ear=0.32, mar=0.10, monotonic_ms=t)
        self.assertIn("LONG_BLINK_DETECTED", res_reopen["reason_codes"])

        # Drive normally for 10 seconds
        for _ in range(300):
            t += 33.33
            res = self.engine.update(ear=0.32, mar=0.10, monotonic_ms=t)

        # After 10s of open eyes, LONG_BLINK_DETECTED must have expired from reason codes
        self.assertNotIn("LONG_BLINK_DETECTED", res["reason_codes"])
        self.assertNotIn("RECENT_LONG_BLINK", res["reason_codes"])
        self.assertEqual(res["state"], "NORMAL")

    def test_stale_yawn_decays_and_expires(self):
        """
        One sustained yawn (2.5s) occurs.
        After yawn ends and prolonged normal driving, active and recent yawn evidence must decay to zero.
        """
        t = 1000.0
        # Normal baseline 2s
        for _ in range(60):
            self.engine.update(ear=0.30, mar=0.10, monotonic_ms=t)
            t += 33.33

        # 2.5s yawn
        for _ in range(75):
            res_yawn = self.engine.update(ear=0.30, mar=0.62, monotonic_ms=t)
            t += 33.33

        self.assertIn("SUSTAINED_YAWN", res_yawn["reason_codes"])

        # Close mouth and drive normally for 15 seconds
        for _ in range(450):
            t += 33.33
            res = self.engine.update(ear=0.30, mar=0.10, monotonic_ms=t)

        # Active yawn must be False and SUSTAINED_YAWN must not be in reason codes
        self.assertFalse(res["active_signals"]["active_yawn"])
        self.assertNotIn("SUSTAINED_YAWN", res["reason_codes"])
        self.assertNotIn("ACTIVE_YAWN", res["reason_codes"])
        self.assertEqual(res["state"], "NORMAL")

    def test_repeated_long_blinks_accumulate_fatigue(self):
        """
        Repeated long blinks (3 long blinks within 15s) must escalate state to FATIGUE_RISK or DROWSY.
        """
        t = 1000.0
        for _ in range(60): # 2s baseline
            self.engine.update(ear=0.30, mar=0.10, monotonic_ms=t)
            t += 33.33

        # 3 cycles of long blink + brief pause
        for _ in range(3):
            for _ in range(16): # 530ms long blink
                self.engine.update(ear=0.08, mar=0.10, monotonic_ms=t)
                t += 33.33
            for _ in range(40): # 1.3s open
                res = self.engine.update(ear=0.30, mar=0.10, monotonic_ms=t)
                t += 33.33

        self.assertGreaterEqual(res["recent_signals"]["long_blinks_30s"], 2)
        self.assertIn(res["state"], ["FATIGUE_RISK", "DROWSY"])
        self.assertGreaterEqual(res["fatigue_risk_score"], 0.25)

    def test_yawn_frequency_comparison(self):
        """
        Multiple yawns in short succession must produce significantly higher fatigue risk than one isolated yawn.
        """
        engine_single = DriverStateEngineV2()
        engine_multi = DriverStateEngineV2()

        # Engine 1: 1 yawn
        t = 1000.0
        for _ in range(60): # 2s baseline
            engine_single.update(ear=0.30, mar=0.10, monotonic_ms=t)
            engine_multi.update(ear=0.30, mar=0.10, monotonic_ms=t)
            t += 33.33

        # Single yawn in engine 1
        for _ in range(35): # 1.1s yawn
            engine_single.update(ear=0.30, mar=0.60, monotonic_ms=t)
            t += 33.33
        # End yawn
        for _ in range(60):
            res_single = engine_single.update(ear=0.30, mar=0.10, monotonic_ms=t)
            t += 33.33

        # Multi yawns in engine 2
        t = 3000.0
        for _ in range(3):
            for _ in range(35): # 1.1s yawn
                engine_multi.update(ear=0.30, mar=0.60, monotonic_ms=t)
                t += 33.33
            for _ in range(30): # 1.0s between yawns
                engine_multi.update(ear=0.30, mar=0.10, monotonic_ms=t)
                t += 33.33
        res_multi = engine_multi.update(ear=0.30, mar=0.10, monotonic_ms=t)

        self.assertGreater(
            res_multi["recent_signals"]["effective_yawn_weight"],
            res_single["recent_signals"]["effective_yawn_weight"]
        )

    def test_cnn_noise_oscillation_does_not_flicker_state(self):
        """
        Oscillating CNN probability [0.45, 0.72, 0.51, 0.68, 0.42] with alert open eyes
        must NOT cause state flickering to DROWSY or CRITICAL.
        """
        t = 1000.0
        # 5s baseline
        for _ in range(150):
            self.engine.update(ear=0.35, mar=0.10, cnn_drowsy_prob=0.10, monotonic_ms=t)
            t += 33.33

        noisy_cnn_probs = [0.45, 0.72, 0.51, 0.68, 0.42, 0.75, 0.38, 0.69, 0.44]
        states_seen = []
        for cnn_val in noisy_cnn_probs * 3:
            for _ in range(5):
                res = self.engine.update(ear=0.35, mar=0.10, cnn_drowsy_prob=cnn_val, monotonic_ms=t)
                t += 33.33
                states_seen.append(res["state"])

        self.assertNotIn("DROWSY", states_seen)
        self.assertNotIn("CRITICAL", states_seen)

    def test_multi_hazard_brain_fusion_driver_awake_with_pothole(self):
        """
        When driver is NORMAL (fatigue low), and severe pothole is detected:
        Unified state represents the ROAD HAZARD, but must NOT contain false drowsy driver reason.
        """
        driver_data = {
            "state": "NORMAL",
            "fatigue_risk_score": 0.08,
            "cnn_drowsy_probability": 0.64, # CNN is high but driver engine output is NORMAL
            "reason_codes": ["BASELINE_NORMAL", "SIGNAL_DISAGREEMENT"],
        }
        collision_data = {
            "global_collision_state": "SAFE",
            "global_collision_risk_score": 0.05,
            "reason_codes": ["CLEAR_PATH"],
        }
        hazard_data = {
            "global_hazard_state": "CRITICAL",
            "global_hazard_risk_score": 0.85,
            "reason_codes": ["SEVERE_POTHOLE_IN_PATH"],
            "primary_hazard": {"estimated_distance_m": 8.5},
        }

        assessment = self.brain.compute(
            driver_data=driver_data,
            collision_data=collision_data,
            hazard_data=hazard_data,
            monotonic_ms=1000.0,
        )

        self.assertEqual(assessment.primary_risk_source, "ROAD_HAZARD")
        self.assertEqual(assessment.unified_state, "CRITICAL")
        self.assertNotIn("FATIGUED_DRIVER_APPROACHING_HAZARD", assessment.reason_codes)
        self.assertNotIn("DRIVER_FATIGUE", assessment.secondary_risk_sources)

    def test_multi_hazard_brain_fusion_driver_drowsy_with_pothole(self):
        """
        When driver genuinely outputs DROWSY and severe pothole is detected:
        Brain compounds risk and adds FATIGUED_DRIVER_APPROACHING_HAZARD.
        """
        driver_data = {
            "state": "DROWSY",
            "fatigue_risk_score": 0.72,
            "cnn_drowsy_probability": 0.85,
            "reason_codes": ["HIGH_PERCLOS", "FREQUENT_LONG_BLINKS"],
        }
        collision_data = {
            "global_collision_state": "SAFE",
            "global_collision_risk_score": 0.05,
            "reason_codes": ["CLEAR_PATH"],
        }
        hazard_data = {
            "global_hazard_state": "CRITICAL",
            "global_hazard_risk_score": 0.80,
            "reason_codes": ["SEVERE_POTHOLE_IN_PATH"],
            "primary_hazard": {"estimated_distance_m": 8.5},
        }

        assessment = self.brain.compute(
            driver_data=driver_data,
            collision_data=collision_data,
            hazard_data=hazard_data,
            monotonic_ms=1000.0,
        )

        self.assertEqual(assessment.unified_state, "CRITICAL")
        self.assertIn("FATIGUED_DRIVER_APPROACHING_HAZARD", assessment.reason_codes)
        self.assertIn("DRIVER_FATIGUE", assessment.secondary_risk_sources)


if __name__ == "__main__":
    unittest.main()
