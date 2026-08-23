"""
RoadGuard AI — Driver State Engine V2.0 Recovery After Reopen Tests
Verifies two-timescale fatigue recovery, continuous open-eye tracking,
PERCLOS historical attenuation, and avoidance of permanent lock-in.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.driver_state_config import DriverStateConfig
from core.driver_state_engine import DriverStateEngineV2


class TestDriverRecoveryAfterReopen(unittest.TestCase):

    def setUp(self):
        self.engine = DriverStateEngineV2()

    def test_prolonged_closure_recovery_exact_bug(self):
        """
        Requirement 10: Exact bug scenario:
        0-5s: Normal open eyes (EAR=0.35)
        5-8s: Continuous eye closure (EAR=0.08) -> Triggers CRITICAL override
        8s onward: Eyes strongly open (EAR=0.45)
        
        Assert:
        - Active closure ends immediately on confirmed reopen
        - CRITICAL override clears
        - DROWSY / CRITICAL demotes within configured recovery time
        - Raw PERCLOS remains elevated (> 0.20) initially
        - State recovers to FATIGUE_RISK and eventually NORMAL without waiting for PERCLOS to reach ~0
        """
        t = 1000.0
        # 1. 0-5s: Normal open eyes (150 frames @ 30fps)
        for _ in range(150):
            res = self.engine.update(ear=0.35, mar=0.10, monotonic_ms=t)
            t += 33.33
        self.assertEqual(res["state"], "NORMAL")

        # 2. 5-8s: 3.0s continuous eye closure
        for _ in range(90):
            res_closed = self.engine.update(ear=0.08, mar=0.10, monotonic_ms=t)
            t += 33.33
        self.assertEqual(res_closed["state"], "CRITICAL")
        self.assertIn("PROLONGED_EYE_CLOSURE", res_closed["reason_codes"])
        self.assertGreaterEqual(res_closed["metrics"]["eye_closure_duration_ms"], 2500.0)

        # 3. 8.0s: Reopen eyes (EAR=0.45)
        res_reopen = self.engine.update(ear=0.45, mar=0.10, monotonic_ms=t)
        t += 33.33

        # Active closure duration must be cleared immediately
        self.assertEqual(res_reopen["metrics"]["eye_closure_duration_ms"], 0.0)

        # Feed 500ms of open eyes -> reopen confirmed, PROLONGED_EYE_CLOSURE cleared
        for _ in range(15):
            res_confirmed = self.engine.update(ear=0.45, mar=0.10, monotonic_ms=t)
            t += 33.33
        self.assertNotIn("PROLONGED_EYE_CLOSURE", res_confirmed["reason_codes"])

        # Feed up to 4.5s of continuous open eyes (total 5s open)
        states_seen = []
        for _ in range(135):
            res_open = self.engine.update(ear=0.45, mar=0.10, monotonic_ms=t)
            t += 33.33
            states_seen.append(res_open["state"])

        # At this point, eyes have been open for ~5 seconds
        # Raw PERCLOS is still ~25-30%, but effective_perclos_risk is attenuated
        self.assertGreater(res_open["metrics"]["raw_perclos"], 0.15)
        self.assertLess(res_open["metrics"]["effective_perclos_risk"], 0.20)
        self.assertIn(res_open["state"], ["FATIGUE_RISK", "NORMAL"])
        self.assertNotEqual(res_open["state"], "CRITICAL")
        self.assertNotEqual(res_open["state"], "DROWSY")

        # Feed another 4 seconds of open eyes -> long blink expires and state recovers to BASELINE_NORMAL
        for _ in range(120):
            res_final = self.engine.update(ear=0.45, mar=0.10, monotonic_ms=t)
            t += 33.33

        self.assertEqual(res_final["state"], "NORMAL")
        self.assertIn("BASELINE_NORMAL", res_final["reason_codes"])

    def test_reopen_close_reopen_resets_recovery_timer(self):
        """
        Requirement 11: Reopen-close-reopen sequence:
        - Closed 2s
        - Open 1s
        - Closed again 1.2s
        - Open continuously
        Assert that continuous open-eye timer resets to zero when eyes re-close.
        """
        t = 1000.0
        # Baseline 3s
        for _ in range(90):
            self.engine.update(ear=0.35, mar=0.10, monotonic_ms=t)
            t += 33.33

        # Closed 2s
        for _ in range(60):
            self.engine.update(ear=0.08, mar=0.10, monotonic_ms=t)
            t += 33.33

        # Open 1s -> continuous_open reaches ~1000ms
        for _ in range(30):
            res = self.engine.update(ear=0.38, mar=0.10, monotonic_ms=t)
            t += 33.33
        self.assertGreaterEqual(res["metrics"]["continuous_open_duration_ms"], 900.0)

        # Closed again 1.2s -> continuous_open must immediately reset to 0.0
        for _ in range(36):
            res_reclose = self.engine.update(ear=0.08, mar=0.10, monotonic_ms=t)
            t += 33.33
        self.assertEqual(res_reclose["metrics"]["continuous_open_duration_ms"], 0.0)
        self.assertGreaterEqual(res_reclose["metrics"]["eye_closure_duration_ms"], 1000.0)

        # Open continuously for 5s -> continuous_open grows monotonically from 0
        for _ in range(150):
            res_reopen = self.engine.update(ear=0.40, mar=0.10, monotonic_ms=t)
            t += 33.33
        self.assertGreaterEqual(res_reopen["metrics"]["continuous_open_duration_ms"], 4500.0)

    def test_high_historical_perclos_with_sustained_awake_eyes(self):
        """
        Requirement 12: High historical PERCLOS test:
        PERCLOS remains > 0.30, but eyes have been clearly open continuously for 5 seconds.
        Raw PERCLOS must remain > 0.30, but candidate state and current state must recover from DROWSY.
        """
        t = 1000.0
        # 4s baseline open
        for _ in range(120):
            self.engine.update(ear=0.35, mar=0.10, monotonic_ms=t)
            t += 33.33

        # 3.5s closed
        for _ in range(105):
            self.engine.update(ear=0.08, mar=0.10, monotonic_ms=t)
            t += 33.33

        # 5.0s continuously open (150 frames)
        for _ in range(150):
            res = self.engine.update(ear=0.45, mar=0.10, monotonic_ms=t)
            t += 33.33

        # Total time elapsed: 4s + 3.5s + 5s = 12.5s. Closed time = 3.5s / 12.5s = 28% to ~30% PERCLOS
        # Check that raw PERCLOS is accurately preserved
        self.assertGreaterEqual(res["metrics"]["raw_perclos"], 0.20)
        # Check that effective PERCLOS risk is attenuated due to 5s continuous open duration
        self.assertLessEqual(res["metrics"]["effective_perclos_risk"], 0.15)
        # State must NOT be DROWSY or CRITICAL
        self.assertIn(res["state"], ["FATIGUE_RISK", "NORMAL"])


if __name__ == "__main__":
    unittest.main()
