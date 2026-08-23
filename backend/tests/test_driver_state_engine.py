"""
Unit tests and benchmark validation for Driver State Engine V2.0.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.driver_benchmark import DriverBenchmark
from core.driver_state_config import DriverStateConfig
from core.driver_state_engine import DriverStateEngineV2
from core.perception_event import PerceptionEvent


class TestDriverStateEngineV2(unittest.TestCase):

    def setUp(self):
        self.config = DriverStateConfig.from_yaml()
        self.engine = DriverStateEngineV2(config=self.config)

    def test_normal_blinking_maintains_normal(self):
        """Physiological blinks (140ms every 3s) must keep state NORMAL."""
        t_ms = 0.0
        dt_ms = 33.33
        for i in range(300):
            t_ms += dt_ms
            is_blink = (t_ms % 3000.0) < 140.0
            ear = 0.08 if is_blink else 0.32
            res = self.engine.update(ear=ear, mar=0.10, face_detected=True, monotonic_ms=t_ms)
            self.assertEqual(res["state"], "NORMAL", f"Failed at t={t_ms}ms with state={res['state']}")

        self.assertGreater(res["metrics"]["total_blink_count"], 0)
        self.assertLess(res["metrics"]["avg_blink_duration_ms"], 400.0)

    def test_prolonged_eye_closure_triggers_critical(self):
        """Eyes closed >= 1500ms must immediately trigger CRITICAL with PROLONGED_EYE_CLOSURE."""
        t_ms = 0.0
        dt_ms = 33.33
        for _ in range(150):
            t_ms += dt_ms
            self.engine.update(ear=0.30, face_detected=True, monotonic_ms=t_ms)

        closure_states = []
        for _ in range(55):  # ~1833ms
            t_ms += dt_ms
            res = self.engine.update(ear=0.08, face_detected=True, monotonic_ms=t_ms)
            closure_states.append(res["state"])

        self.assertIn("CRITICAL", closure_states)
        self.assertEqual(res["state"], "CRITICAL")
        self.assertIn("PROLONGED_EYE_CLOSURE", res["reason_codes"])

    def test_repeated_long_blinks_escalate_state(self):
        """Repeated 650ms long blinks must escalate state to FATIGUE_RISK or DROWSY."""
        t_ms = 0.0
        dt_ms = 33.33
        observed_states = []
        for i in range(450):  # 15 seconds
            t_ms += dt_ms
            cycle_pos = t_ms % 3500.0
            ear = 0.08 if cycle_pos < 650.0 else 0.32
            res = self.engine.update(ear=ear, face_detected=True, monotonic_ms=t_ms)
            if res["state"] not in observed_states:
                observed_states.append(res["state"])

        self.assertTrue("FATIGUE_RISK" in observed_states or "DROWSY" in observed_states)
        self.assertGreaterEqual(res["metrics"]["long_blink_count"], 2)

    def test_sustained_yawn_triggers_fatigue_risk(self):
        """Sustained yawn >= 2.0s must trigger FATIGUE_RISK with SUSTAINED_YAWN."""
        t_ms = 0.0
        dt_ms = 33.33
        for _ in range(150):
            t_ms += dt_ms
            self.engine.update(ear=0.30, mar=0.10, face_detected=True, monotonic_ms=t_ms)

        for _ in range(75):
            t_ms += dt_ms
            res = self.engine.update(ear=0.30, mar=0.65, face_detected=True, monotonic_ms=t_ms)

        self.assertEqual(res["state"], "FATIGUE_RISK")
        self.assertIn("SUSTAINED_YAWN", res["reason_codes"])

    def test_head_nod_alone_capped_at_fatigue_risk(self):
        """Head nod alone without eye closure must NEVER trigger DROWSY or CRITICAL."""
        t_ms = 0.0
        dt_ms = 33.33
        observed_states = []
        for i in range(180):  # 6 seconds
            t_ms += dt_ms
            pitch = -25.0 if (2000.0 <= t_ms <= 5000.0) else 0.0
            res = self.engine.update(ear=0.30, mar=0.10, head_pose={"yaw": 0.0, "pitch": pitch, "roll": 0.0},
                                     face_detected=True, monotonic_ms=t_ms)
            if res["state"] not in observed_states:
                observed_states.append(res["state"])

        self.assertNotIn("CRITICAL", observed_states)
        self.assertNotIn("DROWSY", observed_states)

    def test_head_nod_with_eye_closure_triggers_critical(self):
        """Head nod corroborated with closed eyes must trigger CRITICAL."""
        t_ms = 0.0
        dt_ms = 33.33
        observed_states = []
        for i in range(180):
            t_ms += dt_ms
            is_event = (2000.0 <= t_ms <= 4500.0)
            pitch = -25.0 if is_event else 0.0
            ear = 0.08 if is_event else 0.30
            res = self.engine.update(ear=ear, mar=0.10, head_pose={"yaw": 0.0, "pitch": pitch, "roll": 0.0},
                                     face_detected=True, monotonic_ms=t_ms)
            if res["state"] not in observed_states:
                observed_states.append(res["state"])

        self.assertIn("CRITICAL", observed_states)

    def test_noisy_boundary_ear_does_not_create_fake_blinks(self):
        """Borderline oscillating EAR values around 0.20 must not generate multiple fake blinks."""
        t_ms = 0.0
        dt_ms = 33.33
        for i in range(100):
            t_ms += dt_ms
            noisy_ear = 0.198 if (i % 2 == 0) else 0.205
            res = self.engine.update(ear=noisy_ear, face_detected=True, monotonic_ms=t_ms)

        # Hysteresis requires EAR >= 0.25 to count as reopened; no blinks completed
        self.assertEqual(res["metrics"]["total_blink_count"], 0)

    def test_missing_face_observations_sets_tracking_lost(self):
        """Face loss must set tracking_state=LOST without triggering false drowsiness."""
        t_ms = 0.0
        dt_ms = 33.33
        for _ in range(90):
            t_ms += dt_ms
            self.engine.update(ear=0.30, face_detected=True, monotonic_ms=t_ms)

        for _ in range(90):
            t_ms += dt_ms
            res = self.engine.update(ear=0.0, face_detected=False, monotonic_ms=t_ms)

        self.assertEqual(res["tracking_state"], "LOST")
        self.assertIn("TRACKING_LOST", res["reason_codes"])
        self.assertNotEqual(res["state"], "CRITICAL")
        self.assertFalse(res["metrics"]["perclos_reliable"])

    def test_startup_protection_disables_false_perclos(self):
        """Closed eyes for 400ms during first 2s must not produce PERCLOS >= 0.5 or trigger CRITICAL."""
        t_ms = 0.0
        dt_ms = 33.33
        states = []
        for i in range(60):
            t_ms += dt_ms
            ear = 0.10 if (300.0 <= t_ms <= 700.0) else 0.30
            res = self.engine.update(ear=ear, face_detected=True, monotonic_ms=t_ms)
            states.append(res["state"])

        self.assertNotIn("CRITICAL", states)
        self.assertFalse(res["metrics"]["perclos_reliable"])

    def test_perception_event_generation(self):
        """Engine output converts to valid PerceptionEvent."""
        res = self.engine.update(ear=0.30, mar=0.10, face_detected=True, monotonic_ms=1000.0)
        event = self.engine.to_perception_event(res, frame_index=42)

        self.assertIsInstance(event, PerceptionEvent)
        self.assertEqual(event.source, "driver_state_engine_v2")
        self.assertEqual(event.event_type, "driver_state")
        self.assertEqual(event.frame_index, 42)
        self.assertFalse(event.is_error)
        self.assertIn("state", event.data)
        self.assertIn("reason_codes", event.data)


class TestDriverBenchmarkValidationSuite(unittest.TestCase):

    def test_all_20_benchmarks_pass_with_invariants(self):
        """Run all 20 benchmark scenarios and verify mathematical invariants and expectations."""
        benchmark = DriverBenchmark()
        reports = benchmark.run_all_benchmarks()

        self.assertEqual(len(reports), 20)
        for r in reports:
            # 1. Non-negative timing invariant assertion
            if r.recovery_delay_ms is not None:
                self.assertGreaterEqual(
                    r.recovery_delay_ms, 0.0,
                    f"Invariant violated in {r.scenario_name}: recovery_delay_ms={r.recovery_delay_ms} < 0"
                )
            if r.detection_delay_ms is not None:
                self.assertGreaterEqual(
                    r.detection_delay_ms, 0.0,
                    f"Invariant violated in {r.scenario_name}: detection_delay_ms={r.detection_delay_ms} < 0"
                )

            # 2. Scenario pass assertion
            self.assertTrue(
                r.passed,
                f"Benchmark scenario '{r.scenario_name}' FAILED!\n"
                f"Reasons: {r.failure_reasons}\n"
                f"Observed states: {r.states_observed}\n"
                f"False alerts/min: {r.false_alerts_per_minute}\n"
                f"False critical time: {r.false_critical_time_ms}ms"
            )


if __name__ == "__main__":
    unittest.main()
