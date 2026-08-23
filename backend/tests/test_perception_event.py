"""
Tests for core.perception_event.PerceptionEvent (stdlib unittest, no pytest dependency)
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.perception_event import PerceptionEvent


def _ok(**kwargs) -> PerceptionEvent:
    defaults = dict(
        source="road_monitor", event_type="road_hazard",
        frame_index=1, latency_ms=28.4, confidence=0.82, data={}
    )
    defaults.update(kwargs)
    return PerceptionEvent.ok(**defaults)


class TestPerceptionEventOk(unittest.TestCase):

    def test_required_fields_populated(self):
        e = _ok(data={"pothole_detected": True})
        self.assertEqual(e.source, "road_monitor")
        self.assertEqual(e.event_type, "road_hazard")
        self.assertEqual(e.frame_index, 1)
        self.assertAlmostEqual(e.latency_ms, 28.4, places=1)
        self.assertAlmostEqual(e.confidence, 0.82, places=3)
        self.assertEqual(e.data, {"pothole_detected": True})
        self.assertIsNone(e.error)

    def test_timestamp_utc_is_iso8601_string(self):
        e = _ok()
        self.assertIn("T", e.timestamp_utc)
        self.assertTrue("+" in e.timestamp_utc or "Z" in e.timestamp_utc)

    def test_monotonic_ms_is_positive(self):
        e = _ok()
        self.assertGreater(e.monotonic_ms, 0.0)

    def test_explicit_monotonic_ms_preserved(self):
        mono = time.monotonic() * 1000.0
        e = _ok(monotonic_ms=mono)
        self.assertAlmostEqual(e.monotonic_ms, mono, places=3)

    def test_confidence_clamped_above_1(self):
        e = _ok(confidence=1.5)
        self.assertAlmostEqual(e.confidence, 1.0, places=3)

    def test_confidence_clamped_below_0(self):
        e = _ok(confidence=-0.3)
        self.assertAlmostEqual(e.confidence, 0.0, places=3)

    def test_is_error_false(self):
        e = _ok()
        self.assertFalse(e.is_error)

    def test_frozen_immutable(self):
        e = _ok()
        with self.assertRaises((AttributeError, TypeError)):
            e.source = "changed"  # type: ignore[misc]

    def test_to_dict_contains_all_keys(self):
        e = _ok(frame_index=7, data={"x": 1})
        d = e.to_dict()
        for key in ("source", "event_type", "timestamp_utc", "monotonic_ms",
                    "frame_index", "latency_ms", "confidence", "data", "error"):
            self.assertIn(key, d, msg=f"Missing key: {key}")


class TestPerceptionEventFromError(unittest.TestCase):

    def test_event_type_is_module_error(self):
        e = PerceptionEvent.from_error(
            source="driver_monitor", frame_index=5, error=RuntimeError("cam off")
        )
        self.assertEqual(e.event_type, "module_error")

    def test_error_message_captured(self):
        e = PerceptionEvent.from_error(
            source="x", frame_index=0, error=ValueError("bad shape")
        )
        self.assertIn("bad shape", e.error)

    def test_data_is_empty_dict(self):
        e = PerceptionEvent.from_error(
            source="x", frame_index=0, error=Exception("fail")
        )
        self.assertEqual(e.data, {})

    def test_is_error_true(self):
        e = PerceptionEvent.from_error(
            source="x", frame_index=0, error=Exception("fail")
        )
        self.assertTrue(e.is_error)

    def test_confidence_is_zero(self):
        e = PerceptionEvent.from_error(
            source="x", frame_index=0, error=Exception("fail")
        )
        self.assertEqual(e.confidence, 0.0)

    def test_explicit_monotonic_ms_preserved(self):
        mono = 999_000.0
        e = PerceptionEvent.from_error(
            source="x", frame_index=0, error=Exception("fail"), monotonic_ms=mono
        )
        self.assertAlmostEqual(e.monotonic_ms, mono, places=3)


if __name__ == "__main__":
    unittest.main()
