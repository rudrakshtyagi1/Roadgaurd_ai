"""
Tests for core.inference_logger.InferenceLogger (stdlib unittest, no pytest dependency)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference_logger import InferenceLogger
from core.perception_event import PerceptionEvent

_REQUIRED_KEYS = {
    "ts", "monotonic_ms", "source", "event_type",
    "frame_index", "latency_ms", "confidence", "is_error", "error",
}


def _make_logger() -> tuple:
    tmp = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
    log_path = Path(tmp.name)
    tmp.close()
    return InferenceLogger(log_path=log_path), log_path


def _read(path: Path) -> list:
    text = path.read_text(encoding="utf-8").strip()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _ok_event(**kwargs) -> PerceptionEvent:
    defaults = dict(source="road_monitor", event_type="road_hazard",
                    frame_index=1, latency_ms=28.4, confidence=0.82, data={})
    defaults.update(kwargs)
    return PerceptionEvent.ok(**defaults)


class TestInferenceLoggerFromEvent(unittest.TestCase):

    def test_writes_one_line_per_event(self):
        logger, path = _make_logger()
        logger.log(_ok_event())
        self.assertEqual(len(_read(path)), 1)

    def test_all_required_keys_present(self):
        logger, path = _make_logger()
        logger.log(_ok_event())
        record = _read(path)[0]
        for key in _REQUIRED_KEYS:
            self.assertIn(key, record, msg=f"Missing key: {key}")

    def test_source_matches_event(self):
        logger, path = _make_logger()
        logger.log(_ok_event(source="driver_monitor"))
        self.assertEqual(_read(path)[0]["source"], "driver_monitor")

    def test_latency_ms_matches_event(self):
        logger, path = _make_logger()
        logger.log(_ok_event(latency_ms=15.5))
        self.assertAlmostEqual(_read(path)[0]["latency_ms"], 15.5, places=1)

    def test_confidence_matches_event(self):
        logger, path = _make_logger()
        logger.log(_ok_event(confidence=0.73))
        self.assertAlmostEqual(_read(path)[0]["confidence"], 0.73, places=3)

    def test_frame_index_matches_event(self):
        logger, path = _make_logger()
        logger.log(_ok_event(frame_index=42))
        self.assertEqual(_read(path)[0]["frame_index"], 42)

    def test_monotonic_ms_is_positive_float(self):
        logger, path = _make_logger()
        logger.log(_ok_event())
        val = _read(path)[0]["monotonic_ms"]
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0.0)

    def test_multiple_events_multiple_lines(self):
        logger, path = _make_logger()
        for i in range(5):
            logger.log(_ok_event(frame_index=i, latency_ms=float(i)))
        self.assertEqual(len(_read(path)), 5)


class TestInferenceLoggerErrorEvent(unittest.TestCase):

    def test_is_error_true(self):
        logger, path = _make_logger()
        event = PerceptionEvent.from_error(
            source="driver_monitor", frame_index=3, error=RuntimeError("cam offline")
        )
        logger.log(event)
        self.assertTrue(_read(path)[0]["is_error"])

    def test_error_message_in_log(self):
        logger, path = _make_logger()
        event = PerceptionEvent.from_error(
            source="driver_monitor", frame_index=3, error=RuntimeError("cam offline")
        )
        logger.log(event)
        self.assertIn("cam offline", _read(path)[0]["error"])

    def test_ok_event_is_error_false(self):
        logger, path = _make_logger()
        logger.log(_ok_event())
        record = _read(path)[0]
        self.assertFalse(record["is_error"])
        self.assertIsNone(record["error"])


class TestInferenceLoggerRaw(unittest.TestCase):

    def test_writes_one_line(self):
        logger, path = _make_logger()
        logger.log_raw(source="video_router", latency_ms=12.3, confidence=0.60)
        self.assertEqual(len(_read(path)), 1)

    def test_all_required_keys_present(self):
        logger, path = _make_logger()
        logger.log_raw(source="video_router", latency_ms=12.3, confidence=0.60)
        for key in _REQUIRED_KEYS:
            self.assertIn(key, _read(path)[0], msg=f"Missing: {key}")

    def test_source_correct(self):
        logger, path = _make_logger()
        logger.log_raw(source="my_module", latency_ms=1.0, confidence=0.5)
        self.assertEqual(_read(path)[0]["source"], "my_module")

    def test_with_error_sets_is_error(self):
        logger, path = _make_logger()
        logger.log_raw(source="x", latency_ms=0.0, confidence=0.0, error="oops")
        record = _read(path)[0]
        self.assertTrue(record["is_error"])
        self.assertEqual(record["error"], "oops")

    def test_without_error_is_error_false(self):
        logger, path = _make_logger()
        logger.log_raw(source="x", latency_ms=1.0, confidence=0.5)
        record = _read(path)[0]
        self.assertFalse(record["is_error"])
        self.assertIsNone(record["error"])

    def test_custom_frame_index(self):
        logger, path = _make_logger()
        logger.log_raw(source="x", latency_ms=1.0, confidence=0.5, frame_index=77)
        self.assertEqual(_read(path)[0]["frame_index"], 77)


if __name__ == "__main__":
    unittest.main()
