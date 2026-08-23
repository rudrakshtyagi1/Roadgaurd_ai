"""
Tests for core.interfaces Protocols and pipeline.runner.PipelineRunner
(stdlib unittest, no pytest dependency)
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference_logger import InferenceLogger
from core.interfaces import DriverStateDetector, RoadHazardDetector, VehicleDetector
from core.perception_event import PerceptionEvent
from pipeline.runner import PipelineRunner


# ---------------------------------------------------------------------------
# Mock detectors
# ---------------------------------------------------------------------------

class MockDriverDetector:
    def process(self, frame: np.ndarray) -> dict:
        return {
            "ear": 0.30, "mar": 0.10,
            "head_pose": {"yaw": 2.0, "pitch": 1.0, "roll": 0.0},
            "eye_state": "OPEN", "face_detected": True,
            "drowsy_probability": 0.15, "smoothed_fatigue": 0.12,
            "state": "ALERT", "latency_ms": 3.1,
        }


class MockRoadDetector:
    def process(self, frame: np.ndarray) -> dict:
        return {
            "pothole_detected": True, "pothole_confidence": 0.78,
            "potholes": [{"confidence": 0.78, "severity": "MEDIUM", "distance_m": 15.0}],
            "hazard_score": 42.0, "nearest_distance_m": 15.0,
            "relative_proximity": "MEDIUM", "frame_data": "",
        }


class MockVehicleDetector:
    def process(self, frame: np.ndarray) -> dict:
        return {"vehicles": 2, "detections": [], "nearest_distance_m": 30.0, "frame_data": ""}


class FaultyDetector:
    def process(self, frame: np.ndarray) -> dict:
        raise RuntimeError("simulated hardware failure")


_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _make_runner() -> PipelineRunner:
    tmp = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
    logger = InferenceLogger(log_path=Path(tmp.name))
    return PipelineRunner(logger=logger)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance(unittest.TestCase):

    def test_mock_driver_satisfies_driver_state_detector(self):
        self.assertIsInstance(MockDriverDetector(), DriverStateDetector)

    def test_mock_road_satisfies_road_hazard_detector(self):
        self.assertIsInstance(MockRoadDetector(), RoadHazardDetector)

    def test_mock_vehicle_satisfies_vehicle_detector(self):
        self.assertIsInstance(MockVehicleDetector(), VehicleDetector)

    def test_faulty_satisfies_protocol(self):
        # Protocol checks signature existence, not success
        self.assertIsInstance(FaultyDetector(), RoadHazardDetector)

    def test_class_without_process_fails_protocol(self):
        class NoProcess:
            pass
        self.assertNotIsInstance(NoProcess(), DriverStateDetector)

    def test_minimal_class_with_process_satisfies(self):
        class Minimal:
            def process(self, frame): return {}
        self.assertIsInstance(Minimal(), DriverStateDetector)


# ---------------------------------------------------------------------------
# Normal operation
# ---------------------------------------------------------------------------

class TestPipelineRunnerNormal(unittest.TestCase):

    def test_two_modules_produce_two_events(self):
        r = _make_runner()
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        r.register("driver_monitor", MockDriverDetector(), event_type="driver_state", confidence_key="drowsy_probability")
        events, frame_mono = r.tick(_FRAME, frame_index=1)
        self.assertEqual(len(events), 2)
        self.assertGreater(frame_mono, 0.0)

    def test_event_source_correct(self):
        r = _make_runner()
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=1)
        e = PipelineRunner.get(events, "road_monitor")
        self.assertIsNotNone(e)
        self.assertEqual(e.source, "road_monitor")

    def test_confidence_extracted_by_key(self):
        r = _make_runner()
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=1)
        e = PipelineRunner.get(events, "road_monitor")
        self.assertAlmostEqual(e.confidence, 0.78, places=3)

    def test_data_dict_is_full_output(self):
        r = _make_runner()
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=1)
        e = PipelineRunner.get(events, "road_monitor")
        self.assertTrue(e.data["pothole_detected"])
        self.assertAlmostEqual(e.data["hazard_score"], 42.0)

    def test_frame_index_propagated(self):
        r = _make_runner()
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=99)
        self.assertEqual(events[0].frame_index, 99)

    def test_latency_ms_positive(self):
        r = _make_runner()
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=1)
        self.assertGreaterEqual(events[0].latency_ms, 0.0)

    def test_registered_sources_ordered(self):
        r = _make_runner()
        r.register("a", MockRoadDetector(), event_type="t", confidence_key="pothole_confidence")
        r.register("b", MockDriverDetector(), event_type="t", confidence_key="drowsy_probability")
        self.assertEqual(r.registered_sources(), ["a", "b"])


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------

class TestPipelineRunnerIsolation(unittest.TestCase):

    def test_faulty_first_does_not_stop_second(self):
        r = _make_runner()
        r.register("faulty", FaultyDetector(), event_type="t", confidence_key="x")
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=1)
        self.assertEqual(len(events), 2)

    def test_faulty_event_is_error(self):
        r = _make_runner()
        r.register("faulty", FaultyDetector(), event_type="t", confidence_key="x")
        events, _ = r.tick(_FRAME, frame_index=1)
        e = PipelineRunner.get(events, "faulty")
        self.assertTrue(e.is_error)
        self.assertIn("simulated hardware failure", e.error)

    def test_second_module_succeeds_after_faulty_first(self):
        r = _make_runner()
        r.register("faulty", FaultyDetector(), event_type="t", confidence_key="x")
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=1)
        road_e = PipelineRunner.get(events, "road_monitor")
        self.assertFalse(road_e.is_error)

    def test_faulty_event_data_empty(self):
        r = _make_runner()
        r.register("faulty", FaultyDetector(), event_type="t", confidence_key="x")
        events, _ = r.tick(_FRAME, frame_index=1)
        self.assertEqual(PipelineRunner.get(events, "faulty").data, {})

    def test_all_faulty_all_errors(self):
        r = _make_runner()
        r.register("f1", FaultyDetector(), event_type="t", confidence_key="x")
        r.register("f2", FaultyDetector(), event_type="t", confidence_key="x")
        events, _ = r.tick(_FRAME, frame_index=1)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e.is_error for e in events))

    def test_successful_and_errors_filter(self):
        r = _make_runner()
        r.register("faulty", FaultyDetector(), event_type="t", confidence_key="x")
        r.register("road_monitor", MockRoadDetector(), event_type="road_hazard", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=1)
        self.assertEqual(len(PipelineRunner.successful(events)), 1)
        self.assertEqual(len(PipelineRunner.errors(events)), 1)

    def test_get_returns_none_for_unknown_source(self):
        r = _make_runner()
        r.register("road_monitor", MockRoadDetector(), event_type="t", confidence_key="pothole_confidence")
        events, _ = r.tick(_FRAME, frame_index=1)
        self.assertIsNone(PipelineRunner.get(events, "nonexistent"))

    def test_unregister_removes_module(self):
        r = _make_runner()
        r.register("road_monitor", MockRoadDetector(), event_type="t", confidence_key="pothole_confidence")
        r.unregister("road_monitor")
        events, _ = r.tick(_FRAME, frame_index=1)
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
