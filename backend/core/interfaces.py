"""
RoadGuard AI — Perception Module Interfaces
Abstract Protocol contracts for all current and future detector modules.

Usage
-----
Implement any of these Protocols to register a new perception module with the
PipelineRunner.  The runner only calls ``process(frame)``, so any class that
defines that method with the correct signature is a valid detector — no
inheritance required.

    class MyNewDetector:
        def process(self, frame: np.ndarray) -> dict:
            ...

When passing to PipelineRunner.register(), also supply the module's source
name and default event_type so the runner can wrap the raw dict in a
PerceptionEvent without coupling to each module's internals.

Note on typing
--------------
Python's ``typing.Protocol`` (PEP 544) is used here for structural subtyping.
``runtime_checkable=True`` allows ``isinstance()`` checks in tests without
requiring explicit subclassing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DriverStateDetector(Protocol):
    """
    Structural contract for driver-monitoring modules.

    A conforming class must expose a ``process`` method that accepts a single
    BGR numpy frame and returns a dict with at least the canonical keys used
    by the rest of the pipeline.

    Canonical output keys (all optional, pipeline is defensive):
        ear : float            Eye Aspect Ratio [0, 1]
        mar : float            Mouth Aspect Ratio [0, 1]
        head_pose : dict       {yaw, pitch, roll}  in degrees
        eye_state : str        "OPEN" | "CLOSING" | "CLOSED"
        face_detected : bool
        drowsy_probability : float   raw CNN P(Drowsy) [0, 1]
        smoothed_fatigue : float     temporal-buffer smoothed value
        state : str            "ALERT" | "LOW_VIGILANCE" | "DROWSY"
        latency_ms : float
    """

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        ...


@runtime_checkable
class VehicleDetector(Protocol):
    """
    Structural contract for vehicle-detection modules.

    Canonical output keys:
        vehicles : int                  count of detected vehicles
        detections : list[dict]         per-vehicle {class_name, confidence, bbox, distance_m}
        nearest_distance_m : float
        frame_data : str                base64 annotated frame (optional)
        latency_ms : float
    """

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        ...


@runtime_checkable
class RoadHazardDetector(Protocol):
    """
    Structural contract for road-hazard detection modules (potholes, debris, etc.).

    Canonical output keys:
        pothole_detected : bool
        pothole_confidence : float      max confidence across detections
        potholes : list[dict]           per-pothole {confidence, severity, distance_m}
        hazard_score : float            composite score [0, 100]
        nearest_distance_m : float
        relative_proximity : str        "FAR" | "MEDIUM" | "NEAR"
        frame_data : str                base64 annotated frame
        latency_ms : float
    """

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        ...
