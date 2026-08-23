"""
RoadGuard AI — Central Real-Time Pipeline Runner
Coordinates multiple perception modules on each video frame tick.

Design principles
-----------------
1. **Module isolation** — if any detector's ``process()`` raises, the exception
   is caught, logged as a ``PerceptionEvent`` with ``is_error=True``, and the
   remaining modules continue unaffected.  A single faulty detector never kills
   the whole pipeline.

2. **Ordering preserved** — events are returned in registration order so
   downstream consumers (inference_loop, tests) can rely on stable indexing.

3. **No frame copying** — the same numpy frame reference is passed to every
   module.  Modules that draw on the frame (e.g. road_monitor bounding boxes)
   operate in-place; register draw-first modules before analytics-only ones if
   isolation matters.

4. **Monotonic timing** — ``monotonic_ms`` in each PerceptionEvent uses
   ``time.monotonic()`` captured at the start of each module call for
   precision.  The frame-level start is also captured so downstream code can
   compute pipeline-level latency without re-calling the clock.

Usage
-----
    runner = PipelineRunner(logger=InferenceLogger())

    runner.register("road_monitor", road_monitor_instance,
                    event_type="road_hazard",
                    confidence_key="pothole_confidence")

    runner.register("driver_monitor", driver_monitor_instance,
                    event_type="driver_state",
                    confidence_key="drowsy_probability")

    # In async inference loop:
    events, frame_mono_ms = runner.tick(frame, frame_index=n)
    road_event  = runner.get(events, "road_monitor")
    driver_event = runner.get(events, "driver_monitor")
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from core.inference_logger import InferenceLogger
from core.perception_event import PerceptionEvent


@dataclass
class _ModuleEntry:
    """Internal descriptor for a registered perception module."""
    source: str
    module: Any                   # any object with .process(frame) -> dict
    event_type: str
    confidence_key: str           # key in the module's output dict to use as confidence
    preprocess: Optional[Callable[[np.ndarray], np.ndarray]]  # optional frame transform


class PipelineRunner:
    """
    Thin synchronous pipeline that calls each registered detector in order,
    wraps their raw dict output into PerceptionEvents, and logs each event.

    Parameters
    ----------
    logger : InferenceLogger, optional
        If omitted, a default InferenceLogger writing to ``logs/inference.log``
        is created automatically.
    """

    def __init__(self, logger: Optional[InferenceLogger] = None) -> None:
        self._modules: list[_ModuleEntry] = []
        self._logger = logger or InferenceLogger()

    # ------------------------------------------------------------------
    # Module registration
    # ------------------------------------------------------------------

    def register(
        self,
        source: str,
        module: Any,
        event_type: str = "detection",
        confidence_key: str = "confidence",
        preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """
        Register a perception module with the pipeline.

        Parameters
        ----------
        source : str
            Unique module name, e.g. ``"road_monitor"``.
        module : object
            Any object with a ``process(frame: np.ndarray) -> dict`` method.
        event_type : str
            Semantic label placed in the emitted PerceptionEvent.
        confidence_key : str
            Key to extract from the module's output dict to populate
            ``PerceptionEvent.confidence``.  Defaults to ``"confidence"``.
        preprocess : callable, optional
            Optional transform applied to the frame before passing to this
            module.  Useful for resizing, colour-space conversion, etc.
        """
        entry = _ModuleEntry(
            source=source,
            module=module,
            event_type=event_type,
            confidence_key=confidence_key,
            preprocess=preprocess,
        )
        self._modules.append(entry)

    def unregister(self, source: str) -> None:
        """Remove a module by source name (idempotent)."""
        self._modules = [m for m in self._modules if m.source != source]

    # ------------------------------------------------------------------
    # Per-frame tick
    # ------------------------------------------------------------------

    def tick(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> tuple[list[PerceptionEvent], float]:
        """
        Run all registered modules on the current frame.

        For each module:
        - If ``process()`` succeeds → wrap result in ``PerceptionEvent.ok()``.
        - If ``process()`` raises → wrap exception in ``PerceptionEvent.from_error()``.
        - Either way, the event is logged and appended to the results list.
        - The NEXT module always runs regardless of the previous module's outcome.

        Parameters
        ----------
        frame : np.ndarray
            BGR video frame (may be annotated in-place by draw-first modules).
        frame_index : int
            Monotonically increasing counter from the caller's loop.

        Returns
        -------
        events : list[PerceptionEvent]
            One event per registered module, in registration order.
        frame_monotonic_ms : float
            ``time.monotonic() * 1000`` captured at the start of this tick.
            Useful for computing overall pipeline latency in the calling loop.
        """
        frame_monotonic_ms = time.monotonic() * 1000.0
        events: list[PerceptionEvent] = []

        for entry in self._modules:
            t0 = time.perf_counter()
            module_mono = time.monotonic() * 1000.0

            try:
                # Apply optional per-module frame transform
                input_frame = entry.preprocess(frame) if entry.preprocess else frame

                raw: dict[str, Any] = entry.module.process(input_frame)

                latency_ms = (time.perf_counter() - t0) * 1000.0
                confidence = float(raw.get(entry.confidence_key, 0.0) or 0.0)

                event = PerceptionEvent.ok(
                    source=entry.source,
                    event_type=entry.event_type,
                    frame_index=frame_index,
                    latency_ms=latency_ms,
                    confidence=confidence,
                    data=raw,
                    monotonic_ms=module_mono,
                )

            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"[PipelineRunner] ⚠ Module '{entry.source}' raised on frame "
                    f"{frame_index}: {exc}\n{traceback.format_exc()}",
                    flush=True,
                )
                event = PerceptionEvent.from_error(
                    source=entry.source,
                    frame_index=frame_index,
                    error=exc,
                    monotonic_ms=module_mono,
                )

            self._logger.log(event)
            events.append(event)

        return events, frame_monotonic_ms

    # ------------------------------------------------------------------
    # Convenience query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get(events: list[PerceptionEvent], source: str) -> Optional[PerceptionEvent]:
        """Return the first event matching ``source``, or None."""
        for e in events:
            if e.source == source:
                return e
        return None

    @staticmethod
    def successful(events: list[PerceptionEvent]) -> list[PerceptionEvent]:
        """Return only non-error events."""
        return [e for e in events if not e.is_error]

    @staticmethod
    def errors(events: list[PerceptionEvent]) -> list[PerceptionEvent]:
        """Return only error events."""
        return [e for e in events if e.is_error]

    def registered_sources(self) -> list[str]:
        """Return names of all registered modules in registration order."""
        return [m.source for m in self._modules]
