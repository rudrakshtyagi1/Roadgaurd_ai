"""
RoadGuard AI — PerceptionEvent
Common timestamped output schema shared by all perception modules.

Design notes:
  - source / event_type are plain strings for now. Enums can be introduced
    later without a breaking schema change.
  - monotonic_ms uses time.monotonic() which is clock-drift-free. This is the
    reliable basis for TTC / temporal fusion in Phase 2+.
  - timestamp_utc is retained for log readability and cross-system correlation.
  - The dataclass is frozen so events are treated as immutable value objects
    once produced by a detector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class PerceptionEvent:
    """
    Immutable output container for a single perception module inference call.

    Fields
    ------
    source : str
        Name of the producing module, e.g. ``"road_monitor"`` or ``"driver_monitor"``.
    event_type : str
        Semantic label for the event, e.g. ``"pothole_detected"``, ``"driver_alert"``.
    timestamp_utc : str
        ISO-8601 UTC wall-clock timestamp at the moment of event construction.
    monotonic_ms : float
        ``time.monotonic() * 1000`` at event construction.  Use this for all
        latency measurements, TTC, and temporal-fusion calculations.  Wall-clock
        timestamps can drift; monotonic timestamps cannot.
    frame_index : int
        Monotonically increasing frame counter from the pipeline runner.
        Allows downstream consumers to correlate events from the same tick.
    latency_ms : float
        Elapsed time in milliseconds for the detector's ``process()`` call.
    confidence : float
        Primary confidence scalar in [0.0, 1.0].  Module-specific interpretation:
        e.g. YOLO detection confidence for road hazards; P(Drowsy) for driver state.
    data : dict
        Full raw output dict returned by the underlying module.  Carry-forward
        for all existing keys so no downstream code needs to change.
    error : str or None
        If the module raised an exception this cycle, the exception message is
        stored here and ``data`` will be an empty dict.
    """

    source: str
    event_type: str
    timestamp_utc: str
    monotonic_ms: float
    frame_index: int
    latency_ms: float
    confidence: float
    data: dict
    error: str | None = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def ok(
        cls,
        source: str,
        event_type: str,
        frame_index: int,
        latency_ms: float,
        confidence: float,
        data: dict,
        monotonic_ms: float | None = None,
    ) -> "PerceptionEvent":
        """Construct a successful PerceptionEvent."""
        return cls(
            source=source,
            event_type=event_type,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            monotonic_ms=monotonic_ms if monotonic_ms is not None else time.monotonic() * 1000.0,
            frame_index=frame_index,
            latency_ms=round(latency_ms, 3),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            data=data,
            error=None,
        )

    @classmethod
    def from_error(
        cls,
        source: str,
        frame_index: int,
        error: Exception,
        monotonic_ms: float | None = None,
    ) -> "PerceptionEvent":
        """Construct a PerceptionEvent representing a module failure."""
        return cls(
            source=source,
            event_type="module_error",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            monotonic_ms=monotonic_ms if monotonic_ms is not None else time.monotonic() * 1000.0,
            frame_index=frame_index,
            latency_ms=0.0,
            confidence=0.0,
            data={},
            error=str(error),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    @property
    def is_error(self) -> bool:
        return self.error is not None
