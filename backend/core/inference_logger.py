"""
RoadGuard AI — Structured Inference Logger
Writes one JSON-lines entry per inference call to a rotating log file.

Output format (one JSON object per line):
    {
        "ts":           "2026-08-23T12:34:56.789+00:00",   # UTC ISO-8601
        "monotonic_ms": 182345.123,                         # time.monotonic() * 1000
        "source":       "road_monitor",                     # module name
        "event_type":   "pothole_detected",
        "frame_index":  42,
        "latency_ms":   28.4,
        "confidence":   0.82,
        "is_error":     false,
        "error":        null
    }

Usage
-----
    from core.inference_logger import InferenceLogger
    from core.perception_event import PerceptionEvent

    logger = InferenceLogger()              # writes to logs/inference.log
    logger.log(event)                       # PerceptionEvent -> one log line
    logger.log_raw(source, latency, conf)  # lightweight variant without a full event

Rotation
--------
The default log file rotates at 5 MB with up to 3 backup files retained.
Override via constructor arguments.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Optional

from core.perception_event import PerceptionEvent

# ---------------------------------------------------------------------------
# Default log location (relative to backend/ working directory)
# ---------------------------------------------------------------------------
_DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "inference.log"
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 3


class InferenceLogger:
    """
    Thin wrapper around Python's rotating file handler that writes structured
    JSON-lines inference telemetry.

    Thread-safe due to underlying ``logging`` module locking.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        max_bytes: int = _MAX_BYTES,
        backup_count: int = _BACKUP_COUNT,
    ) -> None:
        log_path = Path(log_path) if log_path else _DEFAULT_LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a named logger specific to this instance so multiple logger
        # objects (e.g. in tests) don't duplicate handlers on the root logger.
        logger_name = f"roadguard.inference.{log_path.stem}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # don't pollute root logger

        if not self._logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

        self.log_path = log_path

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def log(self, event: PerceptionEvent) -> None:
        """Write one JSON line from a PerceptionEvent."""
        record = {
            "ts": event.timestamp_utc,
            "monotonic_ms": round(event.monotonic_ms, 3),
            "source": event.source,
            "event_type": event.event_type,
            "frame_index": event.frame_index,
            "latency_ms": event.latency_ms,
            "confidence": event.confidence,
            "is_error": event.is_error,
            "error": event.error,
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))

    def log_raw(
        self,
        source: str,
        latency_ms: float,
        confidence: float,
        frame_index: int = -1,
        event_type: str = "raw",
        error: Optional[str] = None,
    ) -> None:
        """
        Lightweight log call for callers that don't have a PerceptionEvent.
        Useful for one-off REST endpoints that run inference outside the pipeline.
        """
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monotonic_ms": round(time.monotonic() * 1000.0, 3),
            "source": source,
            "event_type": event_type,
            "frame_index": frame_index,
            "latency_ms": round(latency_ms, 3),
            "confidence": round(confidence, 4),
            "is_error": error is not None,
            "error": error,
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))
