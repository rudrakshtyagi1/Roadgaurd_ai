"""
RoadGuard AI — Hard Case Auto-Mining Engine (Phase 8 Data Flywheel)
Identifies ambiguous, uncertain, or near-miss events during drives for automated dataset retraining.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MinedHardCase:
    case_id: str
    category: str # "LOW_CONFIDENCE", "STATE_JUMP", "TRACKING_LOSS", "NEAR_MISS"
    confidence: float
    reason: str
    timestamp_utc: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HardCaseMiner:
    """Monitors live pipeline events and captures edge cases into retraining queue."""

    def __init__(self, max_queue_size: int = 100) -> None:
        self.max_queue_size = max_queue_size
        self._mined_cases: list[MinedHardCase] = []
        self._next_id = 1

    def inspect_and_mine(
        self,
        event_source: str,
        confidence: float,
        state: str,
        metadata: dict[str, Any],
        is_tracking_loss: bool = False,
    ) -> Optional[MinedHardCase]:
        category: Optional[str] = None
        reason: Optional[str] = None

        # Rule 1: Ambiguous Confidence (0.25 <= conf <= 0.40)
        if metadata.get("signal_disagreement") or "SIGNAL_DISAGREEMENT" in metadata.get("reason_codes", []):
            category = "DRIVER_CNN_PHYSIOLOGY_DISAGREEMENT"
            reason = "CNN predicted drowsiness while physiological markers indicated alert wakefulness"
        elif 0.25 <= confidence <= 0.40:
            category = "LOW_CONFIDENCE"
            reason = f"Marginal detection confidence {confidence:.2f} in {event_source}"

        # Rule 2: Tracking Loss
        elif is_tracking_loss:
            category = "TRACKING_LOSS"
            reason = f"Unexpected track drop in {event_source}"

        # Rule 3: Near-miss Critical Alert
        elif state == "CRITICAL":
            category = "NEAR_MISS"
            reason = f"High-risk critical safety event captured in {event_source}"

        if category is not None:
            cid = f"case_{self._next_id}"
            self._next_id += 1
            case = MinedHardCase(
                case_id=cid,
                category=category,
                confidence=round(confidence, 4),
                reason=reason or "",
                timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                metadata=metadata,
            )
            self._mined_cases.append(case)
            if len(self._mined_cases) > self.max_queue_size:
                self._mined_cases.pop(0)
            return case

        return None

    def get_mined_cases(self) -> list[MinedHardCase]:
        return list(self._mined_cases)
