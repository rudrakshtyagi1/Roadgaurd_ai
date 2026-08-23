"""
RoadGuard AI — Collision State Machine (Phase 2.5 Hardened)
Hysteretic global collision state transitions with immediate escalation and cooldown hold-times.
"""

from __future__ import annotations

from typing import Optional

from core.collision_config import CollisionStateMachineConfig
from core.collision.types import TrackCollisionAssessment


class CollisionStateMachine:
    """
    Manages global collision state (UNKNOWN, SAFE, CAUTION, WARNING, CRITICAL) with hysteresis.
    """

    STATE_RANKS = {"UNKNOWN": 0, "SAFE": 1, "CAUTION": 2, "WARNING": 3, "CRITICAL": 4}

    def __init__(self, config: Optional[CollisionStateMachineConfig] = None) -> None:
        self.config = config or CollisionStateMachineConfig()
        self._current_state: str = "SAFE"
        self._pending_lower_state: Optional[str] = None
        self._pending_start_mono_ms: Optional[float] = None

    def reset(self) -> None:
        """Reset internal state machine."""
        self._current_state = "SAFE"
        self._pending_lower_state = None
        self._pending_start_mono_ms = None

    def update(
        self,
        primary_threat: Optional[TrackCollisionAssessment],
        monotonic_ms: float,
    ) -> tuple[str, float, list[str]]:
        """
        Update global state given the frame's primary threat track.
        Returns: (global_state: str, global_risk_score: float, reason_codes: list[str]).
        """
        if primary_threat is None:
            candidate_state = "SAFE"
            candidate_score = 0.0
            reason_codes = ["NO_OBJECTS_DETECTED"]
        elif primary_threat.collision_state == "UNKNOWN":
            candidate_state = "SAFE"  # Global defaults to SAFE during unconfirmed initial track
            candidate_score = 0.0
            reason_codes = list(primary_threat.reason_codes)
        else:
            candidate_state = primary_threat.collision_state
            candidate_score = primary_threat.collision_risk_score
            reason_codes = list(primary_threat.reason_codes)

        current_rank = self.STATE_RANKS.get(self._current_state, 1)
        candidate_rank = self.STATE_RANKS.get(candidate_state, 1)

        if candidate_rank > current_rank:
            # Immediate escalation
            self._current_state = candidate_state
            self._pending_lower_state = None
            self._pending_start_mono_ms = None
        elif candidate_rank < current_rank:
            # De-escalation with cooldown
            if self._pending_lower_state != candidate_state:
                self._pending_lower_state = candidate_state
                self._pending_start_mono_ms = monotonic_ms

            if self._current_state == "CRITICAL":
                cooldown = self.config.cooldown_critical_to_warning_ms
            elif self._current_state == "WARNING":
                cooldown = self.config.cooldown_warning_to_caution_ms
            else:
                cooldown = self.config.cooldown_caution_to_safe_ms

            dwell_ms = monotonic_ms - (self._pending_start_mono_ms or monotonic_ms)
            if dwell_ms >= cooldown:
                self._current_state = candidate_state
                self._pending_lower_state = None
                self._pending_start_mono_ms = None
        else:
            self._pending_lower_state = None
            self._pending_start_mono_ms = None

        return self._current_state, candidate_score, reason_codes
