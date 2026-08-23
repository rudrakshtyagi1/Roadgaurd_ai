"""
RoadGuard AI — Decision & Alert Prioritizer Data Types (Phase 5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlertIntervention:
    """Actionable multimodal intervention payload for the driver."""
    priority_level: str # "CRITICAL", "HIGH", "ELEVATED", "NONE"
    alert_type: str # "COLLISION_IMMINENT", "POTHOLE_AHEAD", "FATIGUE_WARNING", "STANDBY"

    visual_cue: str # "PULSE_RED", "FLASH_AMBER", "STEADY_GREEN"
    audio_cue: str # "CRITICAL_COLLISION_ALARM", "POTHOLE_ALERT_CHIME", "FATIGUE_BEEP", "NONE"
    voice_prompt: Optional[str] # "Warning: Collision threat ahead!", etc.
    haptic_pattern: str # "BURST_TRIPLE", "LONG_PULSE", "NONE"

    primary_cause: str
    action_directive: str # "BRAKE IMMEDIATELY", "STEER CLEAR / SLOW DOWN", "PULL OVER & REST", "MAINTAIN LANE"
    suppress_other_alerts: bool
    monotonic_ms: float
    reason_codes: list[str] = field(default_factory=list)
