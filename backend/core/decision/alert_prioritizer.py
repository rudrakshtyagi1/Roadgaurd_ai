"""
RoadGuard AI — Intelligent Alert Prioritizer & Intervention Engine (Phase 5)
Arbitrates competing alerts, manages audio exclusivity, anti-spam debouncing, and hysteresis.
"""

from __future__ import annotations

import time
from typing import Optional

from core.brain.types import UnifiedRiskAssessment
from core.decision.types import AlertIntervention


class AlertPrioritizerEngine:
    """
    Intelligent decision engine deciding WHAT RoadGuard communicates to the driver.
    
    Priority Hierarchy:
      1. CRITICAL COLLISION (TTC <= 1.6s) -> Immediate Loud Alarm + Voice Override
      2. CRITICAL ROAD HAZARD (Pothole < 12m) -> Pothole Chime + Voice "Pothole ahead"
      3. CRITICAL / DROWSY DRIVER -> Fatigue Warning Voice "Driver drowsiness detected"
      4. ELEVATED CAUTION -> Amber Visual Cue (No Audio spam)
    """

    def __init__(self, debounce_cooldown_s: float = 2.5) -> None:
        self.debounce_cooldown_s = debounce_cooldown_s
        self._last_alert_times: dict[str, float] = {}
        self._current_alert_type: str = "STANDBY"

    def reset(self) -> None:
        self._last_alert_times.clear()
        self._current_alert_type = "STANDBY"

    def decide(self, brain_assessment: UnifiedRiskAssessment, monotonic_ms: Optional[float] = None) -> AlertIntervention:
        mono_ms = monotonic_ms if monotonic_ms is not None else (time.monotonic() * 1000.0)
        now_s = mono_ms / 1000.0

        state = brain_assessment.unified_state
        primary = brain_assessment.primary_risk_source
        codes = brain_assessment.reason_codes

        # ── 1. Priority Selection ────────────────────────────────────
        if state == "CRITICAL" and primary == "COLLISION":
            alert_type = "COLLISION_IMMINENT"
            priority = "CRITICAL"
            visual = "PULSE_RED"
            audio = "CRITICAL_COLLISION_ALARM"
            voice = "Warning! Brake immediately!"
            haptic = "BURST_TRIPLE"
            directive = "BRAKE IMMEDIATELY"
            suppress = True

        elif (state == "CRITICAL" or state == "HIGH") and primary == "ROAD_HAZARD":
            alert_type = "POTHOLE_AHEAD"
            priority = "CRITICAL" if state == "CRITICAL" else "HIGH"
            visual = "PULSE_RED" if state == "CRITICAL" else "FLASH_AMBER"
            audio = "POTHOLE_ALERT_CHIME"
            voice = "Caution: Severe pothole in driving lane."
            haptic = "LONG_PULSE"
            directive = "SLOW DOWN OR STEER CLEAR"
            suppress = True

        elif (state == "CRITICAL" or state == "HIGH") and primary == "DRIVER_FATIGUE":
            alert_type = "FATIGUE_WARNING"
            priority = "CRITICAL" if state == "CRITICAL" else "HIGH"
            visual = "PULSE_RED" if state == "CRITICAL" else "FLASH_AMBER"
            audio = "FATIGUE_BEEP"
            voice = "Driver drowsiness detected. Please pull over and rest."
            haptic = "BURST_TRIPLE"
            directive = "PULL OVER & TAKE A BREAK"
            suppress = True

        elif state == "ELEVATED":
            alert_type = "CAUTION_ADVISORY"
            priority = "ELEVATED"
            visual = "FLASH_AMBER"
            audio = "NONE" # Never spam audio on mild caution
            voice = None
            haptic = "NONE"
            directive = "MAINTAIN AWARENESS"
            suppress = False

        else:
            alert_type = "STANDBY"
            priority = "NONE"
            visual = "STEADY_GREEN"
            audio = "NONE"
            voice = None
            haptic = "NONE"
            directive = "ROAD CLEAR"
            suppress = False

        # ── 2. Debouncing / Cooldown check for Audio & Voice ────────
        is_debounced = False
        if alert_type in self._last_alert_times:
            last_played = self._last_alert_times[alert_type]
            is_debounced = (now_s - last_played) < self.debounce_cooldown_s

        if is_debounced and alert_type != "COLLISION_IMMINENT":
            # Retain visual & directive, but mute repeating audio spam
            audio = "NONE"
            voice = None
        else:
            if audio != "NONE":
                self._last_alert_times[alert_type] = now_s

        self._current_alert_type = alert_type

        return AlertIntervention(
            priority_level=priority,
            alert_type=alert_type,
            visual_cue=visual,
            audio_cue=audio,
            voice_prompt=voice,
            haptic_pattern=haptic,
            primary_cause=primary,
            action_directive=directive,
            suppress_other_alerts=suppress,
            monotonic_ms=mono_ms,
            reason_codes=codes,
        )
