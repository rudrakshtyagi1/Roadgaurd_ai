"""
RoadGuard AI — RoadGuard Brain (Phase 4 Unified Temporal Risk Fusion)
Combines Driver Intelligence, Collision Intelligence, and Road Hazard Intelligence
via contextual gating, temporal arbitration, and explainable reason coding.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from core.brain.types import UnifiedRiskAssessment
from core.perception_event import PerceptionEvent


class RoadGuardBrain:
    """
    Central Risk Fusion Brain for RoadGuard AI.
    
    Contextual Gating Rules:
      1. Imminent Collision (Low TTC in-path) ALWAYS takes primary urgency priority.
      2. Driver Fatigue + Road Hazard compounding: when driver is distracted/drowsy,
         road hazard threat is amplified because driver response time is degraded.
      3. Strict sub-engine decoupling: Brain consumes the temporal fused driver state &
         fatigue_risk_score, never raw single-frame sensor probabilities.
    """

    STATE_RANKS = {"SAFE": 1, "ELEVATED": 2, "HIGH": 3, "CRITICAL": 4}

    def __init__(self) -> None:
        self._frame_index = 0

    def compute(
        self,
        driver_data: dict[str, Any],
        collision_data: dict[str, Any],
        hazard_data: dict[str, Any],
        monotonic_ms: Optional[float] = None,
    ) -> UnifiedRiskAssessment:
        self._frame_index += 1
        mono_ms = monotonic_ms if monotonic_ms is not None else (time.monotonic() * 1000.0)

        # ── 1. Extract Sub-Engine States ─────────────────────────────
        d_state = driver_data.get("state", "NORMAL")
        d_score = float(driver_data.get("fatigue_risk_score", 0.0))
        d_codes = driver_data.get("reason_codes", [])
        d_prob = float(driver_data.get("cnn_drowsy_probability", driver_data.get("drowsy_probability", 0.0)))

        c_state = collision_data.get("global_collision_state", "SAFE")
        c_score = float(collision_data.get("global_collision_risk_score", 0.0))
        c_codes = collision_data.get("reason_codes", [])
        pt = collision_data.get("primary_threat")
        ttc_s = pt.get("ttc_seconds") if pt else None

        h_state = hazard_data.get("global_hazard_state", "SAFE")
        h_score = float(hazard_data.get("global_hazard_risk_score", 0.0))
        h_codes = hazard_data.get("reason_codes", [])
        ph = hazard_data.get("primary_hazard")
        haz_dist = ph.get("estimated_distance_m") if ph else None

        reason_codes: list[str] = []
        secondary_risks: list[str] = []

        # ── 2. Contextual Gating & Urgency Arbitration ───────────────
        # Case A: Imminent Collision Priority
        if c_state in ["CRITICAL", "WARNING"] or (ttc_s is not None and ttc_s <= 2.8):
            primary_source = "COLLISION"
            unified_state = "CRITICAL" if c_state == "CRITICAL" else "HIGH"
            unified_score = max(c_score, 0.85 if unified_state == "CRITICAL" else 0.70)
            reason_codes.extend([c for c in c_codes if c not in reason_codes])

            if d_state in ["DROWSY", "CRITICAL", "FATIGUE_RISK"]:
                secondary_risks.append("DRIVER_FATIGUE")
                reason_codes.append("DRIVER_INATTENTIVE_DURING_COLLISION_RISK")
            if h_state in ["WARNING", "CRITICAL"]:
                secondary_risks.append("ROAD_HAZARD")

        # Case B: Severe Road Hazard in Ego Path
        elif h_state in ["CRITICAL", "WARNING"]:
            primary_source = "ROAD_HAZARD"
            # If driver is drowsy/fatigued, amplify road hazard risk
            if d_state in ["DROWSY", "CRITICAL"]:
                unified_state = "CRITICAL"
                unified_score = min(1.0, h_score * 1.35)
                secondary_risks.append("DRIVER_FATIGUE")
                reason_codes.append("FATIGUED_DRIVER_APPROACHING_HAZARD")
            else:
                unified_state = "CRITICAL" if h_state == "CRITICAL" else "HIGH"
                unified_score = h_score
            reason_codes.extend([c for c in h_codes if c not in reason_codes])

        # Case C: Driver Fatigue / Drowsiness Primary
        elif d_state in ["CRITICAL", "DROWSY"]:
            primary_source = "DRIVER_FATIGUE"
            unified_state = "CRITICAL" if d_state == "CRITICAL" else "HIGH"
            unified_score = d_score
            reason_codes.extend([c for c in d_codes if c not in reason_codes])
            if c_state == "CAUTION":
                secondary_risks.append("COLLISION")
            if h_state == "CAUTION":
                secondary_risks.append("ROAD_HAZARD")

        # Case D: Elevated / Caution Conditions
        elif c_state == "CAUTION" or h_state == "CAUTION" or d_state == "FATIGUE_RISK":
            scores = {"COLLISION": c_score, "ROAD_HAZARD": h_score, "DRIVER_FATIGUE": d_score}
            primary_source = max(scores, key=scores.get)
            unified_state = "ELEVATED"
            unified_score = max(c_score, h_score, d_score, 0.40)
            for src, sc in scores.items():
                if src != primary_source and sc >= 0.20:
                    secondary_risks.append(src)
            reason_codes.extend(c_codes + h_codes + d_codes)

        # Case E: Clear & Safe
        else:
            primary_source = "NONE"
            unified_state = "SAFE"
            unified_score = max(0.0, max(c_score, h_score, d_score))
            reason_codes.append("ALL_SYSTEMS_NORMAL")

        # Deduplicate reason codes
        seen = set()
        dedup_codes = []
        for rc in reason_codes:
            if rc not in seen:
                seen.add(rc)
                dedup_codes.append(rc)

        return UnifiedRiskAssessment(
            unified_state=unified_state,
            unified_risk_score=round(unified_score, 4),
            primary_risk_source=primary_source,
            secondary_risk_sources=secondary_risks,
            reason_codes=dedup_codes,
            driver_state=d_state,
            collision_state=c_state,
            hazard_state=h_state,
            ttc_seconds=ttc_s,
            hazard_distance_m=haz_dist,
            drowsy_probability=d_prob,
            monotonic_ms=mono_ms,
            frame_index=self._frame_index,
            contributing_signals={
                "driver_fatigue_score": d_score,
                "collision_risk_score": c_score,
                "hazard_risk_score": h_score,
            },
        )

    def to_perception_event(self, assessment: UnifiedRiskAssessment) -> PerceptionEvent:
        return PerceptionEvent.ok(
            source="roadguard_brain",
            event_type="unified_risk",
            monotonic_ms=assessment.monotonic_ms,
            frame_index=assessment.frame_index,
            latency_ms=0.5,
            confidence=assessment.unified_risk_score,
            data=assessment.__dict__,
        )
