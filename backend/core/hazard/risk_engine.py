"""
RoadGuard AI — Hazard Risk Engine & State Machine (Phase 3)
Computes hazard risk score, reason codes, and manages hysteretic hazard states.
"""

from __future__ import annotations

from typing import Optional

from core.hazard_config import HazardEngineConfig
from core.hazard.types import HazardAssessment, TrackedHazard


class HazardRiskEngine:
    """Computes risk and manages state transitions for road hazards."""

    STATE_RANKS = {"SAFE": 1, "CAUTION": 2, "WARNING": 3, "CRITICAL": 4}

    def __init__(self, config: Optional[HazardEngineConfig] = None) -> None:
        self.config = config or HazardEngineConfig()
        self._current_state = "SAFE"
        self._pending_lower_state: Optional[str] = None
        self._pending_start_ms: Optional[float] = None

    def reset(self) -> None:
        self._current_state = "SAFE"
        self._pending_lower_state = None
        self._pending_start_ms = None

    def assess_hazard(
        self,
        track: TrackedHazard,
        ego_path_relevance: float,
        path_relation: str,
        severity_level: str,
        severity_score: float,
        growth_rate: float,
    ) -> HazardAssessment:
        cfg = self.config
        w = cfg.weights
        reason_codes: list[str] = []

        # Distance heuristic based on y position (near bottom = closer)
        latest_y = track.history[-1].bottom_center_y if track.history else 0.5
        est_dist = max(2.0, min(50.0, 5.0 / max(0.1, latest_y - 0.45)))

        # ── Gate Unconfirmed Tracks (1-frame noise) ──────────────
        if track.tracking_state == "NEW" or track.frames_seen < cfg.tracker.min_hits_to_confirm:
            reason_codes.append("UNCONFIRMED_HAZARD")
            return HazardAssessment(
                hazard_id=track.hazard_id,
                class_name=track.class_name,
                confidence=track.history[-1].confidence if track.history else 0.50,
                tracking_state=track.tracking_state,
                track_age_ms=track.age_ms,
                ego_path_relevance=ego_path_relevance,
                path_relation=path_relation,
                severity_level=severity_level,
                severity_score=0.0,
                growth_rate=0.0,
                estimated_distance_m=round(est_dist, 1),
                hazard_risk_score=0.0,
                hazard_state="SAFE",
                reason_codes=reason_codes,
            )

        # Sub-risks
        path_risk = ego_path_relevance
        sev_risk = severity_score
        app_risk = min(1.0, max(0.0, growth_rate / 0.10))
        prox_risk = min(1.0, max(0.0, (1.0 - (est_dist / 30.0))))

        if path_relation == "OUT_OF_PATH":
            hazard_risk_score = 0.0
            reason_codes.append("OFF_PATH_HAZARD_IGNORED")
        else:
            raw = (
                w.severity * sev_risk +
                w.approach_motion * app_risk +
                w.proximity_scale * prox_risk
            )
            hazard_risk_score = path_risk * raw

        hazard_risk_score = round(max(0.0, min(1.0, hazard_risk_score)), 4)

        # Reason codes
        if path_relation == "IN_PATH":
            reason_codes.append("POTHOLE_IN_EGO_PATH")
        elif path_relation == "NEAR_PATH":
            reason_codes.append("POTHOLE_NEAR_EGO_PATH")

        if severity_level == "SEVERE":
            reason_codes.append("SEVERE_POTHOLE_SURFACE")
        elif severity_level == "HIGH":
            reason_codes.append("HIGH_SEVERITY_POTHOLE")
        elif severity_level == "MEDIUM":
            reason_codes.append("MODERATE_ROAD_ANOMALY")

        if est_dist <= 10.0 and path_relation == "IN_PATH":
            reason_codes.append("IMMINENT_ROAD_IMPACT")

        # Candidate State
        if path_relation == "OUT_OF_PATH":
            cand_state = "SAFE"
        elif path_relation == "IN_PATH" and severity_level in ["SEVERE", "HIGH"] and est_dist <= 12.0:
            cand_state = "CRITICAL"
        elif (path_relation in ["IN_PATH", "NEAR_PATH"] and severity_level in ["SEVERE", "HIGH"]) or hazard_risk_score >= cfg.state_machine.warning_score_threshold:
            cand_state = "WARNING"
        elif (path_relation in ["IN_PATH", "NEAR_PATH"] and severity_level == "MEDIUM") or hazard_risk_score >= cfg.state_machine.caution_score_threshold:
            cand_state = "CAUTION"
        else:
            cand_state = "SAFE"

        return HazardAssessment(
            hazard_id=track.hazard_id,
            class_name=track.class_name,
            confidence=track.history[-1].confidence if track.history else 0.50,
            tracking_state=track.tracking_state,
            track_age_ms=track.age_ms,
            ego_path_relevance=ego_path_relevance,
            path_relation=path_relation,
            severity_level=severity_level,
            severity_score=severity_score,
            growth_rate=growth_rate,
            estimated_distance_m=round(est_dist, 1),
            hazard_risk_score=hazard_risk_score,
            hazard_state=cand_state,
            reason_codes=reason_codes,
        )

    def update_global_state(
        self,
        assessments: list[HazardAssessment],
        monotonic_ms: float,
    ) -> tuple[str, float, Optional[HazardAssessment], list[str]]:
        if not assessments:
            candidate_state = "SAFE"
            candidate_score = 0.0
            primary_hazard = None
            reason_codes = ["ROAD_CLEAR"]
        else:
            # Sort by severity & risk
            sorted_hazards = sorted(assessments, key=lambda h: (self.STATE_RANKS.get(h.hazard_state, 1), h.hazard_risk_score), reverse=True)
            primary_hazard = sorted_hazards[0]
            candidate_state = primary_hazard.hazard_state
            candidate_score = primary_hazard.hazard_risk_score
            reason_codes = list(primary_hazard.reason_codes)

        current_rank = self.STATE_RANKS.get(self._current_state, 1)
        cand_rank = self.STATE_RANKS.get(candidate_state, 1)

        if cand_rank > current_rank:
            self._current_state = candidate_state
            self._pending_lower_state = None
            self._pending_start_ms = None
        elif cand_rank < current_rank:
            if self._pending_lower_state != candidate_state:
                self._pending_lower_state = candidate_state
                self._pending_start_ms = monotonic_ms

            if self._current_state == "CRITICAL":
                cooldown = self.config.state_machine.cooldown_critical_to_warning_ms
            elif self._current_state == "WARNING":
                cooldown = self.config.state_machine.cooldown_warning_to_caution_ms
            else:
                cooldown = self.config.state_machine.cooldown_caution_to_safe_ms

            if (monotonic_ms - (self._pending_start_ms or monotonic_ms)) >= cooldown:
                self._current_state = candidate_state
                self._pending_lower_state = None
                self._pending_start_ms = None

        return self._current_state, candidate_score, primary_hazard, reason_codes
