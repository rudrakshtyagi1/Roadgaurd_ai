"""
RoadGuard AI — Collision Risk Engine (Phase 2.5 Hardened)
Fuses multi-modal collision sub-signals into an interpretable collision risk score,
enforces strict state semantics (UNKNOWN, SAFE, CAUTION, WARNING, CRITICAL),
and selects the primary threat track.
"""

from __future__ import annotations

from typing import Optional

from core.collision_config import CollisionEngineConfig
from core.collision.types import TrackCollisionAssessment, TrackedObject


class CollisionRiskEngine:
    """
    Computes per-track collision risk scores and selects primary threat.
    
    Principles:
      - New / unconfirmed tracks produce UNKNOWN with zero risk score.
      - Stable vehicles ahead, receding vehicles, and adjacent lane vehicles produce SAFE.
      - Path relevance acts as a strict multiplicative gate.
      - Low TTC outside ego path is suppressed from critical warnings.
    """

    def __init__(self, config: Optional[CollisionEngineConfig] = None) -> None:
        self.config = config or CollisionEngineConfig()

    def assess_track(
        self,
        track: TrackedObject,
        ego_path_relevance: float,
        path_relation: str,
        relative_motion: str,
        motion_strength: float,
        growth_rate: float,
        ttc_seconds: Optional[float],
        ttc_reliable: bool,
        ttc_signal_strength: float,
    ) -> TrackCollisionAssessment:
        """
        Compute full collision assessment and reason codes for a single track.
        """
        cfg = self.config
        w = cfg.weights
        reason_codes: list[str] = []

        # ── 1. Insufficient History / Unconfirmed Track Check ────────
        if track.tracking_state == "NEW" or track.age_ms < cfg.ttc.min_track_age_ms or len(track.history) < cfg.motion.min_samples:
            reason_codes.append("INSUFFICIENT_TRACK_HISTORY")
            latest_sample = track.history[-1] if track.history else None
            return TrackCollisionAssessment(
                track_id=track.track_id,
                class_name=track.class_name,
                detector_confidence=latest_sample.confidence if latest_sample else 0.50,
                tracking_state=track.tracking_state,
                track_age_ms=track.age_ms,
                ego_path_relevance=ego_path_relevance,
                path_relation=path_relation,
                relative_motion="UNKNOWN",
                motion_strength=0.0,
                bbox_growth_rate=0.0,
                ttc_seconds=None,
                ttc_reliable=False,
                ttc_signal_strength=0.0,
                collision_risk_score=0.0,
                collision_state="UNKNOWN",
                reason_codes=reason_codes,
            )

        # ── 2. Sub-risks Computation ─────────────────────────────────
        # Sub-risk: Ego Path
        path_risk = ego_path_relevance

        # Sub-risk: TTC
        ttc_risk = 0.0
        if ttc_reliable and ttc_seconds is not None:
            if ttc_seconds <= cfg.ttc.critical_ttc_s:
                ttc_risk = 1.0
            elif ttc_seconds <= cfg.ttc.warning_ttc_s:
                ttc_risk = 0.70 + 0.30 * (cfg.ttc.warning_ttc_s - ttc_seconds) / (cfg.ttc.warning_ttc_s - cfg.ttc.critical_ttc_s)
            elif ttc_seconds <= cfg.ttc.caution_ttc_s:
                ttc_risk = 0.35 + 0.35 * (cfg.ttc.caution_ttc_s - ttc_seconds) / (cfg.ttc.caution_ttc_s - cfg.ttc.warning_ttc_s)
            else:
                ttc_risk = max(0.0, 0.35 * (cfg.ttc.max_valid_ttc_s - ttc_seconds) / (cfg.ttc.max_valid_ttc_s - cfg.ttc.caution_ttc_s))

        # Sub-risk: Motion (Only approaching adds positive closing risk)
        motion_risk = motion_strength if relative_motion == "APPROACHING" else 0.0

        # Sub-risk: Proximity Scale
        curr_scale = track.history[-1].scale if track.history else 0.1
        prox_risk = min(1.0, max(0.0, curr_scale / 0.50))

        # Sub-risk: Vulnerability (Pedestrian / Cyclist)
        vuln_risk = 0.20 if track.class_name in ["person", "bicycle"] else 0.0

        # ── 3. Multiplicative Gating & Weighted Risk Score ───────────
        if relative_motion == "RECEDING" or path_relation == "OUT_OF_PATH":
            collision_risk_score = 0.0
        elif relative_motion == "STABLE":
            # Stable vehicle ahead only has modest proximity risk if very close
            collision_risk_score = path_risk * 0.25 * prox_risk
        else:
            raw_weighted_risk = (
                w.ttc * ttc_risk +
                w.closing_motion * motion_risk +
                w.proximity_scale * prox_risk +
                w.object_vulnerability * vuln_risk
            )
            collision_risk_score = path_risk * raw_weighted_risk

        collision_risk_score = round(max(0.0, min(1.0, collision_risk_score)), 4)

        # ── 4. Collect Reason Codes ──────────────────────────────────
        if path_relation == "IN_PATH":
            reason_codes.append("OBJECT_IN_EGO_PATH")
        elif path_relation == "NEAR_PATH":
            reason_codes.append("OBJECT_NEAR_EGO_PATH")
        elif path_relation == "OUT_OF_PATH":
            reason_codes.append("ADJACENT_OBJECT_SUPPRESSED")

        if relative_motion == "APPROACHING":
            reason_codes.append("APPROACHING_TRAJECTORY")
        elif relative_motion == "RECEDING":
            reason_codes.append("RECEDING_TRAJECTORY")
        elif relative_motion == "STABLE":
            reason_codes.append("STABLE_TRAJECTORY")

        if ttc_reliable and ttc_seconds is not None:
            if ttc_seconds <= cfg.ttc.critical_ttc_s:
                reason_codes.append("VERY_LOW_TTC")
            elif ttc_seconds <= cfg.ttc.warning_ttc_s:
                reason_codes.append("LOW_TTC")
            elif ttc_seconds <= cfg.ttc.caution_ttc_s:
                reason_codes.append("MODERATE_TTC")

        # ── 5. Candidate Collision State ─────────────────────────────
        if path_relation == "OUT_OF_PATH" or relative_motion in ["RECEDING", "STABLE"]:
            candidate_state = "SAFE"
        elif path_relation == "IN_PATH" and ttc_reliable and ttc_seconds is not None and ttc_seconds <= cfg.ttc.critical_ttc_s:
            candidate_state = "CRITICAL"
        elif (path_relation == "IN_PATH" and ttc_reliable and ttc_seconds is not None and ttc_seconds <= cfg.ttc.warning_ttc_s) or (path_relation in ["IN_PATH", "NEAR_PATH"] and collision_risk_score >= cfg.state_machine.warning_score_threshold):
            candidate_state = "WARNING"
        elif (path_relation == "IN_PATH" and relative_motion == "APPROACHING" and (ttc_reliable or motion_strength >= 0.50)) or (path_relation == "IN_PATH" and collision_risk_score >= cfg.state_machine.caution_score_threshold):
            candidate_state = "CAUTION"
        else:
            candidate_state = "SAFE"

        if not reason_codes and candidate_state == "SAFE":
            reason_codes.append("SAFE_FOLLOWING")

        latest_sample = track.history[-1] if track.history else None
        detector_conf = latest_sample.confidence if latest_sample else 0.50

        return TrackCollisionAssessment(
            track_id=track.track_id,
            class_name=track.class_name,
            detector_confidence=detector_conf,
            tracking_state=track.tracking_state,
            track_age_ms=track.age_ms,
            ego_path_relevance=ego_path_relevance,
            path_relation=path_relation,
            relative_motion=relative_motion,
            motion_strength=motion_strength,
            bbox_growth_rate=growth_rate,
            ttc_seconds=ttc_seconds,
            ttc_reliable=ttc_reliable,
            ttc_signal_strength=ttc_signal_strength,
            collision_risk_score=collision_risk_score,
            collision_state=candidate_state,
            reason_codes=reason_codes,
        )

    def select_primary_threat(self, assessments: list[TrackCollisionAssessment]) -> Optional[TrackCollisionAssessment]:
        """
        Rank all active tracks and select the primary threat.
        """
        if not assessments:
            return None

        # Filter out LOST and UNKNOWN tracks
        valid_assessments = [a for a in assessments if a.tracking_state != "LOST" and a.collision_state != "UNKNOWN"]
        if not valid_assessments:
            # Fallback to any non-lost assessment
            valid_assessments = [a for a in assessments if a.tracking_state != "LOST"]
            if not valid_assessments:
                return None

        state_priority = {"CRITICAL": 4, "WARNING": 3, "CAUTION": 2, "SAFE": 1, "UNKNOWN": 0}

        def threat_sort_key(a: TrackCollisionAssessment) -> tuple:
            ttc_val = a.ttc_seconds if (a.ttc_reliable and a.ttc_seconds is not None) else 999.0
            return (state_priority.get(a.collision_state, 0), a.collision_risk_score, -ttc_val)

        valid_assessments.sort(key=threat_sort_key, reverse=True)
        return valid_assessments[0]
