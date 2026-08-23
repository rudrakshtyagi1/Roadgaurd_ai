"""
RoadGuard AI — Driver State Engine V2.0 (Lifecycle, Temporal Fusion & Recovery Hardened)
Temporal driver state evaluation module with monotonic timing, rolling PERCLOS,
two-timescale fatigue reasoning, explicit sustained-open-eye tracking,
wakefulness recovery gating, hysteretic state machine transitions, and explainable reason codes.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Any, Optional

from core.driver_state_config import DriverStateConfig
from core.perception_event import PerceptionEvent


@dataclass
class _Sample:
    """Internal historical observation record."""
    monotonic_ms: float
    dt_ms: float
    is_closed: bool
    mar: float
    yaw: float
    pitch: float
    cnn_prob: float
    face_detected: bool


@dataclass
class _BlinkRecord:
    """Historical completed blink record with temporal decay."""
    end_monotonic_ms: float
    duration_ms: float
    is_long_blink: bool

    def decay_weight(self, now_ms: float, strong_window_ms: float, history_window_ms: float) -> float:
        age = max(0.0, now_ms - self.end_monotonic_ms)
        if age <= strong_window_ms:
            return 1.0
        if age >= history_window_ms:
            return 0.0
        return max(0.0, 1.0 - (age - strong_window_ms) / max(1.0, history_window_ms - strong_window_ms))


@dataclass
class _YawnRecord:
    """Historical completed yawn record with temporal decay."""
    start_monotonic_ms: float
    end_monotonic_ms: float
    duration_ms: float

    def decay_weight(self, now_ms: float, strong_window_ms: float, history_window_ms: float) -> float:
        age = max(0.0, now_ms - self.end_monotonic_ms)
        if age <= strong_window_ms:
            return 1.0
        if age >= history_window_ms:
            return 0.0
        return max(0.0, 1.0 - (age - strong_window_ms) / max(1.0, history_window_ms - strong_window_ms))


class DriverStateEngineV2:
    """
    Temporal Driver State Engine V2.0 for RoadGuard AI.
    
    Key Hardening Principles:
      1. Two-timescale fatigue reasoning: 30-second rolling PERCLOS is preserved mathematically
         as long-horizon history, while continuous open-eye duration and immediate wakefulness
         control responsive state recovery.
      2. Explicit sustained-open-eye tracking: monotonic timer tracks continuous wakefulness
         since eye reopening, resetting immediately on eye closure.
      3. Active closure vs. historical recovery: active PROLONGED_EYE_CLOSURE override clears
         immediately once confirmed reopened, ending persistent latching.
      4. Time-decaying evidence lifecycles: past long blinks and yawns decay and expire.
      5. State machine hysteresis: fast escalation on danger, controlled dwell hold times on recovery.
    """

    def __init__(self, config: Optional[DriverStateConfig] = None) -> None:
        self.config = config or DriverStateConfig.from_yaml()

        # Rolling observation deque for PERCLOS & Quality
        self._history: collections.deque[_Sample] = collections.deque(maxlen=3000)
        self._blink_history: collections.deque[_BlinkRecord] = collections.deque(maxlen=200)
        self._yawn_history: collections.deque[_YawnRecord] = collections.deque(maxlen=50)

        # Monotonic time tracking
        self._last_mono_ms: Optional[float] = None
        self._engine_start_mono_ms: Optional[float] = None
        self._last_face_seen_mono_ms: Optional[float] = None

        # Eye & Blink State Machine
        self._eye_state: str = "OPEN"
        self._eye_closure_start_mono_ms: Optional[float] = None
        self._eyes_open_since_mono_ms: Optional[float] = None
        self._last_completed_blink_duration_ms: float = 0.0
        self._total_blink_count: int = 0

        # Yawn State Machine
        self._mouth_state: str = "NORMAL"
        self._yawn_start_mono_ms: Optional[float] = None
        self._total_yawn_count: int = 0

        # Head Pose Ongoing Durations
        self._nodding_start_mono_ms: Optional[float] = None
        self._head_turn_start_mono_ms: Optional[float] = None

        # Sustained CNN tracking
        self._cnn_high_start_mono_ms: Optional[float] = None

        # State Machine & Hysteresis
        self._current_state: str = "NORMAL"
        self._pending_lower_state: Optional[str] = None
        self._lower_state_candidate_start_mono_ms: Optional[float] = None

    def reset(self) -> None:
        """Reset all internal state and buffers."""
        self._history.clear()
        self._blink_history.clear()
        self._yawn_history.clear()
        self._last_mono_ms = None
        self._engine_start_mono_ms = None
        self._last_face_seen_mono_ms = None
        self._eye_state = "OPEN"
        self._eye_closure_start_mono_ms = None
        self._eyes_open_since_mono_ms = None
        self._last_completed_blink_duration_ms = 0.0
        self._total_blink_count = 0
        self._mouth_state = "NORMAL"
        self._yawn_start_mono_ms = None
        self._total_yawn_count = 0
        self._nodding_start_mono_ms = None
        self._head_turn_start_mono_ms = None
        self._cnn_high_start_mono_ms = None
        self._current_state = "NORMAL"
        self._pending_lower_state = None
        self._lower_state_candidate_start_mono_ms = None

    def update(
        self,
        ear: float = 0.30,
        mar: float = 0.10,
        head_pose: Optional[dict[str, float]] = None,
        cnn_drowsy_prob: float = 0.0,
        face_detected: bool = True,
        monotonic_ms: Optional[float] = None,
        frame_index: int = 0,
    ) -> dict[str, Any]:
        """
        Process a single time-stamped driver observation and update temporal state.
        """
        t_now = monotonic_ms if monotonic_ms is not None else time.monotonic() * 1000.0

        if self._engine_start_mono_ms is None:
            self._engine_start_mono_ms = t_now

        # Compute dt
        if self._last_mono_ms is None:
            dt_ms = 33.33  # ~30fps default for first frame
        else:
            dt_ms = min(self.config.windows.max_sample_dt_ms, max(0.1, t_now - self._last_mono_ms))
        self._last_mono_ms = t_now

        head_pose = head_pose or {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        yaw = float(head_pose.get("yaw", 0.0))
        pitch = float(head_pose.get("pitch", 0.0))
        roll = float(head_pose.get("roll", 0.0))

        # ── 1. Update Face Tracking Quality ──────────────────────────
        if face_detected:
            self._last_face_seen_mono_ms = t_now

        is_eye_closed = (ear < self.config.eye.ear_close_threshold) if face_detected else False

        # Append sample to rolling history
        self._history.append(_Sample(
            monotonic_ms=t_now,
            dt_ms=dt_ms,
            is_closed=is_eye_closed,
            mar=mar if face_detected else 0.0,
            yaw=yaw,
            pitch=pitch,
            cnn_prob=cnn_drowsy_prob,
            face_detected=face_detected,
        ))

        # Prune history older than max window (60s)
        cutoff_ms = t_now - max(self.config.windows.perclos_window_ms, self.config.windows.blink_history_window_ms)
        while self._history and self._history[0].monotonic_ms < cutoff_ms:
            self._history.popleft()

        # Quality metrics over quality_window_ms
        quality_cutoff_ms = t_now - self.config.windows.quality_window_ms
        quality_samples = [s for s in self._history if s.monotonic_ms >= quality_cutoff_ms]
        total_q_dt = sum(s.dt_ms for s in quality_samples)
        valid_q_dt = sum(s.dt_ms for s in quality_samples if s.face_detected)
        valid_observation_ratio = (valid_q_dt / total_q_dt) if total_q_dt > 0 else 1.0

        consecutive_dropout_ms = (t_now - self._last_face_seen_mono_ms) if self._last_face_seen_mono_ms is not None else 0.0
        if not face_detected and consecutive_dropout_ms >= self.config.quality.max_consecutive_dropouts_ms:
            tracking_state = "LOST"
        elif valid_observation_ratio >= self.config.quality.tracked_ratio_threshold:
            tracking_state = "TRACKED"
        elif valid_observation_ratio >= self.config.quality.degraded_ratio_threshold:
            tracking_state = "DEGRADED"
        else:
            tracking_state = "LOST"

        signal_strength = valid_observation_ratio if tracking_state != "LOST" else 0.0

        # ── 2. Blink & Open-Eye State Machine with Continuous Tracking ──
        ongoing_eye_closure_ms = 0.0
        continuous_open_duration_ms = 0.0

        if face_detected:
            if self._eye_state == "OPEN":
                if ear < self.config.eye.ear_close_threshold:
                    # Eye just closed -> reset open-eye timer and start closure timer
                    self._eye_state = "CLOSED"
                    self._eye_closure_start_mono_ms = t_now
                    self._eyes_open_since_mono_ms = None
                else:
                    # Eyes remain open -> continue/start open-eye timer
                    if self._eyes_open_since_mono_ms is None:
                        self._eyes_open_since_mono_ms = t_now
                    continuous_open_duration_ms = t_now - self._eyes_open_since_mono_ms
            else:  # currently "CLOSED"
                if ear >= self.config.eye.ear_reopen_threshold:
                    # Eye reopened -> completed blink and start open-eye timer immediately
                    self._eye_state = "OPEN"
                    self._eyes_open_since_mono_ms = t_now
                    continuous_open_duration_ms = 0.0
                    if self._eye_closure_start_mono_ms is not None:
                        blink_duration_ms = t_now - self._eye_closure_start_mono_ms
                        self._eye_closure_start_mono_ms = None
                        if blink_duration_ms >= self.config.eye.normal_blink_min_ms:
                            self._total_blink_count += 1
                            self._last_completed_blink_duration_ms = blink_duration_ms
                            is_long = blink_duration_ms >= self.config.eye.long_blink_min_ms
                            self._blink_history.append(_BlinkRecord(
                                end_monotonic_ms=t_now,
                                duration_ms=blink_duration_ms,
                                is_long_blink=is_long,
                            ))
                else:
                    # Eyes remain closed
                    self._eyes_open_since_mono_ms = None

            if self._eye_state == "CLOSED" and self._eye_closure_start_mono_ms is not None:
                ongoing_eye_closure_ms = t_now - self._eye_closure_start_mono_ms
        else:
            # Face dropout: reset active closure & open-eye state
            self._eye_state = "OPEN"
            self._eye_closure_start_mono_ms = None
            self._eyes_open_since_mono_ms = None

        # Clean blink history past max window
        blink_history_win = self.config.evidence.long_blink_history_window_ms
        while self._blink_history and (t_now - self._blink_history[0].end_monotonic_ms) > blink_history_win:
            self._blink_history.popleft()

        # Compute Decayed Long Blink Evidence
        strong_blink_win = self.config.evidence.long_blink_strong_window_ms
        long_blinks_30s = 0
        effective_long_blink_weight = 0.0
        latest_long_blink_age_ms: Optional[float] = None

        for b in self._blink_history:
            if b.is_long_blink:
                long_blinks_30s += 1
                w_decay = b.decay_weight(t_now, strong_blink_win, blink_history_win)
                effective_long_blink_weight += w_decay
                age = t_now - b.end_monotonic_ms
                if latest_long_blink_age_ms is None or age < latest_long_blink_age_ms:
                    latest_long_blink_age_ms = age

        total_recent_blinks = len(self._blink_history)
        blink_window_s = self.config.windows.blink_history_window_ms / 1000.0
        blink_rate_per_min = (total_recent_blinks / blink_window_s) * 60.0 if blink_window_s > 0 else 0.0
        avg_blink_duration_ms = (
            sum(b.duration_ms for b in self._blink_history) / total_recent_blinks
            if total_recent_blinks > 0 else 150.0
        )

        # ── 3. Yawn State Machine with Hysteresis & Decay ────────────
        ongoing_yawn_ms = 0.0
        if face_detected:
            if self._mouth_state == "NORMAL":
                if mar >= self.config.yawn.yawn_start_mar:
                    self._mouth_state = "YAWNING"
                    self._yawn_start_mono_ms = t_now
            else:  # currently "YAWNING"
                if mar < self.config.yawn.yawn_end_mar:
                    self._mouth_state = "NORMAL"
                    if self._yawn_start_mono_ms is not None:
                        yawn_dur = t_now - self._yawn_start_mono_ms
                        yawn_start = self._yawn_start_mono_ms
                        self._yawn_start_mono_ms = None
                        if yawn_dur >= self.config.yawn.min_yawn_duration_ms:
                            self._total_yawn_count += 1
                            self._yawn_history.append(_YawnRecord(
                                start_monotonic_ms=yawn_start,
                                end_monotonic_ms=t_now,
                                duration_ms=yawn_dur,
                            ))

            if self._mouth_state == "YAWNING" and self._yawn_start_mono_ms is not None:
                ongoing_yawn_ms = t_now - self._yawn_start_mono_ms
        else:
            self._mouth_state = "NORMAL"
            self._yawn_start_mono_ms = None

        # Clean yawn history
        yawn_history_win = self.config.evidence.yawn_history_window_ms
        strong_yawn_win = self.config.evidence.yawn_strong_window_ms
        while self._yawn_history and (t_now - self._yawn_history[0].end_monotonic_ms) > yawn_history_win:
            self._yawn_history.popleft()

        yawns_60s = len(self._yawn_history)
        effective_yawn_weight = sum(
            y.decay_weight(t_now, strong_yawn_win, yawn_history_win) for y in self._yawn_history
        )

        # ── 4. Head Pose Nodding & Inattention Trackers ──────────────
        ongoing_nodding_ms = 0.0
        ongoing_head_turn_ms = 0.0

        if face_detected:
            if pitch <= self.config.head_pose.nodding_pitch_down_deg:
                if self._nodding_start_mono_ms is None:
                    self._nodding_start_mono_ms = t_now
                ongoing_nodding_ms = t_now - self._nodding_start_mono_ms
            else:
                self._nodding_start_mono_ms = None

            if abs(yaw) >= self.config.head_pose.inattention_yaw_deg or pitch >= self.config.head_pose.inattention_pitch_up_deg:
                if self._head_turn_start_mono_ms is None:
                    self._head_turn_start_mono_ms = t_now
                ongoing_head_turn_ms = t_now - self._head_turn_start_mono_ms
            else:
                self._head_turn_start_mono_ms = None
        else:
            self._nodding_start_mono_ms = None
            self._head_turn_start_mono_ms = None

        # ── 5. Mathematical 30s Rolling PERCLOS (Unmodified Definition) ─
        perclos_cutoff_ms = t_now - self.config.windows.perclos_window_ms
        perclos_samples = [s for s in self._history if s.monotonic_ms >= perclos_cutoff_ms]
        
        valid_perclos_dt = sum(s.dt_ms for s in perclos_samples if s.face_detected)
        closed_perclos_dt = sum(s.dt_ms for s in perclos_samples if s.face_detected and s.is_closed)

        perclos_reliable = (valid_perclos_dt >= self.config.windows.min_valid_history_ms)
        if perclos_reliable and valid_perclos_dt > 0:
            raw_perclos = closed_perclos_dt / valid_perclos_dt
        else:
            raw_perclos = 0.0

        # ── 6. Sustained CNN Tracking ────────────────────────────────
        if cnn_drowsy_prob >= self.config.evidence.strong_cnn_threshold and face_detected:
            if self._cnn_high_start_mono_ms is None:
                self._cnn_high_start_mono_ms = t_now
            cnn_sustained_ms = t_now - self._cnn_high_start_mono_ms
        else:
            self._cnn_high_start_mono_ms = None
            cnn_sustained_ms = 0.0

        # ── 7. Wakefulness Support & Physiological Contradiction ─────
        ear_awake_factor = min(1.0, max(0.0, (ear - self.config.eye.ear_close_threshold) / 0.15))
        perclos_awake_factor = min(1.0, max(0.0, (0.15 - raw_perclos) / 0.15)) if perclos_reliable else 0.5
        no_closure_factor = 1.0 if ongoing_eye_closure_ms < 300.0 else 0.0
        no_yawn_factor = 1.0 if ongoing_yawn_ms < 1000.0 else 0.0
        head_stable_factor = 1.0 if (abs(yaw) < 18.0 and abs(pitch) < 15.0) else 0.5

        wakefulness_support = (
            0.40 * ear_awake_factor +
            0.30 * perclos_awake_factor +
            0.15 * no_closure_factor +
            0.15 * no_yawn_factor
        ) * head_stable_factor
        wakefulness_support = min(1.0, max(0.0, wakefulness_support))

        # Check for signal disagreement (CNN says drowsy, but physiology says awake)
        signal_disagreement = (
            cnn_drowsy_prob >= 0.55
            and ear >= self.config.evidence.wakefulness_ear_min
            and raw_perclos <= self.config.evidence.wakefulness_perclos_max
            and ongoing_eye_closure_ms < 300.0
            and ongoing_yawn_ms < 1000.0
        )

        # ── 8. Two-Timescale Sub-Risk Fusion (Recovery Attenuation) ──
        if perclos_reliable:
            raw_perclos_risk = min(1.0, max(0.0, raw_perclos / 0.40))
        else:
            raw_perclos_risk = 0.0

        # Compute recovery context attenuation factor on historical PERCLOS
        # During confirmed continuous open eyes, PERCLOS acts as long-term history rather than current danger
        reopen_conf_ms = self.config.recovery.reopen_confirmation_ms
        drowsy_rec_ms = self.config.recovery.drowsy_recovery_ms
        normal_rec_ms = self.config.recovery.normal_recovery_ms

        if ongoing_eye_closure_ms > 0.0 or continuous_open_duration_ms < reopen_conf_ms:
            # Active closure or unconfirmed reopen -> full PERCLOS risk
            recovery_attenuation = 1.0
        else:
            # Reopen confirmed (continuous_open >= reopen_conf_ms)
            if continuous_open_duration_ms < drowsy_rec_ms:
                # Decaying from 1.0 down to 0.45
                prog = (continuous_open_duration_ms - reopen_conf_ms) / max(1.0, drowsy_rec_ms - reopen_conf_ms)
                recovery_attenuation = 1.0 - 0.55 * prog
            elif continuous_open_duration_ms < normal_rec_ms:
                # Decaying from 0.45 down to 0.15
                prog = (continuous_open_duration_ms - drowsy_rec_ms) / max(1.0, normal_rec_ms - drowsy_rec_ms)
                recovery_attenuation = 0.45 - 0.30 * prog
            else:
                # Sustained open >= normal_rec_ms (>= 4s)
                recovery_attenuation = 0.0 if wakefulness_support >= 0.70 else 0.10

        effective_perclos_risk = raw_perclos_risk * recovery_attenuation

        # Sub-risk 2: Prolonged eye closure ongoing risk (strictly ACTIVE only)
        closure_warn_ms = self.config.eye.prolonged_closure_warning_ms
        closure_crit_ms = self.config.eye.prolonged_closure_critical_ms
        if ongoing_eye_closure_ms > closure_warn_ms:
            eye_closure_risk = min(1.0, (ongoing_eye_closure_ms - closure_warn_ms) / (closure_crit_ms - closure_warn_ms))
        else:
            eye_closure_risk = 0.0

        # Sub-risk 3: Decayed Long blinks risk
        long_blink_risk = min(1.0, effective_long_blink_weight / 2.0)

        # Sub-risk 4: Yawn risk (active ongoing + decayed recent)
        sustained_yawn_thresh_ms = self.config.yawn.sustained_yawn_ms
        if ongoing_yawn_ms >= sustained_yawn_thresh_ms:
            active_yawn_risk = min(1.0, (ongoing_yawn_ms - sustained_yawn_thresh_ms) / 2000.0 + 0.5)
        else:
            active_yawn_risk = 0.0
        recent_yawn_risk = min(0.6, effective_yawn_weight / 2.0)
        yawn_risk = max(active_yawn_risk, recent_yawn_risk)

        # Sub-risk 5: Head nodding risk
        nod_thresh_ms = self.config.head_pose.nodding_duration_warning_ms
        if ongoing_nodding_ms >= nod_thresh_ms:
            head_nod_risk = min(0.6, (ongoing_nodding_ms - nod_thresh_ms) / 2000.0 + 0.3)
        else:
            head_nod_risk = 0.0

        # Sub-risk 6: CNN deep feature risk (modulated by wakefulness corroboration)
        if face_detected:
            if wakefulness_support >= 0.70 and ongoing_eye_closure_ms < 400.0:
                cnn_attenuation = 1.0 - (0.65 * wakefulness_support)
                cnn_risk = cnn_drowsy_prob * cnn_attenuation
            else:
                cnn_risk = cnn_drowsy_prob
        else:
            cnn_risk = 0.0

        w = self.config.weights
        raw_fatigue_score = (
            w.perclos * effective_perclos_risk +
            w.eye_closure * eye_closure_risk +
            w.long_blinks * long_blink_risk +
            w.yawn_persistence * yawn_risk +
            w.head_nodding * head_nod_risk +
            w.cnn_drowsy * cnn_risk
        )
        
        # Apply wakefulness attenuation to final score if eyes clearly open
        if wakefulness_support >= 0.80 and eye_closure_risk == 0.0:
            fatigue_risk_score = max(0.0, raw_fatigue_score - 0.10 * wakefulness_support)
        else:
            fatigue_risk_score = raw_fatigue_score

        fatigue_risk_score = min(1.0, max(0.0, fatigue_risk_score))

        # ── 9. Critical Overrides & Candidate State Evaluation ───────
        reason_codes: list[str] = []
        is_critical_override = False

        # Override 1: Active Prolonged Eye Closure >= 1.5s
        if face_detected and ongoing_eye_closure_ms >= self.config.eye.prolonged_closure_critical_ms:
            is_critical_override = True
            reason_codes.append("PROLONGED_EYE_CLOSURE")

        # Override 2: Extreme PERCLOS >= 0.50 (active while unconfirmed or not sustained open)
        if perclos_reliable and raw_perclos >= 0.50 and continuous_open_duration_ms < drowsy_rec_ms:
            is_critical_override = True
            reason_codes.append("EXTREME_PERCLOS")

        # Override 3: Sustained head nod with eye closure corroboration
        if ongoing_nodding_ms >= 2000.0 and ongoing_eye_closure_ms >= 800.0:
            is_critical_override = True
            reason_codes.append("SUSTAINED_HEAD_DROP_EYES_CLOSED")

        # Candidate State Determination
        # Determine recovery eligibility
        is_sustained_drowsy_recovery = (continuous_open_duration_ms >= drowsy_rec_ms)
        is_sustained_normal_recovery = (continuous_open_duration_ms >= normal_rec_ms)

        has_active_fatigue = (
            ongoing_eye_closure_ms >= 400.0
            or ongoing_yawn_ms >= sustained_yawn_thresh_ms
            or ongoing_nodding_ms >= nod_thresh_ms
            or effective_long_blink_weight >= 1.8
        )

        if is_critical_override or (fatigue_risk_score >= self.config.state_machine.critical_score_threshold and not is_sustained_drowsy_recovery):
            candidate_state = "CRITICAL"
        elif (
            (not is_sustained_drowsy_recovery and perclos_reliable and raw_perclos >= 0.30)
            or (effective_long_blink_weight >= 2.2)
            or (cnn_sustained_ms >= self.config.evidence.cnn_confirmation_window_ms and (raw_perclos >= 0.15 or effective_long_blink_weight >= 1.0))
            or (fatigue_risk_score >= self.config.state_machine.drowsy_threshold and (has_active_fatigue or wakefulness_support < 0.60))
        ):
            candidate_state = "DROWSY"
        elif (
            fatigue_risk_score >= self.config.state_machine.fatigue_risk_threshold
            or (not is_sustained_normal_recovery and perclos_reliable and raw_perclos >= 0.15)
            or (ongoing_yawn_ms >= sustained_yawn_thresh_ms)
            or (effective_long_blink_weight >= 1.5)
            or (ongoing_nodding_ms >= self.config.head_pose.nodding_duration_warning_ms)
        ):
            candidate_state = "FATIGUE_RISK"
        else:
            candidate_state = "NORMAL"

        # Collect Explainability Reason Codes (Current & Recovery Evidence)
        if raw_perclos >= 0.30 and perclos_reliable and "EXTREME_PERCLOS" not in reason_codes:
            if continuous_open_duration_ms >= reopen_conf_ms and continuous_open_duration_ms < normal_rec_ms:
                reason_codes.append("RECOVERING_FROM_EYE_CLOSURE")
                reason_codes.append("HIGH_RECENT_PERCLOS")
            elif continuous_open_duration_ms < reopen_conf_ms:
                reason_codes.append("HIGH_PERCLOS")
        elif perclos_reliable and raw_perclos >= 0.15 and continuous_open_duration_ms < normal_rec_ms:
            reason_codes.append("MODERATE_PERCLOS")

        if ongoing_eye_closure_ms >= self.config.eye.prolonged_closure_warning_ms and "PROLONGED_EYE_CLOSURE" not in reason_codes:
            reason_codes.append("ACTIVE_PROLONGED_EYE_CLOSURE")

        if effective_long_blink_weight >= 2.0:
            reason_codes.append("FREQUENT_LONG_BLINKS")
        elif effective_long_blink_weight >= 0.35 and (latest_long_blink_age_ms is not None and latest_long_blink_age_ms < 8000.0):
            reason_codes.append("LONG_BLINK_DETECTED")
            reason_codes.append("RECENT_LONG_BLINK")

        if ongoing_yawn_ms >= self.config.yawn.sustained_yawn_ms:
            reason_codes.append("SUSTAINED_YAWN")
            reason_codes.append("ACTIVE_YAWN")
        elif effective_yawn_weight >= 0.50:
            reason_codes.append("RECENT_YAWNS")

        if ongoing_nodding_ms >= self.config.head_pose.nodding_duration_warning_ms:
            reason_codes.append("HEAD_NODDING")

        if ongoing_head_turn_ms >= self.config.head_pose.head_turn_duration_warning_ms:
            reason_codes.append("HEAD_TURN_INATTENTION")

        if cnn_drowsy_prob >= 0.70 and face_detected and not signal_disagreement:
            reason_codes.append("CNN_DROWSINESS")

        if signal_disagreement:
            reason_codes.append("SIGNAL_DISAGREEMENT")

        if tracking_state == "LOST":
            reason_codes.append("TRACKING_LOST")
        elif not reason_codes and candidate_state == "NORMAL":
            reason_codes.append("BASELINE_NORMAL")

        # ── 10. Hysteresis State Transition Machine ──────────────────
        state_ranks = {"NORMAL": 0, "FATIGUE_RISK": 1, "DROWSY": 2, "CRITICAL": 3}
        current_rank = state_ranks[self._current_state]
        candidate_rank = state_ranks[candidate_state]

        if candidate_rank > current_rank:
            # Immediate escalation
            self._current_state = candidate_state
            self._pending_lower_state = None
            self._lower_state_candidate_start_mono_ms = None
        elif candidate_rank < current_rank:
            # De-escalation hold times: start timer if not already running
            if self._lower_state_candidate_start_mono_ms is None:
                self._lower_state_candidate_start_mono_ms = t_now
                self._pending_lower_state = candidate_state
            else:
                self._pending_lower_state = candidate_state

            if self._current_state == "CRITICAL":
                cooldown_needed = self.config.state_machine.cooldown_critical_to_drowsy_ms
            elif self._current_state == "DROWSY":
                cooldown_needed = self.config.state_machine.cooldown_drowsy_to_fatigue_ms
            else:
                cooldown_needed = self.config.state_machine.cooldown_fatigue_to_normal_ms

            dwell_ms = t_now - self._lower_state_candidate_start_mono_ms
            if dwell_ms >= cooldown_needed:
                # Step down towards candidate state
                if current_rank == 3: # CRITICAL
                    self._current_state = "FATIGUE_RISK" if candidate_rank <= 1 else "DROWSY"
                elif current_rank == 2: # DROWSY
                    self._current_state = candidate_state
                else: # FATIGUE_RISK
                    self._current_state = candidate_state

                if self._current_state == candidate_state:
                    self._pending_lower_state = None
                    self._lower_state_candidate_start_mono_ms = None
                else:
                    self._lower_state_candidate_start_mono_ms = t_now
        else:
            self._pending_lower_state = None
            self._lower_state_candidate_start_mono_ms = None

        return {
            "state": self._current_state,
            "reason_codes": reason_codes,
            "fatigue_risk_score": round(fatigue_risk_score, 4),
            "signal_strength": round(signal_strength, 4),
            "tracking_state": tracking_state,
            "cnn_drowsy_prob": round(cnn_drowsy_prob, 4),
            "wakefulness_support": round(wakefulness_support, 4),
            "signal_disagreement": signal_disagreement,
            "active_signals": {
                "eyes_closed": is_eye_closed,
                "active_yawn": (self._mouth_state == "YAWNING"),
                "head_drop": (ongoing_nodding_ms > 0),
            },
            "recent_signals": {
                "long_blinks_30s": long_blinks_30s,
                "yawns_60s": yawns_60s,
                "effective_long_blink_weight": round(effective_long_blink_weight, 3),
                "effective_yawn_weight": round(effective_yawn_weight, 3),
            },
            "metrics": {
                "perclos": round(raw_perclos, 4),
                "raw_perclos": round(raw_perclos, 4),
                "perclos_reliable": perclos_reliable,
                "effective_perclos_risk": round(effective_perclos_risk, 4),
                "continuous_open_duration_ms": round(continuous_open_duration_ms, 1),
                "valid_observation_ratio": round(valid_observation_ratio, 4),
                "eye_closure_duration_ms": round(ongoing_eye_closure_ms, 1),
                "yawn_duration_ms": round(ongoing_yawn_ms, 1),
                "blink_rate_per_min": round(blink_rate_per_min, 1),
                "avg_blink_duration_ms": round(avg_blink_duration_ms, 1),
                "long_blink_count": long_blinks_30s,
                "total_blink_count": self._total_blink_count,
                "total_yawn_count": self._total_yawn_count,
                "nodding_duration_ms": round(ongoing_nodding_ms, 1),
                "head_turn_duration_ms": round(ongoing_head_turn_ms, 1),
            },
            "sub_risks": {
                "perclos_risk": round(effective_perclos_risk, 4),
                "raw_perclos_risk": round(raw_perclos_risk, 4),
                "eye_closure_risk": round(eye_closure_risk, 4),
                "long_blinks_risk": round(long_blink_risk, 4),
                "yawn_risk": round(yawn_risk, 4),
                "head_nod_risk": round(head_nod_risk, 4),
                "cnn_risk": round(cnn_risk, 4),
            },
            "raw_signals": {
                "ear": round(ear, 4),
                "mar": round(mar, 4),
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "roll": round(roll, 2),
                "face_detected": face_detected,
            },
            "monotonic_ms": round(t_now, 3),
            "frame_index": frame_index,
        }

    def to_perception_event(self, result: dict[str, Any], frame_index: int = 0) -> PerceptionEvent:
        """Convert engine result dict into a standardized PerceptionEvent."""
        return PerceptionEvent.ok(
            source="driver_state_engine_v2",
            event_type="driver_state",
            frame_index=frame_index,
            latency_ms=0.0,
            confidence=result.get("fatigue_risk_score", 0.0),
            data=result,
            monotonic_ms=result.get("monotonic_ms"),
        )
