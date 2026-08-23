"""
RoadGuard AI — Road Hazard Multi-Object Tracker (Phase 3)
Tracks road hazards across frames with persistent hazard_id to prevent duplicate alerts.
"""

from __future__ import annotations

import math
from typing import Optional

from core.hazard_config import HazardTrackerConfig
from core.hazard.types import HazardDetection, HazardSample, TrackedHazard


class HazardTracker:
    """Associates detections across consecutive frames using spatial proximity & IoU."""

    def __init__(self, config: Optional[HazardTrackerConfig] = None) -> None:
        self.config = config or HazardTrackerConfig()
        self._tracks: dict[int, TrackedHazard] = {}
        self._next_hazard_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_hazard_id = 1

    @staticmethod
    def _compute_iou(b1: list[float], b2: list[float]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = a1 + a2 - inter_area
        return inter_area / union_area if union_area > 1e-6 else 0.0

    @staticmethod
    def _compute_distance(c1: list[float], c2: list[float]) -> float:
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def update(self, detections: list[HazardDetection], monotonic_ms: float) -> list[TrackedHazard]:
        matched_track_ids: set[int] = set()
        unmatched_detections: list[int] = list(range(len(detections)))

        active_track_ids = [tid for tid, trk in self._tracks.items() if trk.tracking_state != "LOST"]

        # 1. Greedy match via IoU or Spatial Center Distance
        if active_track_ids and detections:
            cost_matrix: list[tuple[float, int, int]] = []
            for tid in active_track_ids:
                trk = self._tracks[tid]
                trk_cx, trk_cy = trk.history[-1].center_x, trk.history[-1].center_y
                for d_idx, det in enumerate(detections):
                    iou = self._compute_iou(trk.current_bbox, det.bbox_xyxy)
                    dist = self._compute_distance([trk_cx, trk_cy], det.bbox_center)
                    if iou >= self.config.iou_threshold or dist <= self.config.spatial_distance_threshold:
                        score = iou + max(0.0, 1.0 - dist)
                        cost_matrix.append((score, tid, d_idx))

            cost_matrix.sort(key=lambda x: x[0], reverse=True)
            for _, tid, d_idx in cost_matrix:
                if tid in matched_track_ids or d_idx not in unmatched_detections:
                    continue
                matched_track_ids.add(tid)
                unmatched_detections.remove(d_idx)

                det = detections[d_idx]
                trk = self._tracks[tid]
                trk.current_bbox = det.bbox_xyxy
                trk.last_seen_mono_ms = monotonic_ms
                trk.frames_seen += 1
                trk.frames_missing = 0
                if trk.frames_seen >= self.config.min_hits_to_confirm:
                    trk.tracking_state = "TRACKED"

                trk.history.append(HazardSample(
                    monotonic_ms=monotonic_ms,
                    center_x=det.bbox_center[0],
                    center_y=det.bbox_center[1],
                    bottom_center_x=det.bbox_center[0],
                    bottom_center_y=det.bbox_xyxy[3],
                    width=det.bbox_width,
                    height=det.bbox_height,
                    area=det.bbox_area,
                    confidence=det.confidence,
                ))

                cutoff = monotonic_ms - self.config.max_track_history_ms
                trk.history = [s for s in trk.history if s.monotonic_ms >= cutoff]

        # 2. Missing tracks
        for tid in active_track_ids:
            if tid not in matched_track_ids:
                trk = self._tracks[tid]
                trk.frames_missing += 1
                if trk.frames_missing > self.config.max_missing_frames:
                    trk.tracking_state = "LOST"
                else:
                    trk.tracking_state = "DEGRADED"

        # 3. New hazard tracks
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            hid = self._next_hazard_id
            self._next_hazard_id += 1

            sample = HazardSample(
                monotonic_ms=monotonic_ms,
                center_x=det.bbox_center[0],
                center_y=det.bbox_center[1],
                bottom_center_x=det.bbox_center[0],
                bottom_center_y=det.bbox_xyxy[3],
                width=det.bbox_width,
                height=det.bbox_height,
                area=det.bbox_area,
                confidence=det.confidence,
            )
            self._tracks[hid] = TrackedHazard(
                hazard_id=hid,
                class_name=det.class_name,
                tracking_state="NEW",
                current_bbox=det.bbox_xyxy,
                first_seen_mono_ms=monotonic_ms,
                last_seen_mono_ms=monotonic_ms,
                frames_seen=1,
                frames_missing=0,
                history=[sample],
            )

        # Cleanup lost tracks older than 5 seconds
        to_del = [tid for tid, trk in self._tracks.items() if trk.tracking_state == "LOST" and (monotonic_ms - trk.last_seen_mono_ms) > 5000.0]
        for tid in to_del:
            del self._tracks[tid]

        return [trk for trk in self._tracks.values() if trk.tracking_state != "LOST"]
