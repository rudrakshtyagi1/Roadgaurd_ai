"""
RoadGuard AI — Multi-Object Tracker
Maintains persistent track IDs and lifecycle states (NEW, TRACKED, DEGRADED, LOST)
using IoU association and temporal sample buffers.
"""

from __future__ import annotations

import math
from typing import Optional

from core.collision_config import TrackerConfig
from core.collision.types import ObjectDetection, TrackedObject, TrackSample


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    return (inter_area / union_area) if union_area > 0 else 0.0


class MultiObjectTracker:
    """
    Associates incoming detections with active tracks using IoU matching.
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self._next_track_id = 1
        self._tracks: dict[int, TrackedObject] = {}

    def reset(self) -> None:
        """Reset all active tracks."""
        self._next_track_id = 1
        self._tracks.clear()

    def update(self, detections: list[ObjectDetection], monotonic_ms: float) -> list[TrackedObject]:
        """
        Match detections to existing tracks and update tracking lifecycles.
        """
        active_track_ids = list(self._tracks.keys())
        unmatched_detections = list(range(len(detections)))
        matched_pairs: list[tuple[int, int]] = []  # (track_id, det_idx)

        # 1. Greedy IoU matching (favoring same-class)
        if active_track_ids and detections:
            # Build cost pairs: (iou, track_id, det_idx)
            cost_pairs: list[tuple[float, int, int]] = []
            for tid in active_track_ids:
                track = self._tracks[tid]
                for det_idx in range(len(detections)):
                    det = detections[det_idx]
                    iou = compute_iou(track.current_bbox, det.bbox_xyxy)
                    if track.class_id == det.class_id:
                        iou += 0.10  # Class bonus
                    if iou >= self.config.iou_threshold:
                        cost_pairs.append((iou, tid, det_idx))

            # Sort descending by score
            cost_pairs.sort(key=lambda x: x[0], reverse=True)
            used_tracks: set[int] = set()
            used_dets: set[int] = set()

            for _, tid, det_idx in cost_pairs:
                if tid not in used_tracks and det_idx not in used_dets:
                    used_tracks.add(tid)
                    used_dets.add(det_idx)
                    matched_pairs.append((tid, det_idx))

            unmatched_detections = [i for i in range(len(detections)) if i not in used_dets]
            unmatched_track_ids = [tid for tid in active_track_ids if tid not in used_tracks]
        else:
            unmatched_track_ids = active_track_ids

        # 2. Update matched tracks
        for tid, det_idx in matched_pairs:
            track = self._tracks[tid]
            det = detections[det_idx]

            track.current_bbox = det.bbox_xyxy
            track.last_seen_mono_ms = monotonic_ms
            track.frames_seen += 1
            track.frames_missing = 0

            # State transition NEW -> TRACKED
            if track.tracking_state == "NEW" and track.frames_seen >= self.config.min_hits_to_track:
                track.tracking_state = "TRACKED"
            elif track.tracking_state == "DEGRADED":
                track.tracking_state = "TRACKED"

            # Append sample
            cx, cy = det.bbox_center
            bx = cx
            by = det.bbox_xyxy[3]  # bottom center
            sqrt_area = math.sqrt(max(1e-6, det.bbox_area))
            fused_scale = 0.60 * det.bbox_height + 0.40 * sqrt_area

            track.history.append(TrackSample(
                monotonic_ms=monotonic_ms,
                center_x=cx,
                center_y=cy,
                bottom_center_x=bx,
                bottom_center_y=by,
                width=det.bbox_width,
                height=det.bbox_height,
                area=det.bbox_area,
                scale=fused_scale,
                confidence=det.confidence,
            ))

            # Prune old samples
            cutoff_ms = monotonic_ms - self.config.max_track_history_ms
            track.history = [s for s in track.history if s.monotonic_ms >= cutoff_ms]

        # 3. Handle unmatched active tracks (missing frames)
        for tid in unmatched_track_ids:
            track = self._tracks[tid]
            track.frames_missing += 1
            if track.frames_missing > self.config.max_missing_frames:
                track.tracking_state = "LOST"
            else:
                track.tracking_state = "DEGRADED"

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            tid = self._next_track_id
            self._next_track_id += 1

            cx, cy = det.bbox_center
            sqrt_area = math.sqrt(max(1e-6, det.bbox_area))
            fused_scale = 0.60 * det.bbox_height + 0.40 * sqrt_area
            sample = TrackSample(
                monotonic_ms=monotonic_ms,
                center_x=cx,
                center_y=cy,
                bottom_center_x=cx,
                bottom_center_y=det.bbox_xyxy[3],
                width=det.bbox_width,
                height=det.bbox_height,
                area=det.bbox_area,
                scale=fused_scale,
                confidence=det.confidence,
            )

            self._tracks[tid] = TrackedObject(
                track_id=tid,
                class_name=det.class_name,
                class_id=det.class_id,
                tracking_state="NEW",
                current_bbox=det.bbox_xyxy,
                first_seen_mono_ms=monotonic_ms,
                last_seen_mono_ms=monotonic_ms,
                frames_seen=1,
                frames_missing=0,
                history=[sample],
            )

        # 5. Clean up LOST tracks older than tolerance
        pruned_tracks = {
            tid: trk for tid, trk in self._tracks.items()
            if trk.tracking_state != "LOST" or trk.frames_missing <= self.config.max_missing_frames + 5
        }
        self._tracks = pruned_tracks

        return list(self._tracks.values())
