"""
Tracking utilities for assigning persistent IDs to detected persons.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort

    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("DeepSORT not available. Falling back to IoU tracker.")


class IoUTracker:
    """Fast IoU-based tracker for crowded scenes when speed matters most."""

    def __init__(self, max_age: int = 30, match_iou_threshold: float = 0.2):
        self.max_age = max_age
        self.match_iou_threshold = match_iou_threshold
        self.next_id = 0
        self.tracks: Dict[int, Dict] = {}
        self.track_history: Dict[int, List[Dict]] = {}

    def update(self, detections: List[Dict], frame=None) -> List[Dict]:
        del frame
        results = []
        matched_track_ids = set()
        used_detection_indices = set()

        track_items = sorted(
            self.tracks.items(), key=lambda item: item[1].get("age", 0), reverse=False
        )

        for track_id, track in track_items:
            best_det_idx = None
            best_iou = 0.0
            for det_idx, det in enumerate(detections):
                if det_idx in used_detection_indices:
                    continue
                iou = self._compute_iou(track["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx is None or best_iou < self.match_iou_threshold:
                self.tracks[track_id]["age"] += 1
                continue

            det = detections[best_det_idx]
            used_detection_indices.add(best_det_idx)
            matched_track_ids.add(track_id)
            hit_streak = self.tracks[track_id].get("hit_streak", 0) + 1
            self.tracks[track_id] = {
                "bbox": det["bbox"],
                "age": 0,
                "hit_streak": hit_streak,
                "confidence": det["confidence"],
            }
            results.append(
                {
                    "track_id": track_id,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "age": 0,
                    "hit_streak": hit_streak,
                }
            )

        for det_idx, det in enumerate(detections):
            if det_idx in used_detection_indices:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "bbox": det["bbox"],
                "age": 0,
                "hit_streak": 1,
                "confidence": det["confidence"],
            }
            matched_track_ids.add(track_id)
            results.append(
                {
                    "track_id": track_id,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "age": 0,
                    "hit_streak": 1,
                }
            )

        expired_track_ids = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids and track["age"] > self.max_age:
                expired_track_ids.append(track_id)
        for track_id in expired_track_ids:
            del self.tracks[track_id]

        for result in results:
            self.track_history.setdefault(result["track_id"], []).append(result)

        return results

    def reset(self):
        self.next_id = 0
        self.tracks = {}
        self.track_history = {}

    def get_active_track_ids(self) -> List[int]:
        return list(self.tracks.keys())

    @staticmethod
    def _compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0


class PersonTracker:
    """
    Tracker for assigning persistent IDs to detected persons.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: Optional[int] = 100,
        tracker_type: str = "iou",
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.tracker_type = tracker_type
        self.track_history: Dict[int, List[Dict]] = {}
        self.use_deepsort = False

        self._build_tracker()

    def _build_tracker(self):
        if self.tracker_type == "deepsort" and DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                embedder="mobilenet",
                half=torch.cuda.is_available(),
                bgr=True,
                embedder_gpu=torch.cuda.is_available(),
            )
            self.use_deepsort = True
            logger.info(f"Initialized DeepSORT tracker with max_age={self.max_age}")
            return

        self.tracker = IoUTracker(
            max_age=self.max_age,
            match_iou_threshold=max(0.1, 1.0 - self.max_iou_distance),
        )
        self.use_deepsort = False
        if self.tracker_type == "deepsort" and not DEEPSORT_AVAILABLE:
            logger.warning("DeepSORT requested but unavailable. Using IoU tracker.")
        logger.info(f"Initialized IoU tracker with max_age={self.max_age}")

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        if self.use_deepsort:
            if not detections:
                tracked_objects = self.tracker.update_tracks([], frame=frame)
            else:
                deepsort_detections = []
                for det in detections:
                    bbox = det["bbox"]
                    ltwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    deepsort_detections.append(
                        (ltwh, det["confidence"], det.get("class_id", 0))
                    )
                tracked_objects = self.tracker.update_tracks(
                    deepsort_detections, frame=frame
                )

            results = []
            for track in tracked_objects:
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb()
                hit_streak = getattr(track, "hit_streak", getattr(track, "hits", 1))
                results.append(
                    {
                        "track_id": track.track_id,
                        "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                        "confidence": float(getattr(track, "det_conf", 1.0)),
                        "age": int(getattr(track, "age", 0)),
                        "hit_streak": int(hit_streak),
                    }
                )
        else:
            results = self.tracker.update(detections, frame)

        for result in results:
            self.track_history.setdefault(result["track_id"], []).append(result)

        logger.debug(f"Tracking: {len(detections)} detections -> {len(results)} tracks")
        return results

    def get_track_by_id(self, track_id: int) -> Optional[Dict]:
        if track_id not in self.track_history:
            return None
        history = self.track_history[track_id]
        return {"track_id": track_id, "history": history, "length": len(history)}

    def remove_old_tracks(self, max_history_length: int = 1000):
        for track_id in list(self.track_history.keys()):
            if len(self.track_history[track_id]) > max_history_length:
                self.track_history[track_id] = self.track_history[track_id][
                    -max_history_length:
                ]

    def get_active_track_ids(self) -> List[int]:
        if self.use_deepsort:
            active_ids = []
            for track in self.tracker.tracker.tracks:
                if track.is_confirmed() and track.time_since_update == 0:
                    active_ids.append(track.track_id)
            return active_ids
        return self.tracker.get_active_track_ids()

    def reset(self):
        self._build_tracker()
        self.track_history = {}
        logger.info("Tracker reset")
