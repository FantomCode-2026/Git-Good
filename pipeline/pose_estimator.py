"""
Pose estimation utilities for fall-risk features.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


class PoseEstimator:
    """
    Extract per-person pose features from an image crop.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enabled: bool = True,
        task_model_path: Optional[str] = None,
    ):
        self.enabled = enabled and MEDIAPIPE_AVAILABLE
        self.landmark_names: List[str] = []
        self.pose = None
        self.mp_pose = None
        self.mode = None
        self.status_message = "Pose features are disabled."

        if self.enabled:
            solutions = getattr(mp, "solutions", None)
            pose_module = getattr(solutions, "pose", None) if solutions is not None else None
            if pose_module is not None:
                self.mp_pose = pose_module
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.landmark_names = [
                    landmark.name for landmark in self.mp_pose.PoseLandmark
                ]
                self.mode = "solutions"
                self.status_message = "Pose enabled via MediaPipe solutions pose."
                logger.info("Initialized MediaPipe pose estimator")
            else:
                self._try_init_tasks_api(
                    min_detection_confidence,
                    min_tracking_confidence,
                    task_model_path,
                )
        elif enabled:
            self.status_message = "Pose disabled: MediaPipe is not installed."
            logger.warning("MediaPipe not installed. Pose-based features are disabled.")

    def _try_init_tasks_api(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        task_model_path: Optional[str],
    ):
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.core.base_options import BaseOptions as BaseOptionsLib
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
        except Exception as exc:
            self.enabled = False
            self.status_message = (
                f"Pose disabled: MediaPipe tasks API unavailable ({exc})."
            )
            logger.warning(
                f"MediaPipe tasks API is unavailable ({exc}). Pose-based features are disabled."
            )
            return

        model_path = Path(task_model_path) if task_model_path else None
        if model_path is None or not model_path.exists():
            self.enabled = False
            self.status_message = (
                "Pose disabled: pose_landmarker.task model file not found."
            )
            logger.warning(
                "MediaPipe tasks API is installed, but no pose landmarker model was found. "
                "Set config.paths.pose_landmarker_task or place pose_landmarker.task in the project root."
            )
            return

        try:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=BaseOptionsLib.Delegate.CPU,
                ),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
            self.landmark_names = [landmark.name for landmark in vision.PoseLandmark]
            self.mode = "tasks"
            self.status_message = (
                f"Pose enabled via MediaPipe Tasks using model {model_path}."
            )
            logger.info(f"Initialized MediaPipe Tasks pose estimator using {model_path}")
        except Exception as exc:
            self.enabled = False
            self.status_message = (
                f"Pose disabled: MediaPipe Tasks initialization failed ({exc})."
            )
            logger.warning(
                f"Failed to initialize MediaPipe Tasks pose estimator ({exc}). "
                "Pose-based features are disabled."
            )

    def extract_pose_features(
        self, frame: np.ndarray, bbox: List[int]
    ) -> Optional[Dict[str, float]]:
        """
        Extract pose landmarks and derived features for a tracked person.
        """
        if not self.enabled or self.pose is None:
            return None

        crop, offset = self._crop_with_margin(frame, bbox)
        if crop is None:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        landmarks = self._extract_landmarks_from_crop(crop_rgb, crop.shape[:2], offset)
        if not landmarks:
            return None

        derived = self._compute_pose_features(landmarks, bbox)
        derived["landmarks"] = landmarks
        return derived

    def _extract_landmarks_from_crop(
        self,
        crop_rgb: np.ndarray,
        crop_shape: Tuple[int, int],
        offset: Tuple[int, int],
    ) -> Optional[Dict[str, Tuple[float, float, float]]]:
        h, w = crop_shape

        if self.mode == "solutions":
            result = self.pose.process(crop_rgb)
            if not result.pose_landmarks:
                return None

            landmarks = {}
            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                if idx >= len(self.landmark_names):
                    continue
                landmarks[self.landmark_names[idx]] = (
                    landmark.x * w + offset[0],
                    landmark.y * h + offset[1],
                    landmark.visibility,
                )
            return landmarks

        if self.mode == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            result = self.pose.detect(mp_image)
            if not result.pose_landmarks:
                return None

            landmarks = {}
            first_pose = result.pose_landmarks[0]
            for idx, landmark in enumerate(first_pose):
                if idx >= len(self.landmark_names):
                    continue
                landmarks[self.landmark_names[idx]] = (
                    landmark.x * w + offset[0],
                    landmark.y * h + offset[1],
                    getattr(landmark, "visibility", 1.0),
                )
            return landmarks

        return None

    def _crop_with_margin(
        self, frame: np.ndarray, bbox: List[int], margin_ratio: float = 0.15
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)

        return frame[y1:y2, x1:x2], (x1, y1)

    @staticmethod
    def _midpoint(
        a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float]]:
        if a is None or b is None:
            return None
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    def _compute_pose_features(
        self, landmarks: Dict[str, Tuple[float, float, float]], bbox: List[int]
    ) -> Dict[str, float]:
        left_shoulder = landmarks.get("LEFT_SHOULDER")
        right_shoulder = landmarks.get("RIGHT_SHOULDER")
        left_hip = landmarks.get("LEFT_HIP")
        right_hip = landmarks.get("RIGHT_HIP")
        left_ankle = landmarks.get("LEFT_ANKLE")
        right_ankle = landmarks.get("RIGHT_ANKLE")
        nose = landmarks.get("NOSE")

        shoulder_mid = self._midpoint(left_shoulder, right_shoulder)
        hip_mid = self._midpoint(left_hip, right_hip)
        ankle_mid = self._midpoint(left_ankle, right_ankle)

        bbox_width = max(1.0, bbox[2] - bbox[0])
        bbox_height = max(1.0, bbox[3] - bbox[1])
        bbox_center_x = (bbox[0] + bbox[2]) / 2.0

        body_angle = 0.0
        center_of_mass_shift = 0.0
        posture_drop = 0.0
        step_irregularity = 0.0

        if shoulder_mid is not None and hip_mid is not None:
            dx = shoulder_mid[0] - hip_mid[0]
            dy = hip_mid[1] - shoulder_mid[1]
            body_angle = float(np.degrees(np.arctan2(dx, max(dy, 1e-6))))

        if shoulder_mid is not None and hip_mid is not None:
            com_x = (shoulder_mid[0] + hip_mid[0]) / 2.0
            center_of_mass_shift = float(abs(com_x - bbox_center_x) / bbox_width)

        if nose is not None and ankle_mid is not None:
            posture_drop = float(np.clip((ankle_mid[1] - nose[1]) / bbox_height, 0.0, 1.5))

        if left_ankle is not None and right_ankle is not None:
            step_irregularity = float(abs(left_ankle[1] - right_ankle[1]) / bbox_height)

        return {
            "body_angle_deg": body_angle,
            "center_of_mass_shift": center_of_mass_shift,
            "posture_height_ratio": posture_drop,
            "step_irregularity": step_irregularity,
        }

    def close(self):
        if self.pose is not None:
            self.pose.close()

    def get_status(self) -> Dict[str, Optional[str]]:
        """
        Return a machine-readable pose subsystem status.
        """
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "message": self.status_message,
        }
