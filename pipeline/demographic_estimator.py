"""
Lightweight demographic estimator for tracked persons.

Assigns gender (and optionally age-group) to each tracked person using
bounding-box heuristics and pose-based cues.  Designed to be swapped out
for a proper classifier model in the future.

Gender estimation strategies
----------------------------
1. **heuristic** – Uses shoulder-width-to-hip-width ratio from pose landmarks
   when available, falls back to bbox height/width ratio statistics.
2. **random_seeded** – Assigns gender deterministically from track_id so results
   are reproducible but respect the configured male_ratio.
3. **model** – (placeholder) Load a pretrained gender classifier.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from pipeline.behavior_profile import (
    BehaviorProfile,
    build_profile_from_config,
    DEFAULT_PROFILE,
)


class DemographicEstimator:
    """
    Estimates gender and age-group for each tracked person and returns
    a `BehaviorProfile` that parameterises downstream metrics.
    """

    def __init__(self, demo_cfg):
        """
        Args:
            demo_cfg: DemographicConfig from config.py
        """
        self.cfg = demo_cfg
        self.enabled = demo_cfg.enabled
        self.method = demo_cfg.gender_method
        self.male_ratio = demo_cfg.male_ratio

        # Cache: track_id → BehaviorProfile (assigned once per track lifetime)
        self._profile_cache: Dict[int, BehaviorProfile] = {}

        logger.info(
            f"DemographicEstimator initialised  enabled={self.enabled}  "
            f"method={self.method}  male_ratio={self.male_ratio}"
        )

    # ── Public API ───────────────────────────────────────────────────
    def assign_profile(
        self,
        track_id: int,
        bbox: List[int],
        pose_features: Optional[Dict[str, float]] = None,
    ) -> BehaviorProfile:
        """
        Return the BehaviorProfile for *track_id*.

        On first call for a track_id the profile is estimated and cached.
        Subsequent calls return the cached profile for consistency.
        """
        if not self.enabled:
            return DEFAULT_PROFILE

        if track_id in self._profile_cache:
            return self._profile_cache[track_id]

        gender = self._estimate_gender(track_id, bbox, pose_features)
        age_group = self._estimate_age_group(track_id, bbox)
        profile = build_profile_from_config(gender, age_group, self.cfg)
        self._profile_cache[track_id] = profile
        return profile

    def get_cached_profile(self, track_id: int) -> BehaviorProfile:
        """Return cached profile or DEFAULT_PROFILE."""
        return self._profile_cache.get(track_id, DEFAULT_PROFILE)

    def get_demographic_summary(self) -> Dict[str, int]:
        """Return counts of male / female / unknown across active profiles."""
        counts = {"male": 0, "female": 0, "unknown": 0}
        for profile in self._profile_cache.values():
            counts[profile.gender] = counts.get(profile.gender, 0) + 1
        return counts

    def cleanup(self, active_track_ids: List[int]):
        """Remove stale profiles that are no longer tracked."""
        stale = [tid for tid in self._profile_cache if tid not in active_track_ids]
        for tid in stale:
            del self._profile_cache[tid]

    def reset(self):
        """Clear all cached profiles."""
        self._profile_cache.clear()
        logger.info("DemographicEstimator reset")

    # ── Gender estimation strategies ─────────────────────────────────
    def _estimate_gender(
        self,
        track_id: int,
        bbox: List[int],
        pose_features: Optional[Dict[str, float]] = None,
    ) -> str:
        if self.method == "heuristic":
            return self._gender_heuristic(track_id, bbox, pose_features)
        elif self.method == "random_seeded":
            return self._gender_random(track_id)
        else:
            # 'model' or future — fall back to random
            return self._gender_random(track_id)

    def _gender_heuristic(
        self,
        track_id: int,
        bbox: List[int],
        pose_features: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Use pose shoulder-to-hip ratio when present; otherwise use a
        deterministic hash of the track_id so the assignment is stable.

        Men tend to have broader shoulders relative to hips (ratio > 1.0),
        while women tend to have a ratio closer to or below 1.0.

        When pose data is unavailable we use a bbox height/width ratio
        as a very weak signal combined with the seeded fallback.
        """
        # Try pose-based shoulder/hip ratio
        if pose_features is not None:
            landmarks = pose_features.get("landmarks")
            if landmarks:
                ls = landmarks.get("LEFT_SHOULDER")
                rs = landmarks.get("RIGHT_SHOULDER")
                lh = landmarks.get("LEFT_HIP")
                rh = landmarks.get("RIGHT_HIP")
                if ls and rs and lh and rh:
                    shoulder_w = abs(ls[0] - rs[0])
                    hip_w = abs(lh[0] - rh[0])
                    if hip_w > 1e-3:
                        ratio = shoulder_w / hip_w
                        if ratio > 1.05:
                            return "male"
                        elif ratio < 0.95:
                            return "female"
                        # ambiguous – fall through to hash

        # Height/width bbox heuristic: taller/narrower → slight male bias
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        hw_ratio = height / width

        # Combine with seeded hash for determinism
        seed = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16) % 1000
        bias = 0.0
        if hw_ratio > 3.2:
            bias = 0.08  # slightly more likely male (taller relative)
        elif hw_ratio < 2.5:
            bias = -0.08  # slightly more likely female

        threshold = self.male_ratio + bias
        return "male" if (seed / 1000.0) < threshold else "female"

    def _gender_random(self, track_id: int) -> str:
        """Deterministic pseudo-random assignment based on track_id."""
        seed = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16) % 1000
        return "male" if (seed / 1000.0) < self.male_ratio else "female"

    # ── Age-group estimation (placeholder) ───────────────────────────
    def _estimate_age_group(self, track_id: int, bbox: List[int]) -> str:
        """
        Placeholder age-group estimator.

        Uses bbox height as a very rough proxy:
        - Very short bboxes → child
        - Normal → adult
        - (Future: use face-based age estimator)
        """
        if not self.cfg.age_estimation_enabled:
            return "adult"

        height = max(1, bbox[3] - bbox[1])
        if height < 80:
            return "child"
        # Can extend with a model later
        return "adult"
