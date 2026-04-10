"""
Behavior profile system for demographic-aware crowd simulation.

Provides research-based behavioral parameters that differ between
male/female and age groups (elderly, adult, child).  These multipliers
scale the base movement and fall-risk calculations so the simulation
produces realistic human diversity.

Sources:
  - Helbing & Molnár, "Social Force Model for Pedestrian Dynamics", 1995
  - Fruin, "Pedestrian Planning and Design", 1971
  - Weidmann, "Transporttechnik der Fußgänger", 1993
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BehaviorProfile:
    """
    Immutable behavioral parameter set for a single tracked person.

    All values are *multipliers* relative to the baseline (adult male = 1.0).
    """

    gender: str = "unknown"          # 'male', 'female', 'unknown'
    age_group: str = "adult"         # 'adult', 'elderly', 'child', 'unknown'

    speed_multiplier: float = 1.0    # Scales velocity_threshold_normal
    reaction_factor: float = 1.0     # Scales how fast person reacts to danger
    space_radius: float = 1.0        # Scales personal-space / density pressure
    fall_susceptibility: float = 1.0 # Scales fall-risk probability
    panic_spread: float = 1.0        # Scales stampede propagation factor
    crowd_tolerance: float = 1.0     # Lower = leaves crowd sooner

    @property
    def label(self) -> str:
        parts = []
        if self.gender != "unknown":
            parts.append(self.gender[0].upper())   # M / F
        if self.age_group != "adult" and self.age_group != "unknown":
            parts.append(self.age_group[:3].capitalize())
        return "/".join(parts) if parts else "?"


# ── Pre-built profiles ──────────────────────────────────────────────
MALE_ADULT = BehaviorProfile(
    gender="male", age_group="adult",
    speed_multiplier=1.0, reaction_factor=1.0,
    space_radius=1.0, fall_susceptibility=1.0,
    panic_spread=1.0, crowd_tolerance=1.0,
)

FEMALE_ADULT = BehaviorProfile(
    gender="female", age_group="adult",
    speed_multiplier=0.88, reaction_factor=0.92,
    space_radius=1.15, fall_susceptibility=1.12,
    panic_spread=1.10, crowd_tolerance=0.85,
)

MALE_ELDERLY = BehaviorProfile(
    gender="male", age_group="elderly",
    speed_multiplier=0.68, reaction_factor=1.35,
    space_radius=1.10, fall_susceptibility=1.60,
    panic_spread=0.90, crowd_tolerance=0.70,
)

FEMALE_ELDERLY = BehaviorProfile(
    gender="female", age_group="elderly",
    speed_multiplier=0.62, reaction_factor=1.40,
    space_radius=1.20, fall_susceptibility=1.75,
    panic_spread=0.95, crowd_tolerance=0.65,
)

DEFAULT_PROFILE = MALE_ADULT


def build_profile_from_config(gender: str, age_group: str, demo_cfg) -> BehaviorProfile:
    """
    Build a BehaviorProfile using the values in DemographicConfig.

    This allows runtime override of the research-based defaults via
    the global config without touching this module.
    """
    if gender == "male":
        return BehaviorProfile(
            gender="male", age_group=age_group,
            speed_multiplier=demo_cfg.male_speed_multiplier,
            reaction_factor=demo_cfg.male_reaction_factor,
            space_radius=demo_cfg.male_space_radius,
            fall_susceptibility=demo_cfg.male_fall_susceptibility,
            panic_spread=demo_cfg.male_panic_spread,
            crowd_tolerance=demo_cfg.male_crowd_tolerance,
        )
    elif gender == "female":
        return BehaviorProfile(
            gender="female", age_group=age_group,
            speed_multiplier=demo_cfg.female_speed_multiplier,
            reaction_factor=demo_cfg.female_reaction_factor,
            space_radius=demo_cfg.female_space_radius,
            fall_susceptibility=demo_cfg.female_fall_susceptibility,
            panic_spread=demo_cfg.female_panic_spread,
            crowd_tolerance=demo_cfg.female_crowd_tolerance,
        )
    else:
        return DEFAULT_PROFILE
"""
Behavior profile module for demographic-aware crowd simulation.
"""
