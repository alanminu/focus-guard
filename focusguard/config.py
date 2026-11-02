"""Central configuration for FocusGuard.

Every threshold used by the rule engine lives here so it can be reported in the
paper, tuned per context, and saved alongside results for reproducibility.

Time-based rules follow published protocols where they exist:
  * Long distraction >= 3 s single glance away  (Euro NCAP DSM protocol, Palao et al. 2023)
  * Visual-attention time-sharing (VATS): >= 10 s cumulative off-target in a 30 s window
  * PERCLOS (proportion of time eyes >= 80 % closed) over a 60 s window, drowsy if > 0.15
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass
class Config:
    # ---------------- context ----------------
    context: str = "screen"          # "screen" (e-learning / desk) or "driving"

    # ---------------- calibration (Module C) ----------------
    calibration_s: float = 8.0        # seconds of "normal" behaviour used to build a personal baseline
    min_calibration_frames: int = 40
    adaptive_baseline: bool = True    # slowly track drift while the user is FOCUSED
    adapt_alpha: float = 0.002        # EMA rate for drift tracking (per frame)

    # population defaults (used before / without calibration)
    default_ear: float = 0.28
    default_mar: float = 0.05

    # ---------------- eye closure / drowsiness ----------------
    ear_closed_ratio: float = 0.65    # eyes "closed" when EAR < ratio * personal open-eye EAR
    ear_closed_floor: float = 0.12
    ear_closed_ceil: float = 0.25
    blink_bs_thresh: float = 0.55     # MediaPipe eyeBlink blendshape agreement threshold
    blink_max_s: float = 0.5          # closures shorter than this are blinks
    long_closure_s: float = 1.0       # closure longer than this -> microsleep / drowsy
    perclos_window_s: float = 60.0
    perclos_drowsy: float = 0.15

    # ---------------- mouth / yawning ----------------
    mar_yawn_abs: float = 0.45        # absolute MAR above which the mouth is "wide open"
    mar_yawn_delta: float = 0.30      # ... or baseline + delta, whichever is larger
    jaw_open_bs_thresh: float = 0.45
    yawn_min_s: float = 1.5
    talk_std_thresh: float = 0.035    # MAR oscillation (std over 1 s) that indicates talking
    talk_window_s: float = 1.0
    yawn_window_s: float = 180.0
    yawns_for_drowsy: int = 2

    # ---------------- head pose / gaze (visual distraction) ----------------
    yaw_limit_deg: float = 28.0       # head yaw deviation from personal zero
    pitch_down_limit_deg: float = 18.0
    pitch_up_limit_deg: float = 20.0
    gaze_limit_deg: float = 22.0      # combined head+eye gaze deviation
    eye_gain_deg: float = 35.0        # maps blendshape eye-in-head gaze (-1..1) to degrees
    gap_merge_s: float = 0.25         # short gaps inside an off-target run are bridged

    long_distraction_s: float = 3.0   # Euro NCAP "long distraction"
    vats_window_s: float = 30.0       # Euro NCAP visual attention time sharing
    vats_cumulative_s: float = 10.0
    head_down_s: float = 2.0          # sustained head-down (phone on lap / desk)
    face_absent_s: float = 2.0

    # ---------------- phone (optional object detector) ----------------
    use_phone_detector: bool = False
    phone_every_n: int = 5
    phone_conf: float = 0.35
    phone_min_s: float = 1.0

    # ---------------- attention score ----------------
    score_tau_s: float = 1.5          # EMA time constant for the 0-100 attention index

    extras: dict = field(default_factory=dict)

    # ---------------------------------------------------------------
    @classmethod
    def for_context(cls, context: str) -> "Config":
        c = cls(context=context)
        if context == "driving":
            # Driving: wider tolerated gaze zone (mirrors/instrument cluster are brief glances,
            # handled by the 3 s rule), stricter head-down (phone on lap).
            c.yaw_limit_deg = 32.0
            c.gaze_limit_deg = 25.0
            c.pitch_down_limit_deg = 15.0
            c.head_down_s = 1.5
        return c

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))
