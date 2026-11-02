"""Module C - Threshold Model with personal calibration.

The first prototype used either fixed thresholds or a Random Forest trained on
hand-labelled frames. Neither generalises: open-eye EAR differs by ~40 %
between people (Dewi et al. 2022 found the optimal threshold moved from 0.2 to
0.18 for smaller eyes), and "looking at the screen" depends on camera placement.

Here the threshold model is *annotation-free*: during the first few seconds the
user behaves normally and the calibrator builds a robust personal baseline
(median + MAD, blink frames rejected). All decision thresholds are then
expressed relative to that baseline. The baseline can be saved as a user
profile and slowly adapted while the user is FOCUSED (drift in posture,
lighting, camera).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np

from .config import Config

BASE_KEYS = ["ear", "mar", "yaw", "pitch", "roll", "bs_eye_yaw", "bs_eye_pitch", "jaw_open", "face_scale"]


@dataclass
class Baseline:
    median: Dict[str, float] = field(default_factory=dict)
    mad: Dict[str, float] = field(default_factory=dict)
    n_frames: int = 0
    calibrated: bool = False
    warnings: List[str] = field(default_factory=list)

    # ---------- derived personal thresholds ----------
    def ear_closed(self, cfg: Config) -> float:
        e = self.median.get("ear", cfg.default_ear)
        return float(np.clip(cfg.ear_closed_ratio * e, cfg.ear_closed_floor, cfg.ear_closed_ceil))

    def mar_open(self, cfg: Config) -> float:
        m = self.median.get("mar", cfg.default_mar)
        return float(max(cfg.mar_yawn_abs, m + cfg.mar_yawn_delta))

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Baseline":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def population_default(cls, cfg: Config) -> "Baseline":
        med = {k: 0.0 for k in BASE_KEYS}
        med.update(ear=cfg.default_ear, mar=cfg.default_mar, face_scale=0.1)
        return cls(median=med, mad={k: 0.0 for k in BASE_KEYS}, n_frames=0, calibrated=False)


class Calibrator:
    """Collects frames for ``cfg.calibration_s`` seconds of face-present video."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.buf: List[Dict[str, float]] = []
        self.t0: Optional[float] = None
        self.done = False
        self.baseline = Baseline.population_default(cfg)

    def progress(self, t: float) -> float:
        if self.t0 is None:
            return 0.0
        return min(1.0, (t - self.t0) / self.cfg.calibration_s)

    def add(self, t: float, feats: Optional[Dict[str, float]]) -> bool:
        """Feed one frame. Returns True once calibration has just completed."""
        if self.done or feats is None:
            return False
        if self.t0 is None:
            self.t0 = t
        self.buf.append(feats)
        if t - self.t0 >= self.cfg.calibration_s and len(self.buf) >= self.cfg.min_calibration_frames:
            self.baseline = self.fit(self.buf)
            self.done = True
            return True
        return False

    @staticmethod
    def fit(frames: List[Dict[str, float]]) -> Baseline:
        med, mad = {}, {}
        ear = np.array([f.get("ear", np.nan) for f in frames], dtype=float)
        ear_med0 = np.nanmedian(ear)
        # reject blink frames when estimating open-eye EAR
        open_mask = ear > 0.8 * ear_med0
        jaw = np.array([f.get("jaw_open", 0.0) for f in frames], dtype=float)
        mar = np.array([f.get("mar", np.nan) for f in frames], dtype=float)
        # reject yawning / talking-wide frames when estimating the resting mouth
        rest_mouth = (jaw < 0.3) & (mar < 0.35)
        for k in BASE_KEYS:
            v = np.array([f.get(k, np.nan) for f in frames], dtype=float)
            if k == "ear":
                v = v[open_mask]
            elif k in ("mar", "jaw_open") and rest_mouth.sum() >= 10:
                v = v[rest_mouth]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            m = float(np.median(v))
            med[k] = m
            mad[k] = float(1.4826 * np.median(np.abs(v - m)))
        quality = []
        if mad.get("yaw", 0) > 10 or mad.get("pitch", 0) > 10:
            quality.append("head moved a lot during calibration")
        if rest_mouth.mean() < 0.6:
            quality.append("mouth open/yawning during calibration")
        if open_mask.mean() < 0.7:
            quality.append("eyes closed often during calibration")
        return Baseline(median=med, mad=mad, n_frames=len(frames), calibrated=True, warnings=quality)


def adapt(baseline: Baseline, feats: Dict[str, float], alpha: float,
          keys=("ear", "mar", "yaw", "pitch", "bs_eye_yaw", "bs_eye_pitch")) -> None:
    """Exponential drift tracking; call only on frames judged FOCUSED."""
    for k in keys:
        v = feats.get(k)
        if v is None or not np.isfinite(v) or k not in baseline.median:
            continue
        baseline.median[k] = (1 - alpha) * baseline.median[k] + alpha * v
