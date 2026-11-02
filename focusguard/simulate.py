"""Synthetic attribute streams with ground-truth events.

Used to unit-test and benchmark the *decision logic* (Modules C-D) in isolation
from the vision front-end: we know exactly when a long distraction or a
microsleep happens, so detection recall, false alarms per hour and latency
can be measured. This does NOT validate landmark accuracy - that needs real
video (see scripts/eval_sust.py and the recording protocol).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .features import FEATURE_NAMES

EVENT_TYPES = ["short_glance", "long_distraction", "head_down", "microsleep", "yawn"]
ALERT_EVENTS = {"long_distraction": 3.0, "head_down": 2.0, "microsleep": 1.0, "yawn": 1.5, "vats": 0.0}


@dataclass
class Subject:
    ear_open: float
    yaw0: float
    pitch0: float
    mar0: float
    blink_rate: float  # per minute


@dataclass
class Scenario:
    t: np.ndarray
    X: np.ndarray                      # (n, len(FEATURE_NAMES)); NaN rows = face missing
    events: List[Dict] = field(default_factory=list)
    subject: Optional[Subject] = None

    def feats(self, i: int) -> Optional[Dict[str, float]]:
        row = self.X[i]
        if not np.isfinite(row[0]):
            return None
        return dict(zip(FEATURE_NAMES, map(float, row)))


def random_subject(rng: np.random.Generator) -> Subject:
    return Subject(ear_open=rng.uniform(0.22, 0.34), yaw0=rng.uniform(-15, 15), pitch0=rng.uniform(-10, 15),
                   mar0=rng.uniform(0.02, 0.12), blink_rate=rng.uniform(10, 22))


def make_scenario(duration_s: float = 600.0, fps: float = 30.0, seed: int = 0, calib_s: float = 10.0,
                  n_events: Optional[Dict[str, int]] = None, dropout_rate: float = 0.01,
                  spike_rate: float = 0.005) -> Scenario:
    rng = np.random.default_rng(seed)
    s = random_subject(rng)
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    idx = {k: i for i, k in enumerate(FEATURE_NAMES)}
    X = np.zeros((n, len(FEATURE_NAMES)))
    # ---- attentive baseline with physiological noise ----
    drift = np.cumsum(rng.normal(0, 0.03, n))
    drift -= np.convolve(drift, np.ones(int(fps * 20)) / int(fps * 20), mode="same")
    ear = s.ear_open + rng.normal(0, 0.012, n)
    yaw = s.yaw0 + rng.normal(0, 2.0, n) + 2 * drift
    pitch = s.pitch0 + rng.normal(0, 1.8, n)
    mar = np.clip(s.mar0 + rng.normal(0, 0.012, n), 0, None)
    jaw = np.clip(rng.normal(0.03, 0.015, n), 0, 1)
    eye_yaw = rng.normal(0, 0.03, n)
    eye_pitch = -0.3 + rng.normal(0, 0.03, n)
    blink_bs = np.clip(rng.normal(0.12, 0.04, n), 0, 1)

    # ---- ground-truth events (non-overlapping, after calibration) ----
    if n_events is None:
        k = duration_s / 600.0
        # alert-worthy events are placed first so they always fit
        n_events = {"long_distraction": round(8 * k), "head_down": round(5 * k), "microsleep": round(5 * k),
                    "yawn": round(5 * k), "short_glance": round(30 * k)}
    dur_range = {"short_glance": (0.6, 2.2), "long_distraction": (3.6, 8.0), "head_down": (2.8, 6.0),
                 "microsleep": (1.3, 3.0), "yawn": (2.5, 5.0)}
    occupied = np.zeros(n, bool)
    occupied[: int((calib_s + 2) * fps)] = True
    events = []
    for kind, k in n_events.items():
        for _ in range(k):
            for _try in range(200):
                d = rng.uniform(*dur_range[kind])
                st = rng.uniform(calib_s + 2, duration_s - d - 2)
                a, b = int(st * fps), int((st + d) * fps)
                pad = int(3 * fps)
                if not occupied[max(0, a - pad):b + pad].any():
                    occupied[a:b] = True
                    events.append({"type": kind, "start": st, "end": st + d})
                    break
    for e in events:
        a, b = int(e["start"] * fps), int(e["end"] * fps)
        ramp = np.minimum(1, np.minimum(np.arange(b - a), np.arange(b - a)[::-1]) / (0.15 * fps))
        if e["type"] in ("short_glance", "long_distraction"):
            side = rng.choice([-1, 1])
            amp = rng.uniform(35, 60)
            yaw[a:b] = s.yaw0 + side * amp * ramp + rng.normal(0, 2, b - a)
            eye_yaw[a:b] = -side * 0.25 * ramp          # eyes lead the head in the same direction
        elif e["type"] == "head_down":
            pitch[a:b] = s.pitch0 - rng.uniform(25, 38) * ramp
            eye_pitch[a:b] = -0.3 - 0.35 * ramp
            ear[a:b] = s.ear_open * (1 - 0.25 * ramp)     # lids lower when looking down
        elif e["type"] == "microsleep":
            ear[a:b] = s.ear_open * (1 - 0.75 * ramp) + rng.normal(0, 0.01, b - a)
            blink_bs[a:b] = 0.12 + 0.75 * ramp
            pitch[a:b] -= rng.uniform(0, 12) * ramp        # possible head nod
        elif e["type"] == "yawn":
            mar[a:b] = s.mar0 + rng.uniform(0.55, 0.85) * ramp
            jaw[a:b] = 0.03 + 0.65 * ramp
            ear[a:b] = s.ear_open * (1 - 0.3 * ramp)
    # ---- natural blinks everywhere (normal behaviour, must NOT alert) ----
    tb = 0.0
    while True:
        tb += rng.exponential(60.0 / s.blink_rate)
        if tb >= duration_s:
            break
        d = rng.uniform(0.1, 0.35)
        a, b = int(tb * fps), int((tb + d) * fps)
        if b - a < 1:
            continue
        ear[a:b] = s.ear_open * rng.uniform(0.15, 0.35)
        blink_bs[a:b] = rng.uniform(0.6, 0.9)
    # ---- tracker artefacts ----
    spikes = rng.random(n) < spike_rate
    yaw[spikes] += rng.choice([-1, 1], spikes.sum()) * rng.uniform(30, 50, spikes.sum())
    ear[rng.random(n) < spike_rate] *= 0.5
    X[:, idx["ear"]] = X[:, idx["ear_l"]] = X[:, idx["ear_r"]] = ear
    X[:, idx["mar"]], X[:, idx["yaw"]], X[:, idx["pitch"]] = mar, yaw, pitch
    X[:, idx["roll"]] = rng.normal(0, 2, n)
    X[:, idx["bs_eye_yaw"]], X[:, idx["bs_eye_pitch"]] = eye_yaw, eye_pitch
    X[:, idx["blink_bs"]], X[:, idx["jaw_open"]] = blink_bs, jaw
    X[:, idx["face_scale"]], X[:, idx["face_cx"]], X[:, idx["face_cy"]] = 0.1, 0.5, 0.5
    X[:, idx["iris_h"]] = X[:, idx["iris_v"]] = 0.0
    # dropouts in short bursts
    i = 0
    while i < n:
        if rng.random() < dropout_rate / 3:
            L = int(rng.integers(1, 6))
            X[i:i + L] = np.nan
            i += L
        i += 1
    events += vats_ground_truth(events, t)
    return Scenario(t=t, X=X, events=sorted(events, key=lambda e: e["start"]), subject=s)


def vats_ground_truth(events: List[Dict], t: np.ndarray, window_s: float = 30.0, cum_s: float = 10.0) -> List[Dict]:
    """Euro NCAP visual-attention-time-sharing ground truth: intervals where the true
    cumulative eyes-off-road time in the last 30 s reaches 10 s while the eyes are off."""
    fps = 1.0 / (t[1] - t[0])
    off = np.zeros(len(t), bool)
    for e in events:
        if e["type"] in ("short_glance", "long_distraction", "head_down"):
            off[(t >= e["start"]) & (t <= e["end"])] = True
    w = int(window_s * fps)
    cs = np.cumsum(np.r_[0, off.astype(float)])
    roll = (cs[1:] - cs[np.maximum(0, np.arange(len(t)) + 1 - w)]) / fps
    m = off & (roll >= cum_s)
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            out.append({"type": "vats", "start": float(t[i]), "end": float(t[j])})
            i = j + 1
        else:
            i += 1
    return out
