"""Module D - Artificial Emotional-Intelligence Behaviour Model (temporal rule engine).

Per-frame cues (eyes closed, off-target gaze, head down, mouth wide open,
talking, face absent, phone) are converted into *behavioural events* using
explicit temporal logic, then fused into a state and a 0-100 attention index.

    state priority:  ABSENT > DROWSY > DISTRACTED > GLANCE > FOCUSED

Why temporal logic instead of a per-frame classifier: attention is a state
that exists over time. A single frame with low EAR is a blink (normal); the
same value for 1 s is a microsleep. A 1 s mirror check is normal driving; a
3 s glance away is a Euro NCAP "long distraction". The first prototype
classified isolated frames, so it flagged every blink as distraction.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .calibration import Baseline, adapt
from .config import Config

STATES = ["FOCUSED", "GLANCE", "DISTRACTED", "DROWSY", "ABSENT", "CALIBRATING"]
STATE_TARGET_SCORE = {"FOCUSED": 100.0, "GLANCE": 75.0, "DISTRACTED": 25.0,
                      "DROWSY": 15.0, "ABSENT": 0.0, "CALIBRATING": 100.0}


# ---------------------------------------------------------------- helpers
class RunTracker:
    """Tracks continuous runs of a boolean signal; bridges gaps shorter than ``gap_s``."""

    def __init__(self, gap_s: float = 0.0):
        self.gap_s = gap_s
        self.active = False
        self.start = 0.0
        self.last_true = 0.0

    def update(self, t: float, flag: bool) -> Optional[Tuple[float, float]]:
        ended = None
        if flag:
            if not self.active:
                self.active, self.start = True, t
            self.last_true = t
        elif self.active and (t - self.last_true) > self.gap_s:
            ended = (self.start, self.last_true)
            self.active = False
        return ended

    def duration(self, t: float) -> float:
        return (self.last_true - self.start) if self.active else 0.0

    def is_on(self, t: float) -> bool:
        """True only while the underlying signal is currently true (not during a bridged gap)."""
        return self.active and (t - self.last_true) < 1e-9


class WindowSum:
    """Time-weighted sum of a boolean over a sliding window."""

    def __init__(self, window_s: float):
        self.w = window_s
        self.q: Deque[Tuple[float, float, float]] = deque()
        self.on = 0.0
        self.total = 0.0

    def add(self, t: float, dt: float, flag: bool):
        v = dt if flag else 0.0
        self.q.append((t, v, dt))
        self.on += v
        self.total += dt
        while self.q and self.q[0][0] < t - self.w:
            _, v0, d0 = self.q.popleft()
            self.on -= v0
            self.total -= d0

    def fraction(self) -> float:
        return self.on / self.total if self.total > 1e-6 else 0.0


@dataclass
class FrameState:
    t: float
    state: str
    score: float
    reasons: List[str] = field(default_factory=list)
    cues: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- engine
class BehaviorEngine:
    def __init__(self, cfg: Config, baseline: Optional[Baseline] = None):
        self.cfg = cfg
        self.baseline = baseline or Baseline.population_default(cfg)
        c = cfg
        self.closure = RunTracker(gap_s=0.07)
        self.mouth = RunTracker(gap_s=0.3)
        self.off = RunTracker(gap_s=c.gap_merge_s)
        self.down = RunTracker(gap_s=c.gap_merge_s)
        self.absent = RunTracker(gap_s=0.0)
        self.phone = RunTracker(gap_s=0.5)
        self.talk = RunTracker(gap_s=0.5)
        self.perclos = WindowSum(c.perclos_window_s)
        self.vats = WindowSum(c.vats_window_s)
        self.mar_hist: Deque[Tuple[float, float]] = deque()
        self.blinks: Deque[float] = deque()
        self.yawns: Deque[float] = deque()
        self.events: List[Dict] = []
        self.last_t: Optional[float] = None
        self.last_off_t = -1e9
        self.score = 100.0
        self.t_first = None

    # ------------------------------------------------------------
    def _event(self, kind: str, start: float, end: float):
        self.events.append({"type": kind, "start": round(start, 3), "end": round(end, 3),
                            "duration": round(end - start, 3)})

    def update(self, t: float, feats: Optional[Dict[str, float]], phone_seen: bool = False) -> FrameState:
        c, b = self.cfg, self.baseline
        if self.t_first is None:
            self.t_first = t
        dt = 0.0 if self.last_t is None else float(np.clip(t - self.last_t, 0.0, 0.5))
        self.last_t = t
        cues = dict(eyes_closed=False, off_target=False, head_down=False, mouth_wide=False,
                    talking=False, face_absent=feats is None, phone=bool(phone_seen))
        m: Dict[str, float] = {}

        # ---- face presence ----
        ended = self.absent.update(t, feats is None)
        if ended and ended[1] - ended[0] >= c.face_absent_s:
            self._event("face_absent", *ended)
        off_ended = None
        if feats is None:
            # A face missing for > 0.5 s counts as "eyes off target" (driver turned fully away /
            # left the desk). Shorter dropouts are tracker glitches and are ignored.
            if self.absent.duration(t) >= 0.5:
                self.vats.add(t, dt, True)
                off_ended = self.off.update(t, True)
                self.last_off_t = t
        else:
            # ---- eyes ----
            ear_thr = b.ear_closed(c)
            ear = feats.get("ear", np.nan)
            closed = bool(np.isfinite(ear) and ear < ear_thr and
                          (feats.get("blink_bs", 1.0) > 0.35 or ear < 0.8 * ear_thr))
            cues["eyes_closed"] = closed
            self.perclos.add(t, dt, closed)
            ended = self.closure.update(t, closed)
            if ended:
                d = ended[1] - ended[0] + dt
                if d <= c.blink_max_s:
                    self.blinks.append(t)
                    self._event("blink", ended[0], ended[1])
                elif d >= c.long_closure_s:
                    self._event("long_eye_closure", ended[0], ended[1])

            # ---- mouth ----
            mar = feats.get("mar", np.nan)
            wide = bool((np.isfinite(mar) and mar > b.mar_open(c)) or feats.get("jaw_open", 0.0) > c.jaw_open_bs_thresh)
            cues["mouth_wide"] = wide
            ended = self.mouth.update(t, wide)
            if ended and (ended[1] - ended[0]) >= c.yawn_min_s:
                self.yawns.append(ended[1])
                self._event("yawn", ended[0], ended[1])
            self.mar_hist.append((t, mar if np.isfinite(mar) else 0.0))
            while self.mar_hist and self.mar_hist[0][0] < t - c.talk_window_s:
                self.mar_hist.popleft()
            mar_std = float(np.std([v for _, v in self.mar_hist])) if len(self.mar_hist) > 3 else 0.0
            talking = (mar_std > c.talk_std_thresh) and not wide
            cues["talking"] = talking
            ended = self.talk.update(t, talking)
            if ended and ended[1] - ended[0] >= c.extras.get("talk_min_s", 3.0):
                self._event("talking", *ended)
            m["mar_std"] = mar_std

            # ---- head pose + gaze ----
            yaw_dev = feats.get("yaw", 0.0) - b.median.get("yaw", 0.0)
            pitch_dev = feats.get("pitch", 0.0) - b.median.get("pitch", 0.0)
            if closed:   # eye-in-head blendshapes are meaningless with closed lids
                ey = ep = 0.0
            else:
                ey = feats.get("bs_eye_yaw", 0.0) - b.median.get("bs_eye_yaw", 0.0)
                ep = feats.get("bs_eye_pitch", 0.0) - b.median.get("bs_eye_pitch", 0.0)
            gaze_yaw = yaw_dev - c.eye_gain_deg * ey
            gaze_pitch = pitch_dev + c.eye_gain_deg * ep
            off = (abs(gaze_yaw) > c.gaze_limit_deg or abs(yaw_dev) > c.yaw_limit_deg or
                   gaze_pitch < -c.pitch_down_limit_deg - 4 or gaze_pitch > c.pitch_up_limit_deg)
            head_down = pitch_dev < -c.pitch_down_limit_deg
            cues["off_target"], cues["head_down"] = bool(off), bool(head_down)
            off_ended = self.off.update(t, off)
            if off:
                self.last_off_t = t
            self.vats.add(t, dt, off)
            ended = self.down.update(t, head_down)
            if ended and ended[1] - ended[0] >= c.head_down_s:
                self._event("head_down", *ended)
            m.update(yaw_dev=yaw_dev, pitch_dev=pitch_dev, gaze_yaw=gaze_yaw, gaze_pitch=gaze_pitch,
                     ear_thr=ear_thr)

        # completed off-target runs -> events
        if off_ended:
            s0, e0 = off_ended
            self._event("glance" if e0 - s0 < c.long_distraction_s else "long_distraction", s0, e0)

        ended = self.phone.update(t, cues["phone"])
        if ended and ended[1] - ended[0] >= c.phone_min_s:
            self._event("phone", *ended)

        # ---- windows ----
        while self.blinks and self.blinks[0] < t - 60.0:
            self.blinks.popleft()
        while self.yawns and self.yawns[0] < t - c.yawn_window_s:
            self.yawns.popleft()
        elapsed = max(1e-6, t - self.t_first)
        m.update(
            perclos=self.perclos.fraction(),
            blink_rate=len(self.blinks) * 60.0 / min(60.0, max(elapsed, 10.0)),
            yawns_recent=float(len(self.yawns)),
            closure_s=self.closure.duration(t),
            off_s=self.off.duration(t) if self.off.active else 0.0,
            vats_s=self.vats.on,
            absent_s=self.absent.duration(t),
            head_down_s=self.down.duration(t),
        )

        # ---- fuse into state ----
        reasons: List[str] = []
        if m["absent_s"] >= c.face_absent_s:
            state = "ABSENT"
            reasons.append("face_absent")
        else:
            drowsy = []
            if m["closure_s"] >= c.long_closure_s:
                drowsy.append("eyes_closed")
            if self.perclos.total >= 0.95 * c.perclos_window_s and m["perclos"] > c.perclos_drowsy:
                drowsy.append("perclos")
            if self.mouth.duration(t) >= c.yawn_min_s:
                drowsy.append("yawning")
            distracted = []
            if m["off_s"] >= c.long_distraction_s:
                distracted.append("long_glance")
            if m["vats_s"] >= c.vats_cumulative_s and (t - self.last_off_t) < 2.0:
                distracted.append("vats")
            if m["head_down_s"] >= c.head_down_s:
                distracted.append("head_down")
            if self.phone.duration(t) >= c.phone_min_s:
                distracted.append("phone")
            talk_min = c.extras.get("talk_min_s", 3.0)
            if c.context == "screen" and self.talk.duration(t) >= talk_min:
                distracted.append("talking")
            if drowsy:
                state, reasons = "DROWSY", drowsy + distracted
            elif distracted:
                state, reasons = "DISTRACTED", distracted
            elif cues["off_target"]:
                state, reasons = "GLANCE", ["glance"]
            else:
                state = "FOCUSED"
        if len(self.yawns) >= c.yawns_for_drowsy:
            reasons.append("frequent_yawns")

        # ---- attention index ----
        target = STATE_TARGET_SCORE[state]
        if state in ("FOCUSED", "GLANCE"):
            target -= min(30.0, 100.0 * max(0.0, m["perclos"] - 0.05))
            if "frequent_yawns" in reasons:
                target -= 10.0
            if cues["talking"] and c.context == "screen":
                target -= 5.0
        a = 1.0 - math.exp(-dt / c.score_tau_s) if dt > 0 else 1.0
        self.score = (1 - a) * self.score + a * target

        # ---- drift adaptation ----
        if (c.adaptive_baseline and b.calibrated and state == "FOCUSED" and feats is not None
                and not cues["eyes_closed"] and not cues["mouth_wide"]):
            adapt(b, feats, c.adapt_alpha)

        return FrameState(t=t, state=state, score=float(self.score), reasons=reasons, cues=cues, metrics=m)
