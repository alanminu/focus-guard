import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from focusguard.behavior import BehaviorEngine, RunTracker, WindowSum
from focusguard.calibration import Baseline, Calibrator
from focusguard.config import Config
from focusguard.features import (FEATURE_NAMES, R_EYE_EAR, euler_to_rotation, eye_aspect_ratio,
                                 rotation_to_euler)
from focusguard.simulate import make_scenario

FPS = 30.0


def base_feats(**kw):
    f = {k: 0.0 for k in FEATURE_NAMES}
    f.update(ear=0.30, ear_l=0.30, ear_r=0.30, mar=0.05, bs_eye_pitch=-0.3, blink_bs=0.1, jaw_open=0.02,
             face_scale=0.1)
    f.update(kw)
    return f


def run(engine, seconds, t0=0.0, **kw):
    states = []
    n = int(seconds * FPS)
    for i in range(n):
        states.append(engine.update(t0 + i / FPS, base_feats(**kw)))
    return states, t0 + n / FPS


def calibrated_engine(cfg=None):
    cfg = cfg or Config.for_context("driving")
    b = Calibrator.fit([base_feats() for _ in range(100)])
    return BehaviorEngine(cfg, b)


# ---------------------------------------------------------------- features
def test_ear_formula():
    p = np.zeros((478, 3))
    # p1 (0,0) p4 (10,0); vertical pairs 3 px apart -> EAR = (3+3)/(2*10) = 0.3
    pts = {0: (0, 0), 1: (3, -1.5), 2: (7, -1.5), 3: (10, 0), 4: (7, 1.5), 5: (3, 1.5)}
    for k, (x, y) in pts.items():
        p[R_EYE_EAR[k]] = (x, y, 0)
    assert eye_aspect_ratio(p, R_EYE_EAR) == pytest.approx(0.3)


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (30, -10, 5), (-45, 20, -12), (10, 35, 0)])
def test_euler_roundtrip(yaw, pitch, roll):
    R = euler_to_rotation(yaw, pitch, roll)
    y, p, r = rotation_to_euler(R)
    assert (y, p, r) == pytest.approx((yaw, pitch, roll), abs=1e-6)


def test_euler_ignores_scale():
    R = 1.7 * euler_to_rotation(20, -5, 3)
    assert rotation_to_euler(R) == pytest.approx((20, -5, 3), abs=1e-6)


# ---------------------------------------------------------------- helpers
def test_run_tracker_bridges_gaps():
    rt = RunTracker(gap_s=0.25)
    flags = [1] * 30 + [0] * 3 + [1] * 30 + [0] * 20
    ended = [rt.update(i / FPS, bool(f)) for i, f in enumerate(flags)]
    runs = [e for e in ended if e]
    assert len(runs) == 1 and runs[0][1] - runs[0][0] == pytest.approx(62 / FPS, abs=1e-6)


def test_window_sum():
    w = WindowSum(10.0)
    for i in range(600):                    # 20 s, flag on for the last 5 s
        w.add(i / FPS, 1 / FPS, i >= 450)
    assert w.on == pytest.approx(5.0, abs=0.05)
    assert w.fraction() == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------- calibration
def test_calibration_rejects_blinks_and_yawns():
    frames = [base_feats(ear=0.26) for _ in range(200)]
    frames += [base_feats(ear=0.05) for _ in range(30)]           # blinks
    frames += [base_feats(mar=0.8, jaw_open=0.7) for _ in range(40)]  # a yawn
    b = Calibrator.fit(frames)
    assert b.median["ear"] == pytest.approx(0.26, abs=1e-6)
    assert b.median["mar"] == pytest.approx(0.05, abs=1e-6)


def test_personal_ear_threshold():
    cfg = Config()
    small = Calibrator.fit([base_feats(ear=0.22) for _ in range(60)])
    large = Calibrator.fit([base_feats(ear=0.34) for _ in range(60)])
    assert small.ear_closed(cfg) < large.ear_closed(cfg)
    # a 0.19 EAR is "closed" for large eyes but "open" for small eyes
    assert 0.19 < large.ear_closed(cfg)
    assert 0.19 > small.ear_closed(cfg)


# ---------------------------------------------------------------- behaviour rules
def test_blink_is_not_distraction():
    eng = calibrated_engine()
    _, t = run(eng, 3)
    st, t = run(eng, 0.2, t, ear=0.05, ear_l=0.05, blink_bs=0.8)
    st2, t = run(eng, 2, t)
    assert all(s.state == "FOCUSED" for s in st + st2)
    assert any(e["type"] == "blink" for e in eng.events)


def test_microsleep_triggers_drowsy_after_1s():
    eng = calibrated_engine()
    _, t = run(eng, 3)
    st, t = run(eng, 2.0, t, ear=0.05, ear_l=0.05, blink_bs=0.9)
    first = next(i for i, s in enumerate(st) if s.state == "DROWSY") / FPS
    assert 0.9 <= first <= 1.1


def test_short_glance_vs_long_distraction():
    eng = calibrated_engine()
    _, t = run(eng, 3)
    st, t = run(eng, 2.0, t, yaw=45)
    assert all(s.state in ("GLANCE", "FOCUSED") for s in st)
    _, t = run(eng, 5, t)
    st, t = run(eng, 5.0, t, yaw=45)
    first = next(i for i, s in enumerate(st) if s.state == "DISTRACTED") / FPS
    assert 2.9 <= first <= 3.2          # Euro NCAP 3 s long-distraction rule
    assert "long_glance" in st[-1].reasons


def test_eye_only_gaze_shift_detected():
    eng = calibrated_engine()
    _, t = run(eng, 3)
    # head still, eyes rotated far to one side (bs units) -> combined gaze beyond limit
    st, t = run(eng, 4.0, t, bs_eye_yaw=-0.9)
    assert st[-1].state == "DISTRACTED"


def test_vats_cumulative_rule():
    eng = calibrated_engine()
    _, t = run(eng, 3)
    states = []
    for _ in range(6):                  # 6 x 2.5 s glances with 2 s looks back = 15 s off in 27 s
        s1, t = run(eng, 2.5, t, yaw=40)
        s2, t = run(eng, 2.0, t)
        states += s1 + s2
    assert any("vats" in s.reasons for s in states)
    assert not any("long_glance" in s.reasons for s in states)


def test_yawn_event():
    eng = calibrated_engine()
    _, t = run(eng, 3)
    _, t = run(eng, 3.0, t, mar=0.8, jaw_open=0.7)
    _, t = run(eng, 1.0, t)
    assert sum(e["type"] == "yawn" for e in eng.events) == 1


def test_face_absent():
    eng = calibrated_engine()
    _, t = run(eng, 2)
    states = [eng.update(t + i / FPS, None) for i in range(int(3 * FPS))]
    assert states[-1].state == "ABSENT"
    assert states[int(1.0 * FPS)].state != "ABSENT"


def test_camera_offset_handled_by_calibration():
    """A camera mounted to the side gives a constant yaw of 30 deg - calibrated engine must not alert."""
    cfg = Config.for_context("screen")
    b = Calibrator.fit([base_feats(yaw=30, pitch=12) for _ in range(100)])
    eng = BehaviorEngine(cfg, b)
    st, _ = run(eng, 10, yaw=30, pitch=12)
    assert st[-1].state == "FOCUSED"
    raw = BehaviorEngine(cfg, Baseline.population_default(cfg))
    st2, _ = run(raw, 10, yaw=30, pitch=12)
    assert st2[-1].state == "DISTRACTED"


def test_synthetic_scenario_runs():
    sc = make_scenario(duration_s=60, seed=3)
    assert sc.X.shape == (1800, len(FEATURE_NAMES))
    eng = BehaviorEngine(Config(), Baseline.population_default(Config()))
    for i, t in enumerate(sc.t):
        eng.update(float(t), sc.feats(i))
    assert np.isfinite(eng.score)
