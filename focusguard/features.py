"""Module B - Attention Attribute Model (feature extraction).

Turns the raw 478 landmarks / blendshapes / pose matrix into a compact,
interpretable vector of attention attributes:

  ear_l, ear_r, ear      Eye Aspect Ratio (Soukupova & Cech 2016), both eyes
  mar                    Mouth Aspect Ratio (inner lips / mouth width)
  yaw, pitch, roll       Head pose in degrees from the facial transformation matrix
  iris_h, iris_v         Iris position inside the eye opening (-0.5 .. 0.5)
  bs_eye_yaw, bs_eye_pitch   Eye-in-head gaze from blendshapes (eyeLookIn/Out/Up/Down)
  blink_bs               Mean eyeBlink blendshape score
  jaw_open               jawOpen blendshape score
  face_scale             Inter-ocular distance / image width (distance-to-camera proxy)

All distances are computed in *pixel* space. The first prototype computed
Euclidean distances on MediaPipe's normalised (x, y), which are scaled by width
and height separately, so EAR/MAR were distorted on non-square (16:9) frames.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .detector import FaceObservation

# ---- MediaPipe Face Mesh indices (478-point topology) ----
# "right" / "left" are from the SUBJECT's point of view.
R_EYE_EAR = [33, 160, 158, 133, 153, 144]   # p1..p6
L_EYE_EAR = [362, 385, 387, 263, 373, 380]
R_EYE_CORNERS = (33, 133)                  # outer, inner
L_EYE_CORNERS = (263, 362)
R_EYE_LIDS = (159, 145)                    # upper, lower
L_EYE_LIDS = (386, 374)
R_IRIS = [468, 469, 470, 471, 472]         # 468 = centre
L_IRIS = [473, 474, 475, 476, 477]         # 473 = centre
MOUTH_VERT = [(13, 14), (82, 87), (312, 317)]
MOUTH_CORNERS = (61, 291)
NOSE_TIP = 1

FEATURE_NAMES = ["ear_l", "ear_r", "ear", "mar", "yaw", "pitch", "roll",
                 "iris_h", "iris_v", "bs_eye_yaw", "bs_eye_pitch", "blink_bs",
                 "jaw_open", "face_scale", "face_cx", "face_cy"]


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(p: np.ndarray, idx) -> float:
    p1, p2, p3, p4, p5, p6 = (p[i] for i in idx)
    horiz = _d(p1, p4)
    if horiz < 1e-6:
        return float("nan")
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * horiz)


def mouth_aspect_ratio(p: np.ndarray) -> float:
    width = _d(p[MOUTH_CORNERS[0]], p[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return float("nan")
    return float(np.mean([_d(p[a], p[b]) for a, b in MOUTH_VERT]) / width)


def rotation_to_euler(R: np.ndarray):
    """Rotation matrix -> (yaw, pitch, roll) in degrees.

    Convention R = Ry(yaw) @ Rx(pitch) @ Rz(roll) in MediaPipe's canonical-face
    frame (x right, y up, z toward the camera).  Signs are normalised so that
    the rule engine only relies on *deviation from the personal baseline*,
    except pitch, which is reported as positive = looking UP, negative = head
    DOWN (verified on driving footage).
    """
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    # remove any scale in the transformation matrix
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    sp = -R[1, 2]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(sp) < 0.9999:
        yaw = math.atan2(R[0, 2], R[2, 2])
        roll = math.atan2(R[1, 0], R[1, 1])
    else:  # gimbal lock
        yaw = math.atan2(-R[2, 0], R[0, 0])
        roll = 0.0
    return math.degrees(yaw), -math.degrees(pitch), math.degrees(roll)


def euler_to_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of :func:`rotation_to_euler` (used by tests)."""
    y, p, r = map(math.radians, (yaw, -pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def _iris_offset(p: np.ndarray, corners, lids, iris_c) -> tuple:
    """Iris centre position within the eye, normalised: h in [-0.5, 0.5] along the
    corner axis (image-left -> image-right), v perpendicular (down positive)."""
    a, b = p[corners[0]][:2], p[corners[1]][:2]
    if a[0] > b[0]:
        a, b = b, a                                  # make axis point image-left -> image-right
    axis = b - a
    L2 = float(axis @ axis)
    if L2 < 1e-6:
        return float("nan"), float("nan")
    c = p[iris_c][:2]
    h = float((c - a) @ axis) / L2 - 0.5
    mid = (p[lids[0]][:2] + p[lids[1]][:2]) / 2.0
    normal = np.array([-axis[1], axis[0]]) / math.sqrt(L2)
    if normal[1] < 0:
        normal = -normal                             # point "down" in image coordinates
    v = float((c - mid) @ normal) / math.sqrt(L2)
    return h, v


def extract_features(obs: Optional[FaceObservation]) -> Optional[Dict[str, float]]:
    if obs is None:
        return None
    p = obs.landmarks_px
    f: Dict[str, float] = {}
    f["ear_r"] = eye_aspect_ratio(p, R_EYE_EAR)
    f["ear_l"] = eye_aspect_ratio(p, L_EYE_EAR)
    f["ear"] = float(np.nanmean([f["ear_l"], f["ear_r"]]))
    f["mar"] = mouth_aspect_ratio(p)

    if obs.transform is not None:
        f["yaw"], f["pitch"], f["roll"] = rotation_to_euler(obs.transform)
    else:
        f["yaw"] = f["pitch"] = f["roll"] = float("nan")

    hr, vr = _iris_offset(p, R_EYE_CORNERS, R_EYE_LIDS, R_IRIS[0])
    hl, vl = _iris_offset(p, L_EYE_CORNERS, L_EYE_LIDS, L_IRIS[0])
    f["iris_h"] = float(np.nanmean([hr, hl]))
    f["iris_v"] = float(np.nanmean([vr, vl]))

    bs = obs.blendshapes
    g = lambda k: bs.get(k, 0.0)
    # positive = eyes rotate toward the subject's LEFT (image right for a mirrored-free camera)
    f["bs_eye_yaw"] = 0.5 * (g("eyeLookOutLeft") + g("eyeLookInRight") - g("eyeLookInLeft") - g("eyeLookOutRight"))
    f["bs_eye_pitch"] = 0.5 * (g("eyeLookUpLeft") + g("eyeLookUpRight") - g("eyeLookDownLeft") - g("eyeLookDownRight"))
    f["blink_bs"] = 0.5 * (g("eyeBlinkLeft") + g("eyeBlinkRight"))
    f["jaw_open"] = g("jawOpen")

    io = _d(p[R_EYE_CORNERS[0]], p[L_EYE_CORNERS[0]])
    f["face_scale"] = io / obs.width
    f["face_cx"] = float(p[NOSE_TIP][0] / obs.width)
    f["face_cy"] = float(p[NOSE_TIP][1] / obs.height)
    return f
