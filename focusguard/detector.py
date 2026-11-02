"""Module A - Facial Landmark Detector.

Wraps the MediaPipe Tasks *Face Landmarker* (478 3-D landmarks incl. iris,
52 ARKit-style blendshape scores and a 4x4 facial transformation matrix).
The legacy ``mp.solutions.face_mesh`` API used by the first prototype has been
removed from recent MediaPipe releases (>=0.10.30), so this module uses the
supported Tasks API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
except ImportError as e:  # pragma: no cover
    raise ImportError("mediapipe>=0.10.14 is required: pip install mediapipe") from e

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


@dataclass
class FaceObservation:
    """Raw output of Module A for one frame (pixel coordinates)."""
    landmarks_px: np.ndarray          # (478, 3): x, y in pixels, z scaled by image width
    blendshapes: dict                 # name -> score in [0, 1]
    transform: Optional[np.ndarray]   # (4, 4) facial transformation matrix or None
    width: int
    height: int


class FaceLandmarkDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL, num_faces: int = 1,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}. Download it with:\n"
                f"  curl -L -o {model_path} {MODEL_URL}")
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, t_sec: float) -> Optional[FaceObservation]:
        """Run the landmarker on one BGR frame. ``t_sec`` must be monotonic."""
        h, w = frame_bgr.shape[:2]
        ts = int(round(t_sec * 1000))
        if ts <= self._last_ts:           # MediaPipe VIDEO mode requires strictly increasing ts
            ts = self._last_ts + 1
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.face_landmarks:
            return None
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]], dtype=np.float64)
        bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        tm = np.array(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
        return FaceObservation(pts, bs, tm, w, h)

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
