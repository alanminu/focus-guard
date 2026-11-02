"""Optional object-context cue: mobile-phone detection.

Facial cues alone cannot tell "looking down at notes" from "looking down at a
phone". Recent driver-monitoring work fuses face cues with object context
(e.g. YOLO-guided hybrids). We run a COCO-pretrained YOLO11n (class 67 =
"cell phone") through ONNX Runtime every N frames - no PyTorch needed at
runtime. Export once with:  yolo export model=yolo11n.pt format=onnx imgsz=320
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

PHONE_CLASS = 67
DEFAULT_ONNX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "yolo11n.onnx")


class PhoneDetector:
    def __init__(self, onnx_path: str = DEFAULT_ONNX, conf: float = 0.35, imgsz: int = 320):
        import onnxruntime as ort  # imported lazily: optional dependency
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"{onnx_path} missing - see README (phone detector).")
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.imgsz = conf, imgsz

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        return canvas, r, left, top

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[float, Tuple[int, int, int, int]]]:
        img, r, left, top = self._letterbox(frame_bgr)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0]        # (84, N)
        scores = out[4 + PHONE_CLASS]
        keep = np.where(scores > self.conf)[0]
        boxes, confs = [], []
        for i in keep:
            cx, cy, w, h = out[:4, i]
            boxes.append([int((cx - w / 2 - left) / r), int((cy - h / 2 - top) / r), int(w / r), int(h / r)])
            confs.append(float(scores[i]))
        dets = []
        if boxes:
            for j in np.array(cv2.dnn.NMSBoxes(boxes, confs, self.conf, 0.5)).flatten()[:5]:
                x, y, w, h = boxes[j]
                dets.append((confs[j], (x, y, x + w, y + h)))
        return dets


class ThrottledPhoneDetector:
    """Runs the detector every ``every_n`` frames and holds the last result."""

    def __init__(self, every_n: int = 5, **kw):
        self.det = PhoneDetector(**kw)
        self.every_n = every_n
        self.i = 0
        self.last: List = []

    def __call__(self, frame_bgr) -> Tuple[bool, list]:
        if self.i % self.every_n == 0:
            self.last = self.det.detect(frame_bgr)
        self.i += 1
        return bool(self.last), self.last
