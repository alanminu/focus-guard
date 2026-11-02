"""End-to-end pipeline: Module A -> B -> C -> D -> E."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .behavior import BehaviorEngine, FrameState  # noqa: F401
from .calibration import Baseline, Calibrator
from .config import Config
from .detector import DEFAULT_MODEL, FaceLandmarkDetector, FaceObservation
from .features import extract_features
from .monitor import SessionMonitor


@dataclass
class PipelineOutput:
    obs: Optional[FaceObservation]
    feats: Optional[Dict[str, float]]
    state: FrameState
    phone_boxes: list
    timings_ms: Dict[str, float]


class DistractionPipeline:
    def __init__(self, cfg: Optional[Config] = None, model_path: str = DEFAULT_MODEL,
                 profile: Optional[str] = None, calibrate: bool = True):
        self.cfg = cfg or Config()
        self.detector = FaceLandmarkDetector(model_path)
        self.calibrator = Calibrator(self.cfg)
        if profile:                                   # re-use a saved personal baseline
            self.calibrator.baseline = Baseline.from_json(profile)
            self.calibrator.done = True
        elif not calibrate:
            self.calibrator.done = True               # population defaults
        self.engine = BehaviorEngine(self.cfg, self.calibrator.baseline)
        self.monitor = SessionMonitor()
        self.phone = None
        if self.cfg.use_phone_detector:
            from .phone import ThrottledPhoneDetector
            self.phone = ThrottledPhoneDetector(every_n=self.cfg.phone_every_n, conf=self.cfg.phone_conf)

    @property
    def baseline(self) -> Baseline:
        return self.calibrator.baseline

    def process(self, frame_bgr: np.ndarray, t: float) -> PipelineOutput:
        tm = {}
        t0 = time.perf_counter()
        obs = self.detector.detect(frame_bgr, t)
        t1 = time.perf_counter()
        feats = extract_features(obs)
        t2 = time.perf_counter()
        phone_seen, boxes = (False, [])
        if self.phone is not None:
            phone_seen, boxes = self.phone(frame_bgr)
        t3 = time.perf_counter()

        calibrating = not self.calibrator.done
        if calibrating:
            # The engine keeps running on population-default thresholds while the personal
            # baseline is being collected, so nothing that happens in the first seconds is missed.
            if self.calibrator.add(t, feats):
                self.engine.baseline = self.calibrator.baseline
        fs = self.engine.update(t, feats, phone_seen)
        if calibrating:
            fs.reasons.append(f"calibrating {self.calibrator.progress(t)*100:.0f}%")
            fs.metrics["calibrating"] = 1.0
        t4 = time.perf_counter()
        self.monitor.log(fs, feats)
        tm.update(landmarks=(t1 - t0) * 1e3, features=(t2 - t1) * 1e3, phone=(t3 - t2) * 1e3,
                  behaviour=(t4 - t3) * 1e3, total=(time.perf_counter() - t0) * 1e3)
        return PipelineOutput(obs, feats, fs, boxes, tm)

    def save(self, out_dir: str, title: str = "FocusGuard session report") -> Dict:
        import os
        summ = self.monitor.save(out_dir, self.engine.events, title)
        if self.calibrator.baseline.calibrated:
            self.calibrator.baseline.to_json(os.path.join(out_dir, "profile.json"))
        self.cfg.to_json(os.path.join(out_dir, "config.json"))
        return summ

    def close(self):
        self.detector.close()
