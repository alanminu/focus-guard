"""Command-line entry point: live webcam or offline video analysis.

  python -m focusguard.cli --source 0                     # webcam, live HUD
  python -m focusguard.cli --source lecture.mp4 --out runs/lecture --save-video
  python -m focusguard.cli --source 0 --context driving --phone
  python -m focusguard.cli --source 0 --profile runs/me/profile.json   # skip calibration
"""
from __future__ import annotations

import argparse
import os
import time

import cv2

from .config import Config
from .overlay import draw
from .pipeline import DistractionPipeline


def run_stream(source, out_dir=None, display=True, save_video=False, context="screen", phone=False,
               profile=None, no_calibration=False, max_seconds=None, stride=1, mirror=None, quiet=False,
               cfg: Config = None):
    cfg = cfg or Config.for_context(context)
    cfg.use_phone_detector = phone
    live = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    cap = cv2.VideoCapture(int(source) if live else source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if mirror is None:
        mirror = live
    pipe = DistractionPipeline(cfg, profile=profile, calibrate=not no_calibration)
    writer = None
    t_start = time.time()
    i = 0
    fps_ema = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride:
                i += 1
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            t = (time.time() - t_start) if live else i / src_fps
            tic = time.perf_counter()
            out = pipe.process(frame, t)
            dt = time.perf_counter() - tic
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(dt, 1e-3)) if fps_ema else 1.0 / max(dt, 1e-3)
            if display or save_video:
                vis = draw(frame, out, fps_ema)
                if save_video and out_dir:
                    if writer is None:
                        os.makedirs(out_dir, exist_ok=True)
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps / stride, (w, h))
                    writer.write(vis)
                if display:
                    cv2.imshow("FocusGuard (q = quit, r = recalibrate)", vis)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("r"):
                        pipe.calibrator.__init__(cfg)
            i += 1
            if max_seconds and t >= max_seconds:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    summ = pipe.save(out_dir, title=f"FocusGuard - {os.path.basename(str(source))}") if out_dir else pipe.monitor.summary(pipe.engine.events)
    pipe.close()
    if not quiet:
        import json
        print(json.dumps(summ, indent=2))
    return summ, pipe


def main():
    ap = argparse.ArgumentParser(description="FocusGuard real-time distraction detection")
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--out", default=None, help="output folder for logs + report")
    ap.add_argument("--context", default="screen", choices=["screen", "driving"])
    ap.add_argument("--phone", action="store_true", help="enable YOLO phone detector (needs onnxruntime)")
    ap.add_argument("--profile", default=None, help="saved profile.json (skips calibration)")
    ap.add_argument("--no-calibration", action="store_true", help="use population thresholds")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args()
    run_stream(a.source, a.out, display=not a.no_display, save_video=a.save_video, context=a.context,
               phone=a.phone, profile=a.profile, no_calibration=a.no_calibration,
               max_seconds=a.max_seconds, stride=a.stride)


if __name__ == "__main__":
    main()
