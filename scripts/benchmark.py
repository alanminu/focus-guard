"""Per-module latency benchmark (CPU only).

  python scripts/benchmark.py --video data/test.mp4 --frames 600 [--phone]
"""
import argparse, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focusguard.config import Config  # noqa: E402
from focusguard.overlay import draw  # noqa: E402
from focusguard.pipeline import DistractionPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--phone", action="store_true")
    ap.add_argument("--width", type=int, default=0, help="resize width (0 = native)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = Config.for_context("driving")
    cfg.use_phone_detector = a.phone
    pipe = DistractionPipeline(cfg)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rec = {k: [] for k in ("landmarks", "features", "phone", "behaviour", "overlay", "total")}
    i = 0
    while i < a.frames:
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if a.width:
            f = cv2.resize(f, (a.width, int(f.shape[0] * a.width / f.shape[1])))
        out = pipe.process(f, i / fps)
        t0 = cv2.getTickCount()
        draw(f, out)
        ov = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1e3
        if i >= 30:                                   # skip warm-up
            for k in ("landmarks", "features", "phone", "behaviour"):
                rec[k].append(out.timings_ms[k])
            rec["overlay"].append(ov)
            rec["total"].append(out.timings_ms["total"] + ov)
        i += 1
    res = {k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))} for k, v in rec.items() if v}
    res["fps_equivalent"] = 1000.0 / res["total"]["mean_ms"]
    res["resolution"] = f"{int(cap.get(3))}x{int(cap.get(4))}" if not a.width else f"{a.width}w"
    res["phone_detector"] = a.phone
    print(json.dumps(res, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
