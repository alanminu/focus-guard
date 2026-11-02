"""Batch feature extraction (Modules A+B) for a folder of videos.

Stores one compressed .npz per video with per-frame features (NaN when no face)
plus two per-clip descriptors used for *pseudo-subject grouping* when a dataset
does not publish subject IDs:
  shape  - mean aligned landmark geometry (identity cue)
  scene  - mean 12x12 colour thumbnail of the frame (vehicle / room cue)

  python scripts/extract_features.py --videos "data/SUST/**/*.mp4" --out features/sust --stride 2 --workers 2
"""
import argparse, glob, os, sys, time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focusguard.detector import FaceLandmarkDetector  # noqa: E402
from focusguard.features import FEATURE_NAMES, extract_features  # noqa: E402

SHAPE_IDX = np.arange(0, 468, 3)


def _aligned_shape(p):
    le, re = p[33, :2], p[263, :2]
    c = (le + re) / 2
    d = re - le
    s = np.linalg.norm(d) + 1e-6
    ang = -np.arctan2(d[1], d[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return ((p[SHAPE_IDX, :2] - c) @ R.T) / s


def process(args):
    path, out_dir, stride = args
    name = os.path.splitext(os.path.basename(path))[0]
    label = os.path.basename(os.path.dirname(path))
    dst = os.path.join(out_dir, f"{label}__{name}.npz")
    if os.path.exists(dst):
        return dst
    det = FaceLandmarkDetector()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, ts, shapes, thumbs = [], [], [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            t = i / fps
            obs = det.detect(f, t)
            ft = extract_features(obs)
            rows.append([np.nan] * len(FEATURE_NAMES) if ft is None else [ft[k] for k in FEATURE_NAMES])
            ts.append(t)
            if obs is not None and len(shapes) < 40:
                shapes.append(_aligned_shape(obs.landmarks_px))
            if len(thumbs) < 20 and i % (stride * 8) == 0:
                thumbs.append(cv2.resize(f, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32).ravel())
        i += 1
    det.close()
    np.savez_compressed(dst, X=np.array(rows, np.float32), t=np.array(ts, np.float32), fps=fps,
                        names=np.array(FEATURE_NAMES), label=label, clip=name,
                        shape=np.mean(shapes, 0).ravel() if shapes else np.zeros(len(SHAPE_IDX) * 2),
                        scene=np.mean(thumbs, 0) if thumbs else np.zeros(432), size=np.array([cap.get(3), cap.get(4)]))
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vids = sorted(glob.glob(a.videos, recursive=True))
    print(f"{len(vids)} videos", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(process, [(v, a.out, a.stride) for v in vids]), 1):
            if k % 50 == 0 or k == len(vids):
                el = time.time() - t0
                print(f"{k}/{len(vids)}  {el/60:.1f} min elapsed, ETA {(len(vids)-k)*el/k/60:.1f} min", flush=True)
