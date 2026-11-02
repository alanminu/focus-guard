"""Benchmark the decision logic on synthetic streams with known ground truth.

Compares the first prototype's per-frame logic with FocusGuard's temporal,
calibrated engine on N simulated subjects (different eye sizes, camera
placements, blink rates) x 10 min each, containing short glances and blinks
(normal - must NOT alert) and long distractions / head-down / microsleeps /
yawns (must alert).

  python scripts/eval_synthetic.py --subjects 20 --out results/synthetic
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focusguard.behavior import BehaviorEngine  # noqa: E402
from focusguard.calibration import Baseline, Calibrator  # noqa: E402
from focusguard.config import Config  # noqa: E402
from focusguard.features import FEATURE_NAMES  # noqa: E402
from focusguard.simulate import ALERT_EVENTS, make_scenario  # noqa: E402

I = {k: i for i, k in enumerate(FEATURE_NAMES)}


def per_frame(sc, use_pose: bool, smooth: int = 1):
    X = sc.X
    ok = np.isfinite(X[:, 0])
    ear, mar = X[:, I["ear_l"]], X[:, I["mar"]]
    flag = (ear < 0.20) | (mar > 0.45)
    if use_pose:
        flag |= (np.abs(X[:, I["yaw"]]) > 25) | (X[:, I["pitch"]] < -15)
    flag &= ok
    if smooth > 1:
        k = np.ones(smooth) / smooth
        flag = np.convolve(flag.astype(float), k, mode="same") > 0.5
    return flag


def focusguard(sc, calibrate: bool, cfg=None):
    cfg = cfg or Config.for_context("driving")
    cal = Calibrator(cfg)
    eng = BehaviorEngine(cfg, Baseline.population_default(cfg))
    flag = np.zeros(len(sc.t), bool)
    for i, ti in enumerate(sc.t):
        f = sc.feats(i)
        if calibrate and not cal.done and cal.add(ti, f):
            eng.baseline = cal.baseline
        s = eng.update(float(ti), f)
        flag[i] = s.state in ("DISTRACTED", "DROWSY")
    return flag


def episodes(flag, t, gap=0.5):
    idx = np.where(flag)[0]
    if idx.size == 0:
        return []
    eps, a, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[prev] > gap:
            eps.append((t[a], t[prev]))
            a = i
        prev = i
    eps.append((t[a], t[prev]))
    return eps


def score(flag, sc):
    t = sc.t
    res = {}
    alert_ev = [e for e in sc.events if e["type"] in ALERT_EVENTS]
    hits, lat = [], []
    for e in alert_ev:
        m = (t >= e["start"]) & (t <= e["end"] + 0.5) & flag
        hit = bool(m.any())
        hits.append(hit)
        if hit:
            lat.append(float(t[np.argmax(m)] - e["start"]))
    res["recall"] = float(np.mean(hits)) if hits else float("nan")
    res["latency_s"] = float(np.median(lat)) if lat else float("nan")
    for kind in ALERT_EVENTS:
        ev = [h for h, e in zip(hits, alert_ev) if e["type"] == kind]
        res[f"recall_{kind}"] = float(np.mean(ev)) if ev else float("nan")
    eps = episodes(flag, t)
    fa = 0
    for a, b in eps:
        if not any(a <= e["end"] + 1.0 and b >= e["start"] - 1.0 for e in alert_ev):
            fa += 1
    hours = (t[-1] - t[0]) / 3600
    res["false_alarms_per_hour"] = fa / hours
    # isolated short glances: not part of a VATS (cumulative) episode -> should never alert
    sg = [e for e in sc.events if e["type"] == "short_glance" and not any(
        v["type"] == "vats" and v["start"] <= e["end"] + 1 and v["end"] >= e["start"] - 1 for v in sc.events)]
    res["short_glance_alerted"] = float(np.mean([flag[(t >= e["start"]) & (t <= e["end"])].any() for e in sg])) if sg else float("nan")
    res["alert_time_share"] = float(flag.mean())
    return res


SYSTEMS = {
    "V1: EAR+lip, per-frame": lambda sc: per_frame(sc, use_pose=False),
    "Per-frame, all cues": lambda sc: per_frame(sc, use_pose=True),
    "Per-frame + 5-frame smoothing": lambda sc: per_frame(sc, use_pose=True, smooth=5),
    "FocusGuard (no calibration)": lambda sc: focusguard(sc, calibrate=False),
    "FocusGuard (calibrated)": lambda sc: focusguard(sc, calibrate=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--out", default="results/synthetic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for s in range(a.subjects):
        sc = make_scenario(duration_s=a.minutes * 60, seed=1000 + s)
        for name, fn in SYSTEMS.items():
            r = score(fn(sc), sc)
            r.update(system=name, subject=s)
            rows.append(r)
        print(f"subject {s+1}/{a.subjects} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_subject.csv"), index=False)
    cols = ["recall", "recall_long_distraction", "recall_head_down", "recall_microsleep", "recall_yawn", "recall_vats",
            "latency_s", "false_alarms_per_hour", "short_glance_alerted", "alert_time_share"]
    agg = df.groupby("system", sort=False)[cols].agg(["mean", "std"])
    summary = df.groupby("system", sort=False)[cols].mean().round(3)
    summary.to_csv(os.path.join(a.out, "summary.csv"))
    print(summary.to_string())
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({k: {c: agg.loc[k, (c, "mean")] for c in cols} | {c + "_std": agg.loc[k, (c, "std")] for c in cols}
                   for k in agg.index}, f, indent=2)


if __name__ == "__main__":
    main()
