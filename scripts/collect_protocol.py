"""Guided data-collection protocol (replaces manual 'f'/'d' key labelling).

On-screen prompts tell the participant what to do; labels come from the
prompt timeline, so they are consistent across participants and there is no
key-press lag. The first 1.5 s after each prompt is marked 'transition' and
excluded from training/evaluation. Only features are stored (no video) unless
--save-video is given with the participant's written consent.

  python scripts/collect_protocol.py --subject S01 --out data/protocol
  python scripts/collect_protocol.py --subject S02 --context driving --out data/protocol

Recommended: >= 10 participants, 2 sessions each (different day / lighting),
glasses on for at least 3 participants. Then run
  python scripts/train_model.py --data data/protocol
"""
import argparse, csv, os, sys, time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focusguard.config import Config  # noqa: E402
from focusguard.features import FEATURE_NAMES  # noqa: E402
from focusguard.overlay import draw  # noqa: E402
from focusguard.pipeline import DistractionPipeline  # noqa: E402

# (instruction, label, seconds)   label in {focused, glance, distracted, drowsy}
PROTOCOL_SCREEN = [
    ("Look at the screen normally (calibration)", "focused", 15),
    ("Read the text on screen / watch the lecture", "focused", 40),
    ("Look DOWN at your phone on your lap", "distracted", 10),
    ("Back to the screen", "focused", 20),
    ("Look away to your LEFT (window / door)", "distracted", 8),
    ("Back to the screen", "focused", 15),
    ("Look away to your RIGHT", "distracted", 8),
    ("Quick glances: look left for ~1 s and back, repeat", "glance", 20),
    ("Back to the screen", "focused", 15),
    ("Close your eyes as if dozing", "drowsy", 6),
    ("Back to the screen", "focused", 15),
    ("Yawn (a real or acted yawn)", "drowsy", 6),
    ("Turn and talk to someone beside you", "distracted", 15),
    ("Look UP at the ceiling", "distracted", 6),
    ("Back to the screen, stay focused", "focused", 25),
]
PROTOCOL_DRIVING = [
    ("Look at the road (calibration)", "focused", 15),
    ("Drive normally, eyes on the road", "focused", 40),
    ("Check the left mirror briefly, then back (repeat)", "glance", 20),
    ("Look DOWN at the phone / infotainment", "distracted", 8),
    ("Eyes on the road", "focused", 20),
    ("Look out of the side window", "distracted", 6),
    ("Eyes on the road", "focused", 15),
    ("Close your eyes as if falling asleep", "drowsy", 5),
    ("Eyes on the road", "focused", 15),
    ("Yawn", "drowsy", 6),
    ("Talk to the passenger (eyes on road)", "focused", 15),
    ("Turn to talk to the passenger", "distracted", 6),
    ("Eyes on the road", "focused", 25),
]
TRANSITION_S = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", default="1")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--context", default="screen", choices=["screen", "driving"])
    ap.add_argument("--out", default="data/protocol")
    ap.add_argument("--save-video", action="store_true", help="only with written consent")
    a = ap.parse_args()
    steps = PROTOCOL_SCREEN if a.context == "screen" else PROTOCOL_DRIVING
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"{a.subject}_s{a.session}_{a.context}")
    cfg = Config.for_context(a.context)
    pipe = DistractionPipeline(cfg)
    cap = cv2.VideoCapture(a.camera)
    writer = None
    f = open(stem + ".csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["subject", "session", "t", "step", "instruction", "label", "rule_state", "rule_score"] + FEATURE_NAMES)
    bounds, acc = [], 0.0
    for s in steps:
        bounds.append((acc, acc + s[2], s))
        acc += s[2]
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t = time.time() - t0
            if t >= acc:
                break
            k = next(i for i, (a0, b0, _) in enumerate(bounds) if a0 <= t < b0)
            a0, b0, (instr, label, _) = bounds[k]
            lab = "transition" if t - a0 < TRANSITION_S else label
            out = pipe.process(frame, t)
            feats = out.feats or {}
            w.writerow([a.subject, a.session, f"{t:.3f}", k, instr, lab, out.state.state, f"{out.state.score:.1f}"] +
                       [feats.get(n, "") for n in FEATURE_NAMES])
            if a.save_video:
                if writer is None:
                    h, wd = frame.shape[:2]
                    writer = cv2.VideoWriter(stem + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (wd, h))
                writer.write(frame)
            vis = draw(frame.copy(), out)
            cv2.rectangle(vis, (0, vis.shape[0] - 70), (vis.shape[1], vis.shape[0] - 30), (0, 0, 0), -1)
            cv2.putText(vis, f"[{k+1}/{len(steps)}] {instr}  ({b0 - t:4.1f}s)", (10, vis.shape[0] - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("FocusGuard protocol recording (q = abort)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        f.close()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    pipe.save(stem + "_report", title=f"Protocol {a.subject} session {a.session}")
    print("saved", stem + ".csv")


if __name__ == "__main__":
    main()
