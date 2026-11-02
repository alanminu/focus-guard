"""HUD overlay for live / offline visualisation – redesigned for a rich, interactive look."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .features import L_EYE_EAR, L_IRIS, MOUTH_CORNERS, MOUTH_VERT, R_EYE_EAR, R_IRIS, NOSE_TIP

# ── colour palette (BGR) ────────────────────────────────────────────────────
PALETTE = {
    "FOCUSED":     (  60, 200,  80),   # vibrant green
    "GLANCE":      (  30, 180, 230),   # amber-cyan
    "DISTRACTED":  (  40,  80, 230),   # bold red-orange
    "DROWSY":      ( 130,  50, 210),   # purple
    "ABSENT":      ( 100, 100, 100),   # grey
    "CALIBRATING": ( 180, 160,  30),   # gold
}
TEXT_LIGHT  = (240, 240, 240)
TEXT_DIM    = (160, 160, 160)
BG_DARK     = (  15,  15,  20)
ACCENT_TICK = (  80, 200, 120)

_PULSE_START = time.time()


def _lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _alpha_blend(img, overlay, alpha=0.55):
    """Blend a solid-colour overlay onto img in-place."""
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _rounded_rect(img, x1, y1, x2, y2, r, color, alpha=1.0, filled=True):
    """Draw a filled rounded rectangle (anti-aliased via polylines)."""
    overlay = img.copy()
    pts = []
    for cx, cy, a0, a1 in [
        (x1 + r, y1 + r, 180, 270),
        (x2 - r, y1 + r,  270, 360),
        (x2 - r, y2 - r,    0,  90),
        (x1 + r, y2 - r,   90, 180),
    ]:
        for deg in range(a0, a1 + 1, 4):
            rad = math.radians(deg)
            pts.append([int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))])
    pts = np.array(pts, dtype=np.int32)
    if filled:
        cv2.fillPoly(overlay, [pts], color)
    else:
        cv2.polylines(overlay, [pts], True, color, 1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        img[:] = overlay


def _put_text(img, text, x, y, scale, color, thickness=1, shadow=True):
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _arc_gauge(img, cx, cy, radius, value, max_val, col, bg_col=(40, 40, 45), thickness=8):
    """Draw a partial arc gauge (180° sweep from left to right)."""
    start_a, end_a = 200, 340   # degrees (OpenCV: 0=3-o'clock, CCW)
    span = end_a - start_a
    # background arc
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, start_a, end_a, bg_col, thickness, cv2.LINE_AA)
    # filled arc
    fill_end = start_a + int(span * max(0.0, min(1.0, value / max_val)))
    if fill_end > start_a:
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, start_a, fill_end, col, thickness, cv2.LINE_AA)
    # tip dot
    tip_deg = math.radians(fill_end)
    tx = int(cx + radius * math.cos(tip_deg))
    ty = int(cy + radius * math.sin(tip_deg))
    cv2.circle(img, (tx, ty), thickness // 2 + 2, col, -1, cv2.LINE_AA)


def _draw_landmark_mesh(img, out, col):
    """Draw eye contours, iris dots, mouth lines, and gaze arrow."""
    if out.obs is None:
        return
    p = out.obs.landmarks_px
    fs = out.state

    # Eye contours – bright white with glow hint
    for idx in (R_EYE_EAR, L_EYE_EAR):
        pts = p[idx, :2].astype(int)
        cv2.polylines(img, [pts], True, (80, 80, 80), 3, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (220, 220, 255), 1, cv2.LINE_AA)

    # Iris dots – cyan
    for c in (R_IRIS[0], L_IRIS[0]):
        center = tuple(p[c, :2].astype(int))
        cv2.circle(img, center, 5, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, center, 4, (0, 220, 220), -1, cv2.LINE_AA)
        cv2.circle(img, center, 2, (200, 255, 255), -1, cv2.LINE_AA)

    # Mouth lines
    for a, b in MOUTH_VERT + [MOUTH_CORNERS]:
        pa = tuple(p[a, :2].astype(int))
        pb = tuple(p[b, :2].astype(int))
        cv2.line(img, pa, pb, (60, 60, 60), 3, cv2.LINE_AA)
        cv2.line(img, pa, pb, (200, 160, 220), 1, cv2.LINE_AA)

    # Gaze arrow
    gy = fs.metrics.get("gaze_yaw", 0.0)
    gp = fs.metrics.get("gaze_pitch", 0.0)
    n = p[NOSE_TIP, :2]
    h, w = img.shape[:2]
    L = 0.22 * w * min(1.0, out.feats.get("face_scale", 0.1) * 6 if out.feats else 1.0)
    tip = n + L * np.array([-np.sin(np.radians(gy)), -np.sin(np.radians(gp))])
    cv2.arrowedLine(img, tuple(n.astype(int)), tuple(tip.astype(int)),
                    (30, 30, 30), 4, tipLength=0.28)
    cv2.arrowedLine(img, tuple(n.astype(int)), tuple(tip.astype(int)),
                    col, 2, tipLength=0.28, line_type=cv2.LINE_AA)


def draw(frame: np.ndarray, out, fps: float = 0.0) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    fs = out.state
    col = PALETTE.get(fs.state, (180, 180, 180))
    t_now = time.time() - _PULSE_START
    pulse = 0.5 + 0.5 * math.sin(t_now * 3.5)   # 0..1 oscillation

    # ── 1. Landmark mesh ────────────────────────────────────────────────────
    _draw_landmark_mesh(img, out, col)

    # ── 2. Phone bounding boxes ─────────────────────────────────────────────
    for sc, (x1, y1, x2, y2) in out.phone_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 40, 255), 2, cv2.LINE_AA)
        _put_text(img, f"PHONE  {sc:.0%}", x1, max(18, y1 - 6), 0.5, (0, 40, 255), 1)

    # ── 3. Top status bar ───────────────────────────────────────────────────
    BAR_H = max(48, h // 10)
    bar_overlay = img.copy()
    cv2.rectangle(bar_overlay, (0, 0), (w, BAR_H), BG_DARK, -1)
    _alpha_blend(img, bar_overlay, alpha=0.72)

    # Coloured left accent strip
    strip_w = 6
    cv2.rectangle(img, (0, 0), (strip_w, BAR_H), col, -1)

    # State label (large, bold)
    cal = [r for r in fs.reasons if r.startswith("calibrating")]
    label = fs.state
    _put_text(img, label, strip_w + 12, int(BAR_H * 0.70),
              BAR_H / 52, col, thickness=2)

    # Calibrating progress pill
    if cal:
        prog_txt = cal[0].replace("calibrating ", "CAL ") + "%"
        (tw, _), _ = cv2.getTextSize(prog_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        px = strip_w + 14 + int(BAR_H / 52 * 6 * len(label)) + 10
        _rounded_rect(img, px, int(BAR_H * 0.22), px + tw + 12, int(BAR_H * 0.82), 6,
                      PALETTE["CALIBRATING"], alpha=0.75)
        _put_text(img, prog_txt, px + 6, int(BAR_H * 0.68), 0.42, TEXT_LIGHT, 1)

    # Reason tags on right
    reasons = [r for r in fs.reasons if not r.startswith("calibrating")]
    rx = w - 10
    for reason in reversed(reasons):
        rtxt = reason.upper().replace("_", " ")
        (tw, _), _ = cv2.getTextSize(rtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        rx -= tw + 18
        _rounded_rect(img, rx, int(BAR_H * 0.20), rx + tw + 16, int(BAR_H * 0.80), 5,
                      col, alpha=0.55)
        _put_text(img, rtxt, rx + 8, int(BAR_H * 0.66), 0.38, TEXT_LIGHT, 1)
        rx -= 6

    # FPS badge (top-right corner of bar)
    if fps:
        fps_txt = f"{fps:.0f} FPS"
        _put_text(img, fps_txt, w - 68, int(BAR_H * 0.42), 0.38, TEXT_DIM, 1, shadow=False)

    # ── 4. Attention arc gauge (top-right panel) ────────────────────────────
    PANEL_W, PANEL_H = 170, 140
    px0 = w - PANEL_W - 10
    py0 = BAR_H + 10

    # Panel background
    panel_bg = img.copy()
    _rounded_rect(panel_bg, px0, py0, px0 + PANEL_W, py0 + PANEL_H, 10, BG_DARK)
    _alpha_blend(img, panel_bg, alpha=0.65)

    # Arc gauge
    cx, cy = px0 + PANEL_W // 2, py0 + PANEL_H // 2 + 10
    _arc_gauge(img, cx, cy, 48, fs.score, 100, col, thickness=10)

    # Score number
    score_txt = f"{fs.score:.0f}"
    scale = 1.1 if fs.score >= 100 else 1.3
    (tw, th), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    _put_text(img, score_txt, cx - tw // 2, cy + th // 2 - 4, scale, col, thickness=2)

    # "ATTENTION" label
    _put_text(img, "ATTENTION", px0 + 20, py0 + PANEL_H - 10, 0.36, TEXT_DIM, 1, shadow=False)

    # Pulsing indicator dot when not FOCUSED
    if fs.state not in ("FOCUSED", "CALIBRATING"):
        dot_col = _lerp_color(col, (20, 20, 20), 1 - pulse)
        cv2.circle(img, (px0 + PANEL_W - 16, py0 + 16), 7, dot_col, -1, cv2.LINE_AA)
        cv2.circle(img, (px0 + PANEL_W - 16, py0 + 16), 7, col, 1, cv2.LINE_AA)

    # ── 5. Metrics panel (bottom-left) ──────────────────────────────────────
    m, f = fs.metrics, out.feats or {}
    if f or m:
        rows = []
        if f:
            ear = f.get("ear", 0)
            ear_thr = m.get("ear_thr", 0)
            mar = f.get("mar", 0)
            rows.append(("EAR", f"{ear:.2f}", f"thr {ear_thr:.2f}", ear >= ear_thr))
            rows.append(("MAR", f"{mar:.2f}", "", mar < 0.5))
            rows.append(("YAW", f"{m.get('yaw_dev', 0):+.1f}°", "", abs(m.get('yaw_dev', 0)) < 15))
            rows.append(("PITCH", f"{m.get('pitch_dev', 0):+.1f}°", "", abs(m.get('pitch_dev', 0)) < 15))
        if m:
            pct = m.get("perclos", 0) * 100
            rows.append(("PERCLOS", f"{pct:.1f}%", "", pct < 10))
            rows.append(("BLINKS/m", f"{m.get('blink_rate', 0):.1f}", "", True))
            rows.append(("YAWNS", f"{int(m.get('yawns_recent', 0))}", "", int(m.get('yawns_recent', 0)) < 2))

        ROW_H = 20
        MPAD = 10
        MP_W = 175
        n_rows = len(rows)
        mp_h = MPAD * 2 + n_rows * ROW_H + 4

        my0 = h - mp_h - 8
        # background
        mb = img.copy()
        _rounded_rect(mb, 8, my0, 8 + MP_W, my0 + mp_h, 8, BG_DARK)
        _alpha_blend(img, mb, alpha=0.65)

        for i, (lbl, val, hint, ok) in enumerate(rows):
            vy = my0 + MPAD + i * ROW_H + ROW_H - 4
            dot_c = ACCENT_TICK if ok else col
            cv2.circle(img, (20, vy - 5), 3, dot_c, -1, cv2.LINE_AA)
            _put_text(img, lbl, 30, vy, 0.38, TEXT_DIM, 1, shadow=False)
            _put_text(img, val, 100, vy, 0.40, TEXT_LIGHT, 1)
            if hint:
                _put_text(img, hint, 140, vy, 0.32, TEXT_DIM, 1, shadow=False)

    # ── 6. State-change alert flash ─────────────────────────────────────────
    if fs.state in ("DISTRACTED", "DROWSY", "ABSENT"):
        alpha_flash = 0.12 + 0.08 * pulse
        flash = img.copy()
        cv2.rectangle(flash, (0, 0), (w, h), col, -1)
        _alpha_blend(img, flash, alpha=alpha_flash)
        # border
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, 3, cv2.LINE_AA)

    # ── 7. Watermark ────────────────────────────────────────────────────────
    _put_text(img, "FocusGuard v2", w - 120, h - 8, 0.33, (60, 60, 60), 1, shadow=False)

    return img
