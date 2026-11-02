"""Module E - Monitor: session logging, statistics and reports.

Produces, for every session:
  frames.csv     per-frame features, cues, state, attention score
  events.csv     behavioural events (blink, glance, long_distraction, yawn, ...)
  episodes.csv   contiguous state episodes
  summary.json   session statistics (focus %, distraction episodes, PERCLOS, ...)
  timeline.png   attention index + state ribbon + key signals
  report.html    self-contained report (for teachers / parents / fleet managers)
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

STATE_COLORS = {"FOCUSED": "#2e9d5b", "GLANCE": "#e3b341", "DISTRACTED": "#d9534f",
                "DROWSY": "#7b52ab", "ABSENT": "#6c757d", "CALIBRATING": "#5bc0de"}


class SessionMonitor:
    def __init__(self):
        self.rows: List[Dict] = []

    def log(self, fs, feats: Optional[Dict[str, float]]):
        r = {"t": fs.t, "state": fs.state, "score": fs.score, "reasons": "|".join(fs.reasons)}
        r.update({f"cue_{k}": int(v) for k, v in fs.cues.items()})
        r.update({f"m_{k}": v for k, v in fs.metrics.items()})
        if feats:
            r.update(feats)
        self.rows.append(r)

    # ------------------------------------------------------------
    def frames(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @staticmethod
    def episodes(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["state", "start", "end", "duration"])
        s = df["state"].values
        t = df["t"].values
        cut = np.r_[0, np.nonzero(s[1:] != s[:-1])[0] + 1, len(s)]
        eps = []
        for a, b in zip(cut[:-1], cut[1:]):
            end = t[b] if b < len(t) else t[-1]
            eps.append({"state": s[a], "start": float(t[a]), "end": float(end), "duration": float(end - t[a])})
        return pd.DataFrame(eps)

    def summary(self, events: List[Dict]) -> Dict:
        df = self.frames()
        if df.empty:
            return {}
        df = df[df.state != "CALIBRATING"]
        if df.empty:
            return {"note": "session ended during calibration"}
        dt = np.diff(df.t.values, append=df.t.values[-1])
        dt = np.clip(dt, 0, 0.5)
        total = float(dt.sum()) or 1e-6
        share = {s: float(dt[df.state.values == s].sum() / total) for s in STATE_COLORS if s != "CALIBRATING"}
        eps = self.episodes(df)
        ev = pd.DataFrame(events) if events else pd.DataFrame(columns=["type", "duration", "start"])
        count = lambda k: int((ev.type == k).sum()) if len(ev) else 0
        dist = eps[eps.state == "DISTRACTED"]
        foc = eps[eps.state == "FOCUSED"]
        out = {
            "duration_s": round(total, 1),
            "mean_attention_index": round(float(np.average(df.score, weights=dt + 1e-9)), 1),
            "time_share": {k: round(v, 3) for k, v in share.items()},
            "distraction_episodes": int(len(dist)),
            "mean_distraction_episode_s": round(float(dist.duration.mean()), 2) if len(dist) else 0.0,
            "longest_focus_streak_s": round(float(foc.duration.max()), 1) if len(foc) else 0.0,
            "blinks": count("blink"),
            "blink_rate_per_min": round(count("blink") * 60.0 / total, 1),
            "glances": count("glance"),
            "long_distractions": count("long_distraction"),
            "yawns": count("yawn"),
            "long_eye_closures": count("long_eye_closure"),
            "head_down_events": count("head_down"),
            "phone_events": count("phone"),
            "talking_events": count("talking"),
            "face_absent_events": count("face_absent"),
            "max_perclos": round(float(df.get("m_perclos", pd.Series([0])).max()), 3),
        }
        reasons = df.reasons.fillna("").str.split("|").explode()
        reasons = reasons[reasons != ""]
        out["top_reasons"] = reasons.value_counts().head(6).to_dict()
        return out

    # ------------------------------------------------------------
    def save(self, out_dir: str, events: List[Dict], title: str = "FocusGuard session report") -> Dict:
        os.makedirs(out_dir, exist_ok=True)
        df = self.frames()
        df.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
        pd.DataFrame(events).to_csv(os.path.join(out_dir, "events.csv"), index=False)
        self.episodes(df).to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
        summ = self.summary(events)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summ, f, indent=2)
        png = plot_timeline(df, os.path.join(out_dir, "timeline.png"), title)
        write_html_report(summ, png, os.path.join(out_dir, "report.html"), title)
        return summ


def plot_timeline(df: pd.DataFrame, path: str, title: str = "") -> Optional[bytes]:
    if df.empty:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 0.45, 1.2, 1.2]})
    t = df.t.values
    ax = axes[0]
    ax.plot(t, df.score, color="#1f4e79", lw=1.6)
    ax.set_ylim(-3, 103)
    ax.set_ylabel("Attention\nindex")
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=11, loc="left")
    ax = axes[1]
    eps = SessionMonitor.episodes(df)
    for _, e in eps.iterrows():
        ax.axvspan(e.start, max(e.end, e.start + 1e-3), color=STATE_COLORS.get(e.state, "#ccc"), lw=0)
    ax.set_yticks([])
    ax.set_ylabel("State", rotation=0, ha="right", va="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for s, c in STATE_COLORS.items()]
    ax.legend(handles, STATE_COLORS.keys(), ncol=6, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), frameon=False)
    ax = axes[2]
    if "m_gaze_yaw" in df:
        ax.plot(t, df.m_gaze_yaw, lw=0.9, label="gaze yaw dev (°)", color="#e07b39")
        ax.plot(t, df.m_gaze_pitch, lw=0.9, label="gaze pitch dev (°)", color="#3d85c6")
        ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.25)
    ax.set_ylabel("Gaze")
    ax = axes[3]
    if "ear" in df:
        ax.plot(t, df.ear, lw=0.9, label="EAR", color="#2e9d5b")
    if "mar" in df:
        ax.plot(t, df.mar, lw=0.9, label="MAR", color="#a64d79")
    if "m_ear_thr" in df:
        ax.plot(t, df.m_ear_thr, lw=0.8, ls="--", color="#2e9d5b", alpha=0.7, label="EAR closed thr")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Eyes /\nmouth")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return buf.getvalue()


def write_html_report(summ: Dict, png: Optional[bytes], path: str, title: str):
    img = f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}" style="width:100%">' if png else ""
    ts = summ.get("time_share", {})
    share_rows = "".join(f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>" for k, v in ts.items())
    other = "".join(f"<tr><td>{k.replace('_',' ')}</td><td>{v}</td></tr>" for k, v in summ.items()
                    if k not in ("time_share", "top_reasons"))
    reasons = ", ".join(f"{k} ({v})" for k, v in summ.get("top_reasons", {}).items()) or "-"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,Segoe UI,Arial;margin:24px;color:#222;max-width:1100px}}
table{{border-collapse:collapse;margin:8px 0 18px}}td{{border:1px solid #ddd;padding:4px 10px}}
h1{{font-size:20px}}.kpi{{display:inline-block;margin-right:28px;font-size:14px}}.kpi b{{font-size:26px;display:block}}</style></head>
<body><h1>{title}</h1>
<div class="kpi"><b>{summ.get('mean_attention_index','-')}</b>mean attention index</div>
<div class="kpi"><b>{ts.get('FOCUSED',0)*100:.0f}%</b>time focused</div>
<div class="kpi"><b>{summ.get('distraction_episodes','-')}</b>distraction episodes</div>
<div class="kpi"><b>{summ.get('yawns','-')}</b>yawns</div>
{img}
<h3>Time share by state</h3><table>{share_rows}</table>
<h3>Statistics</h3><table>{other}</table>
<p><b>Most frequent reasons:</b> {reasons}</p>
<p style="color:#777;font-size:12px">All processing ran locally; no video frames were stored.</p>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
