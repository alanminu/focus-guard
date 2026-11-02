"""FocusGuard dashboard (Streamlit).

  pip install streamlit
  streamlit run dashboard/app.py

* Session browser: every folder in runs/ written by the CLI / protocol recorder
* Analyse an uploaded video offline (no frames are stored, only features)
* Compare sessions (attention index, focus share, distraction episodes)
"""
import json
import os
import sys
import tempfile

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from focusguard.cli import run_stream  # noqa: E402

RUNS = os.path.join(ROOT, "runs")
st.set_page_config(page_title="FocusGuard", layout="wide")
st.title("FocusGuard - attention monitoring dashboard")


def sessions():
    if not os.path.isdir(RUNS):
        return []
    return sorted(d for d in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, d, "summary.json")))


tab1, tab2, tab3 = st.tabs(["Session report", "Analyse a video", "Compare sessions"])

with tab1:
    ss = sessions()
    if not ss:
        st.info("No sessions yet. Run `python -m focusguard.cli --source 0 --out runs/my_session`.")
    else:
        name = st.selectbox("Session", ss)
        d = os.path.join(RUNS, name)
        summ = json.load(open(os.path.join(d, "summary.json")))
        c = st.columns(5)
        c[0].metric("Mean attention index", summ.get("mean_attention_index"))
        c[1].metric("Time focused", f"{summ.get('time_share', {}).get('FOCUSED', 0)*100:.0f}%")
        c[2].metric("Distraction episodes", summ.get("distraction_episodes"))
        c[3].metric("Yawns", summ.get("yawns"))
        c[4].metric("Blinks / min", summ.get("blink_rate_per_min"))
        st.image(os.path.join(d, "timeline.png"), width='stretch')
        fr = pd.read_csv(os.path.join(d, "frames.csv"))
        st.subheader("Attention index")
        st.line_chart(fr.set_index("t")["score"])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Episodes")
            ep = pd.read_csv(os.path.join(d, "episodes.csv"))
            st.dataframe(ep[ep.state != "FOCUSED"].sort_values("duration", ascending=False), width='stretch')
        with col2:
            st.subheader("Events")
            ev_path = os.path.join(d, "events.csv")
            if os.path.getsize(ev_path) > 1:
                ev = pd.read_csv(ev_path)
                st.dataframe(ev[ev.type != "blink"], width='stretch')
                st.bar_chart(ev.type.value_counts())
        with open(os.path.join(d, "report.html"), "rb") as fh:
            st.download_button("Download HTML report", fh, file_name=f"{name}_report.html")

with tab2:
    up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    ctx = st.radio("Context", ["screen", "driving"], horizontal=True)
    phone = st.checkbox("Phone detector (YOLO11n, ONNX)")
    if up and st.button("Analyse"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tf:
            tf.write(up.read())
        out = os.path.join(RUNS, os.path.splitext(up.name)[0])
        with st.spinner("Processing..."):
            summ, _ = run_stream(tf.name, out, display=False, save_video=True, context=ctx, phone=phone, quiet=True)
        os.unlink(tf.name)
        st.success(f"Saved to runs/{os.path.basename(out)} - open it in the Session report tab")
        st.json(summ)

with tab3:
    ss = sessions()
    if ss:
        rows = []
        for s in ss:
            j = json.load(open(os.path.join(RUNS, s, "summary.json")))
            rows.append({"session": s, "attention": j.get("mean_attention_index"),
                         "focused_%": 100 * j.get("time_share", {}).get("FOCUSED", 0),
                         "distraction_episodes": j.get("distraction_episodes"),
                         "yawns": j.get("yawns"), "duration_s": j.get("duration_s")})
        df = pd.DataFrame(rows).set_index("session")
        st.dataframe(df, width='stretch')
        st.bar_chart(df[["attention", "focused_%"]])
