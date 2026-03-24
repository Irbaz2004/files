"""
╔══════════════════════════════════════════════════════════════╗
║      STUDENT ATTENTION MONITOR — Streamlit Dashboard         ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import csv
import os
import glob
from datetime import datetime
from collections import deque

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Dark Futuristic Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #080c14;
    color: #e0e8ff;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 1rem 2rem; max-width: 100%; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1525 100%);
    border-right: 1px solid #1a2744;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] label { color: #7a9cd4 !important; font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1525 0%, #111d30 100%);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 24px rgba(0,100,255,0.08);
}
div[data-testid="metric-container"] label {
    color: #4a7abf !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #e8f0ff !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #1a3a6e, #0f2550);
    color: #7ab8ff;
    border: 1px solid #2a5090;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    padding: 0.45rem 1.2rem;
    transition: all 0.2s ease;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1f4a90, #142e66);
    border-color: #4a80c0;
    color: #aad4ff;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(74,144,226,0.25);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0d6e3f, #085530);
    color: #4dffaa;
    border-color: #1a8f55;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #108a4e, #0a6a3c);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1e;
    border-bottom: 1px solid #1a2744;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    color: #4a6a9a !important;
    padding: 0.6rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #7ab8ff !important;
    border-bottom: 2px solid #4a90e2 !important;
    background: transparent !important;
}

/* Headers */
h1 { font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 0.05em; }
h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 600; color: #7ab8ff; }

/* Status badges */
.badge-active {
    display: inline-block;
    background: rgba(0, 200, 100, 0.15);
    color: #00e87a;
    border: 1px solid #00c864;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-inactive {
    display: inline-block;
    background: rgba(220, 50, 50, 0.15);
    color: #ff6060;
    border: 1px solid #cc3333;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
}
.badge-warning {
    display: inline-block;
    background: rgba(255, 160, 0, 0.15);
    color: #ffa020;
    border: 1px solid #cc8000;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
}

/* Page title bar */
.title-bar {
    background: linear-gradient(90deg, #0a1628 0%, #0d1e3a 50%, #0a1628 100%);
    border: 1px solid #1a3060;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

/* Dividers */
hr { border-color: #1a2744; margin: 0.8rem 0; }

/* Selectbox / number input */
div[data-baseweb="select"] { background: #0d1525 !important; }
input[type="number"] { background: #0d1525 !important; color: #e0e8ff !important; }

/* Progress bar custom */
.progress-wrap {
    background: #0d1525;
    border-radius: 8px;
    border: 1px solid #1a2744;
    overflow: hidden;
    height: 22px;
    margin: 4px 0;
}
.progress-fill-green {
    height: 100%;
    background: linear-gradient(90deg, #0a8f50, #00d47a);
    transition: width 0.6s ease;
    border-radius: 0 8px 8px 0;
}
.progress-fill-red {
    height: 100%;
    background: linear-gradient(90deg, #8f0a0a, #d43a00);
    transition: width 0.6s ease;
    border-radius: 0 8px 8px 0;
}

/* Camera feed container */
.cam-frame {
    border: 1px solid #1a3060;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 30px rgba(30,80,200,0.15);
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1a3060; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
def init_state():
    defaults = {
        "monitoring": False,
        "session_start": None,
        "snapshots": [],
        "active_pct_history": deque(maxlen=120),  # last 10 min at 5s intervals
        "last_snap_time": 0,
        "total_students": 30,
        "class_name": "B-Tech Final Year",
        "subject": "Python Programming",
        "duration_min": 45,
        "threshold": 0.65,
        "camera_index": 0,
        "current_active": 0,
        "current_visible": 0,
        "current_frame": None,
        "cap": None,
        "face_mesh": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────
# MEDIAPIPE SETUP
# ──────────────────────────────────────────────
@st.cache_resource
def get_face_mesh():
    mp_fm = mp.solutions.face_mesh
    return mp_fm.FaceMesh(
        max_num_faces=50,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

face_mesh = get_face_mesh()


def analyze_frame(frame, threshold):
    """Returns annotated frame, active count, visible count."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    results = face_mesh.process(rgb)

    active_count = 0
    visible_count = 0

    if results.multi_face_landmarks:
        for face_lm in results.multi_face_landmarks:
            lm = face_lm.landmark
            visible_count += 1

            # Head pose
            nose_x = lm[1].x * w
            left_ear_x = lm[234].x * w
            right_ear_x = lm[454].x * w
            ear_span = abs(right_ear_x - left_ear_x)
            nose_mid_offset = abs(nose_x - (left_ear_x + right_ear_x) / 2)
            yaw_score = max(0.0, 1.0 - (nose_mid_offset / (ear_span * 0.6 + 1e-6)))

            forehead_y = lm[10].y * h
            chin_y = lm[175].y * h
            left_eye_y = lm[33].y * h
            right_eye_y = lm[263].y * h
            face_height = abs(chin_y - forehead_y)
            eye_y = (left_eye_y + right_eye_y) / 2
            face_center_y = (forehead_y + chin_y) / 2
            pitch_score = max(0.0, 1.0 - (abs(eye_y - face_center_y) / (face_height * 0.4 + 1e-6)))

            all_x = [l.x for l in lm]; all_y = [l.y for l in lm]
            bw = (max(all_x) - min(all_x)) * w
            bh = (max(all_y) - min(all_y)) * h
            visibility = min(1.0, (bw * bh) / (w * h * 0.04))

            attention = yaw_score * 0.5 + pitch_score * 0.3 + visibility * 0.2
            is_active = attention >= threshold
            if is_active:
                active_count += 1

            # Draw box
            x1 = int(min(all_x) * w) - 8
            y1 = int(min(all_y) * h) - 8
            x2 = int(max(all_x) * w) + 8
            y2 = int(max(all_y) * h) + 8
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            color = (0, 220, 90) if is_active else (50, 50, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{'ACT' if is_active else 'OFF'} {attention:.0%}"
            cv2.rectangle(frame, (x1, y1 - 18), (x1 + 72, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

    return frame, active_count, visible_count


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
        <div style='font-family: Rajdhani; font-size: 1.4rem; font-weight:700; 
                    color: #7ab8ff; letter-spacing: 0.1em;'>🎓 STUDENT MONITOR</div>
        <div style='font-size: 0.7rem; color: #4a6a9a; letter-spacing: 0.15em;'>ATTENTION TRACKING SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**CLASS SETTINGS**")
    st.session_state.class_name = st.text_input("Class Name", st.session_state.class_name)
    st.session_state.subject = st.text_input("Subject", st.session_state.subject)
    st.session_state.total_students = st.number_input(
        "Total Students", min_value=1, max_value=200, value=st.session_state.total_students)
    st.session_state.duration_min = st.number_input(
        "Class Duration (min)", min_value=5, max_value=180, value=st.session_state.duration_min)

    st.markdown("---")
    st.markdown("**DETECTION SETTINGS**")
    st.session_state.threshold = st.slider(
        "Attention Threshold", 0.30, 0.90, st.session_state.threshold, 0.05,
        help="Score above this = Active student")
    st.session_state.camera_index = st.selectbox(
        "Camera", [0, 1, 2], index=st.session_state.camera_index)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ START", type="primary", disabled=st.session_state.monitoring):
            st.session_state.monitoring = True
            st.session_state.session_start = datetime.now()
            st.session_state.snapshots = []
            st.session_state.active_pct_history = deque(maxlen=120)
            st.session_state.last_snap_time = 0
            if st.session_state.cap is None or not st.session_state.cap.isOpened():
                cap = cv2.VideoCapture(st.session_state.camera_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                st.session_state.cap = cap
            st.rerun()

    with col2:
        if st.button("⏹ STOP", disabled=not st.session_state.monitoring):
            st.session_state.monitoring = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.rerun()

    if st.session_state.monitoring and st.session_state.session_start:
        elapsed = int((datetime.now() - st.session_state.session_start).total_seconds())
        mins, secs = elapsed // 60, elapsed % 60
        remain = max(0, st.session_state.duration_min * 60 - elapsed)
        r_min, r_sec = remain // 60, remain % 60
        st.markdown(f"""
        <div style='background:#0a1628; border:1px solid #1a3060; border-radius:8px; 
                    padding:0.8rem; margin-top:0.5rem; text-align:center;'>
            <div style='font-size:0.65rem; color:#4a7abf; letter-spacing:0.1em;'>ELAPSED</div>
            <div style='font-family: JetBrains Mono; font-size:1.4rem; color:#7ab8ff; font-weight:700;'>
                {mins:02d}:{secs:02d}
            </div>
            <div style='font-size:0.65rem; color:#4a7abf; letter-spacing:0.1em; margin-top:4px;'>REMAINING</div>
            <div style='font-family: JetBrains Mono; font-size:1.1rem; color:#ffa020;'>
                {r_min:02d}:{r_sec:02d}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.snapshots:
        st.markdown("---")
        if st.button("💾 Save Report"):
            os.makedirs("session_logs", exist_ok=True)
            sid = st.session_state.session_start.strftime("%Y%m%d_%H%M%S")
            csv_path = f"session_logs/session_{sid}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=st.session_state.snapshots[0].keys())
                writer.writeheader()
                writer.writerows(st.session_state.snapshots)
            st.success(f"Saved: {csv_path}")


# ──────────────────────────────────────────────
# MAIN TITLE
# ──────────────────────────────────────────────
status_badge = (
    '<span class="badge-active">● LIVE</span>' if st.session_state.monitoring
    else '<span class="badge-inactive">● STOPPED</span>'
)
st.markdown(f"""
<div class="title-bar">
    <div>
        <span style='font-size:1.6rem; font-weight:700; letter-spacing:0.05em; color:#e0e8ff;'>
            STUDENT ATTENTION MONITOR
        </span>
        &nbsp;&nbsp;{status_badge}
    </div>
    <div style='margin-left:auto; font-size:0.75rem; color:#4a6a9a; letter-spacing:0.08em;'>
        {st.session_state.class_name} &nbsp;|&nbsp; {st.session_state.subject}
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["📹  LIVE MONITOR", "📊  SESSION REPORT"])


# ══════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════
with tab1:
    if not st.session_state.monitoring:
        st.markdown("""
        <div style='text-align:center; padding:4rem 2rem; background:#0a0f1e; 
                    border:1px dashed #1a3060; border-radius:16px; margin-top:1rem;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>📹</div>
            <div style='font-size:1.2rem; color:#4a6a9a; font-weight:600; letter-spacing:0.05em;'>
                MONITORING NOT STARTED
            </div>
            <div style='font-size:0.85rem; color:#2a4a7a; margin-top:0.5rem;'>
                Configure settings in sidebar then press ▶ START
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Layout: camera left, stats right
        cam_col, stats_col = st.columns([3, 2], gap="medium")

        # Placeholders
        with cam_col:
            st.markdown("<div style='font-size:0.72rem; color:#4a6a9a; letter-spacing:0.12em; margin-bottom:4px;'>CAMERA FEED</div>", unsafe_allow_html=True)
            cam_placeholder = st.empty()

        with stats_col:
            st.markdown("<div style='font-size:0.72rem; color:#4a6a9a; letter-spacing:0.12em; margin-bottom:4px;'>LIVE STATS</div>", unsafe_allow_html=True)
            metric_placeholder = st.empty()
            chart_placeholder = st.empty()

        # ── MAIN LOOP ──
        cap = st.session_state.cap
        snap_interval = 5  # seconds

        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                annotated, active_count, visible_count = analyze_frame(
                    frame.copy(), st.session_state.threshold
                )

                st.session_state.current_active = active_count
                st.session_state.current_visible = visible_count

                total = st.session_state.total_students
                active_pct = (active_count / total * 100) if total > 0 else 0
                inactive_count = total - active_count
                inactive_pct = 100 - active_pct

                # Snapshot
                now = time.time()
                if now - st.session_state.last_snap_time >= snap_interval:
                    elapsed_s = int((datetime.now() - st.session_state.session_start).total_seconds())
                    snap = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "elapsed_sec": elapsed_s,
                        "active": active_count,
                        "visible": visible_count,
                        "total": total,
                        "active_pct": round(active_pct, 1),
                    }
                    st.session_state.snapshots.append(snap)
                    st.session_state.active_pct_history.append(active_pct)
                    st.session_state.last_snap_time = now

                # ── Camera display ──
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                with cam_col:
                    cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # ── Stats panel ──
                with stats_col:
                    grade_color = "#00e87a" if active_pct >= 70 else ("#ffa020" if active_pct >= 40 else "#ff6060")
                    grade_text = "EXCELLENT" if active_pct >= 70 else ("MODERATE" if active_pct >= 40 else "CRITICAL")

                    metric_placeholder.markdown(f"""
                    <div style='display:flex; flex-direction:column; gap:10px;'>
                        <!-- Big pct -->
                        <div style='background:linear-gradient(135deg,#0a1628,#0d1e38); 
                                    border:1px solid #1a3060; border-radius:12px; padding:1.2rem; text-align:center;'>
                            <div style='font-size:0.68rem; color:#4a7abf; letter-spacing:0.15em;'>ATTENTION RATE</div>
                            <div style='font-family:JetBrains Mono; font-size:3rem; font-weight:700; color:{grade_color}; line-height:1.1;'>
                                {active_pct:.1f}%
                            </div>
                            <div style='font-size:0.72rem; font-weight:600; color:{grade_color}; letter-spacing:0.1em;'>{grade_text}</div>
                        </div>
                        <!-- Active bar -->
                        <div style='background:#0d1525; border:1px solid #1a2744; border-radius:10px; padding:0.9rem;'>
                            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                                <span style='font-size:0.7rem; color:#4a7abf; letter-spacing:0.1em;'>ACTIVE</span>
                                <span style='font-family:JetBrains Mono; font-size:0.85rem; color:#00e87a; font-weight:700;'>
                                    {active_count} / {total}
                                </span>
                            </div>
                            <div class='progress-wrap'>
                                <div class='progress-fill-green' style='width:{min(active_pct,100):.1f}%;'></div>
                            </div>
                        </div>
                        <!-- Inactive bar -->
                        <div style='background:#0d1525; border:1px solid #1a2744; border-radius:10px; padding:0.9rem;'>
                            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                                <span style='font-size:0.7rem; color:#4a7abf; letter-spacing:0.1em;'>INACTIVE</span>
                                <span style='font-family:JetBrains Mono; font-size:0.85rem; color:#ff6060; font-weight:700;'>
                                    {inactive_count} / {total}
                                </span>
                            </div>
                            <div class='progress-wrap'>
                                <div class='progress-fill-red' style='width:{min(inactive_pct,100):.1f}%;'></div>
                            </div>
                        </div>
                        <!-- 4 mini metrics -->
                        <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;'>
                            <div style='background:#0a1225; border:1px solid #172040; border-radius:8px; padding:0.7rem; text-align:center;'>
                                <div style='font-size:0.62rem; color:#4a6a9a; letter-spacing:0.1em;'>VISIBLE</div>
                                <div style='font-family:JetBrains Mono; font-size:1.2rem; color:#7ab8ff; font-weight:700;'>{visible_count}</div>
                            </div>
                            <div style='background:#0a1225; border:1px solid #172040; border-radius:8px; padding:0.7rem; text-align:center;'>
                                <div style='font-size:0.62rem; color:#4a6a9a; letter-spacing:0.1em;'>TOTAL</div>
                                <div style='font-family:JetBrains Mono; font-size:1.2rem; color:#7ab8ff; font-weight:700;'>{total}</div>
                            </div>
                            <div style='background:#0a1225; border:1px solid #172040; border-radius:8px; padding:0.7rem; text-align:center;'>
                                <div style='font-size:0.62rem; color:#4a6a9a; letter-spacing:0.1em;'>SNAPS</div>
                                <div style='font-family:JetBrains Mono; font-size:1.2rem; color:#7ab8ff; font-weight:700;'>{len(st.session_state.snapshots)}</div>
                            </div>
                            <div style='background:#0a1225; border:1px solid #172040; border-radius:8px; padding:0.7rem; text-align:center;'>
                                <div style='font-size:0.62rem; color:#4a6a9a; letter-spacing:0.1em;'>AVG%</div>
                                <div style='font-family:JetBrains Mono; font-size:1.2rem; color:#ffa020; font-weight:700;'>
                                    {f"{np.mean(list(st.session_state.active_pct_history)):.0f}%" if st.session_state.active_pct_history else "—"}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Mini live chart
                    if len(st.session_state.active_pct_history) > 1:
                        hist = list(st.session_state.active_pct_history)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=hist,
                            mode="lines",
                            fill="tozeroy",
                            fillcolor="rgba(0,200,100,0.08)",
                            line=dict(color="#00e87a", width=2),
                            name="Active %",
                        ))
                        fig.add_hline(y=st.session_state.threshold * 100,
                                      line_dash="dot", line_color="#ffa020",
                                      annotation_text="Threshold", annotation_font_color="#ffa020",
                                      annotation_font_size=9)
                        fig.update_layout(
                            height=140,
                            margin=dict(l=0, r=0, t=4, b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(10,18,37,0.6)",
                            font=dict(color="#4a7abf", size=9),
                            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                            yaxis=dict(showgrid=True, gridcolor="#1a2744", range=[0, 100],
                                       tickfont=dict(size=9), zeroline=False),
                            showlegend=False,
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            else:
                st.error("❌ Camera frame nahi mili. Camera disconnect check karo.")

        # Auto-refresh
        time.sleep(0.08)
        st.rerun()


# ══════════════════════════════════════════════
# TAB 2 — SESSION REPORT
# ══════════════════════════════════════════════
with tab2:
    snaps = st.session_state.snapshots
    if not snaps:
        st.markdown("""
        <div style='text-align:center; padding:4rem 2rem; background:#0a0f1e; 
                    border:1px dashed #1a3060; border-radius:16px; margin-top:1rem;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>📊</div>
            <div style='font-size:1.2rem; color:#4a6a9a; font-weight:600; letter-spacing:0.05em;'>
                NO DATA YET
            </div>
            <div style='font-size:0.85rem; color:#2a4a7a; margin-top:0.5rem;'>
                Start monitoring to collect data and view reports here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(snaps)
        avg_pct = df["active_pct"].mean()
        max_pct = df["active_pct"].max()
        min_pct = df["active_pct"].min()
        total_time = df["elapsed_sec"].max() if len(df) > 0 else 0

        # ── Summary Metrics ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("AVG ATTENTION", f"{avg_pct:.1f}%",
                      delta=f"{'Good' if avg_pct >= 60 else 'Needs Attention'}")
        with c2:
            st.metric("PEAK", f"{max_pct:.1f}%")
        with c3:
            st.metric("LOWEST", f"{min_pct:.1f}%")
        with c4:
            mins = total_time // 60
            st.metric("SESSION TIME", f"{mins}m {total_time % 60}s")

        st.markdown("---")

        # ── Main Trend Chart ──
        col_chart, col_donut = st.columns([3, 1], gap="large")

        with col_chart:
            st.markdown("<div style='font-size:0.72rem; color:#4a6a9a; letter-spacing:0.12em; margin-bottom:6px;'>ATTENTION TREND OVER SESSION</div>", unsafe_allow_html=True)

            fig = go.Figure()

            # Shaded zones
            fig.add_hrect(y0=70, y1=100, fillcolor="rgba(0,200,80,0.04)",
                          line_width=0, annotation_text="Good Zone",
                          annotation_font_color="rgba(0,200,80,0.4)",
                          annotation_font_size=9)
            fig.add_hrect(y0=0, y1=40, fillcolor="rgba(220,50,50,0.04)",
                          line_width=0, annotation_text="Critical Zone",
                          annotation_font_color="rgba(220,50,50,0.4)",
                          annotation_font_size=9)

            # Active line
            fig.add_trace(go.Scatter(
                x=df["time"], y=df["active_pct"],
                mode="lines+markers",
                name="Active %",
                line=dict(color="#00e87a", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,232,122,0.06)",
                marker=dict(size=4, color="#00e87a"),
            ))

            # Moving average
            if len(df) >= 4:
                ma = df["active_pct"].rolling(4, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df["time"], y=ma,
                    mode="lines", name="Moving Avg",
                    line=dict(color="#ffa020", width=1.5, dash="dash"),
                ))

            fig.add_hline(y=st.session_state.threshold * 100,
                          line_dash="dot", line_color="#4a90e2",
                          annotation_text=f"Threshold ({st.session_state.threshold*100:.0f}%)",
                          annotation_font_color="#4a90e2", annotation_font_size=9)

            fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,18,37,0.8)",
                font=dict(color="#7a9cd4", size=10, family="JetBrains Mono"),
                xaxis=dict(gridcolor="#1a2744", tickfont=dict(size=9),
                           nticks=10, zeroline=False, title_text="Time"),
                yaxis=dict(gridcolor="#1a2744", range=[0, 105],
                           ticksuffix="%", zeroline=False, title_text="Active Students"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_donut:
            st.markdown("<div style='font-size:0.72rem; color:#4a6a9a; letter-spacing:0.12em; margin-bottom:6px;'>AVG DISTRIBUTION</div>", unsafe_allow_html=True)
            fig_d = go.Figure(go.Pie(
                values=[avg_pct, 100 - avg_pct],
                labels=["Active", "Inactive"],
                hole=0.65,
                marker=dict(colors=["#00e87a", "#cc3333"],
                            line=dict(color="#080c14", width=2)),
                textinfo="none",
            ))
            fig_d.add_annotation(
                text=f"<b>{avg_pct:.0f}%</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=22, color="#00e87a", family="JetBrains Mono"),
            )
            fig_d.update_layout(
                height=220,
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(size=9, color="#7a9cd4"),
                            bgcolor="rgba(0,0,0,0)", orientation="h",
                            x=0.1, y=-0.05),
                margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

        # ── Per-minute heatmap ──
        st.markdown("---")
        st.markdown("<div style='font-size:0.72rem; color:#4a6a9a; letter-spacing:0.12em; margin-bottom:6px;'>MINUTE-BY-MINUTE BREAKDOWN</div>", unsafe_allow_html=True)

        if len(df) > 0:
            df_copy = df.copy()
            df_copy["minute"] = (df_copy["elapsed_sec"] // 60).astype(int)
            minute_avg = df_copy.groupby("minute")["active_pct"].mean().reset_index()
            minute_avg.columns = ["Minute", "Avg Active %"]

            fig_bar = go.Figure(go.Bar(
                x=minute_avg["Minute"],
                y=minute_avg["Avg Active %"],
                marker_color=[
                    "#00e87a" if v >= 70 else ("#ffa020" if v >= 40 else "#cc3333")
                    for v in minute_avg["Avg Active %"]
                ],
                text=[f"{v:.0f}%" for v in minute_avg["Avg Active %"]],
                textposition="outside",
                textfont=dict(size=9, color="#7a9cd4"),
            ))
            fig_bar.update_layout(
                height=200,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,18,37,0.8)",
                font=dict(color="#7a9cd4", size=9, family="JetBrains Mono"),
                xaxis=dict(title="Minute", gridcolor="#1a2744", zeroline=False),
                yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#1a2744", zeroline=False),
                margin=dict(l=10, r=10, t=20, b=30),
                bargap=0.3,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # ── Raw data table ──
        st.markdown("---")
        with st.expander("📋 Raw Snapshot Data"):
            st.dataframe(
                df.style.background_gradient(subset=["active_pct"],
                                              cmap="RdYlGn", vmin=0, vmax=100),
                use_container_width=True, height=250
            )

        # ── Download buttons ──
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_str = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv_str,
                               file_name=f"session_{st.session_state.session_start.strftime('%Y%m%d_%H%M%S') if st.session_state.session_start else 'report'}.csv",
                               mime="text/csv")
        with col_dl2:
            summary = {
                "class": st.session_state.class_name,
                "subject": st.session_state.subject,
                "avg_attention_pct": float(round(avg_pct, 1)),
                "max_attention_pct": float(round(max_pct, 1)),
                "min_attention_pct": float(round(min_pct, 1)),
                "total_students": int(st.session_state.total_students),
                "session_duration_sec": int(total_time),
            }
            st.download_button("⬇️ Download JSON Summary",
                               json.dumps(summary, indent=2),
                               file_name="summary.json", mime="application/json")
