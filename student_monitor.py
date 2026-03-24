"""
╔══════════════════════════════════════════════════════════════╗
║         STUDENT ATTENTION MONITORING SYSTEM                  ║
║         Real-time Class Engagement Tracker                   ║
╚══════════════════════════════════════════════════════════════╝

Features:
- Real-time face detection via webcam
- Head pose estimation (looking at camera = active)
- Per-student attention tracking
- Session percentage reports
- Auto-save CSV logs every minute
- Class-wise session management
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "total_students": 30,          # Class me kitne students hain
    "class_name": "Class 10-A",    # Class ka naam
    "subject": "Mathematics",      # Subject
    "class_duration_minutes": 60,  # 1 ghanta
    "attention_threshold": 0.65,   # Face score kitna ho toh "active" maano
    "snapshot_interval": 5,        # Har 5 second mein snapshot lo
    "output_dir": "session_logs",  # Log save kahan ho
    "camera_index": 0,             # Default webcam
}

# ─────────────────────────────────────────────
# COLORS (BGR format for OpenCV)
# ─────────────────────────────────────────────
COLOR = {
    "active":     (0, 220, 100),    # Green
    "inactive":   (0, 60, 220),     # Red
    "warning":    (0, 165, 255),    # Orange
    "text":       (255, 255, 255),  # White
    "bg_dark":    (20, 20, 35),     # Dark BG
    "accent":     (255, 200, 0),    # Yellow accent
    "panel_bg":   (30, 30, 50),     # Panel background
}


# ─────────────────────────────────────────────
# ATTENTION TRACKER CLASS
# ─────────────────────────────────────────────
class AttentionTracker:
    def __init__(self, config):
        self.cfg = config
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.snapshots = []          # List of {time, active_count, total_visible, pct}
        self.last_snapshot_time = time.time()
        self.face_history = []       # Rolling 3s attention scores
        self.total_seen = 0
        self.active_seen = 0
        self.cumulative_active_pct = []

        os.makedirs(config["output_dir"], exist_ok=True)

    def record_snapshot(self, active_count, visible_count):
        """Record a moment-in-time reading."""
        now = datetime.now()
        elapsed = (now - self.session_start).seconds
        pct = (active_count / self.cfg["total_students"]) * 100 if self.cfg["total_students"] > 0 else 0

        snap = {
            "time": now.strftime("%H:%M:%S"),
            "elapsed_sec": elapsed,
            "active": active_count,
            "visible": visible_count,
            "total": self.cfg["total_students"],
            "active_pct": round(pct, 1),
        }
        self.snapshots.append(snap)
        self.cumulative_active_pct.append(pct)
        return snap

    def get_session_summary(self):
        """Calculate overall session statistics."""
        if not self.cumulative_active_pct:
            return {}

        avg_pct = np.mean(self.cumulative_active_pct)
        max_pct = np.max(self.cumulative_active_pct)
        min_pct = np.min(self.cumulative_active_pct)
        elapsed = (datetime.now() - self.session_start).seconds

        return {
            "class": self.cfg["class_name"],
            "subject": self.cfg["subject"],
            "session_id": self.session_id,
            "start_time": self.session_start.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_minutes": elapsed // 60,
            "elapsed_seconds": elapsed % 60,
            "total_students": self.cfg["total_students"],
            "avg_attention_pct": round(avg_pct, 1),
            "max_attention_pct": round(max_pct, 1),
            "min_attention_pct": round(min_pct, 1),
            "total_snapshots": len(self.snapshots),
        }

    def save_csv(self):
        """Save snapshot log to CSV."""
        if not self.snapshots:
            return None
        path = os.path.join(
            self.cfg["output_dir"],
            f"session_{self.session_id}.csv"
        )
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.snapshots[0].keys())
            writer.writeheader()
            writer.writerows(self.snapshots)
        return path

    def save_summary_json(self):
        """Save session summary as JSON."""
        summary = self.get_session_summary()
        if not summary:
            return None
        path = os.path.join(
            self.cfg["output_dir"],
            f"summary_{self.session_id}.json"
        )
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path


# ─────────────────────────────────────────────
# FACE ANALYSIS (MediaPipe)
# ─────────────────────────────────────────────
class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=50,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def analyze_frame(self, frame):
        """
        Returns list of faces with attention scores.
        Score > threshold => student is looking at screen.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.face_mesh.process(rgb)

        faces = []

        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                lm = face_lm.landmark

                # Key landmark indices
                # Nose tip: 1, Left eye: 33, Right eye: 263
                # Left ear: 234, Right ear: 454
                # Chin: 175, Forehead: 10

                nose = np.array([lm[1].x * w, lm[1].y * h, lm[1].z * w])
                left_eye = np.array([lm[33].x * w, lm[33].y * h])
                right_eye = np.array([lm[263].x * w, lm[263].y * h])
                left_ear = np.array([lm[234].x * w, lm[234].y * h])
                right_ear = np.array([lm[454].x * w, lm[454].y * h])
                chin = np.array([lm[175].x * w, lm[175].y * h])
                forehead = np.array([lm[10].x * w, lm[10].y * h])

                # ─── Yaw (left-right rotation) ───
                ear_span = np.linalg.norm(right_ear - left_ear)
                nose_mid_offset = abs(nose[0] - (left_ear[0] + right_ear[0]) / 2)
                yaw_score = max(0, 1 - (nose_mid_offset / (ear_span * 0.6 + 1e-6)))

                # ─── Pitch (up-down tilt) ───
                face_height = np.linalg.norm(forehead - chin)
                eye_level_y = (left_eye[1] + right_eye[1]) / 2
                face_center_y = (forehead[1] + chin[1]) / 2
                pitch_offset = abs(eye_level_y - face_center_y)
                pitch_score = max(0, 1 - (pitch_offset / (face_height * 0.4 + 1e-6)))

                # ─── Visibility score (how complete is the face) ───
                all_x = [l.x for l in lm]
                all_y = [l.y for l in lm]
                bbox_w = (max(all_x) - min(all_x)) * w
                bbox_h = (max(all_y) - min(all_y)) * h
                visibility = min(1.0, (bbox_w * bbox_h) / (w * h * 0.04))

                # ─── Combined attention score ───
                attention = (yaw_score * 0.5 + pitch_score * 0.3 + visibility * 0.2)

                # Bounding box
                x_coords = [l.x for l in lm]
                y_coords = [l.y for l in lm]
                x1 = int(min(x_coords) * w) - 10
                y1 = int(min(y_coords) * h) - 10
                x2 = int(max(x_coords) * w) + 10
                y2 = int(max(y_coords) * h) + 10

                faces.append({
                    "bbox": (max(0, x1), max(0, y1), min(w, x2), min(h, y2)),
                    "attention": round(attention, 3),
                    "yaw_score": round(yaw_score, 3),
                    "pitch_score": round(pitch_score, 3),
                })

        return faces

    def close(self):
        self.face_detection.close()
        self.face_mesh.close()


# ─────────────────────────────────────────────
# HUD RENDERER
# ─────────────────────────────────────────────
class HUDRenderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX

    def draw_face_boxes(self, frame, faces, threshold):
        """Draw bounding boxes around detected faces."""
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            score = face["attention"]
            is_active = score >= threshold

            color = COLOR["active"] if is_active else COLOR["inactive"]
            label = f"{'ACTIVE' if is_active else 'INACTIVE'} {score:.0%}"

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (lw, lh), _ = cv2.getTextSize(label, self.font, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        self.font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def draw_side_panel(self, frame, tracker, active_count, visible_count, elapsed_sec):
        """Draw stats panel on right side."""
        h, w = frame.shape[:2]
        panel_w = 280
        panel_x = w - panel_w

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), COLOR["panel_bg"], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Header
        cv2.rectangle(frame, (panel_x, 0), (w, 55), (15, 15, 30), -1)
        cv2.putText(frame, "CLASS MONITOR", (panel_x + 10, 22),
                    self.font_bold, 0.65, COLOR["accent"], 1, cv2.LINE_AA)
        cv2.putText(frame, self.cfg["class_name"], (panel_x + 10, 45),
                    self.font, 0.5, COLOR["text"], 1, cv2.LINE_AA)

        total = self.cfg["total_students"]
        active_pct = (active_count / total * 100) if total > 0 else 0
        inactive_count = total - active_count
        inactive_pct = 100 - active_pct

        # Time
        mins, secs = elapsed_sec // 60, elapsed_sec % 60
        total_mins = self.cfg["class_duration_minutes"]
        remain = max(0, total_mins * 60 - elapsed_sec)
        r_min, r_sec = remain // 60, remain % 60

        y = 80
        stats = [
            ("SUBJECT", self.cfg["subject"]),
            ("TIME", f"{mins:02d}:{secs:02d} elapsed"),
            ("REMAINING", f"{r_min:02d}:{r_sec:02d}"),
            ("", ""),
            ("TOTAL STUDENTS", str(total)),
            ("VISIBLE NOW", str(visible_count)),
        ]

        for label, value in stats:
            if label == "":
                y += 8
                cv2.line(frame, (panel_x + 10, y), (w - 10, y), (60, 60, 80), 1)
                y += 12
                continue
            cv2.putText(frame, label, (panel_x + 10, y),
                        self.font, 0.38, (150, 150, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, value, (panel_x + 10, y + 18),
                        self.font_bold, 0.52, COLOR["text"], 1, cv2.LINE_AA)
            y += 40

        # Big percentage display
        y += 15
        cv2.line(frame, (panel_x + 10, y), (w - 10, y), (60, 60, 80), 1)
        y += 20

        # Active %
        pct_color = COLOR["active"] if active_pct >= 70 else (
            COLOR["warning"] if active_pct >= 40 else COLOR["inactive"])

        cv2.putText(frame, "ACTIVE STUDENTS", (panel_x + 10, y),
                    self.font, 0.4, (150, 150, 180), 1, cv2.LINE_AA)
        y += 5
        cv2.putText(frame, f"{active_pct:.1f}%", (panel_x + 10, y + 45),
                    self.font_bold, 1.6, pct_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{active_count}/{total} students", (panel_x + 10, y + 65),
                    self.font, 0.45, COLOR["text"], 1, cv2.LINE_AA)
        y += 80

        # Progress bar - Active
        bar_x, bar_y = panel_x + 10, y + 10
        bar_w = panel_w - 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14),
                      (50, 50, 70), -1)
        active_bar = int(bar_w * active_pct / 100)
        if active_bar > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + active_bar, bar_y + 14), COLOR["active"], -1)
        y += 30

        # Inactive %
        y += 35
        cv2.putText(frame, "INACTIVE STUDENTS", (panel_x + 10, y),
                    self.font, 0.4, (150, 150, 180), 1, cv2.LINE_AA)
        y += 5
        cv2.putText(frame, f"{inactive_pct:.1f}%", (panel_x + 10, y + 45),
                    self.font_bold, 1.6, COLOR["inactive"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"{inactive_count}/{total} students", (panel_x + 10, y + 65),
                    self.font, 0.45, COLOR["text"], 1, cv2.LINE_AA)
        y += 80

        # Avg session score
        if tracker.cumulative_active_pct:
            avg = np.mean(tracker.cumulative_active_pct)
            cv2.line(frame, (panel_x + 10, y), (w - 10, y), (60, 60, 80), 1)
            y += 20
            cv2.putText(frame, "SESSION AVG", (panel_x + 10, y),
                        self.font, 0.4, (150, 150, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{avg:.1f}%", (panel_x + 10, y + 25),
                        self.font_bold, 0.9, COLOR["accent"], 1, cv2.LINE_AA)
            y += 45

        # Bottom status
        snap_count = len(tracker.snapshots)
        cv2.putText(frame, f"Snapshots: {snap_count}  |  Press Q to quit",
                    (panel_x + 10, h - 15), self.font, 0.35,
                    (100, 100, 130), 1, cv2.LINE_AA)

        # Vertical divider
        cv2.line(frame, (panel_x, 0), (panel_x, h), (60, 60, 100), 2)

        return frame

    def draw_top_bar(self, frame, active_count, total, active_pct, timestamp):
        """Minimal top bar with key info."""
        h, w = frame.shape[:2]
        panel_w = w - 280  # Left side only

        # BG strip
        cv2.rectangle(frame, (0, 0), (panel_w, 40), (15, 15, 30), -1)

        status = "🟢 GOOD" if active_pct >= 70 else ("🟡 LOW" if active_pct >= 40 else "🔴 CRITICAL")
        text = f"  LIVE  |  {timestamp}  |  Active: {active_count}/{total} ({active_pct:.0f}%)  |  {status}"
        cv2.putText(frame, text, (10, 26), self.font, 0.5,
                    COLOR["accent"], 1, cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
def run_monitor(config=None):
    if config is None:
        config = CONFIG

    print("\n" + "═" * 60)
    print("  STUDENT ATTENTION MONITORING SYSTEM")
    print("═" * 60)
    print(f"  Class    : {config['class_name']}")
    print(f"  Subject  : {config['subject']}")
    print(f"  Students : {config['total_students']}")
    print(f"  Duration : {config['class_duration_minutes']} minutes")
    print(f"  Threshold: {config['attention_threshold']*100:.0f}% attention score")
    print("═" * 60)
    print("\n  Controls:")
    print("  [Q] - Quit & save report")
    print("  [S] - Manual snapshot")
    print("  [R] - Reset session")
    print("  [+/-] - Adjust total student count")
    print("\n  Starting camera...\n")

    # Init
    cap = cv2.VideoCapture(config["camera_index"])
    if not cap.isOpened():
        print("❌ ERROR: Camera nahi mili! Check camera_index in CONFIG.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    analyzer = FaceAnalyzer()
    tracker = AttentionTracker(config)
    hud = HUDRenderer(config)

    threshold = config["attention_threshold"]
    last_save_time = time.time()
    frame_count = 0
    current_faces = []
    process_every_n = 2  # Process every 2nd frame for performance

    print("✅ System ready! Camera feed mein students dikhao.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame nahi mila. Camera disconnect?")
            break

        frame = cv2.flip(frame, 1)  # Mirror
        frame_count += 1
        now = time.time()
        elapsed_sec = int((datetime.now() - tracker.session_start).total_seconds())

        # Analyze every Nth frame
        if frame_count % process_every_n == 0:
            current_faces = analyzer.analyze_frame(frame)

        # Count active
        active_faces = [f for f in current_faces if f["attention"] >= threshold]
        visible_count = len(current_faces)
        active_count = min(len(active_faces), config["total_students"])
        active_pct = (active_count / config["total_students"] * 100) if config["total_students"] > 0 else 0

        # Snapshot
        if now - last_save_time >= config["snapshot_interval"]:
            tracker.record_snapshot(active_count, visible_count)
            last_save_time = now

            # Auto-save CSV every minute
            if len(tracker.snapshots) % 12 == 0:
                path = tracker.save_csv()
                if path:
                    print(f"  💾 Auto-saved: {path}")

        # Draw
        frame = hud.draw_face_boxes(frame, current_faces, threshold)
        frame = hud.draw_top_bar(frame, active_count, config["total_students"],
                                 active_pct, datetime.now().strftime("%H:%M:%S"))
        frame = hud.draw_side_panel(frame, tracker, active_count,
                                    visible_count, elapsed_sec)

        # Class ended?
        if elapsed_sec >= config["class_duration_minutes"] * 60:
            cv2.putText(frame, "CLASS OVER - Press Q to save report",
                        (50, frame.shape[0] // 2), hud.font_bold, 1.2,
                        COLOR["accent"], 2, cv2.LINE_AA)

        cv2.imshow("Student Attention Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("s") or key == ord("S"):
            snap = tracker.record_snapshot(active_count, visible_count)
            print(f"  📸 Manual snapshot: {snap}")
        elif key == ord("r") or key == ord("R"):
            tracker = AttentionTracker(config)
            print("  🔄 Session reset!")
        elif key == ord("+") or key == ord("="):
            config["total_students"] = min(200, config["total_students"] + 1)
            print(f"  👥 Total students: {config['total_students']}")
        elif key == ord("-"):
            config["total_students"] = max(1, config["total_students"] - 1)
            print(f"  👥 Total students: {config['total_students']}")

    # ─── Cleanup & Final Report ───
    cap.release()
    cv2.destroyAllWindows()
    analyzer.close()

    print("\n" + "═" * 60)
    print("  SESSION REPORT")
    print("═" * 60)

    summary = tracker.get_session_summary()
    if summary:
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")

        csv_path = tracker.save_csv()
        json_path = tracker.save_summary_json()
        print(f"\n  💾 CSV  saved : {csv_path}")
        print(f"  💾 JSON saved : {json_path}")
    else:
        print("  No data recorded.")

    print("═" * 60)
    print("  Thanks! Monitoring session complete.\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Student Attention Monitoring System")
    parser.add_argument("--students", type=int, default=30, help="Total students in class")
    parser.add_argument("--class-name", type=str, default="Class 10-A", help="Class name")
    parser.add_argument("--subject", type=str, default="Mathematics", help="Subject name")
    parser.add_argument("--duration", type=int, default=60, help="Class duration in minutes")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0=default)")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Attention threshold (0.0-1.0)")
    args = parser.parse_args()

    custom_config = CONFIG.copy()
    custom_config["total_students"] = args.students
    custom_config["class_name"] = args.class_name
    custom_config["subject"] = args.subject
    custom_config["class_duration_minutes"] = args.duration
    custom_config["camera_index"] = args.camera
    custom_config["attention_threshold"] = args.threshold

    run_monitor(custom_config)
