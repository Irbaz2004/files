"""
Session Report Viewer
Saved CSV/JSON files ko padh ke pretty report dikhata hai
"""

import json
import csv
import os
import glob
from datetime import datetime


def print_bar(value, max_val=100, width=40, fill="█", empty="░"):
    filled = int(width * value / max_val)
    return fill * filled + empty * (width - filled)


def view_latest_report(output_dir="session_logs"):
    """Latest session ki report dikhao."""

    json_files = glob.glob(os.path.join(output_dir, "summary_*.json"))
    if not json_files:
        print("❌ Koi session report nahi mili. Pehle monitor chalao.")
        return

    latest = max(json_files, key=os.path.getmtime)
    with open(latest) as f:
        s = json.load(f)

    # Find matching CSV
    session_id = s.get("session_id", "")
    csv_path = os.path.join(output_dir, f"session_{session_id}.csv")
    snapshots = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            snapshots = list(csv.DictReader(f))

    print("\n" + "═" * 62)
    print("  📊  SESSION REPORT")
    print("═" * 62)
    print(f"  Class      : {s.get('class', 'N/A')}")
    print(f"  Subject    : {s.get('subject', 'N/A')}")
    print(f"  Date/Time  : {s.get('start_time', 'N/A')}")
    print(f"  Duration   : {s.get('elapsed_minutes', 0)}m {s.get('elapsed_seconds', 0)}s")
    print(f"  Students   : {s.get('total_students', 0)}")
    print()

    avg = s.get("avg_attention_pct", 0)
    mx  = s.get("max_attention_pct", 0)
    mn  = s.get("min_attention_pct", 0)

    grade = "EXCELLENT 🟢" if avg >= 75 else ("GOOD 🟡" if avg >= 55 else ("POOR 🔴"))

    print(f"  ┌─ ATTENTION SUMMARY ─────────────────────────────────┐")
    print(f"  │  Average  : {avg:5.1f}%  {print_bar(avg, width=28)}")
    print(f"  │  Peak     : {mx:5.1f}%  {print_bar(mx, width=28)}")
    print(f"  │  Lowest   : {mn:5.1f}%  {print_bar(mn, width=28)}")
    print(f"  │  Grade    : {grade}")
    print(f"  └─────────────────────────────────────────────────────┘")

    if snapshots:
        print(f"\n  📈 Timeline (every {len(snapshots)} snapshots):\n")
        # Show up to 15 key snapshots
        step = max(1, len(snapshots) // 15)
        for i, snap in enumerate(snapshots[::step]):
            pct = float(snap.get("active_pct", 0))
            bar = print_bar(pct, width=25)
            print(f"  {snap['time']}  [{bar}] {pct:5.1f}%  "
                  f"({snap['active']}/{snap['total']} active)")

    print(f"\n  💾 Files: {latest}")
    if os.path.exists(csv_path):
        print(f"           {csv_path}")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "session_logs"
    view_latest_report(folder)
