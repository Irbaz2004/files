# 📹 Student Attention Monitoring System

Real-time student engagement tracker using webcam + AI face analysis.

---

## ⚡ Install karo

```bash
pip install -r requirements.txt
```

---

## 🚀 Chalao

### Basic (default settings):
```bash
python student_monitor.py
```

### Custom class settings:
```bash
python student_monitor.py \
  --students 35 \
  --class-name "Class 9-B" \
  --subject "Physics" \
  --duration 60 \
  --camera 0
```

### Arguments:
| Argument | Default | Description |
|---|---|---|
| `--students` | 30 | Total students in class |
| `--class-name` | "Class 10-A" | Class ka naam |
| `--subject` | "Mathematics" | Subject |
| `--duration` | 60 | Class minutes |
| `--camera` | 0 | Webcam index |
| `--threshold` | 0.65 | Attention cutoff (0.0-1.0) |

---

## 🎮 Controls (while running)

| Key | Action |
|---|---|
| `Q` | Quit + save report |
| `S` | Manual snapshot |
| `R` | Reset session |
| `+` | Increase total student count |
| `-` | Decrease total student count |

---

## 📊 Report dekhne ke liye:

```bash
python view_report.py
```

---

## 📁 Output files

- `session_logs/session_YYYYMMDD_HHMMSS.csv` — har snapshot ka data
- `session_logs/summary_YYYYMMDD_HHMMSS.json` — session summary

---

## 🧠 Kaise kaam karta hai?

1. **MediaPipe Face Mesh** se camera mein dikh rahe faces detect karta hai
2. Har face ke liye **head pose** calculate karta hai:
   - **Yaw** = left-right rotation (kahin aur dekh raha?)
   - **Pitch** = up-down tilt (neecha dekh raha?)
   - **Visibility** = kitna face dikh raha hai
3. In teeno ka **attention score** nikalta hai (0.0 - 1.0)
4. Score ≥ threshold → **ACTIVE** (green box)
5. Har 5 seconds mein snapshot leta hai
6. **Percentage = active_count / total_students × 100**

---

## 📈 Example Output

```
Session Report:
  Average Attention : 67.3%  ████████████████████░░░░░░
  Peak Attention    : 89.0%  ████████████████████████████░░
  Lowest Attention  : 31.0%  ██████████░░░░░░░░░░░░░░░░░░░░
  Grade             : GOOD 🟡
```
