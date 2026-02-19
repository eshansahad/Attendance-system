"""
core/recognize.py
Real-time face recognition with:
  - Blink-based liveness detection (MediaPipe)
  - Confidence score calculation
  - Late / duplicate attendance via db_utils
  - Unknown-face intruder logging
  - Cross-platform audio feedback
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
from datetime import datetime

# ── PATH SETUP ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.alerts import send_intruder_alert
from database.db_utils import mark_attendance

# ── PATHS ────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "data_files", "embeddings", "embeddings.npy")
LABELS_PATH     = os.path.join(PROJECT_ROOT, "data_files", "embeddings", "labels.npy")
INTRUDERS_DIR   = os.path.join(PROJECT_ROOT, "data_files", "intruders")
os.makedirs(INTRUDERS_DIR, exist_ok=True)

# ── LOAD EMBEDDINGS ──────────────────────────────────────────────────────────
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        "embeddings.npy not found. Run core/extract_embeddings.py first."
    )

embeddings = np.load(EMBEDDINGS_PATH)
labels     = np.load(LABELS_PATH)

# ── MEDIAPIPE ────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(eye_coords: np.ndarray) -> float:
    """Eye Aspect Ratio – low value means eye is closing (blink)."""
    A = np.linalg.norm(eye_coords[1] - eye_coords[5])
    B = np.linalg.norm(eye_coords[2] - eye_coords[4])
    C = np.linalg.norm(eye_coords[0] - eye_coords[3])
    return (A + B) / (2.0 * C)


def _beep_success():
    """Cross-platform success beep."""
    try:
        import winsound
        winsound.Beep(800, 200)
    except ImportError:
        # macOS / Linux
        os.system("printf '\\a'")


def _beep_late():
    """Distinct beep for late arrivals."""
    try:
        import winsound
        winsound.Beep(500, 400)
    except ImportError:
        os.system("printf '\\a\\a'")


# ── THRESHOLDS & CONSTANTS ───────────────────────────────────────────────────
BLINK_THRESHOLD  = 0.20   # EAR below this → eye closing
BLINK_FRAMES     = 2      # consecutive frames to confirm blink
MATCH_THRESHOLD  = 0.70   # max distance to accept a face match
UNKNOWN_THRESHOLD= 0.95   # above this → treat as complete stranger
COOLDOWN_SECONDS = 5      # seconds before next person can be scanned

# ── GLOBAL STATE (shared across Flask frames) ─────────────────────────────
blink_counter     = 0
blink_detected    = False
attendance_marked = False
last_reset_time   = 0.0
unknown_counter   = 0

current_name       = "Scanning..."
current_status     = "Not Live"
current_confidence = 0.0       # ← NEW: shown in the video overlay


# ── MAIN PROCESSING FUNCTION ─────────────────────────────────────────────────
def process_frame_for_flask(frame: np.ndarray) -> np.ndarray:
    """
    Process a single BGR frame:
      1. Detect face landmarks.
      2. Check liveness via blink.
      3. Match face against known embeddings.
      4. Mark attendance (with late-detection + duplicate guard).
      5. Draw labelled bounding box.

    Returns the annotated frame.
    """
    global blink_counter, blink_detected, attendance_marked
    global last_reset_time, current_name, current_status
    global unknown_counter, current_confidence

    h, w, _ = frame.shape
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result   = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        coords    = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])

        # ── Bounding box from landmarks ──────────────────────────────────
        x1, y1 = np.min(coords, axis=0)
        x2, y2 = np.max(coords, axis=0)

        # ── Blink detection ──────────────────────────────────────────────
        left_ear  = _ear(coords[LEFT_EYE])
        right_ear = _ear(coords[RIGHT_EYE])
        avg_ear   = (left_ear + right_ear) / 2.0

        if avg_ear < BLINK_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_FRAMES:
                blink_detected = True
            blink_counter = 0

        # ── Face recognition (only after blink confirmed) ─────────────────
        if blink_detected and not attendance_marked:
            all_lm    = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            nose_tip  = all_lm[1]
            embedding = (all_lm - nose_tip).flatten()

            distances = np.linalg.norm(embeddings - embedding, axis=1)
            idx       = np.argmin(distances)
            dist      = distances[idx]

            if dist < MATCH_THRESHOLD:
                # ── Known face ──────────────────────────────────────────
                # Confidence: 100% when dist=0, 0% when dist=MATCH_THRESHOLD
                current_confidence = round((1.0 - dist / MATCH_THRESHOLD) * 100, 1)
                current_name       = labels[idx]

                marked, result_status = mark_attendance(
                    current_name,
                    confidence=current_confidence
                )

                if marked:
                    current_status = result_status   # 'Present' or 'Late'
                    attendance_marked = True
                    last_reset_time   = time.time()

                    if result_status == "Late":
                        _beep_late()
                    else:
                        _beep_success()

                elif result_status == "duplicate":
                    current_status = "Already marked"

            elif dist > UNKNOWN_THRESHOLD:
                # ── Unknown face ─────────────────────────────────────────
                current_name       = "UNKNOWN"
                current_status     = "Not Live"
                current_confidence = 0.0
                unknown_counter   += 1

                if unknown_counter == 5:
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(INTRUDERS_DIR, f"intruder_{ts}.jpg")
                    cv2.imwrite(path, frame)
                    try:
                        send_intruder_alert(path)
                    except Exception:
                        pass
                    unknown_counter = -60   # cooldown before next save

        # ── Reset after cooldown ─────────────────────────────────────────
        if attendance_marked and (time.time() - last_reset_time > COOLDOWN_SECONDS):
            attendance_marked  = False
            blink_detected     = False
            current_name       = "Scanning..."
            current_status     = "Not Live"
            current_confidence = 0.0

        # ── Draw annotated bounding box + label ──────────────────────────
        color = (0, 255, 0) if blink_detected else (0, 100, 255)
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)

        # Status line (name + liveness + confidence)
        if current_confidence > 0:
            label_text = f"{current_name}  |  {current_status}  |  {current_confidence:.1f}%"
        else:
            label_text = f"{current_name}  |  {current_status}"

        # Draw a small background pill for readability
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(
            frame,
            (ix1, iy1 - th - 14),
            (ix1 + tw + 8, iy1),
            (0, 0, 0),
            cv2.FILLED
        )
        cv2.putText(
            frame, label_text,
            (ix1 + 4, iy1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1
        )

        # Liveness hint
        liveness_text = (
            "LIVE ✓" if blink_detected else "Blink to verify"
        )
        cv2.putText(
            frame, liveness_text,
            (ix1, iy2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1
        )

    else:
        # No face detected
        current_name       = "Scanning..."
        current_status     = "Not Live"
        current_confidence = 0.0

    return frame


# ── STANDALONE MODE ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = process_frame_for_flask(frame)
        cv2.imshow("Smart Attend – Security Monitor", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()