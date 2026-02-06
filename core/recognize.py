import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import winsound
from datetime import datetime

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.alerts import send_intruder_alert
from database.db_utils import mark_attendance

# ==============================
# PATHS
# ==============================
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "data_files", "embeddings", "embeddings.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data_files", "embeddings", "labels.npy")
INTRUDERS_DIR = os.path.join(PROJECT_ROOT, "data_files", "intruders")
os.makedirs(INTRUDERS_DIR, exist_ok=True)

# ==============================
# LOAD EMBEDDINGS
# ==============================
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError("Run extract_embeddings.py first")

embeddings = np.load(EMBEDDINGS_PATH)
labels = np.load(LABELS_PATH)

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ==============================
# GLOBAL STATE (Flask-safe)
# ==============================
BLINK_THRESHOLD = 0.20
BLINK_FRAMES = 2
MATCH_THRESHOLD = 0.70
UNKNOWN_THRESHOLD = 0.95
COOLDOWN_SECONDS = 5

blink_counter = 0
blink_detected = False
attendance_marked = False
last_reset_time = 0
unknown_counter = 0

current_name = "Scanning..."
current_status = "Not Live"

# ==============================
# 🔥 MAIN FLASK FUNCTION
# ==============================
def process_frame_for_flask(frame):
    global blink_counter, blink_detected, attendance_marked
    global last_reset_time, current_name, current_status, unknown_counter

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]

        coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])
        x1, y1 = np.min(coords, axis=0)
        x2, y2 = np.max(coords, axis=0)

        # ----- BLINK CHECK -----
        ear = (
            eye_aspect_ratio(coords[LEFT_EYE]) +
            eye_aspect_ratio(coords[RIGHT_EYE])
        ) / 2.0

        if ear < BLINK_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_FRAMES:
                blink_detected = True
            blink_counter = 0

        # ----- FACE RECOGNITION -----
        if blink_detected and not attendance_marked:
            all_lm = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            nose_tip = all_lm[1]
            embedding = (all_lm - nose_tip).flatten()

            distances = np.linalg.norm(embeddings - embedding, axis=1)
            idx = np.argmin(distances)
            dist = distances[idx]

            if dist < MATCH_THRESHOLD:
                current_name = labels[idx]
                current_status = "LIVE"
                mark_attendance(current_name)
                attendance_marked = True
                last_reset_time = time.time()

                try:
                    winsound.Beep(800, 200)
                except:
                    pass

            elif dist > UNKNOWN_THRESHOLD:
                current_name = "UNKNOWN"
                unknown_counter += 1

                if unknown_counter == 5:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(INTRUDERS_DIR, f"intruder_{ts}.jpg")
                    cv2.imwrite(path, frame)
                    send_intruder_alert(path)
                    unknown_counter = -60

        # ----- RESET TIMER -----
        if attendance_marked:
            if time.time() - last_reset_time > COOLDOWN_SECONDS:
                attendance_marked = False
                blink_detected = False
                current_name = "Scanning..."
                current_status = "Not Live"

        # ----- DRAW UI -----
        color = (0, 255, 0) if blink_detected else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            f"{current_name} | {current_status}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    else:
        current_name = "Scanning..."
        current_status = "Not Live"

    return frame


# ==============================
# OPTIONAL: STANDALONE MODE
# ==============================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame_for_flask(frame)
        cv2.imshow("Security AI Monitor", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
