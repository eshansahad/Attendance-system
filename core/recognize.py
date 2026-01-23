import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# ==============================
# UPDATED IMPORT
# ==============================
# Add the project root to system path so we can import from 'database'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db_utils import mark_attendance

# ==============================
# UPDATED PATHS
# ==============================
EMBEDDINGS_PATH = "data_files/embeddings/embeddings.npy"
LABELS_PATH = "data_files/embeddings/labels.npy"

print("[INFO] Loading encodings...")
embeddings = np.load(EMBEDDINGS_PATH)
labels = np.load(LABELS_PATH)

# ==============================
# MediaPipe Face Mesh
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# Eye landmark indices
# ==============================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ==============================
# Camera setup
# ==============================
cap = cv2.VideoCapture(0)

BLINK_THRESHOLD = 0.25
BLINK_FRAMES = 2
blink_counter = 0
blink_detected = False

THRESHOLD = 1.0  # Face recognition sensitivity
attendance_marked = False  # Prevent duplicates

print("[INFO] Face recognition with liveness started...")

# ==============================
# Main loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    name = "Unknown"
    status = "Not Live"

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]

        # Convert landmarks to pixel coordinates
        coords = np.array(
            [[lm.x * frame.shape[1], lm.y * frame.shape[0]]
             for lm in landmarks.landmark]
        )

        # ==============================
        # Blink detection
        # ==============================
        left_eye = coords[LEFT_EYE]
        right_eye = coords[RIGHT_EYE]

        ear = (eye_aspect_ratio(left_eye) +
               eye_aspect_ratio(right_eye)) / 2.0

        if ear < BLINK_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_FRAMES:
                blink_detected = True
            blink_counter = 0

        # ==============================
        # If LIVE -> recognize face
        # ==============================
        if blink_detected:
            status = "Live"

            embedding = []
            for lm in landmarks.landmark:
                embedding.extend([lm.x, lm.y, lm.z])
            embedding = np.array(embedding)

            distances = np.linalg.norm(embeddings - embedding, axis=1)
            min_idx = np.argmin(distances)

            if distances[min_idx] < THRESHOLD:
                name = labels[min_idx]

                # MARK ATTENDANCE ONLY ONCE
                if not attendance_marked:
                    mark_attendance(name)
                    attendance_marked = True

    # ==============================
    # Display output
    # ==============================
    cv2.putText(frame, f"Name: {name}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Liveness: {status}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition with Liveness", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()