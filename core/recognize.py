"""
core/recognize.py
─────────────────────────────────────────────────────────────────────────────
Multi-Face Detection  +  Anti-Spoofing  +  Blink Liveness  +  Recognition

Pipeline per face (in order):
  1. Anti-spoof check  (MiniFASNet ONNX — catches printed photos / screens)
  2. Blink liveness    (EAR-based — catches static real-face bypasses)
  3. Face recognition  (numpy embedding match)
  4. Attendance mark   (DB write via db_utils)

Public API (unchanged from original):
    process_frame_for_flask(frame: np.ndarray) -> np.ndarray

dashboard.py / video_feed.html need ZERO changes.

Model path:
    data_files/models/best_model.onnx   ← MiniFASNetV2SE (128×128 input)

If the model file is missing the system falls back to blink-only liveness
and prints a warning — it does NOT crash.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── PATH SETUP ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.alerts import send_intruder_alert
from database.db_utils import mark_attendance

# ── PATHS ────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH  = os.path.join(PROJECT_ROOT, "data_files", "embeddings", "embeddings.npy")
LABELS_PATH      = os.path.join(PROJECT_ROOT, "data_files", "embeddings", "labels.npy")
INTRUDERS_DIR    = os.path.join(PROJECT_ROOT, "data_files", "intruders")
ANTISPOOF_MODEL  = os.path.join(PROJECT_ROOT, "data_files", "models", "best_model.onnx")
os.makedirs(INTRUDERS_DIR, exist_ok=True)

# ── LOAD EMBEDDINGS ───────────────────────────────────────────────────────────
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        "embeddings.npy not found. Run core/extract_embeddings.py first."
    )

embeddings = np.load(EMBEDDINGS_PATH)
labels     = np.load(LABELS_PATH)

# ── LOAD ANTI-SPOOF MODEL (graceful fallback) ─────────────────────────────────
_antispoof_session = None
ANTISPOOF_AVAILABLE = False

try:
    import onnxruntime as ort
    if os.path.exists(ANTISPOOF_MODEL):
        _antispoof_session = ort.InferenceSession(
            ANTISPOOF_MODEL,
            providers=["CPUExecutionProvider"]
        )
        _antispoof_input_name   = _antispoof_session.get_inputs()[0].name
        _antispoof_input_shape  = _antispoof_session.get_inputs()[0].shape
        # Shape is [batch, C, H, W] — grab H
        ANTISPOOF_SIZE = int(_antispoof_input_shape[2]) if len(_antispoof_input_shape) == 4 else 128
        ANTISPOOF_AVAILABLE = True
        print(f"[ANTISPOOF] Model loaded — input {ANTISPOOF_SIZE}×{ANTISPOOF_SIZE}")
    else:
        print(f"[ANTISPOOF] Model not found at {ANTISPOOF_MODEL} — blink-only mode")
except ImportError:
    print("[ANTISPOOF] onnxruntime not installed — blink-only mode")
except Exception as e:
    print(f"[ANTISPOOF] Failed to load model: {e} — blink-only mode")

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
MAX_FACES    = 10
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ── CONSTANTS ────────────────────────────────────────────────────────────────
BLINK_THRESHOLD      = 0.20
BLINK_FRAMES         = 2
MATCH_THRESHOLD      = 0.70
UNKNOWN_THRESHOLD    = 0.95
COOLDOWN_SECONDS     = 5
STALE_SECONDS        = 3.0
TRACKING_DIST_RATIO  = 0.15
SPOOF_THRESHOLD      = 0.55   # probability above this = REAL face


# ── PER-FACE STATE ────────────────────────────────────────────────────────────
@dataclass
class FaceState:
    nose_tip_px:        np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_seen:          float      = field(default_factory=time.time)

    # Anti-spoof
    spoof_score:        float = 1.0    # 1.0 = real, 0.0 = spoof (smoothed)
    spoof_confirmed:    bool  = False  # True once enough real frames seen
    spoof_frame_count:  int   = 0      # consecutive real frames

    # Liveness
    blink_counter:      int   = 0
    blink_detected:     bool  = False

    # Attendance
    attendance_marked:  bool  = False
    last_reset_time:    float = 0.0

    # Display
    current_name:       str   = "Scanning..."
    current_status:     str   = "Not Live"
    current_confidence: float = 0.0

    # Intruder
    unknown_counter:    int   = 0


_face_states: Dict[int, FaceState] = {}
_next_key: int = 0


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _ear(eye_coords: np.ndarray) -> float:
    A = np.linalg.norm(eye_coords[1] - eye_coords[5])
    B = np.linalg.norm(eye_coords[2] - eye_coords[4])
    C = np.linalg.norm(eye_coords[0] - eye_coords[3])
    return (A + B) / (2.0 * C)


def _beep_success():
    try:
        import winsound; winsound.Beep(800, 200)
    except ImportError:
        os.system("printf '\\a'")


def _beep_late():
    try:
        import winsound; winsound.Beep(500, 400)
    except ImportError:
        os.system("printf '\\a\\a'")


def _check_antispoof(face_crop: np.ndarray) -> float:
    """
    Run the MiniFASNet ONNX model on a cropped face.

    Returns a float:
        > SPOOF_THRESHOLD  →  real face
        < SPOOF_THRESHOLD  →  spoof (photo / screen)

    If the model is unavailable returns 1.0 (treat as real — fallback to
    blink-only liveness).
    """
    if not ANTISPOOF_AVAILABLE or _antispoof_session is None:
        return 1.0

    try:
        # Resize to model input size
        img = cv2.resize(face_crop, (ANTISPOOF_SIZE, ANTISPOOF_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Normalise with ImageNet mean/std (standard for MiniFASNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std

        # HWC → CHW → NCHW
        img = np.transpose(img, (2, 0, 1))[np.newaxis].astype(np.float32)

        outputs = _antispoof_session.run(None, {_antispoof_input_name: img})
        logits  = outputs[0][0]     # shape: (2,)  [spoof, real]

        # Softmax
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()

        # Index 1 = real probability (MiniFASNetV2SE label convention)
        real_prob = float(probs[1]) if len(probs) >= 2 else float(probs[0])
        return real_prob

    except Exception:
        return 1.0   # fail safe — don't block attendance on model errors


# ── FACE TRACKING ─────────────────────────────────────────────────────────────
def _match_faces(
    detected_nose_tips: List[np.ndarray],
    frame_width: int,
) -> List[int]:
    global _face_states, _next_key

    threshold_px  = TRACKING_DIST_RATIO * frame_width
    matched_keys: List[int] = []
    claimed:      set       = set()

    for nose_tip in detected_nose_tips:
        best_key:  Optional[int] = None
        best_dist: float         = float("inf")

        for key, state in _face_states.items():
            if key in claimed:
                continue
            dist = float(np.linalg.norm(nose_tip - state.nose_tip_px))
            if dist < best_dist:
                best_dist = dist
                best_key  = key

        if best_key is not None and best_dist < threshold_px:
            claimed.add(best_key)
            matched_keys.append(best_key)
        else:
            new_state = FaceState(nose_tip_px=nose_tip.copy())
            _face_states[_next_key] = new_state
            claimed.add(_next_key)
            matched_keys.append(_next_key)
            _next_key += 1

    return matched_keys


def _prune_stale_states() -> None:
    now   = time.time()
    stale = [k for k, s in _face_states.items()
             if now - s.last_seen > STALE_SECONDS]
    for k in stale:
        del _face_states[k]


# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
_PALETTE = [
    (0,   255,   0),
    (255, 165,   0),
    (0,   200, 255),
    (180,   0, 255),
    (255,  50,  50),
    (50,  255, 150),
    (255, 220,   0),
    (0,   120, 255),
    (255, 100, 200),
    (100, 255, 100),
]


# ── DRAW HELPER ───────────────────────────────────────────────────────────────
def _put_label(frame, text, x, y, color, font_scale=0.55, thickness=1):
    """Draw text with a dark background pill for readability."""
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        frame,
        (x - 4, y - th - 8),
        (x + tw + 4, y + 4),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame, text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        color, thickness,
    )


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────
def process_frame_for_flask(frame: np.ndarray) -> np.ndarray:
    """
    Process a BGR frame — all faces simultaneously.

    Pipeline per face:
      1. Anti-spoof  →  abort with red SPOOF box if fake
      2. Blink check →  must blink to confirm liveness
      3. Recognition →  embedding match
      4. Attendance  →  DB write
    """
    _prune_stale_states()

    h, w, _ = frame.shape
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result   = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return frame

    # Collect nose-tip positions
    detected_nose_tips: List[np.ndarray] = []
    all_coords_list:    List[np.ndarray] = []

    for landmarks in result.multi_face_landmarks:
        coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])
        detected_nose_tips.append(coords[1][:2])
        all_coords_list.append(coords)

    state_keys = _match_faces(detected_nose_tips, w)

    for detection_idx, (landmarks, coords, key) in enumerate(
        zip(result.multi_face_landmarks, all_coords_list, state_keys)
    ):
        state             = _face_states[key]
        state.last_seen   = time.time()
        state.nose_tip_px = detected_nose_tips[detection_idx].copy()

        # Bounding box
        x1, y1 = np.min(coords, axis=0)
        x2, y2 = np.max(coords, axis=0)
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        # Clamp to frame bounds for safe crop
        cx1 = max(0, ix1); cy1 = max(0, iy1)
        cx2 = min(w, ix2); cy2 = min(h, iy2)

        # ── STEP 1: ANTI-SPOOF CHECK ─────────────────────────────────────────
        is_spoof = False
        if ANTISPOOF_AVAILABLE and (cx2 - cx1) > 20 and (cy2 - cy1) > 20:
            face_crop    = frame[cy1:cy2, cx1:cx2]
            real_prob    = _check_antispoof(face_crop)

            # Smooth score with EMA (α=0.4) to avoid single-frame flickers
            state.spoof_score = 0.4 * real_prob + 0.6 * state.spoof_score

            if state.spoof_score > SPOOF_THRESHOLD:
                state.spoof_frame_count += 1
                if state.spoof_frame_count >= 3:   # 3 consecutive real frames
                    state.spoof_confirmed = True
            else:
                state.spoof_frame_count = 0
                state.spoof_confirmed   = False
                is_spoof = True

        else:
            # Model unavailable — skip spoof check
            state.spoof_confirmed = True

        if is_spoof:
            # Draw red SPOOF box and skip the rest of the pipeline
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)
            _put_label(
                frame,
                f"SPOOF DETECTED  ({state.spoof_score:.0%} real)",
                ix1, iy1 - 6,
                (0, 0, 255),
            )
            cv2.putText(
                frame, "Anti-Spoof FAIL",
                (ix1, iy2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1,
            )
            continue   # next face

        # ── STEP 2: BLINK LIVENESS ───────────────────────────────────────────
        left_ear  = _ear(coords[LEFT_EYE])
        right_ear = _ear(coords[RIGHT_EYE])
        avg_ear   = (left_ear + right_ear) / 2.0

        if avg_ear < BLINK_THRESHOLD:
            state.blink_counter += 1
        else:
            if state.blink_counter >= BLINK_FRAMES:
                state.blink_detected = True
            state.blink_counter = 0

        # ── STEP 3: FACE RECOGNITION ─────────────────────────────────────────
        if state.blink_detected and not state.attendance_marked:
            all_lm    = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            nose_tip  = all_lm[1]
            embedding = (all_lm - nose_tip).flatten()

            distances = np.linalg.norm(embeddings - embedding, axis=1)
            idx       = int(np.argmin(distances))
            dist      = float(distances[idx])

            if dist < MATCH_THRESHOLD:
                state.current_confidence = round(
                    (1.0 - dist / MATCH_THRESHOLD) * 100, 1
                )
                state.current_name = labels[idx]

                marked, result_status = mark_attendance(
                    state.current_name,
                    confidence=state.current_confidence,
                )

                if marked:
                    state.current_status    = result_status
                    state.attendance_marked = True
                    state.last_reset_time   = time.time()
                    if result_status == "Late":
                        _beep_late()
                    else:
                        _beep_success()
                elif result_status == "duplicate":
                    state.current_status = "Already marked"

            elif dist > UNKNOWN_THRESHOLD:
                state.current_name       = "UNKNOWN"
                state.current_status     = "Not Live"
                state.current_confidence = 0.0
                state.unknown_counter   += 1

                if state.unknown_counter == 5:
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(
                        INTRUDERS_DIR, f"intruder_{ts}_{key}.jpg"
                    )
                    cv2.imwrite(path, frame)
                    try:
                        send_intruder_alert(path)
                    except Exception:
                        pass
                    state.unknown_counter = -60

        # ── COOLDOWN RESET ───────────────────────────────────────────────────
        if state.attendance_marked and (
            time.time() - state.last_reset_time > COOLDOWN_SECONDS
        ):
            state.attendance_marked  = False
            state.blink_detected     = False
            state.current_name       = "Scanning..."
            state.current_status     = "Not Live"
            state.current_confidence = 0.0

        # ── STEP 4: DRAW OVERLAY ─────────────────────────────────────────────
        color      = _PALETTE[key % len(_PALETTE)]
        draw_color = color if state.blink_detected else (0, 140, 255)

        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), draw_color, 2)

        # Main label: name | status | confidence
        if state.current_confidence > 0:
            label_text = (
                f"{state.current_name}  |  "
                f"{state.current_status}  |  "
                f"{state.current_confidence:.1f}%"
            )
        else:
            label_text = f"{state.current_name}  |  {state.current_status}"

        _put_label(frame, label_text, ix1 + 4, iy1 - 6, (255, 255, 255))

        # Liveness + spoof status below box
        if ANTISPOOF_AVAILABLE:
            spoof_txt = f"Real {state.spoof_score:.0%}"
            spoof_col = (0, 220, 60) if state.spoof_confirmed else (0, 140, 255)
            cv2.putText(
                frame, spoof_txt,
                (ix1, iy2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                spoof_col, 1,
            )
            cv2.putText(
                frame,
                "LIVE ✓" if state.blink_detected else "Blink to verify",
                (ix1, iy2 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                draw_color, 1,
            )
        else:
            cv2.putText(
                frame,
                "LIVE ✓" if state.blink_detected else "Blink to verify",
                (ix1, iy2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                draw_color, 1,
            )

    return frame


# ── STANDALONE MODE ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("[INFO] Press Q to quit")
    print(f"[INFO] Anti-spoof: {'ACTIVE' if ANTISPOOF_AVAILABLE else 'DISABLED (blink-only)'}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = process_frame_for_flask(frame)
        cv2.imshow("Smart Attend — Multi-Face + Anti-Spoof", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()