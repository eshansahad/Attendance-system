"""
core/smart_register.py
─────────────────────────────────────────────────────────────────────────────
Smart face registration with quality-gated frame capture.

Phases (5 good frames each = 15 total):
  Phase 0 — Look straight at camera
  Phase 1 — Turn slightly LEFT
  Phase 2 — Turn slightly RIGHT

Quality gates per frame (ALL must pass to save):
  • Face detected by MediaPipe face mesh
  • Face bounding box area ≥ MIN_FACE_RATIO of frame area
  • Face centre within CENTRE_TOLERANCE of frame centre
  • Laplacian variance ≥ MIN_SHARPNESS (blur check)
  • Mean brightness between MIN_BRIGHT and MAX_BRIGHT

Public API:
    SmartRegistrar(student_name, dataset_path)
    registrar.process_frame(frame)  →  annotated frame
    registrar.status()              →  dict (for JSON endpoint)
    registrar.done                  →  True when all phases complete
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ── MEDIAPIPE (static mode for registration — more accurate) ──────────────────
_mp_face_mesh = mp.solutions.face_mesh
_face_mesh    = _mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ── QUALITY THRESHOLDS ────────────────────────────────────────────────────────
MIN_FACE_RATIO    = 0.10   # face box must cover ≥10% of frame area
CENTRE_TOLERANCE  = 0.30   # face centre within 30% of frame centre
MIN_SHARPNESS     = 80.0   # Laplacian variance (below = blurry)
MIN_BRIGHT        = 40     # mean pixel brightness (too dark)
MAX_BRIGHT        = 230    # mean pixel brightness (too bright / overexposed)
FRAMES_PER_PHASE  = 5      # good frames required per angle phase
CAPTURE_INTERVAL  = 0.4    # seconds between capture attempts

# ── PHASE DEFINITIONS ─────────────────────────────────────────────────────────
PHASES = [
    {
        "id":          0,
        "label":       "Look straight at the camera",
        "icon":        "→ ◎ ←",
        "short":       "Front",
    },
    {
        "id":          1,
        "label":       "Slowly turn your head slightly LEFT",
        "icon":        "← ◎",
        "short":       "Left",
    },
    {
        "id":          2,
        "label":       "Slowly turn your head slightly RIGHT",
        "icon":        "◎ →",
        "short":       "Right",
    },
]


# ── REGISTRAR ─────────────────────────────────────────────────────────────────
class SmartRegistrar:
    """
    Stateful per-student registration session.
    Call process_frame() once per camera frame.
    """

    def __init__(self, student_name: str, dataset_path: str):
        self.student_name  = student_name
        self.student_path  = os.path.join(dataset_path, student_name)
        os.makedirs(self.student_path, exist_ok=True)

        self.phase_idx      = 0           # 0, 1, 2
        self.phase_counts   = [0, 0, 0]  # good frames captured per phase
        self.total_saved    = 0
        self.done           = False

        self._last_capture  = 0.0        # timestamp of last saved frame
        self._feedback      = ""         # latest quality feedback message
        self._face_visible  = False

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyse frame, maybe save it, draw overlay.
        Returns the annotated frame.
        """
        if self.done:
            return self._draw_done(frame)

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = _face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            self._face_visible = False
            self._feedback     = "No face detected — please face the camera"
            return self._draw_overlay(frame, w, h, face_ok=False)

        self._face_visible = True
        landmarks = res.multi_face_landmarks[0]
        coords    = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])

        # Face bounding box
        x1, y1 = np.min(coords, axis=0)
        x2, y2 = np.max(coords, axis=0)
        fx, fy = (x1 + x2) / 2, (y1 + y2) / 2   # face centre

        # ── Quality checks ────────────────────────────────────────────────────
        quality_ok, reason = self._quality_check(
            frame, w, h, x1, y1, x2, y2, fx, fy
        )
        self._feedback = reason

        # ── Timed capture ─────────────────────────────────────────────────────
        now = time.time()
        if quality_ok and (now - self._last_capture) >= CAPTURE_INTERVAL:
            phase     = PHASES[self.phase_idx]
            img_name  = (
                f"{self.student_name}"
                f"_p{self.phase_idx}"
                f"_{self.phase_counts[self.phase_idx]}.jpg"
            )
            cv2.imwrite(os.path.join(self.student_path, img_name), frame)
            self.phase_counts[self.phase_idx] += 1
            self.total_saved                  += 1
            self._last_capture                 = now
            self._feedback = (
                f"✓ Frame {self.phase_counts[self.phase_idx]}"
                f"/{FRAMES_PER_PHASE} saved"
            )

            # Advance phase if this one is complete
            if self.phase_counts[self.phase_idx] >= FRAMES_PER_PHASE:
                self.phase_idx += 1
                if self.phase_idx >= len(PHASES):
                    self.done = True

        return self._draw_overlay(
            frame, w, h, x1, y1, x2, y2, quality_ok
        )

    def status(self) -> dict:
        """Return JSON-serialisable status dict for the AJAX endpoint."""
        phase = PHASES[self.phase_idx] if not self.done else PHASES[-1]
        return {
            "done":          self.done,
            "phase_idx":     self.phase_idx,
            "phase_label":   phase["label"] if not self.done else "Registration complete!",
            "phase_short":   phase["short"] if not self.done else "Done",
            "phase_counts":  self.phase_counts,
            "total_saved":   self.total_saved,
            "total_needed":  FRAMES_PER_PHASE * len(PHASES),
            "feedback":      self._feedback,
            "face_visible":  self._face_visible,
            "phases":        [
                {
                    "short":  p["short"],
                    "needed": FRAMES_PER_PHASE,
                    "saved":  self.phase_counts[i],
                    "done":   self.phase_counts[i] >= FRAMES_PER_PHASE,
                }
                for i, p in enumerate(PHASES)
            ],
        }

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _quality_check(
        self,
        frame: np.ndarray,
        w: int, h: int,
        x1: float, y1: float,
        x2: float, y2: float,
        fx: float, fy: float,
    ) -> Tuple[bool, str]:
        """Run all quality checks. Returns (passed, feedback_message)."""

        # 1. Face size
        face_area  = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        if face_area / frame_area < MIN_FACE_RATIO:
            return False, "Move closer to the camera"

        # 2. Face centering
        cx_offset = abs(fx - w / 2) / (w / 2)
        cy_offset = abs(fy - h / 2) / (h / 2)
        if cx_offset > CENTRE_TOLERANCE:
            direction = "right" if fx < w / 2 else "left"
            return False, f"Move your face to the {direction}"
        if cy_offset > CENTRE_TOLERANCE:
            direction = "down" if fy < h / 2 else "up"
            return False, f"Move your face {direction}"

        # 3. Sharpness (Laplacian variance on face crop)
        cx1 = max(0, int(x1)); cy1 = max(0, int(y1))
        cx2 = min(w, int(x2)); cy2 = min(h, int(y2))
        face_crop = frame[cy1:cy2, cx1:cx2]
        if face_crop.size > 0:
            gray  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharp < MIN_SHARPNESS:
                return False, "Hold still — image is blurry"

        # 4. Brightness
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray_full))
        if brightness < MIN_BRIGHT:
            return False, "Too dark — move to better lighting"
        if brightness > MAX_BRIGHT:
            return False, "Too bright — avoid direct light"

        return True, "Good — hold position"

    # ── DRAWING ───────────────────────────────────────────────────────────────

    def _draw_overlay(
        self,
        frame: np.ndarray,
        w: int, h: int,
        x1: float = 0, y1: float = 0,
        x2: float = 0, y2: float = 0,
        face_ok: bool = False,
    ) -> np.ndarray:
        """Draw oval guide, feedback text, phase banner, progress bar."""

        # ── Oval face guide ───────────────────────────────────────────────────
        oval_cx, oval_cy = w // 2, h // 2
        oval_rx, oval_ry = int(w * 0.18), int(h * 0.38)
        oval_color = (0, 220, 60) if face_ok else (0, 140, 255)
        cv2.ellipse(frame, (oval_cx, oval_cy), (oval_rx, oval_ry),
                    0, 0, 360, oval_color, 2)

        # Corner tick marks on oval
        for angle_deg in [0, 90, 180, 270]:
            ang_rad = np.deg2rad(angle_deg)
            px = int(oval_cx + oval_rx * np.cos(ang_rad))
            py = int(oval_cy + oval_ry * np.sin(ang_rad))
            cv2.circle(frame, (px, py), 4, oval_color, -1)

        # ── Face bounding box (only if face detected) ─────────────────────────
        if face_ok and x2 > x1:
            cv2.rectangle(
                frame,
                (int(x1), int(y1)), (int(x2), int(y2)),
                (0, 255, 180), 1,
            )

        # ── Top banner — current phase ────────────────────────────────────────
        if not self.done:
            phase_label = PHASES[self.phase_idx]["label"]
        else:
            phase_label = "All done! Processing embeddings..."

        banner_h = 50
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame, phase_label,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 1,
        )

        # ── Feedback line ─────────────────────────────────────────────────────
        feedback_color = (0, 220, 60) if face_ok else (0, 140, 255)
        cv2.putText(
            frame, self._feedback,
            (12, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            feedback_color, 1,
        )

        # ── Progress bar ──────────────────────────────────────────────────────
        bar_x, bar_y  = 12, h - 32
        bar_w, bar_h  = w - 24, 16
        total_needed  = FRAMES_PER_PHASE * len(PHASES)
        fill_w        = int(bar_w * self.total_saved / total_needed)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (40, 40, 40), -1)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h),
                          (0, 200, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), 1)
        cv2.putText(
            frame,
            f"{self.total_saved}/{total_needed} frames",
            (bar_x + 4, bar_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (255, 255, 255), 1,
        )

        # ── Phase progress dots ───────────────────────────────────────────────
        dot_y = h - 52
        for i, p in enumerate(PHASES):
            saved  = self.phase_counts[i]
            needed = FRAMES_PER_PHASE
            done   = saved >= needed

            dot_x  = w - (len(PHASES) - i) * 80 - 12
            color  = (0, 200, 80) if done else (
                (0, 180, 255) if i == self.phase_idx else (60, 60, 60)
            )
            cv2.circle(frame, (dot_x, dot_y), 8, color, -1)
            cv2.putText(
                frame, p["short"],
                (dot_x - 16, dot_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1,
            )
            if not done:
                cv2.putText(
                    frame, f"{saved}/{needed}",
                    (dot_x - 10, dot_y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    color, 1,
                )

        return frame

    def _draw_done(self, frame: np.ndarray) -> np.ndarray:
        """Full-frame overlay shown when registration is complete."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        cv2.putText(
            frame, "Registration Complete!",
            (cx - 200, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 220, 60), 2,
        )
        cv2.putText(
            frame, f"{self.total_saved} quality frames saved",
            (cx - 150, cy + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (180, 180, 180), 1,
        )
        return frame