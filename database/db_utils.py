"""
database/db_utils.py
Central database utilities for the attendance system.

Public API:
    mark_attendance(name, confidence, session)  → (bool, str)
    get_setting(key, default)                   → str
    set_setting(key, value)                     → None
    get_attendance_percentage(student_id)       → dict
    override_attendance(record_id, new_status, reason) → bool
"""
import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, "attendance.db")


# ── INTERNAL ───────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── SETTINGS ───────────────────────────────────────────────────────────────

def get_setting(key: str, default: str = "") -> str:
    """Return the value of a setting key, or *default* if not found."""
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row else default
    except Exception:
        return default


def set_setting(key: str, value: str) -> None:
    """Upsert a setting key-value pair."""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value)
        )
        conn.commit()


# ── ATTENDANCE MARKING ─────────────────────────────────────────────────────

def mark_attendance(
    name:       str,
    confidence: float = 0.0,
    session:    str   = None
) -> tuple[bool, str]:
    """
    Mark a student present (or late) for today's session.

    Returns
    -------
    (True,  'Present') – newly marked as on-time
    (True,  'Late')    – newly marked as late
    (False, 'duplicate')  – already marked in this session today
    (False, 'not_found')  – student name not in DB
    (False, 'error')      – unexpected DB error
    """
    if session is None:
        session = get_setting("session_name", "General")

    try:
        with _conn() as conn:

            # 1. Look up student
            student = conn.execute(
                "SELECT id FROM students WHERE name=?", (name,)
            ).fetchone()
            if not student:
                return False, "not_found"

            student_id = student["id"]
            today      = datetime.now().strftime("%Y-%m-%d")
            now_str    = datetime.now().strftime("%H:%M:%S")

            # 2. Duplicate check (same student + date + session)
            existing = conn.execute(
                "SELECT id FROM attendance"
                " WHERE student_id=? AND date=? AND session=?",
                (student_id, today, session)
            ).fetchone()
            if existing:
                return False, "duplicate"

            # 3. Late-arrival logic
            late_cutoff = get_setting("late_cutoff", "09:00")
            try:
                current_t = datetime.strptime(now_str[:5], "%H:%M")
                cutoff_t  = datetime.strptime(late_cutoff[:5], "%H:%M")
                status    = "Late" if current_t > cutoff_t else "Present"
            except ValueError:
                status = "Present"

            # 4. Insert
            conn.execute(
                """INSERT INTO attendance
                       (student_id, date, time, status, confidence, session)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (student_id, today, now_str, status,
                 round(float(confidence), 4), session)
            )
            conn.commit()

        return True, status

    except sqlite3.IntegrityError:
        # Unique index fired – treat as duplicate
        return False, "duplicate"
    except Exception as e:
        print(f"[DB ERROR] mark_attendance: {e}")
        return False, "error"


# ── ATTENDANCE STATS ────────────────────────────────────────────────────────

def get_attendance_percentage(student_id: int) -> dict:
    """
    Return attendance statistics for a student.

    Returns dict with keys:
        total     – total school days recorded in DB
        present   – days student was Present
        late      – days student was Late
        absent    – total - present - late
        percentage – (present+late) / total * 100   (0 if no school days)
        below_threshold – bool
    """
    with _conn() as conn:

        # Total school days = distinct dates with ANY attendance
        total = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM attendance"
        ).fetchone()[0]

        # This student's present days
        present = conn.execute(
            "SELECT COUNT(*) FROM attendance"
            " WHERE student_id=? AND status='Present'",
            (student_id,)
        ).fetchone()[0]

        # This student's late days
        late = conn.execute(
            "SELECT COUNT(*) FROM attendance"
            " WHERE student_id=? AND status='Late'",
            (student_id,)
        ).fetchone()[0]

    absent     = max(0, total - present - late)
    pct        = round((present + late) / total * 100, 1) if total else 0.0
    threshold  = float(get_setting("absent_threshold", "75"))

    return {
        "total":           total,
        "present":         present,
        "late":            late,
        "absent":          absent,
        "percentage":      pct,
        "below_threshold": pct < threshold and total > 0,
        "threshold":       threshold,
    }


# ── MANUAL OVERRIDE ─────────────────────────────────────────────────────────

def override_attendance(
    record_id:  int,
    new_status: str,
    reason:     str
) -> bool:
    """
    Allow a teacher to change the status of an existing attendance record.

    Allowed statuses: 'Present', 'Late', 'Absent'
    """
    allowed = {"Present", "Late", "Absent"}
    if new_status not in allowed:
        return False
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE attendance"
                " SET status=?, override_reason=?"
                " WHERE id=?",
                (new_status, reason.strip(), record_id)
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB ERROR] override_attendance: {e}")
        return False