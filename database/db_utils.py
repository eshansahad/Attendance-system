"""
database/db_utils.py  ·  Priority 3 upgrade
─────────────────────────────────────────────
Public API (unchanged + additions):
    mark_attendance(name, confidence, session)   → (bool, str)
    get_setting(key, default)                    → str
    set_setting(key, value)                      → None
    get_attendance_percentage(student_id)        → dict
    override_attendance(record_id, new_status, reason) → bool

    [NEW] get_active_session()                   → str
        Returns the name of whichever timetable period is happening right
        now based on current day + time.  Falls back to the settings
        session_name if nothing matches, then to "General".

    [NEW] get_all_subjects()                     → list[dict]
    [NEW] get_timetable()                        → list[dict]
    [NEW] add_subject(name, code, teacher)       → int | None
    [NEW] delete_subject(subject_id)             → bool
    [NEW] add_period(subject_id, day, start, end)→ int | None
    [NEW] delete_period(period_id)               → bool
    [NEW] get_today_schedule()                   → list[dict]
        Returns today's periods in time order, each with a status:
        'upcoming' | 'active' | 'ended'

BACKWARD COMPATIBILITY:
    mark_attendance() still accepts session=None.
    When session=None it now calls get_active_session() instead of
    directly reading the 'session_name' setting.
    The old behaviour (returning the session_name setting) is preserved
    as the fallback inside get_active_session() itself.
"""

import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "attendance.db")


# ── INTERNAL ───────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    # Enable foreign key enforcement so ON DELETE CASCADE works
    c.execute("PRAGMA foreign_keys = ON")
    return c


# ── SETTINGS ───────────────────────────────────────────────────────────────

def get_setting(key: str, default: str = "") -> str:
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row else default
    except Exception:
        return default


def set_setting(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value)
        )
        conn.commit()


# ── PRIORITY 3: ACTIVE SESSION DETECTION ──────────────────────────────────

def get_active_session() -> str:
    """
    Determine which session/subject is currently active.

    Logic:
      1. Query the timetable for today's day_of_week.
      2. Find a period whose start_time <= NOW <= end_time.
      3. Return the subject name for that period.
      4. If no active period: check for an upcoming period within 10 min
         (so the camera is already labelled correctly before class starts).
      5. If still nothing: fall back to settings 'session_name' → 'General'.
    """
    now         = datetime.now()
    day_of_week = now.weekday()   # 0=Monday, 6=Sunday
    now_str     = now.strftime("%H:%M")

    try:
        with _conn() as conn:
            # Check if timetable table exists (graceful fallback for old DBs)
            has_table = conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name='timetable'"
            ).fetchone()

            if not has_table:
                return get_setting("session_name", "General")

            rows = conn.execute("""
                SELECT t.id, t.start_time, t.end_time, s.name AS subject_name
                FROM   timetable t
                JOIN   subjects  s ON t.subject_id = s.id
                WHERE  t.day_of_week = ?
                  AND  t.is_active   = 1
                  AND  s.is_active   = 1
                ORDER  BY t.start_time
            """, (day_of_week,)).fetchall()

        if not rows:
            return get_setting("session_name", "General")

        # 1. Exact match: now is inside a period
        for row in rows:
            if row["start_time"] <= now_str <= row["end_time"]:
                return row["subject_name"]

        # 2. Upcoming within 10 minutes (pre-class grace window)
        from datetime import timedelta
        ten_min_later = (now + timedelta(minutes=10)).strftime("%H:%M")
        for row in rows:
            if now_str < row["start_time"] <= ten_min_later:
                return row["subject_name"]

        # 3. Fallback
        return get_setting("session_name", "General")

    except Exception as e:
        print(f"[DB] get_active_session error: {e}")
        return get_setting("session_name", "General")


def get_today_schedule() -> list:
    """
    Return today's timetable periods in time order.
    Each entry is a dict:
        id, subject_name, subject_code, start_time, end_time, status
    status: 'active' | 'upcoming' | 'ended'
    """
    now         = datetime.now()
    day_of_week = now.weekday()
    now_str     = now.strftime("%H:%M")

    try:
        with _conn() as conn:
            has_table = conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name='timetable'"
            ).fetchone()
            if not has_table:
                return []

            rows = conn.execute("""
                SELECT t.id, t.start_time, t.end_time,
                       s.name AS subject_name, s.code AS subject_code
                FROM   timetable t
                JOIN   subjects  s ON t.subject_id = s.id
                WHERE  t.day_of_week = ?
                  AND  t.is_active   = 1
                  AND  s.is_active   = 1
                ORDER  BY t.start_time
            """, (day_of_week,)).fetchall()

        schedule = []
        for row in rows:
            if row["start_time"] <= now_str <= row["end_time"]:
                status = "active"
            elif now_str < row["start_time"]:
                status = "upcoming"
            else:
                status = "ended"

            schedule.append({
                "id":           row["id"],
                "subject_name": row["subject_name"],
                "subject_code": row["subject_code"],
                "start_time":   row["start_time"],
                "end_time":     row["end_time"],
                "status":       status,
            })
        return schedule

    except Exception as e:
        print(f"[DB] get_today_schedule error: {e}")
        return []


# ── PRIORITY 3: SUBJECT & TIMETABLE CRUD ──────────────────────────────────

def get_all_subjects() -> list:
    """Return all active subjects as a list of dicts."""
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT id, name, code, teacher, is_active"
                " FROM subjects ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_timetable() -> list:
    """
    Return all active timetable entries joined with subject info.
    Sorted by day_of_week then start_time.
    """
    try:
        with _conn() as conn:
            rows = conn.execute("""
                SELECT t.id, t.day_of_week, t.start_time, t.end_time,
                       t.is_active,
                       s.id AS subject_id, s.name AS subject_name,
                       s.code AS subject_code
                FROM   timetable t
                JOIN   subjects  s ON t.subject_id = s.id
                ORDER  BY t.day_of_week, t.start_time
            """).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def add_subject(name: str, code: str = "", teacher: str = "") -> "int | None":
    """Insert a new subject. Returns new subject id or None on failure."""
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO subjects (name, code, teacher) VALUES (?, ?, ?)",
                (name.strip(), code.strip(), teacher.strip())
            )
            conn.commit()
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None   # duplicate name
    except Exception as e:
        print(f"[DB] add_subject error: {e}")
        return None


def delete_subject(subject_id: int) -> bool:
    """
    Soft-delete a subject (sets is_active=0).
    Associated timetable rows are also soft-deleted.
    We soft-delete rather than hard-delete to preserve attendance history
    whose session column references the subject name.
    """
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE subjects  SET is_active=0 WHERE id=?", (subject_id,)
            )
            conn.execute(
                "UPDATE timetable SET is_active=0 WHERE subject_id=?",
                (subject_id,)
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB] delete_subject error: {e}")
        return False


def add_period(
    subject_id: int,
    day_of_week: int,
    start_time:  str,
    end_time:    str,
) -> "int | None":
    """
    Add a timetable period.
    Validates:
      • day_of_week in 0–6
      • start_time < end_time (HH:MM strings compare lexicographically)
    Returns new period id or None on validation/DB failure.
    """
    if not (0 <= day_of_week <= 6):
        return None
    if start_time >= end_time:
        return None
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO timetable"
                " (subject_id, day_of_week, start_time, end_time)"
                " VALUES (?, ?, ?, ?)",
                (subject_id, day_of_week, start_time[:5], end_time[:5])
            )
            conn.commit()
            return cur.lastrowid
    except Exception as e:
        print(f"[DB] add_period error: {e}")
        return None


def delete_period(period_id: int) -> bool:
    """Hard-delete a timetable period (it has no FK children)."""
    try:
        with _conn() as conn:
            conn.execute(
                "DELETE FROM timetable WHERE id=?", (period_id,)
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB] delete_period error: {e}")
        return False


# ── ATTENDANCE MARKING ─────────────────────────────────────────────────────

def mark_attendance(
    name:       str,
    confidence: float = 0.0,
    session:    str   = None,
) -> "tuple[bool, str]":
    """
    Mark a student present (or late) for today's session.

    Priority 3 change: when session=None, calls get_active_session()
    instead of directly reading the 'session_name' setting.
    get_active_session() itself falls back to session_name → 'General',
    so backward compatibility is fully preserved.

    Returns
    -------
    (True,  'Present')   – newly marked on-time
    (True,  'Late')      – newly marked late
    (False, 'duplicate') – already marked this session today
    (False, 'not_found') – student name not in DB
    (False, 'error')     – unexpected DB error
    """
    if session is None:
        # Priority 3: auto-detect from timetable
        session = get_active_session()

    try:
        with _conn() as conn:

            student = conn.execute(
                "SELECT id FROM students WHERE name=?", (name,)
            ).fetchone()
            if not student:
                return False, "not_found"

            student_id = student["id"]
            today      = datetime.now().strftime("%Y-%m-%d")
            now_str    = datetime.now().strftime("%H:%M:%S")

            existing = conn.execute(
                "SELECT id FROM attendance"
                " WHERE student_id=? AND date=? AND session=?",
                (student_id, today, session)
            ).fetchone()
            if existing:
                return False, "duplicate"

            # Late-arrival logic
            late_cutoff = get_setting("late_cutoff", "09:00")
            try:
                current_t = datetime.strptime(now_str[:5], "%H:%M")
                cutoff_t  = datetime.strptime(late_cutoff[:5],  "%H:%M")
                status    = "Late" if current_t > cutoff_t else "Present"
            except ValueError:
                status = "Present"

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
        return False, "duplicate"
    except Exception as e:
        print(f"[DB ERROR] mark_attendance: {e}")
        return False, "error"


# ── ATTENDANCE STATS ────────────────────────────────────────────────────────

def get_attendance_percentage(student_id: int) -> dict:
    """Return attendance statistics for a student (all sessions combined)."""
    with _conn() as conn:

        total = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM attendance"
        ).fetchone()[0]

        present = conn.execute(
            "SELECT COUNT(*) FROM attendance"
            " WHERE student_id=? AND status='Present'",
            (student_id,)
        ).fetchone()[0]

        late = conn.execute(
            "SELECT COUNT(*) FROM attendance"
            " WHERE student_id=? AND status='Late'",
            (student_id,)
        ).fetchone()[0]

    absent    = max(0, total - present - late)
    pct       = round((present + late) / total * 100, 1) if total else 0.0
    threshold = float(get_setting("absent_threshold", "75"))

    return {
        "total":           total,
        "present":         present,
        "late":            late,
        "absent":          absent,
        "percentage":      pct,
        "below_threshold": pct < threshold and total > 0,
        "threshold":       threshold,
    }


def get_attendance_by_subject(student_id: int) -> list:
    """
    Return per-subject attendance stats for a student.
    Used by the student dashboard to show attendance broken down by class.

    Returns list of dicts:
        subject_name, total_classes, attended, percentage
    """
    try:
        with _conn() as conn:
            # All distinct (date, session) pairs = one class session each
            all_sessions = conn.execute(
                "SELECT session, COUNT(DISTINCT date) AS class_days"
                " FROM attendance"
                " GROUP BY session"
            ).fetchall()

            student_sessions = conn.execute(
                "SELECT session, COUNT(*) AS attended"
                " FROM attendance"
                " WHERE student_id=?"
                " GROUP BY session",
                (student_id,)
            ).fetchall()

        attended_map = {r["session"]: r["attended"] for r in student_sessions}

        result = []
        for row in all_sessions:
            sess     = row["session"]
            total    = row["class_days"]
            attended = attended_map.get(sess, 0)
            pct      = round(attended / total * 100, 1) if total else 0.0
            result.append({
                "subject_name":    sess,
                "total_classes":   total,
                "attended":        attended,
                "percentage":      pct,
            })

        return sorted(result, key=lambda x: x["subject_name"])

    except Exception as e:
        print(f"[DB] get_attendance_by_subject error: {e}")
        return []


# ── MANUAL OVERRIDE ─────────────────────────────────────────────────────────

def override_attendance(
    record_id:  int,
    new_status: str,
    reason:     str,
) -> bool:
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