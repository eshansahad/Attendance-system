"""
database/migrate_db.py  ·  Priority 3 addition
───────────────────────────────────────────────
Run this ONCE after pulling the Priority 3 update.
Safe to run multiple times — skips anything that already exists.

New tables added:
  subjects   — subject/course names (e.g. "CS-101", "Mathematics")
  timetable  — period slots: subject + day-of-week + start_time + end_time

Usage:
    python database/migrate_db.py
"""
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "attendance.db")


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found at: {DB_PATH}")
        print("[FIX] Run database/fix_db.py first.")
        return

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── 1. EXISTING COLUMNS (unchanged from before) ──────────────────────────
    existing_cols = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(attendance)").fetchall()
    }

    new_columns = [
        ("status",          "TEXT    NOT NULL DEFAULT 'Present'"),
        ("confidence",      "REAL    NOT NULL DEFAULT 0.0"),
        ("session",         "TEXT    NOT NULL DEFAULT 'General'"),
        ("override_reason", "TEXT             DEFAULT NULL"),
    ]

    for col_name, col_def in new_columns:
        if col_name not in existing_cols:
            cursor.execute(
                f"ALTER TABLE attendance ADD COLUMN {col_name} {col_def}"
            )
            print(f"  [ADD]  attendance.{col_name}")
        else:
            print(f"  [OK]   attendance.{col_name} already exists")

    cursor.execute(
        "UPDATE attendance SET status='Present' WHERE status IS NULL OR status=''"
    )

    # ── 2. SETTINGS TABLE ────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    print("  [OK]   settings table ready")

    defaults = [
        ("late_cutoff",      "09:00"),
        ("absent_threshold", "75"),
        ("session_name",     "General"),
    ]
    for key, value in defaults:
        cursor.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )
        print(f"  [SET]  {key} = {value} (default, won't overwrite)")

    # ── 3. UNIQUE INDEX on attendance ────────────────────────────────────────
    try:
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            uq_attendance_student_date_session
            ON attendance(student_id, date, session)
        """)
        print("  [OK]   unique index on (student_id, date, session)")
    except sqlite3.OperationalError as e:
        print(f"  [WARN] Could not create unique index: {e}")

    # ── 4. students.is_active ────────────────────────────────────────────────
    student_cols = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(students)").fetchall()
    }
    if "is_active" not in student_cols:
        cursor.execute(
            "ALTER TABLE students ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
        )
        print("  [ADD]  students.is_active")
    else:
        print("  [OK]   students.is_active already exists")

    # ── 5. COMPLAINTS TABLE ──────────────────────────────────────────────────
    try:
        complaint_cols = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(complaints)").fetchall()
        }
        if "student_seen" not in complaint_cols:
            cursor.execute(
                "ALTER TABLE complaints"
                " ADD COLUMN student_seen INTEGER NOT NULL DEFAULT 1"
            )
            cursor.execute(
                "UPDATE complaints SET student_seen=1"
                " WHERE status IN ('Accepted','Declined')"
            )
            print("  [ADD]  complaints.student_seen")
        else:
            print("  [OK]   complaints.student_seen already exists")
    except Exception:
        pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  INTEGER NOT NULL,
            date        TEXT    NOT NULL,
            reason      TEXT    NOT NULL,
            description TEXT    NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'Pending',
            teacher_note TEXT            DEFAULT NULL,
            created_at  TEXT    NOT NULL,
            resolved_at TEXT             DEFAULT NULL,
            student_seen INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    print("  [OK]   complaints table ready")

    # ── 6. PRIORITY 3 — SUBJECTS TABLE ───────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL UNIQUE,
            code       TEXT    NOT NULL DEFAULT '',
            teacher    TEXT    NOT NULL DEFAULT '',
            is_active  INTEGER NOT NULL DEFAULT 1,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    print("  [OK]   subjects table ready")

    # ── 7. PRIORITY 3 — TIMETABLE TABLE ──────────────────────────────────────
    # day_of_week: 0=Monday … 6=Sunday  (Python datetime.weekday() convention)
    # start_time / end_time: stored as 'HH:MM' strings (same as late_cutoff)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timetable (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id   INTEGER NOT NULL,
            day_of_week  INTEGER NOT NULL CHECK(day_of_week BETWEEN 0 AND 6),
            start_time   TEXT    NOT NULL,
            end_time     TEXT    NOT NULL,
            is_active    INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        )
    """)
    print("  [OK]   timetable table ready")

    # Seed a default "General" subject so existing attendance records that
    # reference session='General' remain coherent.
    cursor.execute(
        "INSERT OR IGNORE INTO subjects (name, code, teacher) VALUES (?, ?, ?)",
        ("General", "GEN", "")
    )
    print("  [SET]  default subject 'General' seeded")

    conn.commit()
    conn.close()
    print("\n[MIGRATE] All done. Database is up to date.")


if __name__ == "__main__":
    migrate()