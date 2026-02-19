"""
Run this ONCE to upgrade your database schema.
It is safe to run multiple times - it skips columns that already exist.

Usage:
    python database/migrate_db.py
"""
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found at: {DB_PATH}")
        print("[FIX] Run database/database_setup.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── 1. NEW COLUMNS ON attendance TABLE ──────────────────────────────────
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

    # Backfill: mark all existing rows as 'Present' where status is empty/null
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
        ("late_cutoff",         "09:00"),   # HH:MM  – mark 'Late' after this time
        ("absent_threshold",    "75"),       # %      – warn student below this
        ("session_name",        "General"), # default session label
    ]
    for key, value in defaults:
        cursor.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )
        print(f"  [SET]  {key} = {value} (default, won't overwrite existing)")

    # ── 3. UNIQUE CONSTRAINT GUARD (soft – via index) ─────────────────────
    # Prevents duplicate attendance for same student+date+session at DB level
    try:
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            uq_attendance_student_date_session
            ON attendance(student_id, date, session)
        """)
        print("  [OK]   unique index on (student_id, date, session)")
    except sqlite3.OperationalError as e:
        print(f"  [WARN] Could not create unique index: {e}")

    # ── 4. COMPLAINTS TABLE ─────────────────────────────────────────────────
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
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    print("  [OK]   complaints table ready")

    conn.commit()
    conn.close()
    print("\n[MIGRATE] All done. Your database is up to date.")


if __name__ == "__main__":
    migrate()