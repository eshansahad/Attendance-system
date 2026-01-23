import sqlite3
from datetime import datetime

DB_PATH = "database/attendance.db"

def mark_attendance(student_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Insert student if not exists
    cursor.execute(
        "INSERT OR IGNORE INTO students (name) VALUES (?)",
        (student_name,)
    )

    # Get student ID
    cursor.execute(
        "SELECT id FROM students WHERE name = ?",
        (student_name,)
    )
    student_id = cursor.fetchone()[0]

    today = datetime.now().strftime("%Y-%m-%d")

    # Check if attendance already marked today
    cursor.execute(
        "SELECT * FROM attendance WHERE student_id = ? AND date = ?",
        (student_id, today)
    )

    if cursor.fetchone() is None:
        time_now = datetime.now().strftime("%H:%M:%S")
        cursor.execute(
            "INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
            (student_id, today, time_now)
        )
        conn.commit()
        print(f"[ATTENDANCE] Marked for {student_name}")
    else:
        print(f"[INFO] Attendance already marked for {student_name}")

    conn.close()
