import sqlite3
from datetime import datetime
import os

# ==============================
# ABSOLUTE PATH SETUP
# ==============================
# This finds the exact folder where db_utils.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# This ensures the database is always opened in the 'database' folder
DB_PATH = os.path.join(BASE_DIR, "attendance.db")

def mark_attendance(student_name):
    conn = None
    try:
        # Use the absolute path to connect
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1. Insert student if not exists
        cursor.execute(
            "INSERT OR IGNORE INTO students (name) VALUES (?)",
            (student_name,)
        )

        # 2. Get student ID
        cursor.execute(
            "SELECT id FROM students WHERE name = ?",
            (student_name,)
        )
        row = cursor.fetchone()
        
        if row is None:
            print(f"[ERROR] Could not find or create student: {student_name}")
            return
            
        student_id = row[0]
        today = datetime.now().strftime("%Y-%m-%d")

        # 3. Check if attendance already marked today
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
            print(f"[ATTENDANCE] Marked for {student_name} at {time_now}")
        else:
            print(f"[INFO] Attendance already marked for {student_name} today")

    except sqlite3.Error as e:
        print(f"[ERROR] Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Quick test
    mark_attendance("Test_User")