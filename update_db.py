import sqlite3
import os

DB_PATH = "database/attendance.db"

def upgrade_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("[INFO] Upgrading database...")

    # 1. Add 'password' column to students if it doesn't exist
    try:
        cursor.execute("ALTER TABLE students ADD COLUMN password TEXT DEFAULT '1234'")
        print("[SUCCESS] Added 'password' column to students (Default: 1234)")
    except sqlite3.OperationalError:
        print("[INFO] 'password' column already exists in students.")

    # 2. Create a 'teachers' table for Admin Login
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    
    # 3. Create a 'issues' table for Student Reports
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            type TEXT, -- 'Missing Attendance' or 'Tech Error'
            description TEXT,
            date TEXT,
            status TEXT DEFAULT 'Pending'
        )
    """)

    # 4. Add a default Teacher Account (admin / admin123)
    try:
        cursor.execute("INSERT INTO teachers (username, password) VALUES (?, ?)", ("admin", "admin123"))
        print("[SUCCESS] Created default Teacher account: admin / admin123")
    except sqlite3.IntegrityError:
        print("[INFO] Teacher 'admin' already exists.")

    conn.commit()
    conn.close()
    print("[INFO] Database upgrade complete.")

if __name__ == "__main__":
    upgrade_database()