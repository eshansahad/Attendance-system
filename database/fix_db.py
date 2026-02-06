import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")

def rebuild_db():
    # Remove existing if it somehow still exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("[INFO] Deleted old database.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("[PROCESS] Creating tables...")
    
    # 1. Teachers Table
    cursor.execute('''CREATE TABLE teachers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')

    # 2. Students Table
    cursor.execute('''CREATE TABLE students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL DEFAULT '1234'
    )''')

    # 3. Attendance Table
    cursor.execute('''CREATE TABLE attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(id)
    )''')

    # 4. Issues Table
    cursor.execute('''CREATE TABLE issues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        type TEXT,
        description TEXT,
        date TEXT,
        status TEXT DEFAULT 'Pending',
        FOREIGN KEY (student_id) REFERENCES students(id)
    )''')

    # 5. Insert Default Teacher
    cursor.execute("INSERT INTO teachers (username, password) VALUES ('admin', 'admin123')")

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Database created at {DB_PATH}")
    print("[INFO] Default login -> User: admin | Pass: admin123")

if __name__ == "__main__":
    rebuild_db()