import sqlite3
import os

# Create database folder if not exists
os.makedirs("database", exist_ok=True)

# Connect to SQLite database
conn = sqlite3.connect("database/attendance.db")
cursor = conn.cursor()

# Create students table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
)
""")

# Create attendance table
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    date TEXT,
    time TEXT,
    FOREIGN KEY(student_id) REFERENCES students(id)
)
""")

conn.commit()
conn.close()

print("[INFO] Database and tables created successfully")
