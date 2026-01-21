from flask import Flask, render_template, request, send_file
import sqlite3
import csv
from datetime import datetime

app = Flask(__name__)
DB_PATH = "database/attendance.db"

def get_attendance(date=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if date:
        cursor.execute("""
        SELECT students.name, attendance.date, attendance.time
        FROM attendance
        JOIN students ON attendance.student_id = students.id
        WHERE attendance.date = ?
        """, (date,))
    else:
        cursor.execute("""
        SELECT students.name, attendance.date, attendance.time
        FROM attendance
        JOIN students ON attendance.student_id = students.id
        """)

    data = cursor.fetchall()
    conn.close()
    return data

@app.route("/")
def index():
    selected_date = request.args.get("date")
    attendance = get_attendance(selected_date)
    return render_template(
        "index.html",
        attendance=attendance,
        selected_date=selected_date
    )

@app.route("/export")
def export_csv():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT students.name, attendance.date, attendance.time
    FROM attendance
    JOIN students ON attendance.student_id = students.id
    """)

    rows = cursor.fetchall()
    conn.close()

    filename = f"attendance_{datetime.now().date()}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student Name", "Date", "Time"])
        writer.writerows(rows)

    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
