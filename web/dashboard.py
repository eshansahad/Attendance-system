# ==============================
# PATH SETUP (MUST BE FIRST)
# ==============================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# ==============================
# STANDARD IMPORTS
# ==============================
from flask import (
    Flask, render_template, request, redirect,
    session, url_for, Response, send_from_directory
)
import sqlite3
import cv2
import time
from datetime import datetime, timedelta

# ==============================
# PROJECT IMPORTS
# ==============================
from utils.alerts import send_absentee_report
from database.db_utils import mark_attendance
from core.recognize import process_frame_for_flask   # 🔥 IMPORTANT

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "lana_final_master_v11"

# ==============================
# PATHS
# ==============================
DB_PATH = os.path.join(PROJECT_ROOT, "database", "attendance.db")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data_files", "dataset")
INTRUDERS_DIR = os.path.join(PROJECT_ROOT, "data_files", "intruders")

os.makedirs(INTRUDERS_DIR, exist_ok=True)

# ==============================
# DB UTILS
# ==============================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ==============================
# AUTH
# ==============================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form.get("role")
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()

        if role == "teacher":
            user = conn.execute(
                "SELECT * FROM teachers WHERE username=? AND password=?",
                (username, password)
            ).fetchone()
            if user:
                session["user_id"] = user["id"]
                session["role"] = "teacher"
                conn.close()
                return redirect(url_for("index"))

        elif role == "student":
            user = conn.execute(
                "SELECT * FROM students WHERE name=? AND password=?",
                (username, password)
            ).fetchone()
            if user:
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                session["role"] = "student"
                conn.close()
                return redirect(url_for("student_dashboard"))

        conn.close()

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==============================
# TEACHER DASHBOARD
# ==============================
@app.route("/")
def index():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    today = datetime.now().strftime("%Y-%m-%d")

    attendance = conn.execute("""
        SELECT s.name, a.time
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
    """, (today,)).fetchall()

    intruders = [
        {"filename": f}
        for f in sorted(os.listdir(INTRUDERS_DIR), reverse=True)
        if f.endswith((".jpg", ".png"))
    ]

    chart_labels, chart_data = [], []

    for i in range(6, -1, -1):
        day = datetime.now() - timedelta(days=i)
        date = day.strftime("%Y-%m-%d")
        label = day.strftime("%d %b")

        total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        present = conn.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date=?",
            (date,)
        ).fetchone()[0]

        chart_labels.append(label)
        chart_data.append(round((present / total) * 100, 2) if total else 0)

    conn.close()

    return render_template(
        "index.html",
        attendance=attendance,
        intruders=intruders,
        chart_labels=chart_labels,
        chart_data=chart_data
    )

# ==============================
# CAMERA STREAM (🔥 FIXED)
# ==============================
@app.route("/start_camera")
def start_camera():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))
    return render_template("video_feed.html")


def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # 🔥 REAL AI PROCESSING (bounding box, name, blink, alerts)
        frame = process_frame_for_flask(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ==============================
# STUDENTS
# ==============================
@app.route("/students")
def students_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    students = conn.execute("SELECT * FROM students").fetchall()
    conn.close()

    return render_template("students.html", students=students)


@app.route("/delete_student/<int:student_id>")
def delete_student(student_id):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    student = conn.execute(
        "SELECT name FROM students WHERE id=?",
        (student_id,)
    ).fetchone()

    if student:
        conn.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
        conn.execute("DELETE FROM students WHERE id=?", (student_id,))
        conn.commit()

        folder = os.path.join(DATASET_PATH, student["name"])
        if os.path.exists(folder):
            import shutil
            shutil.rmtree(folder)

    conn.close()
    return redirect(url_for("students_page"))

# ==============================
# REGISTER + IMAGE CAPTURE (🔥 FIXED)
# ==============================
@app.route("/register", methods=["GET", "POST"])
def register_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("student_name")
        if not name:
            return redirect(url_for("register_page"))

        folder = os.path.join(DATASET_PATH, name)
        os.makedirs(folder, exist_ok=True)

        conn = get_db_connection()
        conn.execute(
            "INSERT INTO students (name, password) VALUES (?, ?)",
            (name, "1234")
        )
        conn.commit()
        conn.close()

        return redirect(url_for("capture_images", student_name=name))

    return render_template("register_form.html")


@app.route("/capture_images/<student_name>")
def capture_images(student_name):
    folder = os.path.join(DATASET_PATH, student_name)
    cap = cv2.VideoCapture(0)

    count = 0
    while count < 15:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            f"Capturing image {count+1}/15",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imwrite(
            os.path.join(folder, f"{student_name}_{count}.jpg"),
            frame
        )

        count += 1
        time.sleep(0.2)

    cap.release()

    os.system(
        f"python {os.path.join(PROJECT_ROOT, 'core', 'extract_embeddings.py')}"
    )

    return redirect(url_for("students_page"))

# ==============================
# REPORT
# ==============================
@app.route("/report")
def report_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    today = datetime.now().strftime("%Y-%m-%d")

    students = conn.execute("SELECT id, name FROM students").fetchall()
    present = conn.execute(
        "SELECT student_id, time FROM attendance WHERE date=?",
        (today,)
    ).fetchall()

    present_map = {p["student_id"]: p["time"] for p in present}

    report = [{
        "name": s["name"],
        "status": "Present" if s["id"] in present_map else "Absent",
        "time": present_map.get(s["id"], "--:--"),
        "date": today
    } for s in students]

    conn.close()
    return render_template("report.html", attendance_logs=report)

# ==============================
# STUDENT DASHBOARD
# ==============================
@app.route("/student_dashboard")
def student_dashboard():
    if session.get("role") != "student":
        return redirect(url_for("login"))

    conn = get_db_connection()
    attendance = conn.execute(
        "SELECT * FROM attendance WHERE student_id=?",
        (session["user_id"],)
    ).fetchall()
    conn.close()

    return render_template(
        "student_dashboard.html",
        student_name=session["user_name"],
        my_attendance=attendance
    )

# ==============================
# SEND ABSENTEES
# ==============================
@app.route("/send_absentees")
def send_absentees():
    send_absentee_report()
    return redirect(url_for("report_page"))

# ==============================
# SERVE INTRUDER IMAGES
# ==============================
@app.route("/intruders/<filename>")
def serve_intruder(filename):
    return send_from_directory(INTRUDERS_DIR, filename)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
