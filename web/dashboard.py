"""
web/dashboard.py
Flask application for Smart Attend.

Changes from original:
  - Fixed: row[2] bug in teacher dashboard attendance query (now uses named cols)
  - Fixed: /clear_intruders route was missing despite being linked in template
  - Added: /override_attendance  – teacher can correct a status + log reason
  - Added: /export_report        – download filtered attendance as CSV
  - Added: /bulk_import          – import students from a CSV file
  - Added: /settings             – view/update late_cutoff & absence threshold
  - Updated: /student_dashboard  – now passes %, calendar, stats, threshold
  - Updated: /report             – now shows status column
"""

# ── PATH SETUP (must be first) ───────────────────────────────────────────────
import os
import sys

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── STANDARD IMPORTS ─────────────────────────────────────────────────────────
import csv
import io
import calendar
import shutil
from datetime import datetime, timedelta, date

from flask import (
    Flask, render_template, request, redirect,
    session, url_for, Response, send_from_directory,
    flash, make_response
)
import sqlite3
import cv2
import time

# ── PROJECT IMPORTS ──────────────────────────────────────────────────────────
from utils.alerts import send_absentee_report
from database.db_utils import (
    mark_attendance,
    get_setting,
    set_setting,
    get_attendance_percentage,
    override_attendance,
)
from core.recognize import process_frame_for_flask

# ── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "lana_final_master_v11"

# ── PATHS ────────────────────────────────────────────────────────────────────
DB_PATH       = os.path.join(PROJECT_ROOT, "database", "attendance.db")
DATASET_PATH  = os.path.join(PROJECT_ROOT, "data_files", "dataset")
INTRUDERS_DIR = os.path.join(PROJECT_ROOT, "data_files", "intruders")
os.makedirs(INTRUDERS_DIR, exist_ok=True)


# ── DB HELPER ────────────────────────────────────────────────────────────────
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  AUTH                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        role     = request.form.get("role", "")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()

        if role == "teacher":
            user = conn.execute(
                "SELECT * FROM teachers WHERE username=? AND password=?",
                (username, password)
            ).fetchone()
            if user:
                session["user_id"]   = user["id"]
                session["role"]      = "teacher"
                session["user_name"] = user["username"]
                conn.close()
                return redirect(url_for("index"))
            error = "Invalid teacher credentials."

        elif role == "student":
            user = conn.execute(
                "SELECT * FROM students WHERE name=? AND password=?",
                (username, password)
            ).fetchone()
            if user:
                session["user_id"]   = user["id"]
                session["role"]      = "student"
                session["user_name"] = user["name"]
                conn.close()
                return redirect(url_for("student_dashboard"))
            error = "Invalid student credentials."

        else:
            error = "Please select a role."

        conn.close()

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TEACHER DASHBOARD                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/")
def index():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn  = get_db_connection()
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Today's attendance (fixed: was row[2] bug) ──────────────────────────
    #    Now returns name, time, status so template can use row["name"] etc.
    attendance = conn.execute("""
        SELECT s.name, a.time, a.status, a.confidence, a.id AS record_id
        FROM   attendance a
        JOIN   students   s ON a.student_id = s.id
        WHERE  a.date = ?
        ORDER  BY a.time
    """, (today,)).fetchall()

    # ── Security alerts ──────────────────────────────────────────────────────
    intruders = [
        {"filename": f}
        for f in sorted(os.listdir(INTRUDERS_DIR), reverse=True)
        if f.lower().endswith((".jpg", ".png"))
    ]

    # ── 7-day trend data for Highcharts ─────────────────────────────────────
    chart_labels, chart_data = [], []
    total_students = conn.execute(
        "SELECT COUNT(*) FROM students"
    ).fetchone()[0] or 1   # avoid division-by-zero

    for i in range(6, -1, -1):
        day      = datetime.now() - timedelta(days=i)
        d_str    = day.strftime("%Y-%m-%d")
        label    = day.strftime("%d %b")
        present  = conn.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date=?",
            (d_str,)
        ).fetchone()[0]
        chart_labels.append(label)
        chart_data.append(round(present / total_students * 100, 2))

    # ── Low-attendance warnings for teacher ─────────────────────────────────
    threshold  = float(get_setting("absent_threshold", "75"))
    all_students = conn.execute("SELECT id, name FROM students").fetchall()
    at_risk = []
    for s in all_students:
        stats = get_attendance_percentage(s["id"])
        if stats["below_threshold"]:
            at_risk.append({"name": s["name"], "percentage": stats["percentage"]})

    conn.close()

    return render_template(
        "index.html",
        attendance=attendance,
        intruders=intruders,
        chart_labels=chart_labels,
        chart_data=chart_data,
        at_risk=at_risk,
        active_page="dashboard",
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CAMERA                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/start_camera")
def start_camera():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))
    return render_template("video_feed.html", active_page="camera")


def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Yield a simple error frame instead of crashing
        error_frame = 255 * __import__("numpy").ones((240, 640, 3), dtype="uint8")
        cv2.putText(
            error_frame, "Camera not found. Check connection.",
            (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2
        )
        _, buf = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buf.tobytes() + b"\r\n"
        )
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            frame = process_frame_for_flask(frame)
            ret, buf = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STUDENTS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/students")
def students_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    raw_students = conn.execute("SELECT * FROM students").fetchall()
    conn.close()

    # Attach attendance stats to each student
    students = []
    for s in raw_students:
        stats = get_attendance_percentage(s["id"])
        students.append({**dict(s), **stats})

    return render_template(
        "students.html", students=students, active_page="students"
    )


@app.route("/delete_student/<int:student_id>")
def delete_student(student_id):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    student = conn.execute(
        "SELECT name FROM students WHERE id=?", (student_id,)
    ).fetchone()

    if student:
        conn.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
        conn.execute("DELETE FROM students WHERE id=?",           (student_id,))
        conn.commit()
        folder = os.path.join(DATASET_PATH, student["name"])
        if os.path.exists(folder):
            shutil.rmtree(folder)

    conn.close()
    return redirect(url_for("students_page"))


# ── Bulk CSV import ──────────────────────────────────────────────────────────
@app.route("/bulk_import", methods=["GET", "POST"])
def bulk_import():
    """
    Import students from a CSV file.
    Expected CSV format (header row required):
        name,password
        Alice,1234
        Bob,5678
    Students that already exist (by name) are skipped.
    """
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    results = []

    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid .csv file.", "error")
            return redirect(url_for("bulk_import"))

        stream = io.StringIO(f.stream.read().decode("utf-8-sig"), newline=None)
        reader = csv.DictReader(stream)

        conn = get_db_connection()
        for row in reader:
            name     = (row.get("name") or "").strip()
            password = (row.get("password") or "1234").strip()
            if not name:
                continue
            existing = conn.execute(
                "SELECT id FROM students WHERE name=?", (name,)
            ).fetchone()
            if existing:
                results.append({"name": name, "status": "skipped (already exists)"})
            else:
                conn.execute(
                    "INSERT INTO students (name, password) VALUES (?, ?)",
                    (name, password)
                )
                folder = os.path.join(DATASET_PATH, name)
                os.makedirs(folder, exist_ok=True)
                results.append({"name": name, "status": "imported"})

        conn.commit()
        conn.close()

    return render_template(
        "bulk_import.html", results=results, active_page="students"
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REGISTER (single student + webcam capture)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/register", methods=["GET", "POST"])
def register_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    if request.method == "POST":
        name = (request.form.get("student_name") or "").strip()
        if not name:
            return redirect(url_for("register_page"))

        folder = os.path.join(DATASET_PATH, name)
        os.makedirs(folder, exist_ok=True)

        conn = get_db_connection()
        conn.execute(
            "INSERT OR IGNORE INTO students (name, password) VALUES (?, ?)",
            (name, "1234")
        )
        conn.commit()
        conn.close()

        return redirect(url_for("capture_images", student_name=name))

    return render_template("register_form.html", active_page="register")


@app.route("/capture_images/<student_name>")
def capture_images(student_name):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    folder = os.path.join(DATASET_PATH, student_name)
    cap    = cv2.VideoCapture(0)
    count  = 0

    while count < 15:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame, f"Capturing {count+1}/15",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )
        cv2.imwrite(os.path.join(folder, f"{student_name}_{count}.jpg"), frame)
        count += 1
        time.sleep(0.2)

    cap.release()
    os.system(
        f'"{sys.executable}" "{os.path.join(PROJECT_ROOT, "core", "extract_embeddings.py")}"'
    )
    return redirect(url_for("students_page"))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REPORT  (teacher view of today's full class)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/report")
def report_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn      = get_db_connection()
    # Allow filtering by date via ?date=YYYY-MM-DD
    report_date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))

    students = conn.execute("SELECT id, name FROM students").fetchall()
    present_rows = conn.execute(
        "SELECT student_id, time, status, confidence, id AS record_id,"
        "       override_reason"
        " FROM attendance WHERE date=?",
        (report_date,)
    ).fetchall()
    conn.close()

    present_map = {p["student_id"]: dict(p) for p in present_rows}

    report = []
    for s in students:
        if s["id"] in present_map:
            r = present_map[s["id"]]
            report.append({
                "name":            s["name"],
                "student_id":      s["id"],
                "record_id":       r["record_id"],
                "status":          r["status"],
                "time":            r["time"],
                "confidence":      r["confidence"],
                "override_reason": r["override_reason"] or "",
                "date":            report_date,
            })
        else:
            report.append({
                "name":            s["name"],
                "student_id":      s["id"],
                "record_id":       None,
                "status":          "Absent",
                "time":            "--:--",
                "confidence":      0.0,
                "override_reason": "",
                "date":            report_date,
            })

    return render_template(
        "report.html",
        attendance_logs=report,
        report_date=report_date,
        active_page="reports",
    )


# ── Override attendance ──────────────────────────────────────────────────────
@app.route("/override_attendance", methods=["POST"])
def override_attendance_route():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    record_id  = request.form.get("record_id",  type=int)
    new_status = request.form.get("new_status", "").strip()
    reason     = request.form.get("reason",     "").strip()
    back_date  = request.form.get("date",       datetime.now().strftime("%Y-%m-%d"))

    if record_id and new_status:
        override_attendance(record_id, new_status, reason)

    return redirect(url_for("report_page", date=back_date))


# ── Export CSV ───────────────────────────────────────────────────────────────
@app.route("/export_report")
def export_report():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    report_date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT s.name, a.date, a.time, a.status, a.confidence, a.session,
               a.override_reason
        FROM   attendance a
        JOIN   students   s ON a.student_id = s.id
        WHERE  a.date = ?
        ORDER  BY s.name
    """, (report_date,)).fetchall()
    conn.close()

    si  = io.StringIO()
    cw  = csv.writer(si)
    cw.writerow(["Name", "Date", "Time", "Status", "Confidence (%)",
                 "Session", "Override Reason"])
    for r in rows:
        cw.writerow([
            r["name"], r["date"], r["time"], r["status"],
            f"{r['confidence']:.1f}", r["session"],
            r["override_reason"] or ""
        ])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = (
        f'attachment; filename="attendance_{report_date}.csv"'
    )
    output.headers["Content-type"] = "text/csv"
    return output


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SETTINGS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    if request.method == "POST":
        late_cutoff = request.form.get("late_cutoff", "09:00").strip()
        threshold   = request.form.get("absent_threshold", "75").strip()
        session_name = request.form.get("session_name", "General").strip()

        # Basic validation
        try:
            datetime.strptime(late_cutoff, "%H:%M")
        except ValueError:
            late_cutoff = "09:00"
        try:
            t = float(threshold)
            if not (0 <= t <= 100):
                threshold = "75"
        except ValueError:
            threshold = "75"

        set_setting("late_cutoff",      late_cutoff)
        set_setting("absent_threshold", threshold)
        set_setting("session_name",     session_name)

        return redirect(url_for("settings_page"))

    current = {
        "late_cutoff":      get_setting("late_cutoff",      "09:00"),
        "absent_threshold": get_setting("absent_threshold", "75"),
        "session_name":     get_setting("session_name",     "General"),
    }
    return render_template(
        "settings.html", current=current, active_page="settings"
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STUDENT DASHBOARD                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _build_calendar(student_id: int, year: int, month: int) -> list:
    """
    Build a week-by-week calendar for one student for the given month.
    Each day cell:
      None              → padding day (before month starts)
      dict with keys:
        day    (int)    → day number
        status (str)    → 'present' | 'late' | 'absent' | 'no-class' | 'future'
        date   (str)    → YYYY-MM-DD
    """
    conn = get_db_connection()

    # School days this month = any day with attendance recorded for anyone
    school_days_rows = conn.execute(
        "SELECT DISTINCT date FROM attendance WHERE date LIKE ?",
        (f"{year}-{month:02d}-%",)
    ).fetchall()
    school_days = {r["date"] for r in school_days_rows}

    # This student's own records
    my_rows = conn.execute(
        "SELECT date, status FROM attendance WHERE student_id=? AND date LIKE ?",
        (student_id, f"{year}-{month:02d}-%")
    ).fetchall()
    my_map = {r["date"]: r["status"].lower() for r in my_rows}
    conn.close()

    today = date.today()
    weeks = []

    for week in calendar.monthcalendar(year, month):
        row = []
        for day in week:
            if day == 0:
                row.append(None)
                continue
            d      = date(year, month, day)
            d_str  = d.strftime("%Y-%m-%d")

            if d > today:
                status = "future"
            elif d_str in my_map:
                status = my_map[d_str]       # 'present' or 'late'
            elif d_str in school_days:
                status = "absent"
            else:
                status = "no-class"

            row.append({"day": day, "status": status, "date": d_str})
        weeks.append(row)

    return weeks


@app.route("/student_dashboard")
def student_dashboard():
    if session.get("role") != "student":
        return redirect(url_for("login"))

    student_id = session["user_id"]
    now        = datetime.now()

    # ── Month/year from query string (defaults to current month) ─────────────
    try:
        req_year  = int(request.args.get("year",  now.year))
        req_month = int(request.args.get("month", now.month))
        req_month = max(1, min(12, req_month))
        req_year  = max(2020, min(now.year + 1, req_year))
    except (ValueError, TypeError):
        req_year, req_month = now.year, now.month

    # ── Prev / next month for nav arrows ─────────────────────────────────────
    prev_dt    = date(req_year, req_month, 1) - timedelta(days=1)
    next_month = req_month % 12 + 1
    next_year  = req_year + (1 if req_month == 12 else 0)

    # ── Stats (always all-time) ───────────────────────────────────────────────
    stats = get_attendance_percentage(student_id)

    # ── Calendar for selected month ───────────────────────────────────────────
    cal_weeks  = _build_calendar(student_id, req_year, req_month)
    month_year = f"{calendar.month_name[req_month]} {req_year}"

    # ── Recent records (last 10) ──────────────────────────────────────────────
    conn = get_db_connection()
    recent = conn.execute(
        "SELECT date, time, status, confidence, session"
        " FROM attendance WHERE student_id=?"
        " ORDER BY date DESC, time DESC LIMIT 10",
        (student_id,)
    ).fetchall()
    conn.close()

    return render_template(
        "student_dashboard.html",
        student_name  = session["user_name"],
        stats         = stats,
        cal_weeks     = cal_weeks,
        month_year    = month_year,
        current_month = req_month,
        current_year  = req_year,
        prev_month    = prev_dt.month,
        prev_year     = prev_dt.year,
        next_month    = next_month,
        next_year     = next_year,
        month_names   = list(calendar.month_name)[1:],
        year_range    = list(range(2020, now.year + 2)),
        recent        = recent,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INTRUDERS / SECURITY                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/intruders/<filename>")
def serve_intruder(filename):
    return send_from_directory(INTRUDERS_DIR, filename)


@app.route("/clear_intruders")
def clear_intruders():
    """Delete all saved intruder images (was missing from original codebase)."""
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    for f in os.listdir(INTRUDERS_DIR):
        if f.lower().endswith((".jpg", ".png")):
            os.remove(os.path.join(INTRUDERS_DIR, f))

    return redirect(url_for("index"))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ALERTS                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/send_absentees")
def send_absentees():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))
    send_absentee_report()
    return redirect(url_for("report_page"))



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COMPLAINTS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Student: submit a new complaint ─────────────────────────────────────────
@app.route("/complaint/new", methods=["GET", "POST"])
def new_complaint():
    if session.get("role") != "student":
        return redirect(url_for("login"))

    student_id = session["user_id"]

    if request.method == "POST":
        date_of_absence = request.form.get("date", "").strip()
        reason          = request.form.get("reason", "").strip()
        description     = request.form.get("description", "").strip()

        if not date_of_absence or not reason or not description:
            flash("All fields are required.", "error")
            return redirect(url_for("new_complaint"))

        # Check there isn't already a pending complaint for this date
        conn = get_db_connection()
        existing = conn.execute(
            "SELECT id FROM complaints"
            " WHERE student_id=? AND date=? AND status='Pending'",
            (student_id, date_of_absence)
        ).fetchone()

        if existing:
            conn.close()
            flash("You already have a pending complaint for that date.", "error")
            return redirect(url_for("my_complaints"))

        conn.execute(
            """INSERT INTO complaints
                   (student_id, date, reason, description, status, created_at)
               VALUES (?, ?, ?, ?, 'Pending', ?)""",
            (student_id, date_of_absence, reason,
             description, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()

        flash("Complaint submitted successfully. Waiting for teacher review.", "success")
        return redirect(url_for("my_complaints"))

    # GET — pre-fill date from query string if coming from calendar
    prefill_date = request.args.get("date", "")
    return render_template(
        "complaint_form.html",
        prefill_date = prefill_date,
        active_page  = "complaints",
    )


# ── Student: view their own complaints ───────────────────────────────────────
@app.route("/complaint/my")
def my_complaints():
    if session.get("role") != "student":
        return redirect(url_for("login"))

    conn = get_db_connection()
    complaints = conn.execute(
        """SELECT id, date, reason, description, status,
                  teacher_note, created_at, resolved_at
           FROM   complaints
           WHERE  student_id = ?
           ORDER  BY created_at DESC""",
        (session["user_id"],)
    ).fetchall()
    conn.close()

    return render_template(
        "my_complaints.html",
        complaints  = complaints,
        active_page = "my_complaints",
    )


# ── Teacher: view all complaints ─────────────────────────────────────────────
@app.route("/complaints")
def complaints_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    status_filter = request.args.get("status", "Pending")

    conn = get_db_connection()

    if status_filter == "All":
        rows = conn.execute(
            """SELECT c.id, s.name AS student_name, c.date, c.reason,
                      c.description, c.status, c.teacher_note,
                      c.created_at, c.resolved_at
               FROM   complaints c
               JOIN   students   s ON c.student_id = s.id
               ORDER  BY c.created_at DESC"""
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT c.id, s.name AS student_name, c.date, c.reason,
                      c.description, c.status, c.teacher_note,
                      c.created_at, c.resolved_at
               FROM   complaints c
               JOIN   students   s ON c.student_id = s.id
               WHERE  c.status = ?
               ORDER  BY c.created_at DESC""",
            (status_filter,)
        ).fetchall()

    pending_count = conn.execute(
        "SELECT COUNT(*) FROM complaints WHERE status='Pending'"
    ).fetchone()[0]

    conn.close()

    return render_template(
        "complaints.html",
        complaints     = rows,
        status_filter  = status_filter,
        pending_count  = pending_count,
        active_page    = "complaints",
    )


# ── Teacher: accept or decline a complaint ───────────────────────────────────
@app.route("/complaint/resolve/<int:complaint_id>", methods=["POST"])
def resolve_complaint(complaint_id):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    action       = request.form.get("action", "")       # "accept" or "decline"
    teacher_note = request.form.get("teacher_note", "").strip()
    back_filter  = request.form.get("back_filter", "Pending")

    if action not in ("accept", "decline"):
        return redirect(url_for("complaints_page"))

    new_status  = "Accepted" if action == "accept" else "Declined"
    resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()

    # Fetch complaint so we can update attendance if accepted
    complaint = conn.execute(
        "SELECT student_id, date FROM complaints WHERE id=?",
        (complaint_id,)
    ).fetchone()

    if complaint:
        conn.execute(
            """UPDATE complaints
               SET status=?, teacher_note=?, resolved_at=?
               WHERE id=?""",
            (new_status, teacher_note, resolved_at, complaint_id)
        )

        # If accepted → mark the student present for that date
        # (only insert if not already present)
        if action == "accept":
            student_id = complaint["student_id"]
            abs_date   = complaint["date"]
            session_name = get_setting("session_name", "General")

            already = conn.execute(
                "SELECT id FROM attendance"
                " WHERE student_id=? AND date=? AND session=?",
                (student_id, abs_date, session_name)
            ).fetchone()

            if not already:
                conn.execute(
                    """INSERT INTO attendance
                           (student_id, date, time, status, confidence,
                            session, override_reason)
                       VALUES (?, ?, ?, 'Present', 0.0, ?, 'Complaint accepted')""",
                    (student_id, abs_date, "00:00:00", session_name)
                )

        conn.commit()

    conn.close()
    return redirect(url_for("complaints_page", status=back_filter))

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RUN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    app.run(debug=True, port=5000)