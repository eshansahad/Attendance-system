"""
web/dashboard.py  — Smart Attend
Includes all Priority 3 timetable fixes:
  - get_timetable / get_today_schedule passed to both dashboard routes
  - today_dow and active_session_now passed to templates
  - subject_stats passed to student dashboard
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
    get_timetable,
    get_today_schedule,
    get_attendance_by_subject,
)
from core.recognize import process_frame_for_flask
from core.smart_register import SmartRegistrar

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


# ── SHARED: active session helper (used by index + timetable routes) ─────────
def _get_active_session(conn):
    """
    Returns the subject name that is currently scheduled right now,
    or falls back to the Settings session_name.
    """
    now  = datetime.now()
    dow  = now.weekday()
    hhmm = now.strftime("%H:%M")

    row = conn.execute("""
        SELECT s.name
        FROM   timetable t
        JOIN   subjects  s ON t.subject_id = s.id
        WHERE  t.day_of_week = ?
          AND  t.start_time  <= ?
          AND  t.end_time    >  ?
          AND  t.is_active   = 1
          AND  s.is_active   = 1
        LIMIT 1
    """, (dow, hhmm, hhmm)).fetchone()

    if row:
        return row["name"]

    setting = conn.execute(
        "SELECT value FROM settings WHERE key='session_name'"
    ).fetchone()
    return setting["value"] if setting else "General"


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
                if user["is_active"] == 0:
                    error = "Your account has been deactivated. Please contact your teacher."
                else:
                    session["user_id"]   = user["id"]
                    session["role"]      = "student"
                    session["user_name"] = user["name"]
                    conn.close()
                    return redirect(url_for("student_dashboard"))
            else:
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

    attendance = conn.execute("""
        SELECT s.name, a.time, a.status, a.confidence, a.id AS record_id
        FROM   attendance a
        JOIN   students   s ON a.student_id = s.id
        WHERE  a.date = ?
        ORDER  BY a.time
    """, (today,)).fetchall()

    intruders = [
        {"filename": f}
        for f in sorted(os.listdir(INTRUDERS_DIR), reverse=True)
        if f.lower().endswith((".jpg", ".png"))
    ]

    chart_labels, chart_data = [], []
    total_students = conn.execute(
        "SELECT COUNT(*) FROM students"
    ).fetchone()[0] or 1

    for i in range(6, -1, -1):
        day     = datetime.now() - timedelta(days=i)
        d_str   = day.strftime("%Y-%m-%d")
        label   = day.strftime("%d %b")
        present = conn.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date=?",
            (d_str,)
        ).fetchone()[0]
        chart_labels.append(label)
        chart_data.append(round(present / total_students * 100, 2))

    threshold    = float(get_setting("absent_threshold", "75"))
    all_students = conn.execute("SELECT id, name FROM students").fetchall()
    at_risk = []
    for s in all_students:
        stats = get_attendance_percentage(s["id"])
        if stats["below_threshold"]:
            at_risk.append({"name": s["name"], "percentage": stats["percentage"]})

    # ── Timetable data for weekly grid ────────────────────────────────────
    _now            = datetime.now()
    active_session  = _get_active_session(conn)
    conn.close()

    return render_template(
        "index.html",
        attendance        = attendance,
        intruders         = intruders,
        chart_labels      = chart_labels,
        chart_data        = chart_data,
        at_risk           = at_risk,
        timetable         = get_timetable(),
        today_schedule    = get_today_schedule(),
        active_session_now= active_session,
        today_dow         = _now.weekday(),
        active_page       = "dashboard",
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

    active_students   = []
    archived_students = []
    for s in raw_students:
        stats = get_attendance_percentage(s["id"])
        row   = {**dict(s), **stats}
        if s["is_active"] == 1:
            active_students.append(row)
        else:
            archived_students.append(row)

    return render_template(
        "students.html",
        students          = active_students,
        archived_students = archived_students,
        active_page       = "students",
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


@app.route("/bulk_import", methods=["GET", "POST"])
def bulk_import():
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
# ║  REGISTER                                                               ║
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


_reg_sessions: dict = {}

@app.route("/capture_images/<student_name>")
def capture_images(student_name):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))
    _reg_sessions[student_name] = SmartRegistrar(student_name, DATASET_PATH)
    return render_template(
        "video_register.html",
        student_name=student_name,
        active_page="register",
    )


def _gen_register_frames(student_name: str):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    reg = _reg_sessions.get(student_name)
    if reg is None:
        cap.release()
        return
    while not reg.done:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = reg.process_frame(frame)
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )
    cap.release()
    if reg and reg.done:
        import subprocess
        subprocess.Popen(
            [sys.executable,
             os.path.join(PROJECT_ROOT, "core", "extract_embeddings.py")]
        )


@app.route("/register_feed/<student_name>")
def register_feed(student_name):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))
    return Response(
        _gen_register_frames(student_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/register_status/<student_name>")
def register_status(student_name):
    if session.get("role") != "teacher":
        return {"error": "unauthorized"}, 401
    reg = _reg_sessions.get(student_name)
    if reg is None:
        return {"error": "session not found"}, 404
    return reg.status()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REPORT                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/report")
def report_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn           = get_db_connection()
    today          = datetime.now().strftime("%Y-%m-%d")
    report_date    = request.args.get("date",           today)
    session_filter = request.args.get("session_filter", "").strip()
    status_filter  = request.args.get("status_filter",  "").strip()

    # ── All sessions ever recorded (for the filter dropdown) ──────────────
    all_sessions_rows = conn.execute(
        "SELECT DISTINCT session FROM attendance WHERE session IS NOT NULL"
        " ORDER BY session"
    ).fetchall()
    all_sessions = [r["session"] for r in all_sessions_rows if r["session"]]

    # ── Students + attendance for the chosen date ─────────────────────────
    students = conn.execute("SELECT id, name FROM students").fetchall()

    query  = ("SELECT student_id, time, status, confidence, session,"
              " id AS record_id, override_reason"
              " FROM attendance WHERE date=?")
    params = [report_date]

    if session_filter:
        query  += " AND session=?"
        params.append(session_filter)

    present_rows = conn.execute(query, params).fetchall()
    conn.close()

    present_map = {p["student_id"]: dict(p) for p in present_rows}

    report = []
    for s in students:
        if s["id"] in present_map:
            r = present_map[s["id"]]
            row = {
                "name":            s["name"],
                "student_id":      s["id"],
                "record_id":       r["record_id"],
                "status":          r["status"],
                "time":            r["time"],
                "confidence":      r["confidence"],
                "session":         r["session"],
                "override_reason": r["override_reason"] or "",
                "date":            report_date,
            }
        else:
            row = {
                "name":            s["name"],
                "student_id":      s["id"],
                "record_id":       None,
                "status":          "Absent",
                "time":            "--:--",
                "confidence":      0.0,
                "session":         session_filter or "—",
                "override_reason": "",
                "date":            report_date,
            }

        # Client-side status filter (still include all so JS can filter too)
        if not status_filter or row["status"] == status_filter:
            report.append(row)

    return render_template(
        "report.html",
        attendance_logs = report,
        report_date     = report_date,
        today           = today,
        all_sessions    = all_sessions,
        session_filter  = session_filter,
        status_filter   = status_filter,
        active_page     = "reports",
    )


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
        late_cutoff  = request.form.get("late_cutoff",      "09:00").strip()
        threshold    = request.form.get("absent_threshold",  "75").strip()
        session_name = request.form.get("session_name",      "General").strip()

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
    conn = get_db_connection()

    school_days_rows = conn.execute(
        "SELECT DISTINCT date FROM attendance WHERE date LIKE ?",
        (f"{year}-{month:02d}-%",)
    ).fetchall()
    school_days = {r["date"] for r in school_days_rows}

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
            d     = date(year, month, day)
            d_str = d.strftime("%Y-%m-%d")

            if d > today:
                status = "future"
            elif d_str in my_map:
                status = my_map[d_str]
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

    try:
        req_year  = int(request.args.get("year",  now.year))
        req_month = int(request.args.get("month", now.month))
        req_month = max(1, min(12, req_month))
        req_year  = max(2020, min(now.year + 1, req_year))
    except (ValueError, TypeError):
        req_year, req_month = now.year, now.month

    prev_dt    = date(req_year, req_month, 1) - timedelta(days=1)
    next_month = req_month % 12 + 1
    next_year  = req_year + (1 if req_month == 12 else 0)

    stats     = get_attendance_percentage(student_id)
    cal_weeks = _build_calendar(student_id, req_year, req_month)
    month_year = f"{calendar.month_name[req_month]} {req_year}"

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
        student_name   = session["user_name"],
        stats          = stats,
        cal_weeks      = cal_weeks,
        month_year     = month_year,
        current_month  = req_month,
        current_year   = req_year,
        prev_month     = prev_dt.month,
        prev_year      = prev_dt.year,
        next_month     = next_month,
        next_year      = next_year,
        month_names    = list(calendar.month_name)[1:],
        year_range     = list(range(2020, now.year + 2)),
        recent         = recent,
        timetable      = get_timetable(),
        today_schedule = get_today_schedule(),
        today_dow      = now.weekday(),
        subject_stats  = get_attendance_by_subject(student_id),
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TIMETABLE & SUBJECTS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/timetable", methods=["GET"])
def timetable_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    now  = datetime.now()
    dow  = now.weekday()
    hhmm = now.strftime("%H:%M")

    subjects_raw = conn.execute(
        "SELECT * FROM subjects ORDER BY name"
    ).fetchall()

    timetable_raw = conn.execute("""
        SELECT t.*, s.name AS subject_name, s.code AS subject_code
        FROM   timetable t
        JOIN   subjects  s ON t.subject_id = s.id
        WHERE  t.is_active = 1
        ORDER  BY t.day_of_week, t.start_time
    """).fetchall()

    today_schedule = []
    active_period_ids = set()
    for p in timetable_raw:
        if p["day_of_week"] != dow:
            continue
        if hhmm >= p["start_time"] and hhmm < p["end_time"]:
            status = "active"
            active_period_ids.add(p["id"])
        elif hhmm < p["start_time"]:
            status = "upcoming"
        else:
            status = "ended"
        today_schedule.append({**dict(p), "status": status})

    active_session = _get_active_session(conn)
    conn.close()

    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    return render_template(
        "timetable.html",
        subjects          = [dict(s) for s in subjects_raw],
        timetable         = [dict(t) for t in timetable_raw],
        today_schedule    = today_schedule,
        active_period_ids = active_period_ids,
        active_session    = active_session,
        today_name        = day_names[dow],
        today_dow         = dow,
        active_page       = "timetable",
    )


@app.route("/timetable/add_subject", methods=["POST"])
def timetable_add_subject():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    name    = request.form.get("name",    "").strip()
    code    = request.form.get("code",    "").strip()
    teacher = request.form.get("teacher", "").strip()

    if not name:
        return redirect(url_for("timetable_page"))

    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO subjects (name, code, teacher) VALUES (?, ?, ?)",
            (name, code, teacher)
        )
        conn.commit()
    except Exception:
        pass
    conn.close()
    return redirect(url_for("timetable_page"))


@app.route("/timetable/delete_subject/<int:subject_id>")
def timetable_delete_subject(subject_id):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute("UPDATE subjects  SET is_active=0 WHERE id=?", (subject_id,))
    conn.execute("UPDATE timetable SET is_active=0 WHERE subject_id=?", (subject_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("timetable_page"))


@app.route("/timetable/add_period", methods=["POST"])
def timetable_add_period():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    subject_id  = request.form.get("subject_id",  type=int)
    day_of_week = request.form.get("day_of_week", type=int)
    start_time  = request.form.get("start_time",  "").strip()
    end_time    = request.form.get("end_time",    "").strip()

    if not all([subject_id, day_of_week is not None, start_time, end_time]):
        return redirect(url_for("timetable_page"))
    if start_time >= end_time:
        return redirect(url_for("timetable_page"))

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO timetable (subject_id, day_of_week, start_time, end_time)
        VALUES (?, ?, ?, ?)
    """, (subject_id, day_of_week, start_time, end_time))
    conn.commit()
    conn.close()
    return redirect(url_for("timetable_page"))


@app.route("/timetable/delete/<int:period_id>", methods=["POST"])
def timetable_delete_period(period_id):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute("UPDATE timetable SET is_active=0 WHERE id=?", (period_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("timetable_page"))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COMPLAINT NOTIFICATION BADGE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/complaint/badge")
def complaint_badge():
    if session.get("role") != "student":
        return {"count": 0}

    conn = get_db_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM complaints"
        " WHERE student_id=? AND status != 'Pending' AND student_seen=0",
        (session["user_id"],)
    ).fetchone()[0]
    conn.close()
    return {"count": count}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INTRUDERS / SECURITY                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/intruders/<filename>")
def serve_intruder(filename):
    return send_from_directory(INTRUDERS_DIR, filename)


@app.route("/clear_intruders")
def clear_intruders():
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
# ║  STUDENT ARCHIVE / REACTIVATE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/archive_student/<int:student_id>")
def archive_student(student_id):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute("UPDATE students SET is_active=0 WHERE id=?", (student_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("students_page"))


@app.route("/reactivate_student/<int:student_id>")
def reactivate_student(student_id):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute("UPDATE students SET is_active=1 WHERE id=?", (student_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("students_page"))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COMPLAINTS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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

    prefill_date = request.args.get("date", "")
    return render_template(
        "complaint_form.html",
        prefill_date = prefill_date,
        active_page  = "complaints",
    )


@app.route("/complaint/my")
def my_complaints():
    if session.get("role") != "student":
        return redirect(url_for("login"))

    conn = get_db_connection()
    complaints = conn.execute(
        """SELECT id, date, reason, description, status,
                  teacher_note, created_at, resolved_at, student_seen
           FROM   complaints
           WHERE  student_id = ?
           ORDER  BY created_at DESC""",
        (session["user_id"],)
    ).fetchall()

    conn.execute(
        "UPDATE complaints SET student_seen=1"
        " WHERE student_id=? AND status != 'Pending' AND student_seen=0",
        (session["user_id"],)
    )
    conn.commit()
    conn.close()

    return render_template(
        "my_complaints.html",
        complaints  = complaints,
        active_page = "my_complaints",
    )


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
        complaints    = rows,
        status_filter = status_filter,
        pending_count = pending_count,
        active_page   = "complaints",
    )


@app.route("/complaint/resolve/<int:complaint_id>", methods=["POST"])
def resolve_complaint(complaint_id):
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    action       = request.form.get("action", "")
    teacher_note = request.form.get("teacher_note", "").strip()
    back_filter  = request.form.get("back_filter", "Pending")

    if action not in ("accept", "decline"):
        return redirect(url_for("complaints_page"))

    new_status  = "Accepted" if action == "accept" else "Declined"
    resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()

    complaint = conn.execute(
        "SELECT student_id, date FROM complaints WHERE id=?",
        (complaint_id,)
    ).fetchone()

    if complaint:
        conn.execute(
            """UPDATE complaints
               SET status=?, teacher_note=?, resolved_at=?, student_seen=0
               WHERE id=?""",
            (new_status, teacher_note, resolved_at, complaint_id)
        )

        if action == "accept":
            student_id   = complaint["student_id"]
            abs_date     = complaint["date"]
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
# ║  REAL-TIME API                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/api/recent_checkins")
def api_recent_checkins():
    if session.get("role") != "teacher":
        from flask import jsonify
        return jsonify({"error": "unauthorized"}), 401

    from flask import jsonify
    conn  = get_db_connection()
    today = datetime.now().strftime("%Y-%m-%d")

    rows = conn.execute("""
        SELECT s.name, a.time, a.status, a.confidence, a.session
        FROM   attendance a
        JOIN   students   s ON a.student_id = s.id
        WHERE  a.date = ?
        ORDER  BY a.time DESC
        LIMIT  15
    """, (today,)).fetchall()

    total_students = conn.execute(
        "SELECT COUNT(*) FROM students"
    ).fetchone()[0]

    present_count = conn.execute(
        "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date=?",
        (today,)
    ).fetchone()[0]

    conn.close()

    checkins = [{
        "name":       r["name"],
        "time":       r["time"][:5],
        "status":     r["status"],
        "confidence": round(float(r["confidence"]), 1),
        "session":    r["session"],
    } for r in rows]

    return jsonify({
        "checkins":       checkins,
        "present_count":  present_count,
        "total_students": total_students,
        "timestamp":      datetime.now().strftime("%H:%M:%S"),
    })


@app.route("/recent_checkins")
def recent_checkins():
    if session.get("role") != "teacher":
        return {"error": "unauthorized"}, 401

    conn  = get_db_connection()
    today = datetime.now().strftime("%Y-%m-%d")

    rows = conn.execute("""
        SELECT s.name, a.time, a.status, a.confidence, a.session
        FROM   attendance a
        JOIN   students   s ON a.student_id = s.id
        WHERE  a.date = ?
        ORDER  BY a.time DESC
        LIMIT  10
    """, (today,)).fetchall()
    conn.close()

    return {
        "checkins": [
            {
                "name":       r["name"],
                "time":       r["time"][:5],
                "status":     r["status"],
                "confidence": round(r["confidence"], 1),
                "session":    r["session"],
            }
            for r in rows
        ],
        "total": len(rows),
        "date":  today,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PRIORITY 4 — ADVANCED ANALYTICS                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/analytics")
def analytics_page():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    days = request.args.get("days", 30, type=int)
    if days not in (30, 60, 90, 180, 365):
        days = 30

    conn      = get_db_connection()
    today     = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    start_str = (today - timedelta(days=days)).strftime("%Y-%m-%d")

    total_students_n = conn.execute(
        "SELECT COUNT(*) FROM students WHERE is_active=1"
    ).fetchone()[0] or 1

    # ── 1. Class-wide daily trend ──────────────────────────────────────────
    class_rows = conn.execute("""
        SELECT date, COUNT(DISTINCT student_id) AS present
        FROM   attendance
        WHERE  date BETWEEN ? AND ?
        GROUP  BY date
        ORDER  BY date
    """, (start_str, today_str)).fetchall()

    class_trend_labels = [r["date"][5:] for r in class_rows]   # MM-DD
    class_trend_data   = [
        round(r["present"] / total_students_n * 100, 1)
        for r in class_rows
    ]

    # ── 2. Weekday heatmap ─────────────────────────────────────────────────
    DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_totals  = {i: 0   for i in range(7)}
    weekday_present = {i: 0.0 for i in range(7)}
    weekday_counts  = {i: 0   for i in range(7)}

    for r in class_rows:
        d   = datetime.strptime(r["date"], "%Y-%m-%d")
        dow = d.weekday()
        weekday_totals[dow]  += r["present"]
        weekday_counts[dow]  += 1

    weekday_data = []
    for i in range(7):
        c = weekday_counts[i]
        avg = round(weekday_totals[i] / c / total_students_n * 100, 1) if c else 0.0
        weekday_data.append({"name": DAY_NAMES[i], "y": avg})

    non_zero = [w["y"] for w in weekday_data if w["y"] > 0]
    best_day  = DAY_NAMES[max(range(7), key=lambda i: weekday_data[i]["y"])] if non_zero else "—"
    worst_day = DAY_NAMES[min(range(7), key=lambda i: weekday_data[i]["y"] if weekday_data[i]["y"] > 0 else 999)] if non_zero else "—"

    # ── 3. Per-student trend data (for JS chart switcher) ─────────────────
    students_raw = conn.execute(
        "SELECT id, name FROM students WHERE is_active=1 ORDER BY name"
    ).fetchall()

    # Build a set of all school dates in the window
    school_dates = sorted({r["date"] for r in class_rows})

    student_trend_all = {}
    for s in students_raw:
        rows = conn.execute("""
            SELECT date FROM attendance
            WHERE  student_id=? AND date BETWEEN ? AND ?
            ORDER  BY date
        """, (s["id"], start_str, today_str)).fetchall()
        attended_dates = {r["date"] for r in rows}

        # Rolling 7-day window for smoother trend
        actual = []
        labels = []
        for i, d_str in enumerate(school_dates):
            window = school_dates[max(0, i-6):i+1]
            n_present = sum(1 for dd in window if dd in attended_dates)
            pct = round(n_present / len(window) * 100, 1)
            actual.append(pct)
            labels.append(d_str[5:])   # MM-DD

        student_trend_all[str(s["id"])] = {
            "labels": labels,
            "actual": actual,
        }

    # ── 4. At-risk predictions ─────────────────────────────────────────────
    threshold   = float(get_setting("absent_threshold", "75"))
    at_risk_predictions = []

    for s in students_raw:
        all_rows = conn.execute("""
            SELECT date, status FROM attendance
            WHERE  student_id=? AND date BETWEEN ? AND ?
            ORDER  BY date
        """, (s["id"], start_str, today_str)).fetchall()

        attended_dates = {r["date"] for r in all_rows}
        total  = len(school_dates)
        attend = sum(1 for d in school_dates if d in attended_dates)
        pct    = round(attend / total * 100, 1) if total else 0.0

        # Trend: compare first and second half
        half = max(1, total // 2)
        first_half  = school_dates[:half]
        second_half = school_dates[half:]
        r1 = sum(1 for d in first_half  if d in attended_dates) / len(first_half) * 100 if first_half else 0
        r2 = sum(1 for d in second_half if d in attended_dates) / len(second_half)* 100 if second_half else 0
        weekly_trend = round((r2 - r1) / max(1, days / 7), 1)   # % change per week

        # Project 14 days
        projected = round(max(0, min(100, pct + weekly_trend * 2)), 1)

        # Consecutive absences at tail
        consec = 0
        for d in reversed(school_dates):
            if d not in attended_dates:
                consec += 1
            else:
                break

        at_risk_predictions.append({
            "name":                s["name"],
            "pct":                 pct,
            "trend":               weekly_trend,
            "projected":           projected,
            "consecutive_absences": consec,
        })

    # Sort: critical first, then by projected %
    at_risk_predictions.sort(key=lambda x: x["projected"])

    # ── 5. Absentee leaderboard (top 10 worst) ────────────────────────────
    leaderboard = sorted(at_risk_predictions, key=lambda x: x["pct"])[:10]

    # ── 6. KPI strip ──────────────────────────────────────────────────────
    all_pcts = [s["pct"] for s in at_risk_predictions]
    avg_pct  = round(sum(all_pcts) / len(all_pcts), 1) if all_pcts else 0.0
    at_risk_count  = sum(1 for s in at_risk_predictions if s["pct"] < threshold)
    critical_count = sum(1 for s in at_risk_predictions if s["pct"] < 60)

    conn.close()

    kpi = {
        "avg_pct":       avg_pct,
        "at_risk_count": at_risk_count,
        "critical_count":critical_count,
        "best_day":      best_day,
        "worst_day":     worst_day,
        "total_sessions":len(school_dates),
    }

    return render_template(
        "analytics.html",
        days                = days,
        kpi                 = kpi,
        class_trend_labels  = class_trend_labels,
        class_trend_data    = class_trend_data,
        weekday_data        = weekday_data,
        students            = [dict(s) for s in students_raw],
        student_trend_all   = student_trend_all,
        at_risk_predictions = at_risk_predictions,
        leaderboard         = leaderboard,
        today               = today_str,
        active_page         = "analytics",
    )




if __name__ == "__main__":
    app.run(debug=True, port=5000)