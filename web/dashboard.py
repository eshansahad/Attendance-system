from flask import Flask, render_template, request, redirect, session, url_for, Response
import sqlite3
import os
import cv2
import numpy as np
import sys
import shutil  # NEW: Needed to delete folders
from datetime import datetime

# Add the project root to system path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db_utils import mark_attendance
import mediapipe as mp

# Initialize Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "super_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "database", "attendance.db")
DATASET_PATH = os.path.join(BASE_DIR, "..", "data_files", "dataset")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "..", "data_files", "embeddings", "embeddings.npy")
LABELS_PATH = os.path.join(BASE_DIR, "..", "data_files", "embeddings", "labels.npy")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# =========================================
# LOGIN & DASHBOARD ROUTES
# =========================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form['role']
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()

        if role == 'teacher':
            cursor.execute("SELECT * FROM teachers WHERE username = ? AND password = ?", (username, password))
            user = cursor.fetchone()
            if user:
                session['user_id'] = user['id']; session['role'] = 'teacher'
                conn.close(); return redirect(url_for('index'))
        
        elif role == 'student':
            cursor.execute("SELECT * FROM students WHERE name = ? AND password = ?", (username, password))
            user = cursor.fetchone()
            if user:
                session['user_id'] = user['id']; session['user_name'] = user['name']; session['role'] = 'student'
                conn.close(); return redirect(url_for('student_dashboard'))

        conn.close()
        return "Invalid Credentials! <a href='/login'>Try Again</a>"
    return render_template("login.html")

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for('login'))

@app.route("/")
def index():
    if 'role' not in session or session['role'] != 'teacher': return redirect(url_for('login'))
    conn = get_db_connection(); cursor = conn.cursor(); today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT students.name, attendance.date, attendance.time FROM attendance JOIN students ON attendance.student_id = students.id WHERE attendance.date = ?", (today,))
    attendance_data = cursor.fetchall(); conn.close()
    return render_template("index.html", attendance=attendance_data)

@app.route("/students")
def students_page():
    if 'role' not in session or session['role'] != 'teacher': return redirect(url_for('login'))
    conn = get_db_connection(); students = conn.execute("SELECT * FROM students").fetchall(); conn.close()
    return render_template("students.html", students=students)

# --- NEW DELETE ROUTE ---
@app.route("/delete_student/<int:student_id>")
def delete_student(student_id):
    if 'role' not in session or session['role'] != 'teacher': return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Get student name to find their folder
    cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()

    if student:
        student_name = student['name']
        
        # 2. Delete from Database
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM issues WHERE student_id = ?", (student_id,))
        conn.commit()

        # 3. Delete from Dataset Folder
        folder_path = os.path.join(DATASET_PATH, student_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"[INFO] Deleted folder for: {student_name}")
            
    conn.close()
    return redirect(url_for('students_page'))

@app.route("/report")
def report_page():
    if 'role' not in session or session['role'] != 'teacher': return redirect(url_for('login'))
    conn = get_db_connection(); cursor = conn.cursor(); today = datetime.now().strftime("%Y-%m-%d")
    
    cursor.execute("SELECT id, name FROM students"); all_students = cursor.fetchall()
    cursor.execute("SELECT student_id, time FROM attendance WHERE date = ?", (today,)); present_records = cursor.fetchall()
    present_map = {row['student_id']: row['time'] for row in present_records}

    attendance_report = []
    for student in all_students:
        status = "Present" if student['id'] in present_map else "Absent"
        time_in = present_map[student['id']] if student['id'] in present_map else "--:--"
        attendance_report.append({"name": student['name'], "status": status, "time": time_in, "date": today})

    cursor.execute("SELECT issues.id, issues.type, issues.description, issues.date, issues.status, students.name, issues.student_id FROM issues JOIN students ON issues.student_id = students.id ORDER BY issues.date DESC")
    complaints = cursor.fetchall(); conn.close()
    return render_template("report.html", attendance_logs=attendance_report, complaints=complaints, today_date=today)

@app.route("/complaint/<int:issue_id>/<action>")
def manage_complaint(issue_id, action):
    if 'role' not in session or session['role'] != 'teacher': return redirect(url_for('login'))
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM issues WHERE id = ?", (issue_id,)); issue = cursor.fetchone()
    if issue:
        if action == "approve":
            cursor.execute("UPDATE issues SET status = 'Resolved' WHERE id = ?", (issue_id,))
            cursor.execute("SELECT * FROM attendance WHERE student_id = ? AND date = ?", (issue['student_id'], issue['date']))
            if not cursor.fetchone():
                cursor.execute("INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)", (issue['student_id'], issue['date'], "Manual Entry"))
        elif action == "reject":
            cursor.execute("UPDATE issues SET status = 'Rejected' WHERE id = ?", (issue_id,))
    conn.commit(); conn.close()
    return redirect(url_for('report_page'))

@app.route("/student/dashboard")
def student_dashboard():
    if 'role' not in session or session['role'] != 'student': return redirect(url_for('login'))
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance WHERE student_id = ?", (session['user_id'],)); my_attendance = cursor.fetchall()
    cursor.execute("SELECT * FROM issues WHERE student_id = ? ORDER BY date DESC", (session['user_id'],)); my_issues = cursor.fetchall()
    conn.close()
    return render_template("student_dashboard.html", student_name=session['user_name'], my_attendance=my_attendance, my_issues=my_issues)

@app.route("/student/report_issue", methods=["POST"])
def report_issue():
    if 'role' not in session or session['role'] != 'student': return redirect(url_for('login'))
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("INSERT INTO issues (student_id, type, description, date) VALUES (?, ?, ?, ?)", (session['user_id'], request.form['issue_type'], request.form['description'], datetime.now().strftime("%Y-%m-%d")))
    conn.commit(); conn.close()
    return "<script>alert('Report Submitted!'); window.location.href='/student/dashboard';</script>"

# =========================================
# AI CAMERA ROUTES
# =========================================
def gen_recognition_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("[ERROR] Camera not found"); return
    if not os.path.exists(EMBEDDINGS_PATH): print("[ERROR] No embeddings"); cap.release(); return

    embeddings = np.load(EMBEDDINGS_PATH); labels = np.load(LABELS_PATH)
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        try:
            while True:
                success, frame = cap.read()
                if not success: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                
                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        h, w, _ = frame.shape
                        cx, cy = int(face_landmarks.landmark[4].x * w), int(face_landmarks.landmark[4].y * h)
                        embedding = []
                        for lm in face_landmarks.landmark: embedding.extend([lm.x, lm.y, lm.z])
                        embedding = np.array(embedding)
                        distances = np.linalg.norm(embeddings - embedding, axis=1)
                        min_idx = np.argmin(distances)
                        if distances[min_idx] < 0.8:
                            name = labels[min_idx]; mark_attendance(name)
                            cv2.putText(frame, f"{name} (Present)", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.rectangle(frame, (cx-60, cy-60), (cx+60, cy+60), (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Unknown", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit: print("[INFO] Closing Camera"); cap.release()
        except Exception as e: print(e)
        finally: cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_recognition_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera_page():
    if 'role' not in session or session['role'] != 'teacher': return redirect(url_for('login'))
    return render_template("video_feed.html")

@app.route("/register", methods=["GET", "POST"])
def register_page():
    if 'role' not in session or session['role'] != 'teacher': return redirect(url_for('login'))
    if request.method == "POST":
        student_name = request.form['student_name']
        student_path = os.path.join(DATASET_PATH, student_name)
        if not os.path.exists(student_path):
            os.makedirs(student_path)
            conn = get_db_connection(); conn.execute("INSERT INTO students (name, password) VALUES (?, ?)", (student_name, "1234")); conn.commit(); conn.close()
            return render_template("video_register.html", student_name=student_name)
        else: return "Student already exists! <a href='/register'>Go Back</a>"
    return render_template("register_form.html")

@app.route('/capture_images/<student_name>')
def capture_images(student_name):
    student_path = os.path.join(DATASET_PATH, student_name)
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 15:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(student_path, f"{student_name}_{count}.jpg"), frame)
            count += 1
            cv2.waitKey(100)
    cap.release()
    os.system(f"python {os.path.join(BASE_DIR, '..', 'core', 'extract_embeddings.py')}")
    return "captured"

@app.route("/export")
def export_csv(): return "Export feature coming soon! <a href='/'>Back</a>"