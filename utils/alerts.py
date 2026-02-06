import smtplib
from email.message import EmailMessage
import os
import sqlite3
from datetime import datetime

# ======================
# EMAIL CONFIG
# ======================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "eshanrsahad@gmail.com"
SENDER_PASSWORD = "khnanxjfmipwdfxb"   # 16-char app password
ADMIN_EMAIL = "allinoneattime@gmail.com"

# ======================
# SEND INTRUDER ALERT
# ======================
def send_intruder_alert(image_path):
    if not os.path.exists(image_path):
        print("[ALERT] Image not found, email not sent")
        return

    msg = EmailMessage()
    msg["Subject"] = "🚨 SECURITY ALERT: Unauthorized Person Detected"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ADMIN_EMAIL

    msg.set_content(
        "An unauthorized person was detected by the AI Attendance System.\n\n"
        "Please find the attached image for verification.\n\n"
        "– Smart Attendance System"
    )

    with open(image_path, "rb") as img:
        img_data = img.read()

    msg.add_attachment(
        img_data,
        maintype="image",
        subtype="jpeg",
        filename=os.path.basename(image_path)
    )

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("[ALERT] Intruder email sent successfully")

    except Exception as e:
        print("[ERROR] Email alert failed:", e)


DB_PATH = "database/attendance.db"  # relative to project root

def send_absentee_report():
    today = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name FROM students
        WHERE id NOT IN (
            SELECT student_id FROM attendance WHERE date = ?
        )
    """, (today,))

    absentees = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not absentees:
        body = "All students are present today. 🎉"
    else:
        body = "Absent Students:\n\n"
        body += "\n".join(f"• {name}" for name in absentees)

    msg = EmailMessage()
    msg["Subject"] = f"📋 Absentees Report – {today}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ADMIN_EMAIL
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("[REPORT] Absentee email sent")

    except Exception as e:
        print("[ERROR] Absentee email failed:", e)

