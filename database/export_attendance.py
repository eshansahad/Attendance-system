import sqlite3
import pandas as pd
import os
from datetime import datetime

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "reports")

def export_to_excel():
    # 1. Create reports folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        # 2. Connect and fetch data joining students and attendance tables
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT s.name, a.date, a.time 
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        ORDER BY a.date DESC, a.time DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("[INFO] No attendance data found to export.")
            return

        # 3. Generate filename with current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Attendance_Report_{timestamp}.xlsx"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # 4. Save to Excel with a professional look
        df.columns = ["Student Name", "Date", "Time In"]
        df.to_excel(filepath, index=False)
        
        print(f"[SUCCESS] Report generated: {filepath}")

    except Exception as e:
        print(f"[ERROR] Export failed: {e}")

if __name__ == "__main__":
    export_to_excel()