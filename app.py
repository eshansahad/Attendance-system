import sys
import subprocess
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


print("===== Attendance System =====")
print("1. Register Student Face")
print("2. Face Detection (Recognition)")
print("3. Open Dashboard")

choice = input("Enter your choice: ").strip()

if choice == "1":
    print("[INFO] Starting Face Registration...")
    subprocess.run([sys.executable, "core/face_register.py"])

elif choice == "2":
    print("[INFO] Starting Face Recognition...")
    subprocess.run([sys.executable, "core/recognize.py"])

elif choice == "3":
    print("[INFO] Starting Web Dashboard...")
    subprocess.run([sys.executable, "web/dashboard.py"])

else:
    print("[ERROR] Invalid choice")
