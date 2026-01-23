import sys

print("===== Attendance System =====")
print("1. Register Student Face")
print("2. Face Detection (Recognition)")
print("3. Open Dashboard")

choice = input("Enter your choice: ")

if choice == "1":
    # Looks inside 'core' folder
    from core import face_register
elif choice == "2":
    # Looks inside 'core' folder
    from core import recognize
elif choice == "3":
    # Looks inside 'web' folder and STARTS the server
    from web import dashboard
    print("Starting Web Dashboard...")
    dashboard.app.run(debug=True)
else:
    print("Invalid choice")