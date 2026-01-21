print("===== Attendance System =====")
print("1. Register Student Face")
print("2. Face Detection (Recognition)")
print("3. Open Dashboard")

choice = input("Enter your choice: ")

if choice == "1":
    import face_register
elif choice == "2":
    import recognize
elif choice == "3":
    import dashboard
else:
    print("Invalid choice")
