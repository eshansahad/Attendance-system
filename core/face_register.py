import cv2
import os
import time

# Ask student name
student_name = input("Enter student name: ").strip()

# UPDATED: Path points to the new 'data_files' folder
dataset_path = "data_files/dataset"
student_path = os.path.join(dataset_path, student_name)

if not os.path.exists(student_path):
    os.makedirs(student_path)

# Open webcam
cap = cv2.VideoCapture(0)

count = 0
max_images = 15   # Number of images to capture

print("[INFO] Look at the camera. Capturing images...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not working")
        break

    # Show instructions
    cv2.putText(frame, f"Capturing image {count+1}/{max_images}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("Face Registration", frame)

    key = cv2.waitKey(1)

    # Capture image every 0.5 seconds
    if count < max_images:
        img_name = f"{student_name}_{count}.jpg"
        img_path = os.path.join(student_path, img_name)
        cv2.imwrite(img_path, frame)
        print(f"[INFO] Saved {img_name}")
        count += 1
        time.sleep(0.5)

    else:
        print("[INFO] Image capture completed")
        break

    # Press 'q' to quit early
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()