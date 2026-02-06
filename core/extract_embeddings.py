import cv2
import mediapipe as mp
import numpy as np
import os

# ==============================
# ABSOLUTE PATH SETUP
# ==============================
# Get the directory where this script is located (core/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Go up one level to the project root (attendance_system/)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Define absolute paths based on the project root
DATASET_PATH = os.path.join(PROJECT_ROOT, "data_files", "dataset")
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "data_files", "embeddings")

# ==============================
# AI MODELS SETUP
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

embeddings = []
labels = []

print(f"[INFO] Looking for dataset at: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] Dataset folder not found at {DATASET_PATH}")
    print("[FIX] Ensure your images are in attendance_system/data_files/dataset/")
    exit()

# ==============================
# EXTRACTION LOOP
# ==============================
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"[PROCESS] Extracting features for: {person_name}")
    
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            print(f"  [WARN] No face found in {img_name}. Skipping...")
            continue

        landmarks = result.multi_face_landmarks[0]
        
        # --- NORMALIZATION ---
        # Subtract the 'nose tip' (Landmark 1) from all points for position independence
        all_lm = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        nose_tip = all_lm[1] 
        normalized_coords = (all_lm - nose_tip).flatten()
        
        embeddings.append(normalized_coords)
        labels.append(person_name)

# ==============================
# SAVE DATA
# ==============================
os.makedirs(EMBEDDING_DIR, exist_ok=True)

np.save(os.path.join(EMBEDDING_DIR, "embeddings.npy"), np.array(embeddings))
np.save(os.path.join(EMBEDDING_DIR, "labels.npy"), np.array(labels))

print(f"[INFO] Successfully saved {len(labels)} face signatures to {EMBEDDING_DIR}")