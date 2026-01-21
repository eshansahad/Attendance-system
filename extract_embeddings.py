import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh

DATASET_PATH = "dataset"
EMBEDDING_DIR = "embeddings"

embeddings = []
labels = []

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

print("[INFO] Extracting face embeddings...")

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue

        landmarks = result.multi_face_landmarks[0]
        embedding = []

        for lm in landmarks.landmark:
            embedding.extend([lm.x, lm.y, lm.z])

        embeddings.append(embedding)
        labels.append(person_name)

embeddings = np.array(embeddings)

os.makedirs(EMBEDDING_DIR, exist_ok=True)
np.save(os.path.join(EMBEDDING_DIR, "embeddings.npy"), embeddings)
np.save(os.path.join(EMBEDDING_DIR, "labels.npy"), np.array(labels))

print("[INFO] Embeddings extracted and saved successfully")
