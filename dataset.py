import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        return [(lm.x, lm.y) for lm in landmarks.landmark]
    return None

images_folder = 'images'
landmarks_folder = 'landmarks'

os.makedirs(landmarks_folder, exist_ok=True)

for img_name in os.listdir(images_folder):
    img_path = os.path.join(images_folder, img_name)
    image = cv2.imread(img_path)
    landmarks = extract_landmarks(image)
    if landmarks:
        landmarks_array = np.array(landmarks)
        np.save(os.path.join(landmarks_folder, img_name.replace('.png', '.npy')), landmarks_array)
