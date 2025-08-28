import face_recognition
import cv2
import os
from datetime import datetime
import pandas as pd
from deepface import DeepFace
import urllib.request

# === Setup known_faces directory with sample face ===
known_faces_dir = "known_faces"
sample_image_url ="C:/Users/dell/Downloads/face_recognition_project/known_faces/sample_face.jpg"
sample_image_path = os.path.join(known_faces_dir, "sample_face.jpg")

# Create folder if it doesn't exist
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# Download sample face if folder is empty
if not os.listdir(known_faces_dir):
    print("📥 Downloading sample face: sample_face.jpg")
    urllib.request.urlretrieve(sample_image_url, sample_image_path)

# === Load known faces ===
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png')):
        path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
