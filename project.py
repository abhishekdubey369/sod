import cv2
from deepface import DeepFace
import os
import numpy as np

import torch
import pandas as pd
import requests
import psutil
import torchvision
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn

#yolov5 custom model path
path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

def Img_action(path,model):
    

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load reference images for verification
reference_folder = 'Database'
reference_images = {}
for filename in os.listdir(reference_folder):
    path = os.path.join(reference_folder, filename)
    reference_images[filename] = cv2.imread(path)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If a face is detected, extract it and verify it against the reference images
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        for name, reference_image in reference_images.items():
            result = DeepFace.verify(reference_image, face, enforce_detection=False)
            if result['verified']:
                name = os.path.splitext(name)[0]

                # Draw rectangle around the face with name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break
        else:
            # Draw rectangle around the face with Unknown
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # WRITING THE YOLO RESULTS
    results = model(frame)
    frame = np.squeeze(results.render())
    # Display the resulting frame
    cv2.imshow('frame', frame)
    print(results)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()