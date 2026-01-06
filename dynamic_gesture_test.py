import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import pyttsx3
import threading
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk

# -------------------------------
# Load trained model & label encoder
# -------------------------------
model = load_model("dynamic_model.h5")
with open("dynamic_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------------------
# MediaPipe Holistic
# -------------------------------
mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# Drawing style (FACE LINES ONLY)
# -------------------------------
face_line_spec = mp_draw.DrawingSpec(
    color=(255, 0, 0),  # Blue (BGR)
    thickness=3,
    circle_radius=0
)

# -------------------------------
# Text-to-Speech
# -------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# -------------------------------
# Prediction stabilization
# -------------------------------
prev_prediction = None
stable_counter = 0
stable_required = 5
confidence_threshold = 0.8
last_spoken = None

# -------------------------------
# GUI Setup
# -------------------------------
root = tk.Tk()
root.title("ISL Dynamic Gesture Detection")
root.geometry("850x700")
root.configure(bg="#1E1E1E")
root.resizable(False, False)

title_frame = Frame(root, bg="#121212", pady=15)
title_frame.pack(fill="x")
title_label = Label(
    title_frame,
    text="ISL Dynamic Gesture Detection",
    font=("Helvetica", 26, "bold"),
    fg="#FFFFFF",
    bg="#121212"
)
title_label.pack()

video_frame = Frame(root, bg="#2C2C2C", highlightbackground="#FFFFFF",
                    highlightthickness=2, padx=5, pady=5)
video_frame.pack(pady=20)

video_label = Label(video_frame, bg="#2C2C2C")
video_label.pack()

prediction_frame = Frame(root, bg="#2C2C2C", padx=10, pady=10)
prediction_frame.pack(pady=10, fill="x")

prediction_text_label = Label(
    prediction_frame,
    text="Detected Gesture:",
    font=("Helvetica", 30, "bold"),
    fg="#CCCCCC",
    bg="#2C2C2C"
)
prediction_text_label.pack(side="left", padx=10)

prediction_label = Label(
    prediction_frame,
    text="Waiting...",
    font=("Helvetica", 30, "bold"),
    fg="#00FFAB",
    bg="#2C2C2C"
)
prediction_label.pack(side="left", padx=20)

# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -------------------------------
# Update frame loop
# -------------------------------
def update_frame():
    global prev_prediction, stable_counter, last_spoken

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb)

    # ---------------- FACE (CLEAN LINES) ----------------
    if result.face_landmarks:
        # Eyes
        mp_draw.draw_landmarks(
            frame, result.face_landmarks,
            mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_line_spec
        )
        mp_draw.draw_landmarks(
            frame, result.face_landmarks,
            mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_line_spec
        )

        # Mouth
        mp_draw.draw_landmarks(
            frame, result.face_landmarks,
            mp_face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_line_spec
        )

        # Nose bridge (custom minimal)
        nose_pairs = [(168, 6), (6, 197), (197, 195), (195, 5), (5, 4)]
        for start, end in nose_pairs:
            x1 = int(result.face_landmarks.landmark[start].x * frame.shape[1])
            y1 = int(result.face_landmarks.landmark[start].y * frame.shape[0])
            x2 = int(result.face_landmarks.landmark[end].x * frame.shape[1])
            y2 = int(result.face_landmarks.landmark[end].y * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # ---------------- BODY (POSE) ----------------
    if result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    # ---------------- HANDS ----------------
    if result.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            result.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if result.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # ---------------- HAND FEATURES FOR MODEL ----------------
    feature_vector = []

    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            feature_vector.extend([lm.x, lm.y, lm.z])
    else:
        feature_vector.extend([0] * 63)

    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            feature_vector.extend([lm.x, lm.y, lm.z])
    else:
        feature_vector.extend([0] * 63)

    # ---------------- PREDICTION ----------------
    X_input = np.array(feature_vector).reshape(1, -1)
    pred_probs = model.predict(X_input, verbose=0)[0]
    max_conf = np.max(pred_probs)
    pred_label = le.inverse_transform([np.argmax(pred_probs)])[0]

    if max_conf > confidence_threshold:
        if pred_label == prev_prediction:
            stable_counter += 1
        else:
            stable_counter = 1
            prev_prediction = pred_label

        if stable_counter >= stable_required:
            if last_spoken != pred_label:
                prediction_label.config(text=pred_label)
                threading.Thread(
                    target=lambda: engine.say(pred_label) or engine.runAndWait(),
                    daemon=True
                ).start()
                last_spoken = pred_label

    # ---------------- DISPLAY IN GUI ----------------
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# -------------------------------
# Start application
# -------------------------------
update_frame()
root.mainloop()

cap.release()
holistic.close()
