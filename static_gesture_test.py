import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import pyttsx3
import threading
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# -------------------------------
# Load model & label encoder
# -------------------------------
model = load_model("isl_model_static.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------------------
# MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.7)

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
root.title("Static Gesture Detection")
root.geometry("700x550")
root.configure(bg="#2C2C2C")  # dark background
root.resizable(False, False)

video_label = Label(root, bg="#2C2C2C")
video_label.pack()

# Prediction display
prediction_frame = tk.Frame(root, bg="#2C2C2C", padx=10, pady=10)
prediction_frame.pack(pady=10)

label_text = Label(prediction_frame, text="Detected Gesture:", 
                   font=("Helvetica", 20, "bold"), fg="#CCCCCC", bg="#2C2C2C")
label_text.pack(side="left", padx=10)

prediction_label = Label(prediction_frame, text="Waiting...", 
                         font=("Helvetica", 20, "bold"), fg="#00FFAB", bg="#2C2C2C")
prediction_label.pack(side="left", padx=20)

# -------------------------------
# Open webcam
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -------------------------------
# Update frames
# -------------------------------
def update_frame():
    global prev_prediction, stable_counter, last_spoken

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    feature_vector = []

    if result.multi_hand_landmarks:
        hand_count = len(result.multi_hand_landmarks)

        # Hand 1
        if hand_count >= 1:
            for lm in result.multi_hand_landmarks[0].landmark:
                feature_vector.extend([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        else:
            feature_vector.extend([0]*63)

        # Hand 2
        if hand_count == 2:
            for lm in result.multi_hand_landmarks[1].landmark:
                feature_vector.extend([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
        else:
            feature_vector.extend([0]*63)

        # Predict
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
                    threading.Thread(target=lambda: engine.say(pred_label) or engine.runAndWait(), daemon=True).start()
                    last_spoken = pred_label

    # Convert frame to ImageTk for Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# Start loop
update_frame()
root.mainloop()
