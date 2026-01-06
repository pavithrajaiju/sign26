import cv2
import mediapipe as mp
import os
import csv

# -------------------- PATHS --------------------
DATASET_PATH = "Dataset"
CSV_FILE = "isl_all_data.csv"

# -------------------- MEDIAPIPE HOLISTIC --------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- CSV HEADER --------------------
header = []

# Pose landmarks (33 × 3 = 99)
for i in range(33):
    header.extend([f"pose_{i}_x", f"pose_{i}_y", f"pose_{i}_z"])

# Left hand landmarks (21 × 3 = 63)
for i in range(21):
    header.extend([f"left_hand_{i}_x", f"left_hand_{i}_y", f"left_hand_{i}_z"])

# Right hand landmarks (21 × 3 = 63)
for i in range(21):
    header.extend([f"right_hand_{i}_x", f"right_hand_{i}_y", f"right_hand_{i}_z"])

header.append("label")

# Create CSV file if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# -------------------- PROCESS DATASET --------------------
for gesture_type in os.listdir(DATASET_PATH):
    gesture_path = os.path.join(DATASET_PATH, gesture_type)
    if not os.path.isdir(gesture_path):
        continue

    for gesture_class in os.listdir(gesture_path):
        class_path = os.path.join(gesture_path, gesture_class)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing gesture: {gesture_class}")

        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)

            # ---------- VIDEO FILES ----------
            if file_path.lower().endswith((".mp4", ".avi")):
                cap = cv2.VideoCapture(file_path)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(rgb)

                    row = []

                    # Pose
                    if result.pose_landmarks:
                        for lm in result.pose_landmarks.landmark:
                            row.extend([lm.x, lm.y, lm.z])
                    else:
                        row.extend([0] * 99)

                    # Left hand
                    if result.left_hand_landmarks:
                        for lm in result.left_hand_landmarks.landmark:
                            row.extend([lm.x, lm.y, lm.z])
                    else:
                        row.extend([0] * 63)

                    # Right hand
                    if result.right_hand_landmarks:
                        for lm in result.right_hand_landmarks.landmark:
                            row.extend([lm.x, lm.y, lm.z])
                    else:
                        row.extend([0] * 63)

                    row.append(gesture_class)

                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                cap.release()

            # ---------- IMAGE FILES ----------
            elif file_path.lower().endswith((".jpg", ".png", ".jpeg")):
                frame = cv2.imread(file_path)
                if frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)

                row = []

                # Pose
                if result.pose_landmarks:
                    for lm in result.pose_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0] * 99)

                # Left hand
                if result.left_hand_landmarks:
                    for lm in result.left_hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0] * 63)

                # Right hand
                if result.right_hand_landmarks:
                    for lm in result.right_hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0] * 63)

                row.append(gesture_class)

                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

# -------------------- CLEANUP --------------------
holistic.close()
print("✅ Pose + hand landmarks extracted and saved to", CSV_FILE)
