import cv2
import mediapipe as mp
import csv
import os
import time

# List of static ISL letters (skip H and J)
letters = [
    "A","B","C","D","E","F","G","I","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# CSV file
csv_file = "isl_data.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21*3):  # Hand 1
            header.extend([f"hand1_lm{i}_x", f"hand1_lm{i}_y", f"hand1_lm{i}_z"])
        for i in range(21*3):  # Hand 2
            header.extend([f"hand2_lm{i}_x", f"hand2_lm{i}_y", f"hand2_lm{i}_z"])
        header.append("gesture")
        writer.writerow(header)

print("Starting gesture collection for all letters.")
print("You have 5 seconds to get ready for each letter.")

for gesture_name in letters:
    print(f"\nPrepare for gesture '{gesture_name}'...")
    time.sleep(5)  # Give user time to get ready

    print(f"Recording gesture '{gesture_name}'... Press ESC to skip to next letter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        row = []
        if result.multi_hand_landmarks:
            hand_count = len(result.multi_hand_landmarks)

            # Hand 1
            if hand_count >= 1:
                for lm in result.multi_hand_landmarks[0].landmark:
                    row.extend([lm.x, lm.y, lm.z])
                mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            else:
                row.extend([0]*63)

            # Hand 2
            if hand_count == 2:
                for lm in result.multi_hand_landmarks[1].landmark:
                    row.extend([lm.x, lm.y, lm.z])
                mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
            else:
                row.extend([0]*63)

            # Append label
            row.append(gesture_name)

            # Save to CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        cv2.imshow("Collect ISL Gestures", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to skip current letter
            break

cap.release()
cv2.destroyAllWindows()
print(f"All gestures collected and saved to {csv_file}")
