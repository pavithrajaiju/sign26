import cv2
import mediapipe as mp
import csv
import os
import time

# ---------------------------
# CSV setup
# ---------------------------
csv_file = "dynamic_gestures.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21*3):  # hand1
            header.extend([f"hand1_lm{i}_x", f"hand1_lm{i}_y", f"hand1_lm{i}_z"])
        for i in range(21*3):  # hand2
            header.extend([f"hand2_lm{i}_x", f"hand2_lm{i}_y", f"hand2_lm{i}_z"])
        header.append("gesture")
        writer.writerow(header)

# ---------------------------
# MediaPipe Hands
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=== Camera opened. Adjust yourself. Press ENTER when ready to record a gesture. ===")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    cv2.imshow("Camera Preview - Adjust Yourself", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # ENTER key
        break
    elif key == 27:  # ESC key to quit
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")
        exit()

cv2.destroyAllWindows()

print("Camera ready! Type 'exit' to quit anytime.\n")

# ---------------------------
# Recording loop
# ---------------------------
while True:
    gesture_name = input("Enter the gesture/word you want to record: ").strip()
    if gesture_name.lower() == "exit":
        break
    if gesture_name == "":
        print("Please enter a valid gesture name.")
        continue

    print(f"\nGet ready to perform '{gesture_name}' in 5 seconds...")
    time.sleep(5)
    print(f"Recording '{gesture_name}'... Press ESC to skip early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
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

            # Add gesture label
            row.append(gesture_name)

            # Save row to CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        # Show live recording
        cv2.putText(frame, f"Recording: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Recording {gesture_name}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to skip
            break

cap.release()
cv2.destroyAllWindows()
print(f"All dynamic gestures recorded and saved to {csv_file}")
