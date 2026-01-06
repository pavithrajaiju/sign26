import cv2
import mediapipe as mp

# -------------------- MEDIAPIPE IMPORTS --------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# -------------------- HOLISTIC MODEL --------------------
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- WEBCAM --------------------
cap = cv2.VideoCapture(0)

# -------------------- DRAWING SPECS --------------------
# Face: LINES ONLY (no dots)
face_line_spec = mp_drawing.DrawingSpec(
    color=(255, 0, 0),  # Blue lines (BGR)
    thickness=2,
    circle_radius=0
)

# Pose & Hands (unchanged)
pose_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
hand_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = holistic.process(rgb)

    # -------------------- FACE (LINES ONLY – NO DOTS) --------------------
    if result.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.face_landmarks,
            mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_line_spec
        )
        mp_drawing.draw_landmarks(
            frame,
            result.face_landmarks,
            mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_line_spec
        )
        mp_drawing.draw_landmarks(
            frame,
            result.face_landmarks,
            mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_line_spec
        )
        mp_drawing.draw_landmarks(
            frame,
            result.face_landmarks,
            mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_line_spec
        )
        mp_drawing.draw_landmarks(
            frame,
            result.face_landmarks,
            mp_face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_line_spec
        )

    # -------------------- POSE (BODY) --------------------
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            pose_spec,
            pose_spec
        )

    # -------------------- HANDS --------------------
    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            hand_spec,
            hand_spec
        )

    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            hand_spec,
            hand_spec
        )

    cv2.imshow("ISL Holistic Demo (No Face Dots)", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
