import cv2
import mediapipe as mp
import math

# ---------- Function to calculate angle ----------
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    if angle < 0:
        angle += 360

    if angle > 180:
        angle = 360 - angle

    return int(angle)

# ---------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            points = []

            for lm in hand_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

            # Index Finger Angle
            index_angle = calculate_angle(
                points[5],
                points[6],
                points[8]
            )

            # Middle Finger Angle
            middle_angle = calculate_angle(
                points[9],
                points[10],
                points[12]
            )

            # Ring Finger Angle
            ring_angle = calculate_angle(
                points[13],
                points[14],
                points[16]
            )

            # Pinky Finger Angle
            pinky_angle = calculate_angle(
                points[17],
                points[18],
                points[20]
            )

            cv2.putText(
                frame,
                f"Index: {index_angle}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Middle: {middle_angle}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Ring: {ring_angle}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Pinky: {pinky_angle}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("AI Hand Gesture - Angle Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()