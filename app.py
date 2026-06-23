import cv2
import mediapipe as mp
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
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

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            # Show landmark IDs
            for id, lm in enumerate(hand_landmarks.landmark):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.putText(
                    frame,
                    str(id),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            # Thumb tip = 4
            # Index tip = 8
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            x1 = int(thumb_tip.x * w)
            y1 = int(thumb_tip.y * h)

            x2 = int(index_tip.x * w)
            y2 = int(index_tip.y * h)

            # Draw circles
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

            # Draw line
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Calculate distance
            distance = math.hypot(x2 - x1, y2 - y1)

            # Display distance
            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    cv2.imshow("AI Hand Gesture Project", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()