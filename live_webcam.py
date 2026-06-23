import cv2
import av
import math
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="🤖"
)

st.title("🤖 Live AI Hand Gesture Recognition")

st.markdown("""
### Features

✅ Real-time Hand Tracking

✅ 21 Hand Landmarks Detection

✅ Thumb–Index Distance Calculation

✅ Finger Joint Angle Calculation

✅ Live Webcam Streaming using WebRTC

### Units

📏 Distance = Pixels (px)

📐 Angle = Degrees (°)

### Calculations

**Distance Formula**

Distance = √((x₂ − x₁)² + (y₂ − y₁)²)

**Angle Formula**

Angle = atan2(C − B) − atan2(A − B)
""")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    if angle < 0:
        angle += 360

    if angle > 180:
        angle = 360 - angle

    return int(angle)


class HandProcessor(VideoProcessorBase):

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = img.shape

                points = []

                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))

                # Distance Calculation
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                x1 = int(thumb.x * w)
                y1 = int(thumb.y * h)

                x2 = int(index.x * w)
                y2 = int(index.y * h)

                distance = int(
                    math.hypot(
                        x2 - x1,
                        y2 - y1
                    )
                )

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

                # Angle Calculations
                index_angle = calculate_angle(
                    points[5],
                    points[6],
                    points[8]
                )

                middle_angle = calculate_angle(
                    points[9],
                    points[10],
                    points[12]
                )

                ring_angle = calculate_angle(
                    points[13],
                    points[14],
                    points[16]
                )

                pinky_angle = calculate_angle(
                    points[17],
                    points[18],
                    points[20]
                )

                # Display Results
                cv2.putText(
                    img,
                    f"Distance: {distance} px",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"Index Angle: {index_angle} deg",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"Middle Angle: {middle_angle} deg",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"Ring Angle: {ring_angle} deg",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"Pinky Angle: {pinky_angle} deg",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="hand",
    video_processor_factory=HandProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)