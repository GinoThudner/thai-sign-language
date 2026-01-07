import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd

# --- 1. การจัดการ Path และโหลดโมเดล ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

# เรียกใช้งาน MediaPipe นอก Class เพื่อความเสถียร
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

@st.cache_resource
def load_my_model():
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model'] if isinstance(data, dict) else data
        labels = pd.read_csv(label_path, header=None).iloc[:, 0].tolist()
        return model, labels
    except Exception as e:
        st.error(f"Error loading model/labels: {e}")
        return None, None

model, labels = load_my_model()

# --- 2. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Thai Sign Language", layout="centered")
st.title("🖐️ เครื่องแปลภาษามือไทย (On Web)")

# --- 3. ส่วนประมวลผลวิดีโอ (ใช้ระบบใหม่) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and model:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # ทำนายผล
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels[int(prediction[0])]
                
                cv2.putText(img, f"Result: {predicted_character}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            except:
                pass

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame.from_ndarray(img, format="bgr24")

# --- 4. ปุ่มเปิดกล้อง ---
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
