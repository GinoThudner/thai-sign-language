import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd

# 1. จัดการ Path และโหลดโมเดล
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_all_resources():
    # โหลดโมเดลและเลเบล
    with open(model_path, 'rb') as f:
        m_data = pickle.load(f)
        model_obj = m_data['model'] if isinstance(m_data, dict) else m_data
    labels_list = pd.read_csv(label_path, header=None).iloc[:, 0].tolist()
    
    # ตั้งค่า MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_engine = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return model_obj, labels_list, hands_engine, mp_draw, mp_hands

# ดึงทรัพยากรมาใช้งาน
model, labels, hands, mp_drawing, mp_hands_module = load_all_resources()

st.title("🖐️ Thai Sign Language Translator (Public)")
st.write("สถานะระบบ: " + ("✅ พร้อม" if model else "❌ ตรวจสอบโมเดล"))

# 2. ฟังก์ชันประมวลผลวิดีโอ
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # กลับด้านหน้าจอ
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

            try:
                prediction = model.predict([np.asarray(data_aux)])
                result_text = labels[int(prediction[0])]
                cv2.putText(img, result_text, (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            except:
                pass
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands_module.HAND_CONNECTIONS)

    return frame.from_ndarray(img, format="bgr24")

# 3. ปุ่มเปิดกล้อง
webrtc_streamer(
    key="translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
