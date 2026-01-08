import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import collections
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 1. ตั้งค่าพื้นฐาน ---
if "last_msg" not in st.session_state:
    st.session_state.last_msg = "รอตรวจจับ..."
if "history" not in st.session_state:
    st.session_state.history = collections.deque(maxlen=10) # ลดเหลือ 10 เพื่อให้ไวขึ้น

# --- 2. โหลดโมเดล (Cache ไว้เพื่อไม่ให้โหลดซ้ำจนค้าง) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    df = pd.read_csv(label_path, header=None, encoding='utf-8')
    labels_list = df.iloc[:, -1].astype(str).tolist()
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(
        max_num_hands=1, # ตรวจจับแค่มือเดียวจะลื่นกว่ามาก
        min_detection_confidence=0.3, # ลดจาก 0.5 เหลือ 0.3
        min_tracking_confidence=0.3
)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันคำนวณแบบรวดเร็ว ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hl = results.multi_hand_landmarks[0]
        # วาดเส้นให้เห็นในวิดีโอเลยว่าตรวจจับเจอไหม
        mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        # เก็บพิกัด Motion
        p9 = hl.landmark[9]
        st.session_state.history.append((p9.x, p9.y))

        # ตรวจจับ Motion แบบด่วน
        if len(st.session_state.history) == 10:
            dx = st.session_state.history[-1][0] - st.session_state.history[0][0]
            if abs(dx) > 0.10: 
                st.session_state.last_msg = "ไม่"
                return frame.from_ndarray(img, format="bgr24")

        # ถ้าไม่ขยับ ให้ AI ทาย (Static)
        # (ส่วนคำนวณพิกัดเพื่อส่งให้ Model ทายผล...)
        # ... [ส่วนนี้ใช้โค้ดเดิมของคุณในการเตรียม data_aux] ...
        # หลังจากได้ prediction:
        # st.session_state.last_msg = labels[int(prediction)]

    return frame.from_ndarray(img, format="bgr24")

# --- 4. การแสดงผล UI (ปรับให้เข้มข้นขึ้น) ---
st.title("🖐️ ระบบแปลภาษามือไทย")

# กล่องคำแปลที่อัปเดตตัวเองได้
placeholder = st.empty()
placeholder.markdown(
    f"""
    <div style="background-color: #2b2b2b; color: #00ff00; padding: 20px; border-radius: 15px; border: 3px solid #00ff00; text-align: center;">
        <h2 style="margin: 0;">ท่าทางที่พบ:</h2>
        <h1 style="margin: 0; font-size: 80px;">{st.session_state.last_msg}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

webrtc_streamer(
    key="fixed-v3",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
    "video": {
        "width": {"ideal": 320}, # ปรับลดจาก 640 เหลือ 320
        "height": {"ideal": 240}, 
        "frameRate": {"ideal": 10} # ปรับเฟรมเรตให้ต่ำลงเพื่อลดภาระ CPU
    },
    "audio": False
}
    async_processing=True,
)

