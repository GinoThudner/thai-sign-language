import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import collections
import copy
import itertools
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 1. ตั้งค่าพื้นฐาน ---
st.set_page_config(page_title="Sign Language AI", layout="centered")

if "last_msg" not in st.session_state:
    st.session_state.last_msg = "รอตรวจจับ..."
if "history" not in st.session_state:
    st.session_state.history = collections.deque(maxlen=10)

# --- 2. โหลดทรัพยากร ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    # อ่าน Label โดยดึงเฉพาะคอลัมน์ชื่อภาษาไทย
    df = pd.read_csv(label_path, header=None, encoding='utf-8')
    labels_list = df.iloc[:, -1].astype(str).tolist()
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.3, # ลดเพื่อให้ตรวจจับง่ายขึ้นในที่แสงน้อย
        min_tracking_confidence=0.3
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันเตรียมข้อมูลพิกัด ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # ปรับจุดเริ่มต้นให้เป็น (0,0)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    # แปลงเป็นลิสต์เดี่ยว
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # Normalization
    max_val = max(list(map(abs, temp_landmark_list)))
    return [n / max_val if max_val != 0 else 0 for n in temp_landmark_list]

# --- 4. ฟังก์ชันประมวลผลวิดีโอ ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hl = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        # ส่วนที่ 1: ตรวจจับ Motion (ส่ายมือ)
        p9 = hl.landmark[9]
        st.session_state.history.append((p9.x, p9.y))

        if len(st.session_state.history) == 10:
            dx = st.session_state.history[-1][0] - st.session_state.history[0][0]
            # ถ้าส่ายมือซ้าย-ขวา
            if abs(dx) > 0.12: 
                st.session_state.last_msg = "ไม่"
                return frame.from_ndarray(img, format="bgr24")

        # ส่วนที่ 2: ใช้ AI แปลท่าทางนิ่ง
        landmark_list = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
        processed_data = pre_process_landmark(landmark_list)
        
        # ทำนายผล
        input_data = np.array([processed_data], dtype=np.float32)
        prediction = model.predict(input_data)[0]
        conf = model.predict_proba(input_data).max()

        if conf > 0.6:
            st.session_state.last_msg = labels[int(prediction)]

    return frame.from_ndarray(img, format="bgr24")

# --- 5. การแสดงผล UI ---
st.title("🖐️ ระบบแปลภาษามือไทย")

# แสดงกล่องคำแปล
placeholder = st.empty()
placeholder.markdown(
    f"""
    <div style="background-color: #2b2b2b; color: #00ff00; padding: 20px; border-radius: 15px; border: 3px solid #00ff00; text-align: center; margin-bottom: 10px;">
        <h2 style="margin: 0; font-size: 1.5rem;">ท่าทางที่พบ:</h2>
        <h1 style="margin: 0; font-size: 5rem; font-weight: bold;">{st.session_state.last_msg}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# สตรีมวิดีโอขนาดเล็กเพื่อลด Glitch
webrtc_streamer(
    key="fixed-final-v1",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320}, 
            "height": {"ideal": 240}, 
            "frameRate": {"ideal": 12}
        },
        "audio": False
    },
    async_processing=True,
)

if st.button("รีเฟรช / ล้างคำแปล"):
    st.session_state.last_msg = "รอตรวจจับ..."
    st.rerun()
