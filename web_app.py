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

# --- 1. ตั้งค่าสถานะเริ่มต้น ---
if "last_msg" not in st.session_state:
    st.session_state.last_msg = "รอตรวจจับ..."
if "history" not in st.session_state:
    st.session_state.history = collections.deque(maxlen=10)

# --- 2. โหลดทรัพยากร (เพิ่ม error handling) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_resources():
    try:
        with open(model_path, 'rb') as f:
            m = pickle.load(f)
            model_obj = m['model'] if isinstance(m, dict) else m
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, -1].astype(str).tolist()
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None, [], None, None, None

    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.4, # ปรับค่าให้เหมาะสมกับสภาพแสง
        min_tracking_confidence=0.4
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันเตรียมข้อมูล (Pre-process) ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_val = max(list(map(abs, temp_landmark_list)))
    return [n / max_val if max_val != 0 else 0 for n in temp_landmark_list]

# --- 4. ฟังก์ชันจัดการวิดีโอ (Callback) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    # 1. ลดขนาดภาพเพื่อส่งให้ AI ประมวลผล (ลดภาระ CPU)
    img_rgb = cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # 2. วาด Landmark ลงบนภาพต้นฉบับ (img)
        # เราต้องคำนวณสเกลใหม่เพราะตรวจจับจากภาพเล็ก (320x240) แต่จะวาดบนภาพใหญ่
        hl = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        # ... (โค้ดส่วน Motion และ Prediction ของคุณ) ...
        # คำแนะนำ: ในส่วนการหาพิกัด AI static ให้ใช้ค่า l.x, l.y ที่เป็น Ratio (0.0 - 1.0) 
        # จะทำให้แม่นยำกว่าการคูณ w, h ที่เปลี่ยนไปมาครับ

    # 3. ส่งภาพต้นฉบับ (img) กลับไปแสดงผล
    return frame.from_ndarray(img, format="bgr24")

# --- 5. หน้าตา UI ---
st.title("🖐️ ระบบแปลภาษามือไทย")

placeholder = st.empty()
placeholder.markdown(
    f"""
    <div style="background-color: #1e1e1e; color: #00ff00; padding: 20px; border-radius: 15px; border: 2px solid #00ff00; text-align: center; margin-bottom: 10px;">
        <h1 style="margin: 0; font-size: 60px;">{st.session_state.last_msg}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

webrtc_streamer(
    key="stable-v10", # เปลี่ยนชื่อ Key ใหม่เพื่อล้างบัฟเฟอร์
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640}, 
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15, "max": 20}, # ปรับ FrameRate ให้เสถียรขึ้น
            "facingMode": "user"
        },
        "audio": False
    },
    async_processing=True,
)



