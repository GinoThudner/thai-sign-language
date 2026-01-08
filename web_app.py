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
import queue # เพิ่มตัวนี้เพื่อส่งข้อมูลออก
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 1. เตรียมระบบรับส่งข้อมูล ---
# สร้าง Queue สำหรับรับคำแปลจากวิดีโอออกมาที่หน้าเว็บ
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

if "history" not in st.session_state:
    st.session_state.history = collections.deque(maxlen=10)

# --- 2. โหลดโมเดล ---
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
    hands_engine = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันเตรียมข้อมูล ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
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
        
        # Motion Detection
        p9 = hl.landmark[9]
        st.session_state.history.append((p9.x, p9.y))
        if len(st.session_state.history) == 10:
            dx = st.session_state.history[-1][0] - st.session_state.history[0][0]
            if abs(dx) > 0.12:
                st.session_state.result_queue.put("ไม่") # ส่งค่าเข้า Queue
        
        # AI Static Prediction
        pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
        processed = pre_process_landmark(pts)
        prediction = model.predict(np.array([processed]))[0]
        conf = model.predict_proba(np.array([processed])).max()
        
        if conf > 0.7:
            st.session_state.result_queue.put(labels[int(prediction)]) # ส่งค่าเข้า Queue

    return frame.from_ndarray(img, format="bgr24")

# --- 5. ส่วนแสดงผล UI ---
st.title("🖐️ ระบบแปลภาษามือไทย")

# สร้างพื้นที่สำหรับแสดงตัวอักษร
result_placeholder = st.empty()

# ฟังก์ชันแสดงกล่องคำแปล
def display_label(text):
    result_placeholder.markdown(
        f"""
        <div style="background-color: #1e1e1e; color: #00ff00; padding: 20px; border-radius: 15px; border: 3px solid #00ff00; text-align: center; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 20px; color: white;">ท่าทางที่พบ:</p>
            <h1 style="margin: 0; font-size: 80px; font-weight: bold;">{text}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

display_label("รอตรวจจับ...")

webrtc_streamer(
    key="final-check",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},
    async_processing=True,
)

# --- 6. ส่วนดึงข้อมูลจาก Queue มาโชว์บนหน้าเว็บ (จุดตายของงานนี้) ---
while True:
    try:
        # ดึงคำแปลล่าสุดออกมาจาก Queue
        res = st.session_state.result_queue.get(timeout=1.0)
        display_label(res)
    except queue.Empty:
        # ถ้าไม่มีค่าใหม่เข้ามา ไม่ต้องทำอะไร
        continue
