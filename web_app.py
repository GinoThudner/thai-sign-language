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
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="แปลภาษามือไทย", layout="centered")

# ใช้ Queue สำหรับรับคำแปล (ย้ายมาอยู่นอก Session State เพื่อความเสถียรใน Thread)
result_queue = queue.Queue()

if "history" not in st.session_state:
    st.session_state.history = collections.deque(maxlen=10)

# --- 2. โหลดโมเดล ---
@st.cache_resource
def load_resources():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
    label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')
    
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
    
    # ลดขนาดภาพเพื่อประมวลผลให้ไวขึ้น (ป้องกันค้าง)
    img_rgb = cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hl = results.multi_hand_landmarks[0]
        # วาด Landmark ลงบนภาพต้นฉบับ
        mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        # Motion Detection
        p9 = hl.landmark[9]
        st.session_state.history.append((p9.x, p9.y))
        if len(st.session_state.history) == 10:
            dx = st.session_state.history[-1][0] - st.session_state.history[0][0]
            if abs(dx) > 0.12:
                result_queue.put("ไม่")
        
        # AI Static Prediction (ใช้พิกัดจาก Mediapipe โดยตรง)
        landmark_list = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
        processed = pre_process_landmark(landmark_list)
        
        prediction = model.predict(np.array([processed]))[0]
        conf = model.predict_proba(np.array([processed])).max()
        
        if conf > 0.7:
            result_queue.put(labels[int(prediction)])

    return frame.from_ndarray(img, format="bgr24")

# --- 5. ส่วนแสดงผล UI ---
st.title("🖐️ ระบบแปลภาษามือไทย")

# ส่วนแสดงคำแปล
result_placeholder = st.empty()

ctx = webrtc_streamer(
    key="stable-v1",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"width": 640, "height": 480, "frameRate": 15},
        "audio": False
    },
    async_processing=True,
)

# --- 6. อัปเดตคำแปลโดยไม่ใช้ Infinite Loop ---
# ใช้ระบบดึงค่าจาก Queue เมื่อ Component วิดีโอยังทำงานอยู่
if ctx.state.playing:
    while True: # Loop นี้จะทำงานเฉพาะตอนที่โปรแกรมรันเฟรม ซึ่ง Streamlit จัดการให้
        try:
            result = result_queue.get(timeout=0.1)
            result_placeholder.markdown(
                f"""
                <div style="background-color: #1e1e1e; color: #00ff00; padding: 20px; border-radius: 15px; border: 3px solid #00ff00; text-align: center;">
                    <h1 style="margin: 0; font-size: 80px;">{result}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        except queue.Empty:
            break
