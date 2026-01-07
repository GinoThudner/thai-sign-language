import streamlit as st

# ตั้งค่าหน้าเว็บเป็นลำดับแรกสุด
st.set_page_config(page_title="Thai Sign Language", layout="centered")

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools

# --- 1. โหลดโมเดลและรายชื่อท่าทาง ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    labels_list = []
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        # ดึงคอลัมน์ที่ 2 (index 1) มาเป็นคำแปลภาษาไทย
        labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 2. ส่วนแสดงผลคำแปลบนหน้าเว็บ ---
st.title("🖐️ ระบบแปลภาษามือไทย")

# สร้างส่วนแสดงผลคำแปลที่มองเห็นได้ชัดเจน
st.subheader("คำแปลที่ตรวจจับได้:")
result_display = st.empty() 
result_display.info("กำลังรอการตรวจจับท่าทาง...")

# ใช้ Session State เพื่อส่งข้อมูลจาก Callback มาที่หน้าเว็บ
if 'detected_text' not in st.session_state:
    st.session_state['detected_text'] = "รอการตรวจจับ..."

# --- 3. ฟังก์ชันประมวลผลวิดีโอ ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        # Logic การเตรียมข้อมูล 84 features (เหมือนใน app.py ของคุณ)
        data_aux = []
        sorted_hands = sorted(zip(results.multi_hand_landmarks, results.multi_handedness),
                              key=lambda x: x[0].landmark[0].x)
        
        # (ส่วนนี้ข้ามขั้นตอนการแปลงพิกัดเพื่อความกระชับ แต่ใช้ logic เดิมที่คุณมี)
        # ... (โค้ด pre_process_landmark และ get_keypoint_input) ...
        # เมื่อได้ผลลัพธ์:
        # st.session_state['detected_text'] = labels[prediction_id]

    return frame.from_ndarray(img, format="bgr24")

# --- 4. เริ่มต้นสตรีมวิดีโอ (ปิดไมค์) ---
webrtc_streamer(
    key="thai-sign-translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}, # ปิดไมค์ตามต้องการ
    async_processing=True,
)

# แสดงคำแปลภาษาไทยใต้กล้อง
result_display.success(f"ท่าทางที่พบ: {st.session_state['detected_text']}")
