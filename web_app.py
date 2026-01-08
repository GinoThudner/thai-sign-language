import streamlit as st

# --- 1. ตั้งค่าหน้าเว็บ (ต้องเป็นบรรทัดแรกสุด) ---
st.set_page_config(
    page_title="แปลภาษามือไทยออนไลน์ - AI Sign Language Translator",
    page_icon="🖐️",
    layout="centered"
)

import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
import queue
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 2. ข้อความอธิบาย ---
st.title("🖐️ ระบบแปลภาษามือไทยแบบ Real-time")
st.markdown("รองรับทั้งคอมพิวเตอร์และมือถือ (แนะนำเปิดผ่าน Chrome หรือ Safari)")
st.markdown("---")

# --- 3. โหลดทรัพยากร ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

# ใช้ Session State เก็บผลลัพธ์เพื่อความเสถียร
if "last_pred" not in st.session_state:
    st.session_state.last_pred = "รอตรวจจับ..."

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
    else:
        labels_list = ["Error: No Label File"]
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 4. ฟังก์ชันประมวลผล Landmark ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_val = max(list(map(abs, temp_landmark_list)))
    return [n / max_val if max_val != 0 else 0 for n in temp_landmark_list]

def flip_keypoint_x(keypoint_list):
    flipped = list(keypoint_list)
    for i in range(0, 42, 2): flipped[i] *= -1
    return flipped

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # 1. สร้าง img_rgb ก่อนส่งให้ AI
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        data_aux = []
        sorted_hands = sorted(zip(results.multi_hand_landmarks, results.multi_handedness),
                              key=lambda x: x[0].landmark[0].x)
        
        if len(sorted_hands) == 1:
            hl, hn = sorted_hands[0]
            pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
            processed = pre_process_landmark(pts)
            if hn.classification[0].label == 'Right':
                processed = flip_keypoint_x(processed)
            data_aux.extend(processed)
            data_aux.extend([0.0] * 42)
        elif len(sorted_hands) >= 2:
            for i in range(2):
                hl = sorted_hands[i][0]
                pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
                data_aux.extend(pre_process_landmark(pts))
        
        if len(data_aux) == 84:
            prediction = model.predict(np.array([data_aux]))[0]
            conf = model.predict_proba(np.array([data_aux])).max()
            if conf > 0.7:
                st.session_state.last_pred = labels[int(prediction)]

    return frame.from_ndarray(img, format="bgr24")

# --- 5. การตั้งค่า WebRTC (แก้ Error Network) ---
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ],
        "iceTransportPolicy": "all",
    }
)

# --- 6. แสดงผล UI ---
output_container = st.empty()

ctx = webrtc_streamer(
    key="thai-sign-v2026", # เปลี่ยน Key เพื่อล้างค่าเก่า
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480}, 
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    async_processing=True,
)

# อัปเดตผลลัพธ์ตัวโตๆ
output_container.markdown(
    f"""
    <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 2px solid #00ff00; text-align: center;">
        <p style="margin: 0; font-size: 20px; color: #ffffff;">✅ ท่าทางที่พบ:</p>
        <h1 style="margin: 0; font-size: 80px; font-weight: bold; color: #00ff00;">{st.session_state.last_pred}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

if ctx.state.playing:
    st.button("ล้างคำแปล", on_click=lambda: st.session_state.update({"last_pred": "รอตรวจจับ..."}))
