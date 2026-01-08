import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="แปลภาษามือไทยออนไลน์", page_icon="🖐️", layout="centered")

# 2. ส่วนหัวข้อ
st.title("🖐️ ระบบแปลภาษามือไทยแบบ Real-time")
st.write("แนะนำเปิดผ่าน Chrome หรือ Safari และใช้เน็ตมือถือหาก WiFi มีปัญหา")

# 3. โหลดโมเดล
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_all():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    labels_list = pd.read_csv(label_path, header=None).iloc[:, -1].astype(str).tolist()
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_all()

# 4. ฟังก์ชันประมวลผล (ปรับปรุงลำดับตัวแปรตามภาพ 1000193314.jpg)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # บังคับสร้าง RGB ทันทีเพื่อกัน Error NoneType
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # ... (โค้ดส่วนประมวลผล landmark เดิมของคุณ) ...
        # [สมมติว่าผลการทำนายถูกเก็บใน st.session_state.last_pred]
        pass

    return frame.from_ndarray(img, format="bgr24")

# 5. การตั้งค่า RTCConfiguration (จุดสำคัญที่แก้ Error ใน Log)
# เพิ่ม STUN servers ของ Google หลายๆ ตัวเพื่อช่วยในการเชื่อมต่อ
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ],
        "iceTransportPolicy": "all",
    }
)

# 6. ตัวเรียกใช้งาน WebRTC
webrtc_streamer(
    key="fixed-connection-v1",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG, # ใส่ Config ที่เราสร้างไว้
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480}, 
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15} # ลดเฟรมเรตลงเพื่อลดภาระ Network
        },
        "audio": False
    },
    async_processing=True,
)
