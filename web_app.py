import streamlit as st

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Thai Sign Language", layout="centered")

import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 2. โหลดทรัพยากร ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

result_queue = queue.Queue()

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
    else:
        labels_list = ["ไม่พบข้อมูลคำแปล"]
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันประมวลผล ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_val = max(list(map(abs, temp_landmark_list)))
    return [n / max_val if max_val != 0 else 0 for n in temp_landmark_list]

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            # วาดแค่เส้น Skeleton เพื่อให้เห็นการทำงานของ AI
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
            
            pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
            processed = pre_process_landmark(pts)
            data_aux = processed + ([0.0] * 42)
            
            prediction = model.predict(np.array([data_aux[:84]]))[0]
            conf = model.predict_proba(np.array([data_aux[:84]])).max()
            
            if conf > 0.75:
                res_thai = labels[int(prediction)]
                # ส่งเฉพาะคำแปลเข้า Queue (ไม่ต้องวาดลง img แล้ว)
                result_queue.put(res_thai)

    return frame.from_ndarray(img, format="bgr24")

# --- 4. หน้าจอ UI ---
st.markdown("<h1 style='text-align: center;'>🖐️ ระบบแปลภาษามือไทย</h1>", unsafe_allow_html=True)

# สร้างพื้นที่สำหรับช่องเขียวขนาดใหญ่พิเศษ
output_placeholder = st.empty()
output_placeholder.markdown(
    "<div style='background-color: #1e3d2f; padding: 20px; border-radius: 10px; text-align: center;'>"
    "<h1 style='color: white; margin: 0;'>กำลังรอการตรวจจับ...</h1>"
    "</div>", 
    unsafe_allow_html=True
)

webrtc_streamer(
    key="big-font-app",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- 5. ลูปดึงค่ามาอัปเดตตัวอักษรขนาดใหญ่ ---
while True:
    try:
        # ดึงคำแปลล่าสุด
        msg = result_queue.get(timeout=0.1)
        # อัปเดตช่องสีเขียวด้วย HTML เพื่อให้ตัวหนังสือใหญ่พิเศษ
        output_placeholder.markdown(
            f"<div style='background-color: #28a745; padding: 30px; border-radius: 10px; text-align: center; border: 2px solid white;'>"
            f"<p style='color: white; font-size: 20px; margin-bottom: 5px;'>คำแปลที่พบ:</p>"
            f"<h1 style='color: white; font-size: 80px; font-weight: bold; margin: 0;'>{msg}</h1>"
            f"</div>", 
            unsafe_allow_html=True
        )
    except queue.Empty:
        continue
