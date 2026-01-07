import streamlit as st

# 1. ตั้งค่าหน้าเว็บ (ต้องเป็นบรรทัดแรกเสมอ)
st.set_page_config(page_title="Thai Sign Language", layout="centered")

import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
import queue  # สำคัญ: ใช้สำหรับส่งข้อมูลจากกล้องมาที่หน้าจอ
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 2. โหลดทรัพยากร ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

# สร้าง Queue เพื่อรับคำแปลจากวิดีโอ
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
        labels_list = ["No Labels Found"]
    
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
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
            
            pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
            processed = pre_process_landmark(pts)
            
            # เตรียมข้อมูล 84 features (1 มือ)
            data_aux = processed + ([0.0] * 42)
            
            prediction = model.predict(np.array([data_aux[:84]]))[0]
            conf = model.predict_proba(np.array([data_aux[:84]])).max()
            
            if conf > 0.75:
                res_thai = labels[int(prediction)]
                # ส่งคำแปลเข้า Queue
                result_queue.put(f"{res_thai} (มั่นใจ {conf:.2f})")
                
                # แสดง ID บนจอวิดีโอ
                cv2.putText(img, f"ID: {prediction}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame.from_ndarray(img, format="bgr24")

# --- 4. ส่วนหน้าจอแอป (UI) ---
st.title("🖐️ ระบบแปลภาษามือไทย")

# สร้างช่องเขียวแบบ Dynamic (จองพื้นที่ไว้)
output_placeholder = st.empty()
output_placeholder.success("💡 ท่าทางที่พบ: กำลังรอการตรวจจับ...")

webrtc_streamer(
    key="fixed-final-app",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- 5. ลูปดึงค่าจาก Queue มาแสดงผล ---
while True:
    try:
        # ดึงข้อความล่าสุดจาก Queue (รอไม่เกิน 0.1 วินาที)
        result_text = result_queue.get(timeout=0.1)
        # อัปเดตช่องสีเขียวทันทีที่ได้รับข้อมูล
        output_placeholder.success(f"### ✅ ท่าทางที่พบ: {result_text}")
    except queue.Empty:
        # หากไม่มีข้อมูลใหม่ ไม่ต้องทำอะไร
        continue
