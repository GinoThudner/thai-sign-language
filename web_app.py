import streamlit as st

# 1. ตั้งค่าหน้าเว็บ (ต้องเป็นบรรทัดแรก)
st.set_page_config(page_title="Thai Sign Translator", layout="centered")

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

# Queue สำหรับส่งข้อความจากกล้องมาที่ UI
result_queue = queue.Queue()

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    # ดึงคอลัมน์ที่ 2 สำหรับภาษาไทย
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
    else:
        labels_list = ["Error: No Label File"]
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันประมวลผล (Logic เดียวกับในเครื่อง) ---
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
            pts = [[int(l.x * img.shape[1]), int(l.y * img.shape[0])] for l in hl.landmark]
            processed = pre_process_landmark(pts)
            if hn.classification[0].label == 'Right':
                processed = flip_keypoint_x(processed)
            data_aux.extend(processed)
            data_aux.extend([0.0] * 42)
        elif len(sorted_hands) >= 2:
            for i in range(2):
                hl = sorted_hands[i][0]
                pts = [[int(l.x * img.shape[1]), int(l.y * img.shape[0])] for l in hl.landmark]
                data_aux.extend(pre_process_landmark(pts))
        
        if len(data_aux) == 84:
            prediction = model.predict(np.array([data_aux]))[0]
            conf = model.predict_proba(np.array([data_aux])).max()
            
            if conf > 0.7:
                res_text = labels[int(prediction)]
                # ส่งคำแปลเข้า Queue เพื่อไปโชว์ที่หน้าเว็บ
                result_queue.put(f"{res_text} (มั่นใจ {conf:.2f})")
                # วาดแค่ ID ภาษาอังกฤษในจอเพื่อกันตัวอักษรพัง
                cv2.putText(img, f"ID: {prediction}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame.from_ndarray(img, format="bgr24")

# --- 4. หน้าตาเว็บ ---
st.title("🖐️ ระบบแปลภาษามือไทย")
st.markdown("---")

# ส่วนแสดงผลคำแปล (ใช้สีเขียวขนาดใหญ่)
output_text = st.empty()
output_text.success("💡 ท่าทางที่พบ: กำลังรอการตรวจจับ...")

# เริ่มต้นกล้อง (ปิดไมค์)
webrtc_streamer(
    key="translator-v3",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Loop สำหรับดึงข้อมูลจาก Queue มาอัปเดตหน้าเว็บ
while True:
    try:
        msg = result_queue.get(timeout=1.0)
        output_text.success(f"✅ ท่าทางที่พบ: {msg}")
    except queue.Empty:
        pass
