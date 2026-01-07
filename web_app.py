import streamlit as st

# 1. ต้องอยู่บรรทัดแรกเสมอ
st.set_page_config(page_title="Thai Sign Language", layout="centered")

import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 2. โหลดโมเดล (ใช้ Cache เพื่อความไว) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    # ดึงรายชื่อคำแปลภาษาไทย
    labels_list = []
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
    
    mp_hands = mp.solutions.hands
    # ปรับจูนให้ตรวจจับเร็วขึ้น
    hands_engine = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันประมวลผลหลัก ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_val = max(list(map(abs, temp_landmark_list)))
    return [n / max_val if max_val != 0 else 0 for n in temp_landmark_list]

# ใช้ session_state เก็บคำแปลเพื่อโชว์ที่ UI
if "thai_result" not in st.session_state:
    st.session_state.thai_result = "กำลังรอการตรวจจับ..."

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # แปลงสีครั้งเดียวเพื่อความเร็ว
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            # วาดเส้นมือด้วย OpenCV (เบาและไม่ทำให้ค้าง)
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
            
            # เตรียมข้อมูลส่งเข้า Model
            pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
            processed = pre_process_landmark(pts)
            
            # เติมข้อมูลให้ครบ 84 features (สำหรับ 1 มือ)
            data_aux = processed + ([0.0] * 42)
            
            prediction = model.predict(np.array([data_aux]))[0]
            conf = model.predict_proba(np.array([data_aux])).max()
            
            if conf > 0.75:
                # อัปเดตคำแปลไทยเข้าไปในตัวแปรที่จะไปโชว์ด้านบน
                st.session_state.thai_result = labels[int(prediction)]
                
                # แสดงแค่ ID ภาษาอังกฤษในวิดีโอเพื่อกันตัวหนังสือพัง
                cv2.putText(img, f"ID: {prediction} ({conf:.2f})", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame.from_ndarray(img, format="bgr24")

# --- 4. หน้าตาแอป (UI) ---
st.title("🖐️ แปลภาษามือไทย")

# แถบแสดงคำแปลขนาดใหญ่ (แสดงผลภาษาไทยได้ 100% ไม่ค้าง)
st.success(f"### คำแปลที่พบ: {st.session_state.thai_result}")

webrtc_streamer(
    key="thai-sign-stable",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # เปิดโหมดทำงานเบื้องหลังเพื่อลดอาการกระตุก
)

st.write("หมายเหตุ: หากภาพค้าง ให้ลองรีเฟรชหน้าเว็บและขยับมือให้ช้าลงครับ")
