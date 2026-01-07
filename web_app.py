import streamlit as st

# 1. ต้องอยู่บรรทัดแรกสุด
st.set_page_config(page_title="Thai Sign Translator", layout="centered")

import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 2. โหลดทรัพยากร (ใช้ Cache เพื่อไม่ให้โหลดซ้ำจนค้าง) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
    else:
        labels_list = ["No Label"]
    
    mp_hands = mp.solutions.hands
    # ปรับความแม่นยำให้พอดี ไม่ให้หนักเครื่องเกินไป
    hands_engine = mp_hands.Hands(
        max_num_hands=1, # ตรวจจับแค่ 1 มือเพื่อประหยัด CPU
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันประมวลผล (เน้นความไว) ---
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
    img = cv2.flip(img, 1) # กลับด้านภาพ
    h, w, _ = img.shape
    
    # ประมวลผลมือ
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            # วาดเส้นมือด้วย OpenCV (เบาที่สุด)
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
            
            # เตรียมข้อมูล
            pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
            processed = pre_process_landmark(pts)
            
            # ทำนาย (ใส่ 0.0 เติมให้ครบ 84 features ตามสูตรของคุณ)
            input_data = processed + ([0.0] * 42)
            prediction = model.predict(np.array([input_data[:84]]))[0]
            conf = model.predict_proba(np.array([input_data[:84]])).max()
            
            if conf > 0.8:
                res_thai = labels[int(prediction)]
                # แสดง ID และชื่อท่าทางภาษาอังกฤษในจอ (กันค้าง)
                cv2.rectangle(img, (0, h-40), (200, h), (0,0,0), -1)
                cv2.putText(img, f"ID: {prediction} ({conf:.2f})", (10, h-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # ฝากค่าไว้ในคิวหรือแสดงผลที่ UI หลัก
    
    return frame.from_ndarray(img, format="bgr24")

# --- 4. ส่วนหน้าเว็บ ---
st.title("🖐️ ระบบแปลภาษามือไทย")
st.info("💡 คำแปลจะขึ้นที่แถบเขียว และตัวเลข ID จะขึ้นที่หน้าจอ")

# ใช้ WebRTC Streamer แบบประหยัดทรัพยากร
ctx = webrtc_streamer(
    key="fixed-final",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # สำคัญมาก: เพื่อไม่ให้ค้าง
)

# หมายเหตุ: การแสดงภาษาไทยบนจอกลางล่างบนระบบ Cloud มักจะทำให้ค้าง 
# แนะนำให้มองที่แถบ Success หรือใช้การดูเลข ID บนจอแทนครับ
