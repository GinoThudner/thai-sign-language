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
from PIL import ImageFont, ImageDraw, Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 2. โหลดทรัพยากร ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')
font_path = os.path.join(BASE_DIR, 'tahoma.ttf') 

@st.cache_resource
def load_resources():
    # โหลดโมเดล
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    # โหลดเลเบลภาษาไทย
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
    else:
        labels_list = ["Error"]

    # โหลดฟอนต์
    try:
        font = ImageFont.truetype(font_path, 45)
    except:
        font = ImageFont.load_default()
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands, font

model, labels, hands, mp_draw, mp_hands_module, thai_font = load_resources()

# --- 3. ฟังก์ชันประมวลผล Landmark ---
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

# --- 4. ฟังก์ชันหลักสำหรับกล้อง ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # วาดเส้นมือ
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        data_aux = []
        sorted_hands = sorted(zip(results.multi_hand_landmarks, results.multi_handedness),
                              key=lambda x: x[0].landmark[0].x)
        
        # จัดการข้อมูล 1 หรือ 2 มือ
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

        # ทำนายผล
        if len(data_aux) == 84:
            prediction = model.predict(np.array([data_aux]))[0]
            conf = model.predict_proba(np.array([data_aux])).max()
            
            if conf > 0.75:
                res_thai = labels[int(prediction)]

                # --- วาดภาษาไทยลงในวิดีโอ (กึ่งกลางล่าง) ---
                try:
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    
                    # คำนวณจุดกึ่งกลาง
                    bbox = draw.textbbox((0, 0), res_thai, font=thai_font)
                    tw = bbox[2] - bbox[0]
                    tx = (w - tw) // 2
                    ty = h - 80 

                    # วาดแถบพื้นหลังดำจางๆ
                    draw.rectangle([tx - 20, ty - 5, tx + tw + 20, ty + 60], fill=(0, 0, 0, 160))
                    draw.text((tx, ty), res_thai, font=thai_font, fill=(0, 255, 0))
                    
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except:
                    # กันเหนียวถ้า Pillow พัง
                    cv2.putText(img, "Detected", (w//2-50, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame.from_ndarray(img, format="bgr24")

# --- 5. ส่วนแสดงผล UI ---
st.title("🖐️ ระบบแปลภาษามือไทย")
st.write("ยกมือขึ้นมาในเฟรมเพื่อเริ่มการแปล")

webrtc_streamer(
    key="fixed-translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}, # ปิดไมค์
    async_processing=True, # สำคัญ: ช่วยให้ลื่นไหลไม่ค้าง
)
