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
from PIL import ImageFont, ImageDraw, Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 2. โหลดทรัพยากร ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

# กำหนด Path ฟอนต์ (ต้องมีไฟล์ .ttf ในโฟลเดอร์เดียวกับโค้ดหากรันบน Cloud)
font_path = os.path.join(BASE_DIR, 'tahoma.ttf') 

# Queue สำหรับส่งข้อความจากกล้องมาที่ UI
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
        labels_list = ["Error: No Label File"]
    
    # โหลดฟอนต์ภาษาไทย
    try:
        font = ImageFont.truetype(font_path, 40)
    except:
        font = ImageFont.load_default()
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands, font

model, labels, hands, mp_draw, mp_hands_module, thai_font = load_resources()

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

def flip_keypoint_x(keypoint_list):
    flipped = list(keypoint_list)
    for i in range(0, 42, 2): flipped[i] *= -1
    return flipped

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        data_aux = []
        sorted_hands = sorted(zip(results.multi_hand_landmarks, results.multi_handedness),
                              key=lambda x: x[0].landmark[0].x)
        
        # ... (ส่วนเตรียมข้อมูล 84 features เหมือนเดิมที่คุณมี) ...
        # สมมติว่าได้ prediction และ conf มาแล้ว
        
        if conf > 0.7:
            res_thai = labels[int(prediction)]
            result_queue.put(f"{res_thai} ({conf:.2f})") # ส่งไปที่แถบเขียวด้านบน (ใช้ได้แน่นอน)

            # --- ส่วนวาดลงบนวิดีโอ (กลางล่าง) ---
            try:
                # พยายามใช้ Pillow วาดภาษาไทย
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                # คำนวณตำแหน่งกึ่งกลางด้านล่าง
                bbox = draw.textbbox((0, 0), res_thai, font=thai_font)
                text_w = bbox[2] - bbox[0]
                text_x = (w - text_w) // 2
                text_y = h - 100 

                # วาดพื้นหลังดำเพื่อให้ข้อความเด่น
                draw.rectangle([text_x - 20, text_y - 10, text_x + text_w + 20, text_y + 60], fill=(0, 0, 0, 180))
                draw.text((text_x, text_y), res_thai, font=thai_font, fill=(0, 255, 0))
                
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            except:
                # ถ้า Pillow พัง (หาฟอนต์ไทยไม่เจอ) ให้วาด ID เป็นภาษาอังกฤษที่กลางล่างแทน
                # เพื่อป้องกันไม่ให้ขึ้น ????? หรือกล่องดำ
                label_en = f"Gesture ID: {prediction}"
                cv2.rectangle(img, (w//2 - 120, h - 110), (w//2 + 120, h - 40), (0, 0, 0), -1)
                cv2.putText(img, label_en, (w//2 - 100, h - 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame.from_ndarray(img, format="bgr24")

# --- 4. หน้าตาเว็บ ---
st.title("🖐️ ระบบแปลภาษามือไทย")
st.markdown("---")

output_text = st.empty()
output_text.success("💡 ท่าทางที่พบ: กำลังรอการตรวจจับ...")

webrtc_streamer(
    key="thai-sign-final",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

while True:
    try:
        msg = result_queue.get(timeout=1.0)
        output_text.success(f"✅ ท่าทางที่พบ: {msg}")
    except queue.Empty:
        pass



