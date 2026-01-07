import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Thai Sign Language App", layout="centered")
st.title("🖐️ เครื่องแปลภาษามือไทย Real-time")
st.write("สถานะ: กำลังโหลดระบบ... กรุณารอสักครู่")

# --- 2. จัดการ Path และโหลดทรัพยากร (Model & Labels) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_all_resources():
    try:
        with open(model_path, 'rb') as f:
            m_data = pickle.load(f)
            model_obj = m_data['model'] if isinstance(m_data, dict) else m_data
        
        labels_list = pd.read_csv(label_path, header=None).iloc[:, 0].tolist()
        
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands_engine = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        return model_obj, labels_list, hands_engine, mp_draw, mp_hands
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        return None, None, None, None, None

model, labels, hands, mp_drawing, mp_hands_module = load_all_resources()

if model:
    st.success(f"✅ ระบบพร้อมใช้งาน! (โหลด {len(labels)} ท่าทาง)")
else:
    st.error("❌ ไม่สามารถโหลดโมเดลได้")

# --- 3. ฟังก์ชันประมวลผลวิดีโอ ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # สร้างลิสต์ว่างรอไว้ 84 ช่อง (ค่าเริ่มต้นเป็น 0)
        data_aux = [0.0] * 84 
        
        # เราจะประมวลผลเฉพาะมือแรกที่เจอ
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands_module.HAND_CONNECTIONS)
        
        x_ = []
        y_ = []
        for i in range(len(hand_landmarks.landmark)):
            x_.append(hand_landmarks.landmark[i].x)
            y_.append(y_) # (ในโค้ดเดิมมีพิมพ์ผิดนิดหน่อย แก้เป็นด้านล่างครับ)

        # แก้ไขลูปดึงพิกัดใหม่ให้ถูกต้อง
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        min_x = min(x_coords)
        min_y = min(y_coords)

        # ใส่ข้อมูลลงใน 42 ช่องแรก
        for i in range(len(hand_landmarks.landmark)):
            data_aux[i*2] = hand_landmarks.landmark[i].x - min_x
            data_aux[i*2 + 1] = hand_landmarks.landmark[i].y - min_y
        
        # หมายเหตุ: อีก 42 ช่องที่เหลือ (ช่องที่ 42-83) จะเป็น 0.0 
        # เพื่อให้รวมกันได้ 84 features ตามที่ Random Forest ต้องการ

        if model:
            try:
                input_data = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(input_data)
                # ... (ส่วนแสดงผลเหมือนเดิม) ...
                    index = int(prediction[0])
                    
                    if index < len(labels):
                        result_text = str(labels[index])
                    else:
                        result_text = f"Class {index} (No Label)"
                    
                    # วาดผลลัพธ์
                    cv2.rectangle(img, (0, 0), (450, 80), (0, 0, 0), -1) 
                    cv2.putText(img, f"Result: {result_text}", (20, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                
                except Exception as e:
                    # แสดงสาเหตุของ Error บนหน้าจอ
                    error_msg = str(e)
                    cv2.rectangle(img, (0, 0), (640, 40), (0, 0, 255), -1)
                    cv2.putText(img, f"Error: {error_msg[:50]}", (10, 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame.from_ndarray(img, format="bgr24")

# --- 4. ปุ่มเปิดกล้อง ---
webrtc_streamer(
    key="thai-sign-language",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

