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
st.write("สถานะ: กำลังโหลดระบบ...")

# --- 2. จัดการ Path (ใช้ BASE_DIR ชี้ไปที่โฟลเดอร์ src ที่ไฟล์นี้อยู่) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_all_resources():
    try:
        # โหลดโมเดล
        with open(model_path, 'rb') as f:
            m_data = pickle.load(f)
            model_obj = m_data['model'] if isinstance(m_data, dict) else m_data
        
        # โหลดเลเบล (ดึงคำแปลภาษาไทยคอลัมน์ที่ 2)
        if os.path.exists(label_path):
            df = pd.read_csv(label_path, header=None, encoding='utf-8')
            # ถ้ามี 2 คอลัมน์ (เช่น 0,ตกลง) ให้เอาคอลัมน์ index 1
            if df.shape[1] > 1:
                labels_list = df.iloc[:, 1].astype(str).tolist()
            else:
                labels_list = df.iloc[:, 0].astype(str).tolist()
        else:
            labels_list = [str(i) for i in range(20)]
            st.error(f"❌ ไม่พบไฟล์ Label ที่: {label_path}")

        # ตั้งค่า MediaPipe
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
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดทรัพยากร: {e}")
        return None, None, None, None, None

model, labels, hands, mp_drawing, mp_hands_module = load_all_resources()

if model:
    st.success(f"✅ ระบบพร้อมใช้งาน! (โหลดคำแปล {len(labels)} ท่า)")

# --- 3. ฟังก์ชันประมวลผลวิดีโอ (84 Features Logic) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Mirror flip
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # สร้างข้อมูล 84 ช่อง (ตามที่ RandomForest ของคุณต้องการ)
        data_aux = [0.0] * 84 
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # วาดเส้นเชื่อมมือ
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands_module.HAND_CONNECTIONS)
        
        # คำนวณพิกัด Relative
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        min_x, min_y = min(x_coords), min(y_coords)

        for i in range(len(hand_landmarks.landmark)):
            data_aux[i*2] = hand_landmarks.landmark[i].x - min_x
            data_aux[i*2 + 1] = hand_landmarks.landmark[i].y - min_y

        if model:
            try:
                # ทำนายผล
                prediction = model.predict(np.asarray(data_aux).reshape(1, -1))
                index = int(prediction[0])
                
                # ดึงชื่อท่าจาก List Labels
                if index < len(labels):
                    result_text = labels[index]
                else:
                    result_text = f"Class {index}"
                
                # แสดงผลบนหน้าจอวิดีโอ
                cv2.rectangle(img, (0, 0), (450, 80), (0, 0, 0), -1) 
                cv2.putText(img, f"Result: {result_text}", (20, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            except Exception:
                pass

    return frame.from_ndarray(img, format="bgr24")

# --- 4. ปุ่มเปิดกล้อง WebRTC ---
webrtc_streamer(
    key="thai-sign-translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
