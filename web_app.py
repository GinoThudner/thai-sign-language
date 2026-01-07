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
    # โหลดโมเดล
    try:
        with open(model_path, 'rb') as f:
            m_data = pickle.load(f)
            # รองรับทั้งแบบ dict และแบบเก็บ model ตรงๆ
            model_obj = m_data['model'] if isinstance(m_data, dict) else m_data
        
        # โหลดเลเบล
        labels_list = pd.read_csv(label_path, header=None).iloc[:, 0].tolist()
        
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
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        return None, None, None, None, None

model, labels, hands, mp_drawing, mp_hands_module = load_all_resources()

if model:
    st.success("✅ ระบบพร้อมใช้งาน! กรุณากดปุ่ม Start ด้านล่าง")
else:
    st.error("❌ ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์บน GitHub")

# --- 3. ฟังก์ชันประมวลผลวิดีโอ (Core Logic) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # กลับด้านหน้าจอให้เหมือนกระจก
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # วาดจุดเชื่อมมือ (ช่วยเช็คว่า MediaPipe ทำงานไหม)
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands_module.HAND_CONNECTIONS)
            
            data_aux = []
            x_ = []
            y_ = []

            # ดึงพิกัดจุด 21 จุด
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # ปรับพิกัดให้เป็นค่าสัมพัทธ์ (Relative coordinates)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # ทำนายผล
            if model:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    index = int(prediction[0])
                    result_text = str(labels[index])
                    
                    # --- ส่วนแสดงผลบนหน้าจอ ---
                    # วาดกล่องพื้นหลังสีดำเพื่อให้เห็นตัวหนังสือชัด
                    cv2.rectangle(img, (0, 0), (400, 80), (0, 0, 0), -1) 
                    # เขียนคำแปลสีเขียว
                    cv2.putText(img, f"Result: {result_text}", (20, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                except Exception:
                    # ถ้าโมเดลพังจะแสดง Error เล็กๆ ไว้มุมจอ
                    cv2.putText(img, "Prediction Error", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame.from_ndarray(img, format="bgr24")

# --- 4. ปุ่มเปิดกล้อง WebRTC ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="thai-sign-language",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.info("💡 ทิป: หากคำแปลไม่ขึ้น ให้ลองขยับมือให้ห่างจากกล้องพอประมาณและอยู่ในที่ที่มีแสงสว่างเพียงพอ")
