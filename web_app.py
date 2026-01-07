import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd

# --- 1. ตั้งค่า Path ให้ถูกต้องตามโครงสร้างโฟลเดอร์ของคุณ ---
# โครงสร้าง: Root/src/web_app.py -> ต้องถอย 1 ระดับไปที่ Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
            if df.shape[1] > 1:
                labels_list = df.iloc[:, 1].astype(str).tolist()
            else:
                labels_list = df.iloc[:, 0].astype(str).tolist()
        else:
            # ถ้าหาไฟล์ไม่เจอ ให้ใช้ตัวเลขไปก่อนเพื่อไม่ให้ App พัง
            labels_list = [str(i) for i in range(20)] 
            st.error(f"หาไฟล์ไม่เจอที่: {label_path}")

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
        st.error(f"Error loading resources: {e}")
        return None, None, None, None, None

model, labels, hands, mp_drawing, mp_hands_module = load_all_resources()

# --- 2. ส่วนประมวลผลวิดีโอ (84 Features) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        data_aux = [0.0] * 84 
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands_module.HAND_CONNECTIONS)
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        min_x, min_y = min(x_coords), min(y_coords)

        for i in range(len(hand_landmarks.landmark)):
            data_aux[i*2] = hand_landmarks.landmark[i].x - min_x
            data_aux[i*2 + 1] = hand_landmarks.landmark[i].y - min_y

        if model:
            try:
                prediction = model.predict(np.asarray(data_aux).reshape(1, -1))
                index = int(prediction[0])
                
                # ดึงคำแปลจากลิสต์ labels
                result_text = labels[index] if index < len(labels) else f"Class {index}"
                
                # แสดงผลบนหน้าจอ
                cv2.rectangle(img, (0, 0), (450, 80), (0, 0, 0), -1) 
                cv2.putText(img, f"Result: {result_text}", (20, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            except:
                pass

    return frame.from_ndarray(img, format="bgr24")

# --- 3. UI และกล้อง ---
st.title("🖐️ Thai Sign Language Translator")
webrtc_streamer(
    key="translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
