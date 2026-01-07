import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import pickle
import numpy as np

# โหลดโมเดล
model_dict = pickle.load(open('./model.pkl', 'rb'))
model = model_dict['model']

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Thai Sign Language Translator", layout="wide")
st.title("🖐️ เครื่องแปลภาษามือไทยแบบ Real-time")
st.write("วิธีใช้งาน: อนุญาตให้เข้าถึงกล้อง และรอระบบประมวลผล")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- ส่วนนี้คือ Logic การดึงพิกัดและการทำนายเดิมของคุณ ---
                # (ใส่โค้ดการทำนายเดิมที่ดึงพิกัด 21 จุดตรงนี้)
                
                # ตัวอย่างการวาดคำแปลลงบนภาพ
                cv2.putText(img, "Predicted Text", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

webrtc_streamer(key="key", video_processor_factory=VideoProcessor)