import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd

# โหลดโมเดล
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_all_resources():
    # โหลดโมเดล
    with open(model_path, 'rb') as f:
        m_data = pickle.load(f)
        model = m_data['model'] if isinstance(m_data, dict) else m_data
    labels = pd.read_csv(label_path, header=None).iloc[:, 0].tolist()
    
    # โหลด MediaPipe เฉพาะตอนเรียกใช้ฟังก์ชันนี้
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_engine = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    return model, labels, hands_engine, mp_draw, mp_hands

# เรียกใช้งาน
model, labels, hands, mp_drawing, mp_h_module = load_all_resources()

# ... (ส่วนที่เหลือของโค้ด video_frame_callback ให้ใช้ตัวแปรจาก load_all_resources) ...
