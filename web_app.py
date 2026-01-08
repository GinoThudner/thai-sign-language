import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
import collections
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="แปลภาษามือไทยออนไลน์ - AI Sign Language Translator",
    page_icon="🖐️",
    layout="centered"
)

# --- 2. เตรียมระบบเก็บสถานะ (Session State) ---
# ใช้เก็บประวัติพิกัดสำหรับ Motion
if "history" not in st.session_state:
    st.session_state.history = collections.deque(maxlen=15)

# ใช้เก็บคำแปลล่าสุดเพื่อแสดงบนหน้าจอ (แก้ปัญหาค้าง)
if "last_msg" not in st.session_state:
    st.session_state.last_msg = "รอตรวจจับ..."

# --- 3. ส่วนหัวข้อ ---
st.title("🖐️ ระบบแปลภาษามือไทยแบบ Real-time")
st.markdown("""
* **ท่าทางนิ่ง:** ใช้ AI ทำนายตามปกติ
* **ขยับมือซ้าย-ขวา:** แปลว่า **"ไม่"**
* **มือนิ่งสนิท:** แปลว่า **"หยุด"**
""")

# --- 4. โหลดทรัพยากร ---
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
        labels_list = ["Error: No Label File"]
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 5. ฟังก์ชันประมวลผล ---
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
        # เก็บพิกัดจุดที่ 9 (โคนนิ้วกลาง)
        p9 = results.multi_hand_landmarks[0].landmark[9]
        st.session_state.history.append((p9.x, p9.y))

        motion_detected = False
        if len(st.session_state.history) == 15:
            dx = st.session_state.history[-1][0] - st.session_state.history[0][0]
            dy = st.session_state.history[-1][1] - st.session_state.history[0][1]
            speed = (dx**2 + dy**2)**0.5

            if abs(dx) > 0.12: 
                st.session_state.last_msg = "ไม่"
                motion_detected = True
            elif speed < 0.005: 
                st.session_state.last_msg = "หยุด"
                motion_detected = True

        if not motion_detected:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
            
            data_aux = []
            sorted_hands = sorted(zip(results.multi_hand_landmarks, results.multi_handedness),
                                  key=lambda x: x[0].landmark[0].x)
            
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
            
            if len(data_aux) == 84:
                prediction = model.predict(np.array([data_aux]))[0]
                conf = model.predict_proba(np.array([data_aux])).max()
                if conf > 0.6:
                    st.session_state.last_msg = labels[int(prediction)]

    return frame.from_ndarray(img, format="bgr24")

# --- 6. ส่วนการแสดงผล ---
# กล่องแสดงคำแปล (วางไว้ก่อนหน้าจอวิดีโอเพื่อให้เห็นชัด)
output_container = st.empty()
output_container.markdown(
    f"""
    <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 20px;">✅ ท่าทางที่พบ:</p>
        <h1 style="margin: 0; font-size: 70px; font-weight: bold;">{st.session_state.last_msg}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# หน้าจอวิดีโอ
webrtc_streamer(
    key="motion-detect-v2", # เปลี่ยน Key เพื่อรีเซ็ต
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640}, # ใช้ ideal แทน exact เพื่อลดปัญหาภาพลาย
            "height": {"ideal": 480}, 
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    async_processing=True,
)

# ปุ่มกดเผื่อต้องการบังคับ Refresh หน้าจอ
if st.button("รีเฟรชคำแปล"):
    st.rerun()
