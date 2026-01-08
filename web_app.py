import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="แปลภาษามือไทย", page_icon="🖐️", layout="centered")

# 2. ส่วนหัวข้อ
st.title("🖐️ ระบบแปลภาษามือไทยแบบ Real-time")
st.write("รองรับทั้งคอมพิวเตอร์และมือถือ (แนะนำเปิดผ่าน Chrome หรือ Safari)")

# 3. โหลดโมเดลและเครื่องมือ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_all():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    labels_list = pd.read_csv(label_path, header=None).iloc[:, -1].astype(str).tolist()
    
    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_all()

# 4. ฟังก์ชันจัดการ Landmark
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

# 5. ตัวแปรเก็บผลลัพธ์ (ใช้ Session State เพื่อความเสถียรบนมือถือ)
if "last_pred" not in st.session_state:
    st.session_state.last_pred = "รอตรวจจับ..."

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # แก้ไขจุดที่ผิด: ต้องสร้าง RGB ก่อนเรียกใช้ AI
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
            
        # จัดการ Landmark 1 หรือ 2 มือ
        sorted_hands = sorted(zip(results.multi_hand_landmarks, results.multi_handedness),
                              key=lambda x: x[0].landmark[0].x)
        
        if len(sorted_hands) == 1:
            hl, hn = sorted_hands[0]
            pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
            processed = pre_process_landmark(pts)
            if hn.classification[0].label == 'Right':
                processed = flip_keypoint_x(processed)
            data_aux.extend(processed + [0.0] * 42)
        elif len(sorted_hands) >= 2:
            for i in range(2):
                hl = sorted_hands[i][0]
                pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
                data_aux.extend(pre_process_landmark(pts))
        
        if len(data_aux) == 84:
            prediction = model.predict(np.array([data_aux]))[0]
            st.session_state.last_pred = labels[int(prediction)]

    return frame.from_ndarray(img, format="bgr24")

# 6. แสดงผลลัพธ์
res_box = st.empty()

# 7. ตัวรับส่งวิดีโอ (WebRTC)
webrtc_streamer(
    key="universal-sign-v10", # เปลี่ยน Key เพื่อ Reset ระบบ
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"width": {"ideal": 320}, "height": {"ideal": 240}, "frameRate": {"ideal": 15}},
        "audio": False
    },
    async_processing=True,
)

# อัปเดตข้อความบนหน้าจอ
res_box.markdown(f"""
    <div style="background-color:#1e1e1e; padding:20px; border-radius:10px; text-align:center;">
        <h2 style="color:#00ff00; margin:0;">พบท่าทาง: {st.session_state.last_pred}</h2>
    </div>
""", unsafe_allow_html=True)
