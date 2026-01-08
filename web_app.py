import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
import queue
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# 1. ตั้งค่าหน้าเว็บ (SEO & Mobile Optimization)
st.set_page_config(
    page_title="แปลภาษามือไทยออนไลน์ - AI Sign Language Translator",
    page_icon="🖐️",
    layout="centered"
)

# 2. ส่วนหัวข้อเว็บ
st.title("🖐️ ระบบแปลภาษามือไทยแบบ Real-time")
st.markdown("รองรับทั้งคอมพิวเตอร์และมือถือ (แนะนำเปิดผ่าน Chrome หรือ Safari)")

# 3. โหลดทรัพยากร (เพิ่มระบบตรวจจับ Error ของโมเดล)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

# ใช้ Queue เพื่อส่งข้อมูลจาก Video Thread มายังหน้าจอหลัก
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

@st.cache_resource
def load_resources():
    try:
        with open(model_path, 'rb') as f:
            m = pickle.load(f)
            model_obj = m['model'] if isinstance(m, dict) else m
        
        if os.path.exists(label_path):
            df = pd.read_csv(label_path, header=None, encoding='utf-8')
            labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
        else:
            labels_list = ["No Label"]
        
        mp_hands = mp.solutions.hands
        hands_engine = mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None, [], None, None, None

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# 4. ฟังก์ชันประมวลผล Landmark (คงเดิมเพื่อความแม่นยำ)
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

# 5. ฟังก์ชัน Callback สำหรับกล้อง (แก้ลำดับให้ถูกต้อง)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Mirror
    h, w, _ = img.shape
    
    # ประมวลผล AI (ลดเฟรมเรตภายในตัว)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
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
            if conf > 0.5:
                res_thai = labels[int(prediction)]
                st.session_state.result_queue.put(res_thai)

    return frame.from_ndarray(img, format="bgr24")

# 6. UI ส่วนแสดงผล
output_placeholder = st.empty()
output_placeholder.info("💡 พร้อมใช้งาน: กรุณากด Start และอนุญาตให้เข้าถึงกล้อง")

ctx = webrtc_streamer(
    key="sign-v5-final", # เปลี่ยน Key เพื่อล้าง Cache เดิม
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "iceTransportPolicy": "all",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480}, 
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15} # ลด FrameRate ลงเพื่อประหยัด CPU มือถือ
        },
        "audio": False
    },
    async_processing=True,
)

# 7. การดึงผลลัพธ์มาแสดง (ลดภาระ CPU แทนการใช้ while True แบบเดิม)
if ctx.state.playing:
    while True:
        try:
            msg = st.session_state.result_queue.get(timeout=1.0)
            output_placeholder.markdown(
                f"""
                <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #c3e6cb;">
                    <p style="margin: 0; font-size: 18px;">✅ ท่าทางที่พบ:</p>
                    <h1 style="margin: 0; font-size: 60px; font-weight: bold;">{msg}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        except queue.Empty:
            continue
